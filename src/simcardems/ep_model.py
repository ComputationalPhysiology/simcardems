import json
from pathlib import Path

import cbcbeat
import dolfin

from .ORdmm_Land import ORdmm_Land as CellModel


def define_conductivity_tensor(chi, C_m):

    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019  # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conducitivites by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    M = dolfin.as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M


def setup_ep_model(cellmodel, mesh):
    """Set-up cardiac model based on benchmark parameters."""

    # Define time
    time = dolfin.Constant(0.0)

    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Mark stimulation region defined as [0, L]^3
    S1_marker = 1
    S1_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_markers.set_all(S1_marker)  # Mark the whole mesh

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    duration = 2.0  # ms
    A = 50000.0  # mu A/cm^3
    cm2mm = 10.0
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms
    PCL = 1000  # Pacing cycle length [ms]
    stimulus_times = list(range(0, 20000, PCL))  # Stimulus applied at each
    # stimulus_times = [0,20,40] # Stimulus applied at t=0, t=20 and t=40ms
    s = "0.0"
    for t in reversed(stimulus_times):
        s = (
            "(time >= start + "
            + str(t)
            + " ? (time <= (duration + start + "
            + str(t)
            + ") ? amplitude : "
            + s
            + ") : 0.0 )"
        )
    I_s = dolfin.Expression(
        s,
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )
    # Store input parameters in cardiac model
    stimulus = cbcbeat.Markerwise((I_s,), (S1_marker,), S1_markers)

    petsc_options = [
        ["ksp_type", "cg"],
        ["pc_type", "gamg"],
        ["pc_gamg_verbose", "10"],
        ["pc_gamg_square_graph", "0"],
        ["pc_gamg_coarse_eq_limit", "3000"],
        ["mg_coarse_pc_type", "redundant"],
        ["mg_coarse_sub_pc_type", "lu"],
        ["mg_levels_ksp_type", "richardson"],
        ["mg_levels_ksp_max_it", "3"],
        ["mg_levels_pc_type", "sor"],
    ]
    for opt in petsc_options:
        dolfin.PETScOptions.set(*opt)

    heart = cbcbeat.CardiacModel(
        domain=mesh,
        time=time,
        M_i=M,
        M_e=None,
        cell_models=cellmodel,
        stimulus=stimulus,
        applied_current=None,
    )

    return heart


def setup_splitting_solver_parameters(
    dt,
    theta=0.5,
    preconditioner="sor",
    scheme="GRL1",
):
    ps = cbcbeat.SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = theta
    ps["MonodomainSolver"]["preconditioner"] = preconditioner
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    # ps["BasicCardiacODESolver"]["scheme"] = scheme
    ps["CardiacODESolver"]["scheme"] = scheme
    # ps["ode_solver_choice"] = "BasicCardiacODESolver"
    # ps["BasicCardiacODESolver"]["V_polynomial_family"] = "CG"
    # ps["BasicCardiacODESolver"]["V_polynomial_degree"] = 1
    # ps["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
    # ps["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
    return ps


def setup_solver(
    mesh,
    dt,
    coupling,
    scheme="GRL1",
    theta=0.5,
    preconditioner="sor",
    cell_params=None,
    cell_inits=None,
    cell_init_file=None,
):
    ps = setup_splitting_solver_parameters(
        theta=theta,
        preconditioner=preconditioner,
        dt=dt,
        scheme=scheme,
    )

    cell_params_ = CellModel.default_parameters()
    if cell_params is not None:
        cell_params_.update(cell_params)

    cell_inits_ = CellModel.default_initial_conditions()
    if cell_init_file != "":
        if Path(cell_init_file).suffix == ".json":
            with open(cell_init_file, "r") as fid:
                d = json.load(fid)
            cell_inits_.update(d)
        else:
            from .save_load_functions import load_initial_condions_from_h5

            assert Path(cell_init_file).suffix == ".h5", "Expecting .h5 format"
            cell_inits = load_initial_condions_from_h5(cell_init_file)

    if cell_inits is not None:
        cell_inits_.update(cell_inits)

    cell_inits_["lmbda"] = coupling.lmbda
    cell_inits_["Zetas"] = coupling.Zetas
    cell_inits_["Zetaw"] = coupling.Zetaw

    cellmodel = CellModel(init_conditions=cell_inits, params=cell_params)

    # Set-up cardiac model
    ep_heart = setup_ep_model(cellmodel, mesh)
    timer = dolfin.Timer("SplittingSolver: setup")

    solver = cbcbeat.SplittingSolver(ep_heart, ps)

    timer.stop()
    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    # Output some degrees of freedom
    total_dofs = vs.function_space().dim()
    # pde_dofs = V.dim()
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Total degrees of freedom: ", total_dofs)
        # print("PDE degrees of freedom: ", pde_dofs)

    return solver
