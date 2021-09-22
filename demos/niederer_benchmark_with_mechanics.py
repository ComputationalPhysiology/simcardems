try:
    import petsc4py
except:
    print("Cannot import petsc4py")

from dolfin import *
import dolfin
from cbcbeat import *
from ORdmm_Land import ORdmm_Land
import pulse
from ORdmm_Land_em_coupling import ORdmm_Land_em_coupling
import sys
import numpy
import numpy as np

# args = (
#     [sys.argv[0]]
#     + """
#                        --petsc.ksp_type cg
#                        --petsc.pc_type gamg
#                        --petsc.pc_gamg_verbose 10
#                        --petsc.pc_gamg_square_graph 0
#                        --petsc.pc_gamg_coarse_eq_limit 3000
#                        --petsc.mg_coarse_pc_type redundant
#                        --petsc.mg_coarse_sub_pc_type lu
#                        --petsc.mg_levels_ksp_type richardson
#                        --petsc.mg_levels_ksp_max_it 3
#                        --petsc.mg_levels_pc_type sor
#                        """.split()
# )
# parameters.parse(argv=args)

parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["representation"] = "uflacs"


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
    M = as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M


def setup_mechanics_model(mesh):
    left = dolfin.CompiledSubDomain("near(x[0], side) && on_boundary", side=0)
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    left_marker = 1
    left.mark(boundary_markers, left_marker)
    marker_functions = pulse.MarkerFunctions(ffun=boundary_markers)

    # Create mictrotructure
    f0 = dolfin.Expression(("1.0", "0.0", "0.0"), degree=1, cell=mesh.ufl_cell())
    s0 = dolfin.Expression(("0.0", "1.0", "0.0"), degree=1, cell=mesh.ufl_cell())
    n0 = dolfin.Expression(("0.0", "0.0", "1.0"), degree=1, cell=mesh.ufl_cell())

    # Collect the mictrotructure
    microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

    # Create the geometry
    geometry = pulse.Geometry(
        mesh=mesh, marker_functions=marker_functions, microstructure=microstructure
    )

    # Create the material
    material_parameters = pulse.Guccione.default_parameters()
    material_parameters["CC"] = 2.0
    material_parameters["bf"] = 8.0
    material_parameters["bfs"] = 4.0
    material_parameters["bt"] = 2.0
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    Ta = dolfin.Function(V)
    # Ta = dolfin.Constant(0.0)
    material = pulse.Guccione(
        params=material_parameters, active_model="active_stress", activation=Ta
    )

    # Define Dirichlet boundary. Fix at the left boundary
    def dirichlet_bc(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        return dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), left)

    neumann_bc = pulse.NeumannBC(traction=dolfin.Constant(0.0), marker=0)

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,))
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    problem.solve()

    return problem, Ta


def setup_ep_model(cellmodel, mesh):
    """Set-up cardiac model based on benchmark parameters."""

    # Define time
    time = Constant(0.0)

    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Mark stimulation region defined as [0, L]^3
    S1_marker = 1
    L = 1.5
    S1_subdomain = CompiledSubDomain(
        "x[0] <= L + DOLFIN_EPS && x[1] <= L + DOLFIN_EPS && x[2] <= L + DOLFIN_EPS",
        L=L,
    )
    S1_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_subdomain.mark(S1_markers, S1_marker)

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    duration = 2.0  # ms
    A = 50000.0  # mu A/cm^3
    cm2mm = 10.0
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms
    I_s = Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )
    # Store input parameters in cardiac model
    stimulus = Markerwise((I_s,), (1,), S1_markers)
    heart = CardiacModel(mesh, time, M, None, cellmodel, stimulus)

    return heart


def run_splitting_solver(mesh, application_parameters):

    # Extract parameters
    T = application_parameters["T"]
    dt = application_parameters["dt"]
    dx = application_parameters["dx"]
    theta = application_parameters["theta"]
    scheme = application_parameters["scheme"]
    preconditioner = application_parameters["preconditioner"]
    store = application_parameters["store"]
    casedir = application_parameters["casedir"]

    # cell model defined by benchmark specifications
    # CellModel = Tentusscher_panfilov_2006_epi_cell
    # CellModel = ORdmm_Land
    CellModel = ORdmm_Land_em_coupling

    # Set-up solver
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = theta
    ps["MonodomainSolver"]["preconditioner"] = preconditioner
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = scheme

    # Disable adjoint annotating and recording (saves memory)
    import cbcbeat

    if cbcbeat.dolfin_adjoint:
        parameters["adjoint"]["stop_annotating"] = True

    # Customize cell model parameters based on benchmark specifications
    # cell_inits = cell_model_initial_conditions()
    cell_inits = CellModel.default_initial_conditions()
    cellmodel = CellModel(init_conditions=cell_inits)

    # Set-up cardiac model
    ep_heart = setup_ep_model(cellmodel, mesh)
    mech_heart, Ta = setup_mechanics_model(mesh)

    # Set-up solver and time it
    timer = Timer("SplittingSolver: setup")
    solver = SplittingSolver(ep_heart, ps)
    timer.stop()

    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    # Set-up separate potential function for post processing
    VS0 = vs.function_space().sub(0)
    V = VS0.collapse()
    v = Function(V)
    # Ta = Function(V)

    # Set-up object to optimize assignment from a function to subfunction
    assigner = FunctionAssigner(V, VS0)
    assigner.assign(v, vs_.sub(0))

    # Output some degrees of freedom
    total_dofs = vs.function_space().dim()
    pde_dofs = V.dim()
    if MPI.rank(MPI.comm_world) == 0:
        print("Total degrees of freedom: ", total_dofs)
        print("PDE degrees of freedom: ", pde_dofs)

    t0 = 0.0

    # Store initial v
    if store:
        vfile = HDF5File(mesh.mpi_comm(), "%s/v.h5" % casedir, "w")
        vfile.write(v, "/function", t0)
        vfile.write(mesh, "/mesh")

    Tafile = dolfin.XDMFFile(mesh.mpi_comm(), f"{casedir}/Ta.xdmf")
    ufile = dolfin.XDMFFile(mesh.mpi_comm(), f"{casedir}/u.xdmf")
    voltfile = dolfin.XDMFFile(mesh.mpi_comm(), f"{casedir}/voltage.xdmf")

    # Solve
    timer = Timer("SplittingSolver: solve and store")
    solutions = solver.solve((t0, T), dt)
    # pulse.iterate.logger.setLevel(10)
    # pulse.solver.logger.setLevel(10)

    for (i, ((t0, t1), fields)) in enumerate(solutions):
        print("Reached t=%g/%g, dt=%g" % (t0, T, dt))
        if store:
            v_, *s_ = vs.split()
            assigner.assign(v, vs.sub(0))
            vfile.write(v, "/function", t1)
            vfile.flush()

            # Lets only run the mechanics if the change is large
            new_Ta = dolfin.project(cellmodel.Ta(v_, s_, t1), V)
            # FIXME: This would probably fail if we run it with mpirun
            print(
                "diff = ",
                np.linalg.norm(Ta.vector().get_local() - new_Ta.vector().get_local()),
            )
            if (
                np.linalg.norm(Ta.vector().get_local() - new_Ta.vector().get_local())
            ) < 1.0:
                continue

            pulse.iterate.iterate(
                mech_heart,
                Ta,
                new_Ta,
                max_nr_crash=2,
            )
            Tafile.write(Ta, t1)
            u, p = mech_heart.state.split()
            ufile.write(u, t1)
            voltfile.write(v, t1)

    if store:
        vfile.close()

    timer.stop()

    return vs


def create_mesh(dx, refinements=0):
    # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
    # with resolution prescribed by benchmark or more refinements

    # Define geometry parameters
    Lx = 20.0  # mm
    Ly = 7.0  # mm
    Lz = 3.0  # mm

    N = lambda v: int(numpy.rint(v))
    mesh = BoxMesh(
        MPI.comm_world,
        Point(0.0, 0.0, 0.0),
        Point(Lx, Ly, Lz),
        N(Lx / dx),
        N(Ly / dx),
        N(Lz / dx),
    )

    for i in range(refinements):
        print("Performing refinement", i + 1)
        mesh = refine(mesh, redistribute=False)

    return mesh


def forward(application_parameters):

    # Create mesh
    dx = application_parameters["dx"]
    R = application_parameters["refinements"]
    mesh = create_mesh(dx, R)

    # Run solver
    vs = run_splitting_solver(mesh, application_parameters)
    print("Results stored in %s" % application_parameters["casedir"])

    return vs


def init_application_parameters():
    begin("Setting up application parameters")
    application_parameters = Parameters("Niederer-benchmark")
    application_parameters.add("casedir", "new_results")
    application_parameters.add("theta", 0.5)
    application_parameters.add("store", True)
    application_parameters.add("dt", 0.05)
    application_parameters.add("dx", 0.5)
    application_parameters.add("T", 1000.0)
    application_parameters.add("scheme", "GRL1")
    application_parameters.add("preconditioner", "sor")
    application_parameters.add("refinements", 0)
    application_parameters.parse()
    end()

    return application_parameters


if __name__ == "__main__":

    # Default application parameters and parse from command-line
    application_parameters = init_application_parameters()
    application_parameters.parse()

    # Solve benchmark problem with given specifications
    timer = Timer("Total forward time")
    vs = forward(application_parameters)
    timer.stop()

    # List timings
    list_timings(TimingClear.keep, [TimingType.wall])
