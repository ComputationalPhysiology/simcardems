import dolfin
import simcardems
from simcardems.ORdmm_Land import ORdmm_Land as CellModel
import cbcbeat
import pulse
import numpy as np
import matplotlib.pyplot as plt

Lx = 20.0
Ly = 7.0
Lz = 3.0


def run_benchmark():

    # Define time
    time = dolfin.Constant(0.0)
    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # mu F / mm^2

    M = simcardems.ep_model.define_conductivity_tensor(chi, C_m)

    geometry = simcardems.geometry.SlabGeometry()

    coupling = simcardems.em_model.EMCoupling(geometry)

    S1_marker = 1
    L = 1.5
    S1_subdomain_str = f"x[0] <= {L} + DOLFIN_EPS && x[1] <= {L} + DOLFIN_EPS && x[2] <= {L} + DOLFIN_EPS"
    S1_subdomain = dolfin.CompiledSubDomain(S1_subdomain_str)
    S1_markers = dolfin.MeshFunction(
        "size_t",
        geometry.ep_mesh,
        geometry.ep_mesh.topology().dim(),
    )
    S1_markers.set_all(0)
    S1_subdomain.mark(S1_markers, S1_marker)

    duration = 2.0  # ms
    A = 50000.0  # mu A/cm^3
    cm2mm = 10.0
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms
    I_s = dolfin.Expression(
        "time >= start ? (time <= (duration + start) ? amplitude : 0.0) : 0.0",
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        degree=0,
    )
    stimulus = cbcbeat.Markerwise((I_s,), (1,), S1_markers)

    cell_inits = simcardems.ep_model.handle_cell_inits()
    cell_inits["lmbda"] = coupling.lmbda_ep
    cell_inits["Zetas"] = coupling.Zetas_ep
    cell_inits["Zetaw"] = coupling.Zetaw_ep

    cellmodel = CellModel(init_conditions=cell_inits)
    ep_heart = cbcbeat.CardiacModel(
        geometry.ep_mesh,
        time,
        M,
        None,
        cellmodel,
        stimulus,
    )

    dt = 0.05
    ps = cbcbeat.SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = 0.5
    ps["MonodomainSolver"]["preconditioner"] = "sor"
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = 0.5
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    ps["CardiacODESolver"]["scheme"] = "GRL1"
    solver = cbcbeat.SplittingSolver(ep_heart, ps)
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    coupling.register_ep_model(solver)

    # Create the material
    # material_parameters = pulse.HolzapfelOgden.default_parameters()
    # Use parameters from Biaxial test in Holzapfel 2019 (Table 1).
    material_parameters = dict(
        a=2.28,
        a_f=1.686,
        b=9.726,
        b_f=15.779,
        a_s=0.0,
        b_s=0.0,
        a_fs=0.0,
        b_fs=0.0,
    )

    V = dolfin.FunctionSpace(coupling.mech_mesh, "CG", 1)
    active_model = simcardems.land_model.LandModel(
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
        eta=0,
        parameters=solver.ode_solver._model.parameters(),
        XS=coupling.XS_mech,
        XW=coupling.XW_mech,
        function_space=V,
        mesh=coupling.mech_mesh,
    )
    material = pulse.HolzapfelOgden(
        active_model=active_model,
        parameters=material_parameters,
    )

    problem = simcardems.mechanics_model.create_slab_problem(
        material=material,
        geo=geometry,
        bnd_cond="dirichlet",
    )

    problem.solve()

    runner = simcardems.Runner.from_models(
        coupling=coupling,
        ep_solver=solver,
        mech_heart=problem,
        geo=geometry,
    )
    runner.outdir = "benchmark"
    runner.solve(T=1000)


def postprocess():
    simcardems.postprocess.plot_state_traces("benchmark/results.h5", "center")
    simcardems.postprocess.make_xdmffiles("benchmark/results.h5")

    loader = simcardems.DataLoader("benchmark/results.h5")

    points = np.zeros((loader.size, 3))
    for i, t in enumerate(loader.time_stamps):
        u = loader.get("mechanics", "u", t)
        points[i, :] = u(Lx, Ly, Lz)

    times = np.array(loader.time_stamps).astype(float)
    fig, ax = plt.subplots()
    ax.plot(times, points[:, 0], label="$u_x$")
    ax.plot(times, points[:, 1], label="$u_y$")
    ax.plot(times, points[:, 2], label="$u_z$")
    ax.legend()
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Displacement")
    ax.set_title(f"Displacment of corner located at (x,y,z)={(Lx,Ly,Lz)}")
    fig.savefig("benchmark/displacement_corner.png")

    plt.show()


def main():
    run_benchmark()
    # postprocess()


if __name__ == "__main__":
    main()
