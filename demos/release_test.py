from pathlib import Path

import cbcbeat
import dolfin
import pulse
from simcardems import em_model
from simcardems import ep_model
from simcardems import mechanics_model
from simcardems import save_load_functions as io
from simcardems import utils
from simcardems.datacollector import DataCollector
from simcardems.postprocess import plot_state_traces
from tqdm import tqdm

here = Path(__file__).parent.absolute()


def main():

    outdir = "results_release"
    add_release = True
    T = 200
    T_release = 100
    dx = 0.2
    dt = 0.02
    bnd_cond = "dirichlet"
    Lx = 2.0
    Ly = 0.7
    Lz = 0.3
    reset_state = True

    cell_init_file = here.joinpath(
        "initial_conditions",
    ).joinpath("init_5000beats_varlmbda.json")

    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"

    # Disable warnings
    dolfin.set_log_level(40)

    if add_release and bnd_cond != "dirichlet":
        raise RuntimeError(
            "Release can only be added while using dirichlet boundary conditions.",
        )

    state_path = Path(outdir).joinpath("state.h5")

    if not reset_state and state_path.is_file():
        # Load state
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            print("Load previously saved state")
        with dolfin.Timer("[demo] Load previously saved state"):
            coupling, solver, mech_heart, bnd_right_x, mesh, t0 = io.load_state(
                state_path,
            )
    else:
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            print("Create a new state")
        # Create a new state
        with dolfin.Timer("[demo] Create mesh"):
            mesh = utils.create_boxmesh(Lx=Lx, Ly=Ly, Lz=Lz, dx=dx)

        coupling = em_model.EMCoupling(mesh)

        # Set-up solver and time it
        solver = ep_model.setup_solver(
            mesh=mesh,
            dt=dt,
            coupling=coupling,
            cell_init_file=cell_init_file,
        )

        coupling.register_ep_model(solver)

        with dolfin.Timer("[demo] Setup Mech solver"):
            pre_stretch = dolfin.Constant(0.1)
            mech_heart = mechanics_model.setup_mechanics_model(
                mesh=mesh,
                coupling=coupling,
                dt=dt,
                bnd_cond=bnd_cond,
                pre_stretch=pre_stretch,
                cell_params=solver.ode_solver._model.parameters(),
            )
        t0 = 0

    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(f"Starting at t0={t0}")

    vs = solver.solution_fields()[1]
    v, v_assigner = utils.setup_assigner(vs, 0)
    Ca, Ca_assigner = utils.setup_assigner(vs, 45)

    pre_XS, preXS_assigner = utils.setup_assigner(vs, 40)
    pre_XW, preXW_assigner = utils.setup_assigner(vs, 41)

    u_subspace_index = 1 if bnd_cond == "rigid" else 0
    u, u_assigner = utils.setup_assigner(mech_heart.state, u_subspace_index)
    u_assigner.assign(u, mech_heart.state.sub(u_subspace_index))

    collector = DataCollector(outdir, mesh, reset_state=reset_state)
    for name, f in [
        ("u", u),
        ("V", v),
        ("Ca", Ca),
        ("lmbda", coupling.lmbda),
        ("Ta", mech_heart.material.active.Ta_current),
    ]:
        collector.register(name, f)

    time_stepper = cbcbeat.utils.TimeStepper((t0, T), dt, annotate=False)
    save_it = int(1 / dt)  # Save every millisecond

    pbar = tqdm(time_stepper, total=round((T - t0) / dt))
    for (i, (t0, t1)) in enumerate(pbar):

        # Solve EP model
        with dolfin.Timer("[demo] Solve EP model"):
            solver.step((t0, t1))

        # Update these states that are needed in the Mechanics solver
        with dolfin.Timer("[demo] Update mechanics"):
            coupling.update_mechanics()

        with dolfin.Timer("[demo] Compute norm"):
            XS_norm = utils.compute_norm(coupling.XS, pre_XS)
        XW_norm = utils.compute_norm(coupling.XW, pre_XW)

        pbar.set_postfix(
            {
                "XS_norm + XW_norm (solve mechanics if >=0.1)": "{:.2f}".format(
                    XS_norm + XW_norm,
                ),
            },
        )
        if XS_norm + XW_norm >= 0.1:

            preXS_assigner.assign(pre_XS, utils.sub_function(vs, 40))
            preXW_assigner.assign(pre_XW, utils.sub_function(vs, 41))

            # Solve the Mechanics model
            with dolfin.Timer("[demo] Solve mechanics"):
                mech_heart.solve()

            if add_release and t0 >= T_release:
                print("Release")
                pulse.iterate.iterate(mech_heart, bnd_right_x, -0.02 * Lx)

            # Update previous
            mech_heart.material.active.update_prev()
            with dolfin.Timer("[demo] Update EP"):
                coupling.update_ep()

        with dolfin.Timer("[demo] Update vs"):
            solver.vs_.assign(solver.vs)
        # Store every 2 millisecond
        if i % save_it == 0:
            with dolfin.Timer("[demo] Assign u,v and Ca for storage"):
                # Assign u, v and Ca for postprocessing
                v_assigner.assign(v, utils.sub_function(vs, 0))
                Ca_assigner.assign(Ca, utils.sub_function(vs, 45))
                u_assigner.assign(u, mech_heart.state.sub(u_subspace_index))
            with dolfin.Timer("[demo] Store solutions"):
                collector.store(t0)

    with dolfin.Timer("[demo] Save state"):
        io.save_state(
            state_path,
            solver=solver,
            mech_heart=mech_heart,
            dt=dt,
            bnd_cond=bnd_cond,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            t0=t0,
        )

    plot_state_traces(collector.results_file)

    time_table = dolfin.timings(dolfin.TimingClear.keep, [dolfin.TimingType.user])
    print("time table = ", time_table.str(True))
    with open(outdir + "_timings.log", "w+") as out:
        out.write(time_table.str(True))


if __name__ == "__main__":
    main()
