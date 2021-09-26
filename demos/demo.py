import argparse
from pathlib import Path

import cbcbeat
import dolfin
import matplotlib.pyplot as plt
import numpy as np
import pulse
from simcardems import DataCollector
from simcardems import DataLoader
from simcardems import em_model
from simcardems import ep_model
from simcardems import mechanics_model
from simcardems import save_load_functions as io
from simcardems import utils
from simcardems.postprocess import Analysis
from simcardems.postprocess import Boundary
from tqdm import tqdm


here = Path(__file__).parent.absolute()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        default="results/result1",
        type=str,
        help="define output directory",
    )
    parser.add_argument("--dt", default=0.02, type=float, help="define delta t")
    parser.add_argument(
        "-T",
        "--endtime",
        default=2000,
        type=int,
        help="define the endtime of simulation",
    )
    parser.add_argument(
        "--bnd_cond",
        default="dirichlet",
        type=str,
        choices=["dirichlet", "rigid"],
        help="choose boundary conditions",
    )
    parser.add_argument(
        "--reset_state",
        default=True,
        type=bool,
        choices=[True, False],
        help="define if state should be loaded (True) or newly created (False)",
    )
    parser.add_argument(
        "-IC",
        "--cell_init_file",
        default="",
        type=str,
        help="If reset_state=True, define filename of initial conditions (json or h5 file)",
    )
    parser.add_argument(
        "--add_release",
        type=bool,
        default=False,
        choices=[True, False],
        help="define if sudden release should be added",
    )
    parser.add_argument(
        "--T_release",
        type=int,
        default=150,
        help="define time to apply sudden release",
    )
    return parser


dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
dolfin.parameters["form_compiler"]["representation"] = "uflacs"

# Disable warnings
dolfin.set_log_level(40)


def main(
    outdir="results",
    add_release=True,
    T=200,
    T_release=100,
    dx=0.2,
    dt=0.02,
    bnd_cond="dirichlet",
    Lx=2.0,
    Ly=0.7,
    Lz=0.3,
    reset_state=True,
    cell_init_file=None,
):
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

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
            mech_heart, bnd_right_x = mechanics_model.setup_mechanics_model(
                mesh=mesh,
                coupling=coupling,
                dt=dt,
                bnd_cond=bnd_cond,
                cell_params=solver.ode_solver._model.parameters(),
                Lx=Lx,
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

    # code you want to profile

    profiler.stop()

    profiler.print()
    exit()
    pbar = tqdm(time_stepper, total=round((T - t0) / dt))
    for (i, (t0, t1)) in enumerate(pbar):

        # Solve EP model
        with dolfin.Timer("[demo] Solve EP model"):
            solver.step((t0, t1))

        # Extract EP solutions (not needed - vs (pointer) defined earlier)
        # with dolfin.Timer("[demo] Extract EP solution"):
        #     vs = solver.solution_fields()[1]

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


def postprocess(outdir):

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for add_release in [True, False]:
        results_file = Path(outdir).joinpath("results.h5")
        if not results_file.is_file():
            continue

        loader = DataLoader(results_file)
        bnd = Boundary(loader.mesh)

        names = ["lmbda", "Ta", "V", "Ca"]

        values = {name: np.zeros(len(loader.time_stamps)) for name in names}

        for i, t in enumerate(loader.time_stamps):
            for name, val in values.items():
                func = loader.get(name, t)
                dof_coords = func.function_space().tabulate_dof_coordinates()
                dof = np.argmin(
                    np.linalg.norm(dof_coords - np.array(bnd.center), axis=1),
                )
                if np.isclose(dof_coords[dof], np.array(bnd.center)).all():
                    # If we have a dof at the center - evaluation at dof (cheaper)
                    val[i] = func.vector().get_local()[dof]
                else:
                    # Otherwise, evaluation at center coordinates
                    val[i] = func(bnd.center)

        times = np.array(loader.time_stamps, dtype=float)

        if times[-1] > 4000:
            Analysis.plot_peaks(outdir, values["Ca"], 0.0002)

        ax[0, 0].plot(times[1:], values["lmbda"][1:], label=f"release: {add_release}")
        ax[0, 1].plot(times[1:], values["Ta"][1:], label=f"release: {add_release}")
        ax[1, 0].plot(times, values["V"], label=f"release: {add_release}")
        ax[1, 1].plot(times[1:], values["Ca"][1:], label=f"release: {add_release}")

    ax[0, 0].set_title(r"$\lambda$")
    ax[0, 1].set_title("Ta")
    ax[1, 0].set_title("V")
    ax[1, 1].set_title("Ca")
    for axi in ax.flatten():
        axi.grid()
        axi.legend()
        axi.set_xlim([0, 5000])
    ax[1, 0].set_xlabel("Time [ms]")
    ax[1, 1].set_xlabel("Time [ms]")
    ax[0, 0].set_ylim(
        [min(0.9, min(values["lmbda"][1:])), max(1.1, max(values["lmbda"][1:]))],
    )

    # plt.show()
    fig.savefig(outdir + ".png", dpi=300)


if __name__ == "__main__":

    parser = get_parser()
    args = vars(parser.parse_args())

    cell_init_file = args["cell_init_file"] or None
    if args["reset_state"] and cell_init_file:
        cell_init_file = here.parent.joinpath(
            "initial_conditions",
        ).joinpath(cell_init_file)

    main(
        args["outdir"],
        T=args["endtime"],
        T_release=args["T_release"],
        bnd_cond=args["bnd_cond"],
        add_release=args["add_release"],
        cell_init_file=cell_init_file,
        reset_state=args["reset_state"],
    )
    time_table = dolfin.timings(dolfin.TimingClear.keep, [dolfin.TimingType.user])
    print("time table = ", time_table.str(True))
    with open(args["outdir"] + "_timings.log", "w+") as out:
        out.write(time_table.str(True))

    postprocess(args["outdir"])
