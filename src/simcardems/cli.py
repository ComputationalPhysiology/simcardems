import json
import typing
from pathlib import Path

import cbcbeat
import click
import dolfin
from tqdm import tqdm

from . import em_model
from . import ep_model
from . import mechanics_model
from . import save_load_functions as io
from . import utils
from .datacollector import DataCollector
from .postprocess import plot_state_traces


def check_json_path(path):
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find file {path}")
    if not path.suffix == ".json":
        raise ValueError("Invalid file type {path.suffix}, expected .json")


def load_json(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    json_file = click.prompt("Path and name of json file", type=str)
    path = Path(json_file)
    check_json_path(path)
    with open(path, "r") as f:
        kwargs = json.load(f)
        print(kwargs)
    # use some click option to set values from a dictionary


def save_cli_dict(ctx, param, value):
    # Save click to a variable to print and save
    info = ctx.to_info_dict()
    with open("output_dict2.json", "w") as write_file:
        json.dump(info, write_file)
    # or use for-loop to save all param-value pairs
    for key in info["command"]["params"]:
        print(key["name"])


# Create click group to apply to 2 functions: save_dict and main.
@click.command()
@click.option(
    "-o",
    "--outdir",
    default="results/bla",
    type=str,
    help="Define output directory",
)
@click.option("--dt", default=0.02, type=float, help="define delta t")
@click.option(
    "-T",
    "T",
    default=2000,
    type=float,
    help="define the endtime of simulation",
)
@click.option("-dx", default=0.2, type=float, help="Spatial discretization")
@click.option(
    "--bnd_cond",
    default="dirichlet",
    type=str,
    help="choose boundary conditions",
)
@click.option(
    "--reset_state",
    is_flag=True,
    default=True,
    help="define if state should be loaded (True) or newly created (False)",
)
@click.option(
    "-IC",
    "--cell_init_file",
    default="",
    type=str,
    help="If reset_state=True, define filename of initial conditions (json or h5 file)",
)
# Consider using type=click.Path(exists=True),
@click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Indicate if simulations runs on hpc",
)
@click.option(
    "--from_json",
    is_flag=True,
    callback=load_json,
    expose_value=False,
    is_eager=True,
    help="Path to json file",
)
@click.option("--save_cli_dict", callback=save_cli_dict, expose_value=False)
def main(
    outdir,
    T,
    dx,
    dt,
    bnd_cond,
    reset_state,
    cell_init_file,
    hpc,
    Lx=2.0,
    Ly=0.7,
    Lz=0.3,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
):

    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"

    # Disable warnings
    dolfin.set_log_level(40)

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
            mech_heart = mechanics_model.setup_mechanics_model(
                mesh=mesh,
                coupling=coupling,
                dt=dt,
                bnd_cond=bnd_cond,
                cell_params=solver.ode_solver._model.parameters(),
                pre_stretch=pre_stretch,
                traction=traction,
                spring=spring,
                fix_right_plane=fix_right_plane,
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
