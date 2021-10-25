import json
import logging
import os
import typing
from pathlib import Path

import cbcbeat
import click
import dolfin
from tqdm import tqdm

from . import em_model
from . import ep_model
from . import mechanics_model
from . import postprocess as post
from . import save_load_functions as io
from . import utils
from .datacollector import DataCollector
from .version import __version__

logger = logging.getLogger(__name__)

PathLike = typing.Union[os.PathLike, str]


class _tqdm:
    def __init__(self, iterable, *args, **kwargs):
        self._iterable = iterable

    def set_postfix(self, msg):
        logger.info(msg)

    def __iter__(self):
        return iter(self._iterable)


@click.group()
@click.version_option(__version__)
def cli():
    pass


@click.command("run")
@click.option(
    "-o",
    "--outdir",
    default="results",
    type=click.Path(writable=True, resolve_path=True),
    help="Output directory",
)
@click.option("--dt", default=0.02, type=float, help="Time step")
@click.option(
    "-T",
    "--end-time",
    "T",
    default=2000,
    type=float,
    help="Endtime of simulation",
)
@click.option("-dx", default=0.2, type=float, help="Spatial discretization")
@click.option("-lx", default=2.0, type=float, help="Size of mesh in x-direction")
@click.option("-ly", default=0.7, type=float, help="Size of mesh in y-direction")
@click.option("-lz", default=0.3, type=float, help="Size of mesh in z-direction")
@click.option(
    "--bnd_cond",
    default=mechanics_model.BoudaryConditions.dirichlet,
    type=click.Choice(mechanics_model.BoudaryConditions._member_names_),
    help="Boundary conditions for the mechanics problem",
)
@click.option(
    "--load_state",
    is_flag=True,
    default=False,
    help="If load existing state if exists, otherwise create a new state",
)
@click.option(
    "-IC",
    "--cell_init_file",
    default="",
    type=str,
    help=(
        "Path to file containing initial conditions (json or h5 file). "
        "If none is provided then the default initial conditions will be used"
    ),
)
@click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Indicate if simulations runs on hpc. This turns off the progress bar.",
)
def run(
    outdir: PathLike,
    T: float,
    dx: float,
    dt: float,
    bnd_cond: mechanics_model.BoudaryConditions,
    load_state: bool,
    cell_init_file: PathLike,
    hpc: bool,
    lx: float,
    ly: float,
    lz: float,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
):
    main(
        outdir=outdir,
        T=T,
        dx=dx,
        dt=dt,
        bnd_cond=bnd_cond,
        load_state=load_state,
        cell_init_file=cell_init_file,
        hpc=hpc,
        lx=lx,
        ly=ly,
        lz=lz,
        pre_stretch=pre_stretch,
        traction=traction,
        spring=spring,
        fix_right_plane=fix_right_plane,
    )


@click.command("run-json")
@click.argument("path", required=True, type=click.Path(exists=True))
def run_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    main(**data)


def main(
    outdir: PathLike = "results",
    T: float = 1000,
    dx: float = 0.2,
    dt: float = 0.02,
    bnd_cond: mechanics_model.BoudaryConditions = mechanics_model.BoudaryConditions.dirichlet,
    load_state: bool = True,
    cell_init_file: PathLike = "",
    hpc: bool = False,
    lx: float = 2.0,
    ly: float = 0.7,
    lz: float = 0.3,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
):
    # Get all arguments and dump them to a json file
    info_dict = locals()
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    with open(outdir.joinpath("parameters.json"), "w") as f:
        json.dump(info_dict, f)

    # Disable warnings
    dolfin.set_log_level(40)

    state_path = outdir.joinpath("state.h5")

    if load_state and state_path.is_file():
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
            mesh = utils.create_boxmesh(Lx=lx, Ly=ly, Lz=lz, dx=dx)

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

    collector = DataCollector(outdir, mesh, reset_state=not load_state)
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

    if hpc:
        # Turn off progressbar
        pbar = _tqdm(time_stepper, total=round((T - t0) / dt))
    else:
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

        if not hpc:
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
            Lx=lx,
            Ly=ly,
            Lz=lz,
            t0=t0,
        )


@click.command()
@click.argument("folder", required=True, type=click.Path(exists=True))
@click.option(
    "--plot-state-traces",
    is_flag=True,
    default=True,
    help="Plot state traces",
)
def postprocess(folder, plot_state_traces):
    folder = Path(folder)
    if plot_state_traces:
        post.plot_state_traces(folder.joinpath("results.h5"))


cli.add_command(run)
cli.add_command(run_json)
cli.add_command(postprocess)
