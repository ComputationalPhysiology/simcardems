import json
import typing
from pathlib import Path

import cbcbeat
import click
import dolfin
from tqdm import tqdm

from . import mechanics_model
from . import postprocess as post
from . import save_load_functions as io
from . import utils
from .datacollector import DataCollector
from .setup_models import Defaults
from .setup_models import setup_EM_model
from .version import __version__

logger = utils.getLogger(__name__)


class _tqdm:
    def __init__(self, iterable, *args, **kwargs):
        self._iterable = iterable

    def set_postfix(self, msg):
        pass

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
    default=Defaults.outdir,
    type=click.Path(writable=True, resolve_path=True),
    help="Output directory",
)
@click.option("--dt", default=Defaults.dt, type=float, help="Time step")
@click.option(
    "-T",
    "--end-time",
    "T",
    default=Defaults.T,
    type=float,
    help="End-time of simulation",
)
@click.option(
    "-n",
    "--num_refinements",
    default=Defaults.num_refinements,
    type=int,
    help="Number of refinements of for the mesh using in the EP model",
)
@click.option(
    "--save_freq",
    default=Defaults.save_freq,
    type=int,
    help="Set frequency of saving results to file",
)
@click.option(
    "--set_material",
    default=Defaults.set_material,
    type=str,
    help="Choose material properties for mechanics model (default is HolzapfelOgden, option is Guccione",
)
@click.option("-dx", default=Defaults.dx, type=float, help="Spatial discretization")
@click.option(
    "-lx",
    default=Defaults.lx,
    type=float,
    help="Size of mesh in x-direction",
)
@click.option(
    "-ly",
    default=Defaults.ly,
    type=float,
    help="Size of mesh in y-direction",
)
@click.option(
    "-lz",
    default=Defaults.lz,
    type=float,
    help="Size of mesh in z-direction",
)
@click.option(
    "--bnd_cond",
    default=Defaults.bnd_cond,
    type=click.Choice(mechanics_model.BoundaryConditions._member_names_),
    help="Boundary conditions for the mechanics problem",
)
@click.option(
    "--load_state",
    is_flag=True,
    default=Defaults.load_state,
    help="If load existing state if exists, otherwise create a new state",
)
@click.option(
    "-IC",
    "--cell_init_file",
    default=Defaults.cell_init_file,
    type=str,
    help=(
        "Path to file containing initial conditions (json or h5 file). "
        "If none is provided then the default initial conditions will be used"
    ),
)
@click.option(
    "--loglevel",
    default=Defaults.loglevel,
    type=int,
    help="How much printing. DEBUG: 10, INFO:20 (default), WARNING: 30",
)
@click.option(
    "--hpc",
    is_flag=True,
    default=Defaults.hpc,
    help="Indicate if simulations runs on hpc. This turns off the progress bar.",
)
@click.option(
    "--drug_factors_file",
    default=Defaults.drug_factors_file,
    type=str,
    help="Set drugs scaling factors (json file)",
)
@click.option(
    "--popu_factors_file",
    default=Defaults.popu_factors_file,
    type=str,
    help="Set population scaling factors (json file)",
)
@click.option(
    "--disease_state",
    default=Defaults.disease_state,
    type=str,
    help="Indicate disease state. Default is healthy. ",
)
def run(
    outdir: utils.PathLike,
    T: float,
    dx: float,
    dt: float,
    bnd_cond: mechanics_model.BoundaryConditions,
    load_state: bool,
    cell_init_file: utils.PathLike,
    hpc: bool,
    lx: float,
    ly: float,
    lz: float,
    save_freq: int,
    loglevel: int,
    num_refinements: int,
    set_material: str,
    drug_factors_file: str,
    popu_factors_file: str,
    disease_state: str,
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
        save_freq=save_freq,
        loglevel=loglevel,
        num_refinements=num_refinements,
        set_material=set_material,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
    )


@click.command("run-json")
@click.argument("path", required=True, type=click.Path(exists=True))
def run_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    main(**data)


def main(
    outdir: utils.PathLike = Defaults.outdir,
    T: float = Defaults.T,
    dx: float = Defaults.dx,
    dt: float = Defaults.dt,
    cell_init_file: utils.PathLike = Defaults.cell_init_file,
    lx: float = Defaults.lx,
    ly: float = Defaults.ly,
    lz: float = Defaults.lz,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
    num_refinements: int = Defaults.num_refinements,
    set_material: str = Defaults.set_material,
    bnd_cond: mechanics_model.BoundaryConditions = Defaults.bnd_cond,
    load_state: bool = Defaults.load_state,
    hpc: bool = Defaults.hpc,
    save_freq: int = Defaults.save_freq,
    loglevel: int = Defaults.loglevel,
    drug_factors_file: str = Defaults.drug_factors_file,
    popu_factors_file: str = Defaults.popu_factors_file,
    disease_state: str = Defaults.disease_state,
):

    # Get all arguments and dump them to a json file
    info_dict = locals()
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    with open(outdir.joinpath("parameters.json"), "w") as f:
        json.dump(info_dict, f)

    # Disable warnings
    from . import set_log_level

    set_log_level(loglevel)
    dolfin.set_log_level(loglevel + 10)  # TODO: Make it possible to set this?

    state_path = outdir.joinpath("state.h5")

    if load_state and state_path.is_file():
        # Load state
        logger.info("Load previously saved state")
        with dolfin.Timer("[demo] Load previously saved state"):
            coupling, solver, mech_heart, t0 = io.load_state(
                state_path,
                drug_factors_file,
                popu_factors_file,
                disease_state,
            )
    else:
        logger.info("Create a new state")
        # Create a new state
        coupling, solver, mech_heart, t0 = setup_EM_model(
            dx=dx,
            dt=dt,
            bnd_cond=bnd_cond,
            cell_init_file=cell_init_file,
            lx=lx,
            ly=ly,
            lz=lz,
            num_refinements=num_refinements,
            set_material=set_material,
            drug_factors_file=drug_factors_file,
            popu_factors_file=popu_factors_file,
            disease_state=disease_state,
        )

    logger.info(f"Starting at t0={t0}")

    vs = solver.solution_fields()[1]
    v, v_assigner = utils.setup_assigner(vs, 0)
    Ca, Ca_assigner = utils.setup_assigner(vs, 45)

    pre_XS, preXS_assigner = utils.setup_assigner(vs, 40)
    pre_XW, preXW_assigner = utils.setup_assigner(vs, 41)

    u_subspace_index = 1 if bnd_cond == "rigid" else 0
    u, u_assigner = utils.setup_assigner(mech_heart.state, u_subspace_index)
    u_assigner.assign(u, mech_heart.state.sub(u_subspace_index))

    collector = DataCollector(
        outdir,
        coupling.mech_mesh,
        coupling.ep_mesh,
        reset_state=not load_state,
    )
    for group, name, f in [
        ("mechanics", "u", u),
        ("ep", "V", v),
        ("ep", "Ca", Ca),
        ("mechanics", "lmbda", coupling.lmbda_mech),
        ("mechanics", "Ta", mech_heart.material.active.Ta_current),
    ]:
        collector.register(group, name, f)

    time_stepper = cbcbeat.utils.TimeStepper((t0, T), dt, annotate=False)
    save_it = int(save_freq / dt)

    if hpc:
        # Turn off progressbar
        pbar = _tqdm(time_stepper, total=round((T - t0) / dt))
    else:
        pbar = tqdm(time_stepper, total=round((T - t0) / dt))
    for (i, (t0, t1)) in enumerate(pbar):

        logger.debug(f"Solve EP model at step {i} from {t0} to {t1}")

        # Solve EP model
        with dolfin.Timer("[demo] Solve EP model"):
            solver.step((t0, t1))

        # Update these states that are needed in the Mechanics solver
        with dolfin.Timer("[demo] Update mechanics"):
            coupling.update_mechanics()

        with dolfin.Timer("[demo] Compute norm"):
            XS_norm = utils.compute_norm(coupling.XS_ep, pre_XS)
        XW_norm = utils.compute_norm(coupling.XW_ep, pre_XW)

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

            coupling.interpolate_mechanics()

            # Solve the Mechanics model
            with dolfin.Timer("[demo] Solve mechanics"):
                mech_heart.solve()

            coupling.interpolate_ep()
            # Update previous
            mech_heart.material.active.update_prev()
            with dolfin.Timer("[demo] Update EP"):
                coupling.update_ep()

        with dolfin.Timer("[demo] Update vs"):
            solver.vs_.assign(solver.vs)
        # Store every 'save_freq' ms
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
            coupling=coupling,
            dt=dt,
            bnd_cond=bnd_cond,
            t0=t0,
        )


@click.command("postprocess")
@click.argument("folder", required=True, type=click.Path(exists=True))
@click.option(
    "--plot-state-traces",
    is_flag=True,
    default=False,
    help="Plot state traces",
)
@click.option(
    "--population",
    is_flag=True,
    default=False,
    help="Plot population",
)
@click.option("--num_models", default=5, help="Number of models to be analyzed")
def postprocess(folder, num_models, plot_state_traces, population):
    folder = Path(folder)
    if plot_state_traces:
        post.plot_state_traces(folder.joinpath("results.h5"))
    if population:
        print("Execute postprocess for population")
        post.save_popu_json(folder, num_models)


cli.add_command(run)
cli.add_command(run_json)
cli.add_command(postprocess)
