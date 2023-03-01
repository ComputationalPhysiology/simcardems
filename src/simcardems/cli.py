import json
import typing
from pathlib import Path

import click
import typing_extensions

from . import models
from . import postprocess as post
from . import utils
from .config import Config
from .runner import Runner
from .version import __version__

# from . import land_model

logger = utils.getLogger(__name__)


@click.group()
@click.version_option(__version__)
def cli():
    pass


@click.command("run")
@click.argument("geometry-path", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "-s",
    "--geometry-schema-path",
    default=None,
    help=(
        "Schema for the geometry. If not provided it will assume that this file has the "
        "same file name as `geometry-path` but with the suffix `.json`."
    ),
)
@click.option(
    "-o",
    "--outdir",
    default=Config.outdir,
    type=click.Path(writable=True, resolve_path=True),
    help="Output directory",
)
@click.option("--dt", default=Config.dt, type=float, help="Time step")
@click.option(
    "-T",
    "--end-time",
    "T",
    default=Config.T,
    type=float,
    help="End-time of simulation",
)
@click.option(
    "-n",
    "--num_refinements",
    default=Config.num_refinements,
    type=int,
    help="Number of refinements of for the mesh using in the EP model",
)
@click.option(
    "--save_freq",
    default=Config.save_freq,
    type=int,
    help="Set frequency of saving results to file",
)
@click.option(
    "--set_material",
    default=Config.set_material,
    type=str,
    help="Choose material properties for mechanics model (default is HolzapfelOgden, option is Guccione",
)
@click.option(
    "--bnd_rigid",
    is_flag=True,
    default=Config.bnd_rigid,
    help="Flag to set boundary conditions for the mechanics problem to rigid motion condition",
)
@click.option(
    "--load_state",
    is_flag=True,
    default=Config.load_state,
    help="If load existing state if exists, otherwise create a new state",
)
@click.option(
    "-IC",
    "--cell_init_file",
    default=Config.cell_init_file,
    type=click.Path(),
    help=(
        "Path to file containing initial conditions (json or h5 file). "
        "If none is provided then the default initial conditions will be used"
    ),
)
@click.option(
    "--loglevel",
    default=Config.loglevel,
    type=int,
    help="How much printing. DEBUG: 10, INFO:20 (default), WARNING: 30",
)
@click.option(
    "--debug-mode",
    default=Config.debug_mode,
    type=bool,
    help="Run in debug mode. Save more output",
)
@click.option(
    "--show_progress_bar/--hide_progress_bar",
    default=Config.show_progress_bar,
    help="Shows or hide the progress bar.",
)
@click.option(
    "--drug_factors_file",
    default=Config.drug_factors_file,
    type=click.Path(),
    help="Path to drugs scaling factors file (json)",
)
@click.option(
    "--popu_factors_file",
    default=Config.popu_factors_file,
    type=click.Path(),
    help="Path to population scaling factors file (json)",
)
@click.option(
    "--disease_state",
    default=Config.disease_state,
    type=str,
    help="Indicate disease state. Default is healthy. ",
)
@click.option(
    "--mechanics-use-continuation",
    default=Config.mechanics_use_continuation,
    type=bool,
    help="Use continuation based mechanics solver",
)
@click.option(
    "--mechanics-use-custom-newton-solver",
    default=Config.mechanics_use_custom_newton_solver,
    type=bool,
    help="Use custom newton solver and solve ODEs at each Newton iteration",
)
@click.option(
    "--pcl",
    default=Config.PCL,
    type=float,
    help="Pacing cycle length (ms)",
)
@click.option(
    "--spring",
    default=Config.spring,
    type=float,
    help="Set value of spring for Robin boundary condition",
)
@click.option(
    "--traction",
    default=Config.traction,
    type=float,
    help="Set value of traction for Neumann boundary condition",
)
@click.option(
    "--fix_right_plane",
    is_flag=True,
    default=Config.fix_right_plane,
    help="Fix right plane in fiber direction (only usable for slab)",
)
@click.option(
    "--coupling-type",
    default=Config.coupling_type,
    type=click.Choice(models.list_coupling_types()),
    help="Type of coupled model",
)
def run(
    geometry_path: utils.PathLike,
    geometry_schema_path: typing.Optional[utils.PathLike],
    outdir: utils.PathLike,
    T: float,
    dt: float,
    bnd_rigid: bool,
    load_state: bool,
    cell_init_file: utils.PathLike,
    show_progress_bar: bool,
    save_freq: int,
    loglevel: int,
    debug_mode: bool,
    num_refinements: int,
    set_material: str,
    drug_factors_file: utils.PathLike,
    popu_factors_file: utils.PathLike,
    disease_state: str,
    mechanics_use_continuation: bool,
    mechanics_use_custom_newton_solver: bool,
    pcl: float,
    spring: float,
    traction: float,
    fix_right_plane: bool,
    coupling_type: typing_extensions.Literal[
        "fully_coupled_ORdmm_Land",
        "explicit_ORdmm_Land",
        "pureEP_ORdmm_Land",
    ],
):
    config = Config(
        geometry_path=geometry_path,
        geometry_schema_path=geometry_schema_path,
        outdir=outdir,
        T=T,
        dt=dt,
        bnd_rigid=bnd_rigid,
        spring=spring,
        traction=traction,
        fix_right_plane=fix_right_plane,
        load_state=load_state,
        cell_init_file=cell_init_file,
        show_progress_bar=show_progress_bar,
        save_freq=save_freq,
        loglevel=loglevel,
        debug_mode=debug_mode,
        num_refinements=num_refinements,
        set_material=set_material,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
        mechanics_use_continuation=mechanics_use_continuation,
        mechanics_use_custom_newton_solver=mechanics_use_custom_newton_solver,
        PCL=pcl,
        coupling_type=coupling_type,
    )
    main(config=config)


@click.command("run-json")
@click.argument("path", required=True, type=click.Path(exists=True))
def run_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    main(Config(**data))


def main(config: typing.Optional[Config]):
    if config is None:
        config = Config()

    geometry_path = Path(config.geometry_path)
    if not geometry_path.is_file():
        msg = f"Unable to to find geometry path {geometry_path}"
        raise IOError(msg)
    if config.geometry_schema_path is None:
        config.geometry_schema_path = geometry_path.with_suffix(".json")

    # Get all arguments and dump them to a json file
    info_dict = config.__dict__
    outdir = Path(config.outdir)
    outdir.mkdir(exist_ok=True)
    with open(outdir.joinpath("parameters.json"), "w") as f:
        json.dump(info_dict, f, default=post.json_serial)

    runner = Runner(config=config)

    runner.solve(
        T=config.T,
        save_freq=config.save_freq,
        show_progress_bar=config.show_progress_bar,
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
    "--make-xdmf",
    is_flag=True,
    default=False,
    help="Make xdmf files",
)
@click.option(
    "--population",
    is_flag=True,
    default=False,
    help="Plot population",
)
@click.option(
    "--reduction",
    default="average",
    help="What type of reduction to perform when plotting state traces, by default 'average'.",
)
@click.option("--num_models", default=5, help="Number of models to be analyzed")
def postprocess(
    folder,
    num_models,
    plot_state_traces,
    make_xdmf,
    population,
    reduction,
):
    folder = Path(folder)
    if plot_state_traces:
        post.plot_state_traces(folder.joinpath("results.h5"), reduction=reduction)
    if make_xdmf:
        post.make_xdmffiles(folder.joinpath("results.h5"))
    if population:
        logger.info("Execute postprocess for population")
        post.save_popu_json(folder, num_models)


@click.command("gui")
def gui():
    # Make sure we can import the required packages
    from . import gui  # noqa: F401

    gui_path = Path(__file__).parent.joinpath("gui.py")
    import subprocess as sp

    sp.run(["streamlit", "run", gui_path.as_posix()])


@click.command("run-benchmark")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    type=bool,
    help="If True overwrite results in output directory",
)
def run_benchmark(outdir, overwrite):
    # Make sure we can import the required packages
    from . import benchmark  # noqa: F401

    benchmark_path = Path(__file__).parent.joinpath("benchmark.py")
    import subprocess as sp

    path = Path(outdir)
    if path.exists():
        if overwrite:
            import shutil

            shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    sp.run(["python3", benchmark_path, path.as_posix()])


cli.add_command(run)
cli.add_command(run_json)
cli.add_command(postprocess)
cli.add_command(gui)
cli.add_command(run_benchmark)
