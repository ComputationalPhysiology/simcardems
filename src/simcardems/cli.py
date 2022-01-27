import json
import typing
from pathlib import Path

import click
import dolfin

from . import mechanics_model
from . import postprocess as post
from . import utils
from .setup_models import Defaults
from .setup_models import Runner
from .version import __version__

logger = utils.getLogger(__name__)


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

    runner = Runner(
        dx=dx,
        dt=dt,
        bnd_cond=bnd_cond,
        cell_init_file=cell_init_file,
        lx=lx,
        ly=ly,
        lz=lz,
        pre_stretch=pre_stretch,
        spring=spring,
        traction=traction,
        fix_right_plane=fix_right_plane,
        num_refinements=num_refinements,
        set_material=set_material,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
        outdir=outdir,
        reset=not load_state,
    )

    runner.solve(T=T, save_freq=save_freq, hpc=hpc)


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


@click.command("gui")
def gui():
    # Make sure we can import the required packages
    from . import gui  # noqa: F401

    gui_path = Path(__file__).parent.joinpath("gui.py")
    import subprocess as sp

    sp.run(["streamlit", "run", gui_path.as_posix()])


cli.add_command(run)
cli.add_command(run_json)
cli.add_command(postprocess)
cli.add_command(gui)
