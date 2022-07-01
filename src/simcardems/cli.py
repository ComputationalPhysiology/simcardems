import json
import typing
from pathlib import Path

import click
import dolfin

from . import geometry
from . import land_model
from . import mechanics_model
from . import postprocess as post
from . import setup_models
from . import utils
from .config import Config
from .setup_models import TimeStepper
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
@click.option("-dx", default=Config.dx, type=float, help="Spatial discretization")
@click.option(
    "-lx",
    default=Config.lx,
    type=float,
    help="Size of mesh in x-direction",
)
@click.option(
    "-ly",
    default=Config.ly,
    type=float,
    help="Size of mesh in y-direction",
)
@click.option(
    "-lz",
    default=Config.lz,
    type=float,
    help="Size of mesh in z-direction",
)
@click.option(
    "--bnd_cond",
    default=Config.bnd_cond,
    type=click.Choice(mechanics_model.BoundaryConditions._member_names_),
    help="Boundary conditions for the mechanics problem",
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
    type=str,
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
    "--hpc",
    is_flag=True,
    default=Config.hpc,
    help="Indicate if simulations runs on hpc. This turns off the progress bar.",
)
@click.option(
    "--drug_factors_file",
    default=Config.drug_factors_file,
    type=str,
    help="Set drugs scaling factors (json file)",
)
@click.option(
    "--popu_factors_file",
    default=Config.popu_factors_file,
    type=str,
    help="Set population scaling factors (json file)",
)
@click.option(
    "--disease_state",
    default=Config.disease_state,
    type=str,
    help="Indicate disease state. Default is healthy. ",
)
@click.option(
    "--mechanics-ode-scheme",
    default=Config.mechanics_ode_scheme,
    type=click.Choice(land_model.Scheme._member_names_),
    help="Scheme used to solve the ODEs in the mechanics model",
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
    mechanics_ode_scheme: land_model.Scheme,
    mechanics_use_continuation: bool,
    mechanics_use_custom_newton_solver: bool,
):

    config = Config(
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
        mechanics_ode_scheme=mechanics_ode_scheme,
        mechanics_use_continuation=mechanics_use_continuation,
        mechanics_use_custom_newton_solver=mechanics_use_custom_newton_solver,
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

    # Get all arguments and dump them to a json file
    info_dict = config.__dict__
    outdir = Path(config.outdir)
    outdir.mkdir(exist_ok=True)
    with open(outdir.joinpath("parameters.json"), "w") as f:
        json.dump(info_dict, f)

    # Disable warnings
    from . import set_log_level

    set_log_level(config.loglevel)
    dolfin.set_log_level(config.loglevel + 10)  # TODO: Make it possible to set this?

    runner = setup_models.Runner(config=config)

    runner.solve(T=config.T, save_freq=config.save_freq, hpc=config.hpc)


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
@click.option("--num_models", default=5, help="Number of models to be analyzed")
@click.option(
    "--ep",
    is_flag=True,
    default=False,
    help="Postprocess pure EP results",
)
def postprocess(folder, num_models, plot_state_traces, make_xdmf, population):
    folder = Path(folder)
    if plot_state_traces:
        post.plot_state_traces(folder.joinpath("results.h5"))
    if make_xdmf:
        post.make_xdmffiles(folder.joinpath("results.h5"))
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


@click.command("run-ep")
@click.option("--mesh-file", default="", help="Path to file with mesh")
@click.option(
    "-o",
    "--outdir",
    default="results_ep",
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
def run_ep(mesh_file, outdir="results_ep", dt=0.05, T=1000):

    if mesh_file == "":
        # Use some default geometry
        mesh = geometry.create_boxmesh(Lx=Config.lx, Ly=Config.ly, Lz=Config.lz)
    else:
        # Need to figure out a wat to do this
        raise NotImplementedError

    outdir = Path(outdir)

    solver = setup_models.setup_ep_solver(dt=dt, ep_mesh=mesh)

    (vs_, vs, vur) = solver.solution_fields()

    # Set-up separate potential function for post processing
    v, v_assigner = utils.setup_assigner(vs, 0)
    Ca, Ca_assigner = utils.setup_assigner(vs, 45)

    result_filename = outdir / "results.h5"

    def store(t, initial=False):
        v_assigner.assign(v, vs.sub(0))
        Ca_assigner.assign(Ca, vs.sub(45))

        file_mode = "w" if initial else "a"
        with dolfin.HDF5File(
            mesh.mpi_comm(),
            result_filename.as_posix(),
            file_mode,
        ) as result_file:

            result_file.write(v, "/v", t)
            result_file.write(Ca, "/Ca", t)

            if initial:
                result_file.write(mesh, "/mesh")

    t0 = 0.0
    store(t0, initial=True)
    time_stepper = TimeStepper(t0=t0, T=T, dt=dt, use_ns=True)
    pbar = setup_models.create_progressbar(time_stepper=time_stepper, hpc=False)

    for (i, (t0, t)) in enumerate(pbar):

        solver.step((TimeStepper.ns2ms(t0), TimeStepper.ns2ms(t)))

        # Store every 10th step
        if i % 10 == 0:
            store(t0)

        solver.vs_.assign(solver.vs)


cli.add_command(run)
cli.add_command(run_json)
cli.add_command(postprocess)
cli.add_command(gui)
cli.add_command(run_ep)
