import logging
import typing
from collections import namedtuple

import cbcbeat
import dolfin
import pulse

from . import em_model
from . import ep_model
from . import geometry
from . import mechanics_model
from . import utils
from .ORdmm_Land import ORdmm_Land as CellModel

logger = utils.getLogger(__name__)

EMState = namedtuple(
    "EMState",
    ["coupling", "solver", "mech_heart", "t0"],
)


class Defaults:
    outdir: utils.PathLike = "results"
    T: float = 1000
    dx: float = 0.2
    dt: float = 0.05
    bnd_cond: mechanics_model.BoundaryConditions = (
        mechanics_model.BoundaryConditions.dirichlet
    )
    load_state: bool = False
    cell_init_file: utils.PathLike = ""
    hpc: bool = False
    lx: float = 2.0
    ly: float = 0.7
    lz: float = 0.3
    save_freq: int = 1
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None
    traction: typing.Union[dolfin.Constant, float] = None
    spring: typing.Union[dolfin.Constant, float] = None
    fix_right_plane: bool = True
    loglevel = logging.INFO
    num_refinements: int = 1
    set_material: str = ""
    drug_factors_file: str = ""
    popu_factors_file: str = ""
    disease_state: str = "healthy"


def default_parameters():
    return {k: v for k, v in Defaults.__dict__.items() if not k.startswith("_")}


def setup_EM_model(
    lx: float = Defaults.lx,
    ly: float = Defaults.ly,
    lz: float = Defaults.lz,
    dx: float = Defaults.dx,
    dt: float = Defaults.dt,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = True,
    bnd_cond: mechanics_model.BoundaryConditions = Defaults.bnd_cond,
    num_refinements: int = Defaults.num_refinements,
    set_material: str = Defaults.set_material,
    drug_factors_file: str = Defaults.drug_factors_file,
    popu_factors_file: str = Defaults.popu_factors_file,
    disease_state: str = Defaults.disease_state,
    cell_init_file: utils.PathLike = Defaults.cell_init_file,
):

    geo = geometry.SlabGeometry(
        lx=lx,
        ly=ly,
        lz=lz,
        dx=dx,
        num_refinements=num_refinements,
    )

    coupling = em_model.EMCoupling(geo)

    # Set-up solver and time it
    solver = setup_ep_solver(
        dt=dt,
        coupling=coupling,
        cell_init_file=cell_init_file,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
    )

    coupling.register_ep_model(solver)

    mech_heart = setup_mechanics_solver(
        coupling=coupling,
        dt=dt,
        bnd_cond=bnd_cond,
        cell_params=solver.ode_solver._model.parameters(),
        pre_stretch=pre_stretch,
        traction=traction,
        spring=spring,
        fix_right_plane=fix_right_plane,
        set_material=set_material,
    )

    return EMState(
        coupling=coupling,
        solver=solver,
        mech_heart=mech_heart,
        t0=0,
    )


def setup_mechanics_solver(
    coupling: em_model.EMCoupling,
    dt,
    bnd_cond: mechanics_model.BoundaryConditions,
    cell_params,
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None,
    traction: typing.Union[dolfin.Constant, float] = None,
    spring: typing.Union[dolfin.Constant, float] = None,
    fix_right_plane: bool = False,
    set_material: str = "",
    linear_solver="mumps",
):
    """Setup mechanics model with dirichlet boundary conditions or rigid motion."""
    logger.info("Set up mechanics model")

    microstructure = mechanics_model.setup_microstructure(coupling.mech_mesh)

    marker_functions = None
    bcs = None
    if bnd_cond == mechanics_model.BoundaryConditions.dirichlet:
        bcs, marker_functions = mechanics_model.setup_diriclet_bc(
            mesh=coupling.mech_mesh,
            pre_stretch=pre_stretch,
            traction=traction,
            spring=spring,
            fix_right_plane=fix_right_plane,
        )
    # Create the geometry
    geometry = pulse.Geometry(
        mesh=coupling.mech_mesh,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )
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
    active_model = mechanics_model.LandModel(
        f0=microstructure.f0,
        s0=microstructure.s0,
        n0=microstructure.n0,
        eta=0,
        lmbda=coupling.lmbda_mech,
        Zetas=coupling.Zetas_mech,
        Zetaw=coupling.Zetaw_mech,
        parameters=cell_params,
        XS=coupling.XS_mech,
        XW=coupling.XW_mech,
        dt=dt,
        function_space=V,
    )
    material = pulse.HolzapfelOgden(
        active_model=active_model,
        parameters=material_parameters,
    )

    if set_material == "Guccione":
        material_parameters = pulse.Guccione.default_parameters()
        material_parameters["CC"] = 2.0
        material_parameters["bf"] = 8.0
        material_parameters["bfs"] = 4.0
        material_parameters["bt"] = 2.0

        material = pulse.Guccione(
            params=material_parameters,
            active_model=active_model,
        )

    Problem = mechanics_model.MechanicsProblem
    if bnd_cond == mechanics_model.BoundaryConditions.rigid:
        Problem = mechanics_model.RigidMotionProblem

    verbose = logger.getEffectiveLevel() < logging.INFO
    problem = Problem(
        geometry,
        material,
        bcs,
        solver_parameters={"linear_solver": linear_solver, "verbose": verbose},
    )

    problem.solve()

    total_dofs = problem.state.function_space().dim()
    logger.info("Mechanics model")
    utils.print_mesh_info(coupling.mech_mesh, total_dofs)

    return problem


def setup_ep_solver(
    dt,
    coupling,
    scheme="GRL1",
    theta=0.5,
    preconditioner="sor",
    cell_params=None,
    cell_inits=None,
    cell_init_file=None,
    drug_factors_file=None,
    popu_factors_file=None,
    disease_state="healthy",
):
    ps = ep_model.setup_splitting_solver_parameters(
        theta=theta,
        preconditioner=preconditioner,
        dt=dt,
        scheme=scheme,
    )

    cell_params = ep_model.handle_cell_params(
        cell_params=cell_params,
        disease_state=disease_state,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
    )

    cell_inits = ep_model.handle_cell_inits(
        cell_inits=cell_inits,
        cell_init_file=cell_init_file,
    )

    cell_inits["lmbda"] = coupling.lmbda_ep
    cell_inits["Zetas"] = coupling.Zetas_ep
    cell_inits["Zetaw"] = coupling.Zetaw_ep

    cellmodel = CellModel(init_conditions=cell_inits, params=cell_params)

    # Set-up cardiac model
    ep_heart = ep_model.setup_ep_model(cellmodel, coupling.ep_mesh)
    timer = dolfin.Timer("SplittingSolver: setup")

    solver = cbcbeat.SplittingSolver(ep_heart, ps)

    timer.stop()
    # Extract the solution fields and set the initial conditions
    (vs_, vs, vur) = solver.solution_fields()
    vs_.assign(cellmodel.initial_conditions())

    # Output some degrees of freedom
    total_dofs = vs.function_space().dim()
    logger.info("EP model")
    utils.print_mesh_info(coupling.ep_mesh, total_dofs)
    return solver
