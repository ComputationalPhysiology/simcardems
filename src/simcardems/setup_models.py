import logging
import typing
from collections import namedtuple
from pathlib import Path

import cbcbeat
import dolfin
import pulse
from tqdm import tqdm

from . import em_model
from . import ep_model
from . import geometry
from . import mechanics_model
from . import save_load_functions as io
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
        parameters=cell_params,
        XS=coupling.XS_mech,
        XW=coupling.XW_mech,
        function_space=V,
    )
    material = mechanics_model.HolzapfelOgden(
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
    verbose = True
    problem = Problem(
        geometry,
        material,
        bcs,
        solver_parameters={"linear_solver": linear_solver, "verbose": verbose},
    )

    problem.solve()
    coupling.register_mech_model(problem)

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


class Runner:
    def __init__(
        self,
        outdir: utils.PathLike = Defaults.outdir,
        *,
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
        drug_factors_file: str = Defaults.drug_factors_file,
        popu_factors_file: str = Defaults.popu_factors_file,
        disease_state: str = Defaults.disease_state,
        reset: bool = True,
        empty: bool = False,
        **kwargs,
    ) -> None:

        if empty:
            return

        self._state_path = Path(outdir).joinpath("state.h5")

        if not reset and self._state_path.is_file():
            # Load state
            logger.info("Load previously saved state")
            coupling, ep_solver, mech_heart, t0 = io.load_state(
                self._state_path,
                drug_factors_file,
                popu_factors_file,
                disease_state,
            )
        else:
            logger.info("Create a new state")
            # Create a new state
            coupling, ep_solver, mech_heart, t0 = setup_EM_model(
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
            )

        self.coupling: em_model.EMCoupling = coupling
        self.ep_solver: cbcbeat.SplittingSolver = ep_solver
        self.mech_heart: mechanics_model.MechanicsProblem = mech_heart
        self._t0: float = t0

        self._reset = reset
        self._dt = dt
        self._bnd_cond = bnd_cond

        self._setup_assigners()
        self.outdir = outdir

        logger.info(f"Starting at t0={self._t0}")

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, outdir):
        self._outdir = outdir
        self._state_path = Path(outdir).joinpath("state.h5")
        self._setup_datacollector()

    @classmethod
    def from_models(
        cls,
        coupling: em_model.EMCoupling,
        ep_solver: cbcbeat.SplittingSolver,
        mech_heart: mechanics_model.MechanicsProblem,
        reset: bool = True,
        t0: float = 0,
    ):
        obj = cls(empty=True)
        obj.coupling = coupling
        obj.ep_solver = ep_solver
        obj.mech_heart = mech_heart
        obj._t0 = t0

        obj._reset = reset
        obj._dt = ep_solver.parameters["MonodomainSolver"]["default_timestep"]
        obj._bnd_cond = mech_heart.boundary_condition
        obj._setup_assigners()
        return obj

    def _setup_assigners(self):
        self._vs = self.ep_solver.solution_fields()[1]
        self._v, self._v_assigner = utils.setup_assigner(self._vs, 0)
        self._Ca, self._Ca_assigner = utils.setup_assigner(self._vs, 45)

        self._pre_XS, self._preXS_assigner = utils.setup_assigner(self._vs, 40)
        self._pre_XW, self._preXW_assigner = utils.setup_assigner(self._vs, 41)

        self._u_subspace_index = 1 if self._bnd_cond == "rigid" else 0
        self._u, self._u_assigner = utils.setup_assigner(
            self.mech_heart.state,
            self._u_subspace_index,
        )
        self._assign_displacement()

    def _assign_displacement(self):
        self._u_assigner.assign(
            self._u,
            self.mech_heart.state.sub(self._u_subspace_index),
        )

    def _assign_ep(self):
        self._v_assigner.assign(self._v, utils.sub_function(self._vs, 0))
        self._Ca_assigner.assign(self._Ca, utils.sub_function(self._vs, 45))

    def store(self):
        # Assign u, v and Ca for postprocessing
        self._assign_displacement()
        self._assign_ep()
        self.collector.store(self._t)

    def _setup_datacollector(self):
        from .datacollector import DataCollector

        self.collector = DataCollector(
            self._outdir,
            self.coupling.mech_mesh,
            self.coupling.ep_mesh,
            reset_state=self._reset,
        )
        for group, name, f in [
            ("mechanics", "u", self._u),
            ("ep", "V", self._v),
            ("ep", "Ca", self._Ca),
            ("mechanics", "lmbda", self.coupling.lmbda_mech),
            ("mechanics", "Ta", self.mech_heart.material.active.Ta_current),
        ]:
            self.collector.register(group, name, f)

    @property
    def dt_mechanics(self) -> float:
        return float(self._t - self.mech_heart.material.active.t)

    def _solve_mechanics_now(self) -> bool:

        # Update these states that are needed in the Mechanics solver
        self.coupling.ep_to_coupling()

        # XS_norm = utils.compute_norm(self.coupling.XS_ep, self._pre_XS)
        # XW_norm = utils.compute_norm(self.coupling.XW_ep, self._pre_XW)

        # dt for the mechanics model should not be larger than 1 ms
        return True  # (XS_norm + XW_norm >= 0.1) or self.dt_mechanics > 0.1

    def _pre_mechanics_solve(self) -> None:
        self._preXS_assigner.assign(self._pre_XS, utils.sub_function(self._vs, 40))
        self._preXW_assigner.assign(self._pre_XW, utils.sub_function(self._vs, 41))

        self.coupling.coupling_to_mechanics()
        self.mech_heart.material.active.update_time(self._t)

    def _post_mechanics_solve(self) -> None:

        # Update previous active tension
        self.mech_heart.material.active.update_prev()
        self.mech_heart.update_zeta_prev()
        self.mech_heart.update_lmbda_prev()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

    def _solve_mechanics(self):
        self._pre_mechanics_solve()
        self.mech_heart.solve()
        # converged = False

        # current_t = float(self.mech_heart.material.active.t)
        # target_t = self._t
        # dt = self.dt_mechanics
        # while not converged:
        #     t = current_t + dt
        #     self.mech_heart.material.active.update_time(t)
        #     try:
        #         self.mech_heart.solve()
        #     except pulse.mechanicsproblem.SolverDidNotConverge:
        #         logger.warning(f"Failed to solve mechanics problem with dt={dt}")
        #         dt /= 2
        #         logger.warning(f"Try with dt={dt}")
        #         if dt < 1e-6:
        #             logger.warning("dt is too small. Good bye")
        #             raise

        #     else:
        #         if abs(t - target_t) < 1e-12:
        #             # We have reached the target
        #             converged = True
        #         # Update dt so that we hit the target next time
        #         current_t = t
        #         dt = target_t - t
        #         self.mech_heart.material.active.update_prev()

        self._post_mechanics_solve()

    def solve(
        self,
        T: float = Defaults.T,
        save_freq: int = Defaults.save_freq,
        hpc: bool = Defaults.hpc,
    ):
        if not hasattr(self, "_outdir"):
            raise RuntimeError("Please set the output directory")

        save_it = int(save_freq / self._dt)
        pbar = create_progressbar(t0=self._t0, T=T, dt=self._dt, hpc=hpc)

        # Store initial state
        self._t = self._t0
        self.mech_heart.material.active.t = self._t0

        for (i, (t0, self._t)) in enumerate(pbar):

            logger.debug(f"Solve EP model at step {i} from {t0} to {self._t}")

            # Solve EP model
            self.ep_solver.step((t0, self._t))

            if self._solve_mechanics_now():
                logger.debug(
                    f"Solve mechanics model at step {i} from \
                        {self.mech_heart.material.active.t} to {self._t} with timestep \
                        {self._t-self.mech_heart.material.active.t}",
                )
                self._solve_mechanics()

            self.ep_solver.vs_.assign(self.ep_solver.vs)

            # Store every 'save_freq' ms
            if i % save_it == 0:
                self.store()

            # Store state every 5 beats
            if i > 0 and i % int(5000 / self._dt) == 0:
                io.save_state(
                    self._state_path.parent.joinpath(
                        f"state_{int(i*self._dt/1000)}beat.h5",
                    ),
                    solver=self.ep_solver,
                    mech_heart=self.mech_heart,
                    coupling=self.coupling,
                    dt=self._dt,
                    bnd_cond=self._bnd_cond,
                    t0=self._t,
                )

        io.save_state(
            self._state_path,
            solver=self.ep_solver,
            mech_heart=self.mech_heart,
            coupling=self.coupling,
            dt=self._dt,
            bnd_cond=self._bnd_cond,
            t0=self._t,
        )


class _tqdm:
    def __init__(self, iterable, *args, **kwargs):
        self._iterable = iterable

    def set_postfix(self, msg):
        pass

    def __iter__(self):
        return iter(self._iterable)


def create_progressbar(
    t0: float = 0,
    T: float = Defaults.T,
    dt: float = Defaults.dt,
    hpc: bool = Defaults.hpc,
):
    time_stepper = cbcbeat.utils.TimeStepper((t0, T), dt, annotate=False)
    if hpc:
        # Turn off progressbar
        pbar = _tqdm(time_stepper, total=round((T - t0) / dt))
    else:
        pbar = tqdm(time_stepper, total=round((T - t0) / dt))
    return pbar
