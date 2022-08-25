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
from . import land_model
from . import mechanics_model
from . import save_load_functions as io
from . import utils
from .config import Config
from .ORdmm_Land import ORdmm_Land as CellModel

logger = utils.getLogger(__name__)

EMState = namedtuple(
    "EMState",
    ["coupling", "solver", "mech_heart", "t0"],
)


def setup_EM_model(config: Config):

    geo = geometry.SlabGeometry(
        lx=config.lx,
        ly=config.ly,
        lz=config.lz,
        dx=config.dx,
        num_refinements=config.num_refinements,
    )

    coupling = em_model.EMCoupling(geo)

    # Set-up solver and time it
    solver = setup_ep_solver(
        dt=config.dt,
        coupling=coupling,
        cell_init_file=config.cell_init_file,
        drug_factors_file=config.drug_factors_file,
        popu_factors_file=config.popu_factors_file,
        disease_state=config.disease_state,
        PCL=config.PCL,
    )

    coupling.register_ep_model(solver)

    mech_heart = setup_mechanics_solver(
        coupling=coupling,
        bnd_cond=config.bnd_cond,
        cell_params=solver.ode_solver._model.parameters(),
        pre_stretch=config.pre_stretch,
        traction=config.traction,
        spring=config.spring,
        fix_right_plane=config.fix_right_plane,
        set_material=config.set_material,
        mechanics_ode_scheme=config.mechanics_ode_scheme,
        use_custom_newton_solver=config.mechanics_use_custom_newton_solver,
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
    fix_right_plane: bool = Config.fix_right_plane,
    mechanics_ode_scheme: land_model.Scheme = Config.mechanics_ode_scheme,
    set_material: str = "",
    linear_solver="mumps",
    use_custom_newton_solver: bool = Config.mechanics_use_custom_newton_solver,
    Zetas_prev=None,
    Zetaw_prev=None,
    lmbda_prev=None,
    state_prev=None,
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

    active_model = land_model.LandModel(
        f0=microstructure.f0,
        s0=microstructure.s0,
        n0=microstructure.n0,
        eta=0,
        parameters=cell_params,
        XS=coupling.XS_mech,
        XW=coupling.XW_mech,
        mesh=coupling.mech_mesh,
        scheme=mechanics_ode_scheme,
        Zetas=Zetas_prev,
        Zetaw=Zetaw_prev,
        lmbda=lmbda_prev,
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
        use_custom_newton_solver=use_custom_newton_solver,
    )

    if state_prev is not None:
        problem.state.assign(state_prev)

    problem.solve()
    coupling.register_mech_model(problem)

    total_dofs = problem.state.function_space().dim()
    logger.info("Mechanics model")
    utils.print_mesh_info(coupling.mech_mesh, total_dofs)

    return problem


def setup_ep_solver(
    dt,
    coupling,
    scheme=Config.ep_ode_scheme,
    theta=Config.ep_theta,
    preconditioner=Config.ep_preconditioner,
    cell_params=None,
    cell_inits=None,
    cell_init_file=None,
    drug_factors_file=None,
    popu_factors_file=None,
    disease_state=Config.disease_state,
    PCL=Config.PCL,
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
    ep_heart = ep_model.setup_ep_model(cellmodel, coupling.ep_mesh, PCL=PCL)
    timer = dolfin.Timer("SplittingSolver: setup")

    solver = cbcbeat.SplittingSolver(ep_heart, params=ps)

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
        config: typing.Optional[Config] = None,
        empty: bool = False,
        **kwargs,
    ) -> None:

        if config is None:
            config = Config()

        self._config = config

        if empty:
            return

        self._state_path = Path(config.outdir).joinpath("state.h5")
        reset = not config.load_state
        if not reset and self._state_path.is_file():
            # Load state
            logger.info("Load previously saved state")
            coupling, ep_solver, mech_heart, t0 = io.load_state(
                self._state_path,
                config.drug_factors_file,
                config.popu_factors_file,
                config.disease_state,
                config.PCL,  # Set bcl from cli
            )
        else:
            logger.info("Create a new state")
            # Create a new state
            coupling, ep_solver, mech_heart, t0 = setup_EM_model(config)
        self.coupling: em_model.EMCoupling = coupling
        self.ep_solver: cbcbeat.SplittingSolver = ep_solver
        self.mech_heart: mechanics_model.MechanicsProblem = mech_heart
        self._t0: float = t0

        self._reset = reset

        self._setup_assigners()
        self.outdir = config.outdir

        logger.info(f"Starting at t0={self._t0}")

    @property
    def _dt(self):
        return self._config.dt

    @_dt.setter
    def _dt(self, value):
        self._config.dt = value

    @property
    def _bnd_cond(self):
        return self._config.bnd_cond

    @_bnd_cond.setter
    def _bnd_cond(self, value):
        self._config.bnd_cond = value

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, outdir):
        self._outdir = outdir
        self._state_path = Path(outdir).joinpath("state.h5")
        self._setup_datacollector()

    @property
    def t(self) -> float:
        if self._time_stepper is None:
            raise RuntimeError("Please create a time stepper before solving")
        return self._time_stepper.t

    def create_time_stepper(
        self,
        T: float,
        use_ns: bool = True,
        st_progress: typing.Any = None,
    ) -> None:
        self._time_stepper = TimeStepper(
            t0=self._t0,
            T=T,
            dt=self._dt,
            use_ns=True,
            st_progress=st_progress,
        )

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
        self._time_stepper = None
        self._vs = self.ep_solver.solution_fields()[1]
        self._v, self._v_assigner = utils.setup_assigner(self._vs, 0)
        self._Ca, self._Ca_assigner = utils.setup_assigner(self._vs, 45)
        self._XS, self._XS_assigner = utils.setup_assigner(self._vs, 40)
        self._XW, self._XW_assigner = utils.setup_assigner(self._vs, 41)
        self._CaTrpn, self._CaTrpn_assigner = utils.setup_assigner(self._vs, 42)
        self._TmB, self._TmB_assigner = utils.setup_assigner(self._vs, 43)
        self._Cd, self._Cd_assigner = utils.setup_assigner(self._vs, 44)
        self._Zetas, self._Zetas_assigner = utils.setup_assigner(self._vs, 47)
        self._Zetaw, self._Zetaw_assigner = utils.setup_assigner(self._vs, 48)

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
        self._XS_assigner.assign(self._XS, utils.sub_function(self._vs, 40))
        self._XW_assigner.assign(self._XW, utils.sub_function(self._vs, 41))
        self._CaTrpn_assigner.assign(self._CaTrpn, utils.sub_function(self._vs, 42))
        self._TmB_assigner.assign(self._TmB, utils.sub_function(self._vs, 43))
        self._Cd_assigner.assign(self._Cd, utils.sub_function(self._vs, 44))
        self._Zetas_assigner.assign(self._Zetas, utils.sub_function(self._vs, 47))
        self._Zetaw_assigner.assign(self._Zetaw, utils.sub_function(self._vs, 48))

    def store(self):
        # Assign u, v and Ca for postprocessing
        self._assign_displacement()
        self._assign_ep()
        self.collector.store(TimeStepper.ns2ms(self.t))

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
            ("ep", "XS", self._XS),
            ("ep", "XW", self._XW),
            ("ep", "CaTrpn", self._CaTrpn),
            ("ep", "TmB", self._TmB),
            ("ep", "Cd", self._Cd),
            ("ep", "Zetas", self._Zetas),
            ("ep", "Zetaw", self._Zetaw),
            ("mechanics", "Zetas_mech", self.coupling.Zetas_mech),
            ("mechanics", "Zetaw_mech", self.coupling.Zetaw_mech),
            ("mechanics", "XS_mech", self.coupling.XS_mech),
            ("mechanics", "XW_mech", self.coupling.XW_mech),
        ]:
            self.collector.register(group, name, f)

    @property
    def dt_mechanics(self):
        return TimeStepper.ns2ms(float(self.t - self.mech_heart.material.active.t))

    def _solve_mechanics_now(self) -> bool:

        # Update these states that are needed in the Mechanics solver
        self.coupling.ep_to_coupling()

        # XS_norm = utils.compute_norm(self.coupling.XS_ep, self._pre_XS)
        # XW_norm = utils.compute_norm(self.coupling.XW_ep, self._pre_XW)

        # dt for the mechanics model should not be larger than 1 ms
        # return (XS_norm + XW_norm >= 0.05) #or self.dt_mechanics > 0.1
        return True  # self._t <= 10.0 or max(self.coupling.XS_ep.vector()) >= 0.0005 or max(self.coupling.XW_ep.vector()) >= 0.002

    def _pre_mechanics_solve(self) -> None:
        self._preXS_assigner.assign(self._pre_XS, utils.sub_function(self._vs, 40))
        self._preXW_assigner.assign(self._pre_XW, utils.sub_function(self._vs, 41))

        self.coupling.coupling_to_mechanics()
        self.mech_heart.material.active.update_time(self.t)

    def _post_mechanics_solve(self) -> None:

        # Update previous active tension
        self.mech_heart.material.active.update_prev()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

    def _solve_mechanics(self):
        self._pre_mechanics_solve()
        if self._config.mechanics_use_continuation:
            self.mech_heart.solve_for_control(self.coupling.XS_ep)
        else:
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
        T: float = Config.T,
        save_freq: int = Config.save_freq,
        hpc: bool = Config.hpc,
        st_progress: typing.Any = None,
    ):
        if not hasattr(self, "_outdir"):
            raise RuntimeError("Please set the output directory")

        # Truncate residual0 file if exists
        if Path("residual.txt").is_file():
            fr = open("residual.txt", "w")
            fr.truncate(0)
            fr.close()

        save_it = int(save_freq / self._dt)
        self.create_time_stepper(T, use_ns=True, st_progress=st_progress)
        pbar = create_progressbar(time_stepper=self._time_stepper, hpc=hpc)

        # Store initial state
        save_state_every_n_beat = 5  # Save state every fifth beat
        five_beats = (
            TimeStepper.ms2ns(save_state_every_n_beat * 1000.0) / self._time_stepper.dt
        )
        beat_nr = 0
        self.mech_heart.material.active.start_time(self.t)

        for (i, (t0, t)) in enumerate(pbar):
            logger.debug(
                f"Solve EP model at step {i} from {TimeStepper.ns2ms(t0):.2f} ms to {TimeStepper.ns2ms(t):.2f} ms",
            )

            # Solve EP model
            self.ep_solver.step((TimeStepper.ns2ms(t0), TimeStepper.ns2ms(t)))

            if self._solve_mechanics_now():
                logger.debug(
                    (
                        f"Solve mechanics model at step {i} from "
                        f"{TimeStepper.ns2ms(self.mech_heart.material.active.t):.2f} ms"
                        f" to {TimeStepper.ns2ms(self.t):.2f} ms with timestep "
                        f"{self.dt_mechanics:.5f} ms"
                    ),
                )
                self._solve_mechanics()

            self.ep_solver.vs_.assign(self.ep_solver.vs)

            # Store every 'save_freq' ms
            if i % save_it == 0:
                self.store()

            # Store state every 5 beats
            if i > 0 and (i + 1) % five_beats == 0:
                io.save_state(
                    self._state_path.parent.joinpath(
                        f"state_{beat_nr}beat.h5",
                    ),
                    solver=self.ep_solver,
                    mech_heart=self.mech_heart,
                    coupling=self.coupling,
                    dt=self._dt,
                    bnd_cond=self._bnd_cond,
                    t0=TimeStepper.ns2ms(self.t),
                )
                beat_nr += save_state_every_n_beat

            # Residual file : End of line after each time step
            if (
                Path("residual.txt").is_file()
                and dolfin.MPI.rank(dolfin.MPI.comm_world) == 0
            ):
                fr = open("residual.txt", "a")
                fr.write("\n")
                fr.close()

        io.save_state(
            self._state_path,
            solver=self.ep_solver,
            mech_heart=self.mech_heart,
            coupling=self.coupling,
            dt=self._dt,
            bnd_cond=self._bnd_cond,
            t0=TimeStepper.ns2ms(self.t),
        )

        # Copy residual file to output dir (if exists)
        if Path("residual.txt").is_file():
            Path(self._outdir).joinpath("residual.txt").write_text(
                Path("residual.txt").read_text(),
            )


class _tqdm:
    def __init__(self, iterable, *args, **kwargs):
        self._iterable = iterable

    def set_postfix(self, msg):
        pass

    def __iter__(self):
        return iter(self._iterable)


class TimeStepper:
    def __init__(
        self,
        *,
        t0: float,
        T: float,
        dt: float,
        use_ns: bool = True,
        st_progress: typing.Any = None,
    ) -> None:
        """Initialize time stepper

        Parameters
        ----------
        t0 : float
            Start time in milliseconds
        T : float
            End time in milliseconds
        dt : float
            Time step
        use_ns : bool, optional
            Whether to return the time in nanoseconds, by default True
        st_progress:
            Streamlit progress bar
        """

        self._use_ns = use_ns
        self._st_progress = st_progress

        if use_ns:
            self.t0 = TimeStepper.ms2ns(t0)
            self.T = TimeStepper.ms2ns(T)
            self.dt = TimeStepper.ms2ns(dt)
        else:
            self.t0 = t0
            self.T = T
            self.dt = dt

        self.reset()

    def reset(self):
        self.t = self.t0
        self.step = 0

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, T: float) -> None:
        if self.t0 >= T:
            raise ValueError("Start time has to be lower than end time")
        self._T = T

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        dt = min(self.T - self.t0, dt)
        self._dt = dt

    @property
    def total_steps(self) -> int:
        return round((self.T - self.t0) / self.dt)

    def __iter__(self):
        if self.T is None:
            raise RuntimeError("Please assign an end time to time stepper")
        while self.t < self.T:

            prev_t = self.t
            self.t = min(self.t + self.dt, self.T)
            self.step += 1
            if self._st_progress is not None:
                self._st_progress.progress(self.step / self.total_steps)
            yield prev_t, self.t

    @staticmethod
    def ns2ms(t: float) -> float:
        """Convert nanoseconds to milliseconds

        Parameters
        ----------
        t : float
            The time in nanoseconds

        Returns
        -------
        float
            Time in milliseconds
        """
        return t * 1e-6

    @staticmethod
    def ms2ns(t: float) -> float:
        """Convert from milliseconds to nanoseconds

        Parameters
        ----------
        t : float
            Time in milliseconds

        Returns
        -------
        float
            Time in nanoseconds
        """
        return int(t * 1e6)


def create_progressbar(
    time_stepper: TimeStepper,
    hpc: bool = Config.hpc,
):
    if hpc:
        # Turn off progressbar
        pbar = _tqdm(time_stepper, total=time_stepper.total_steps)
    else:
        pbar = tqdm(time_stepper, total=time_stepper.total_steps)
    return pbar
