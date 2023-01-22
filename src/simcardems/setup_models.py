import math
import typing
from pathlib import Path
from typing import NamedTuple

import cbcbeat
import pulse
from tqdm import tqdm

from . import config
from . import ep_model
from . import geometry
from . import mechanics_model
from . import save_load_functions as io
from . import utils
from .models import em_model


logger = utils.getLogger(__name__)


class EMState(NamedTuple):
    coupling: em_model.BaseEMCoupling
    solver: cbcbeat.SplittingSolver
    mech_heart: pulse.MechanicsProblem
    geometry: geometry.BaseGeometry
    t0: float


def setup_EM_model(config: config.Config):

    geo = geometry.load_geometry(
        mesh_path=config.geometry_path,
        schema_path=config.geometry_schema_path,
    )

    if config.coupling_type == "explicit_ORdmm_Land":
        from .models.explicit_ORdmm_Land import EMCoupling, CellModel, ActiveModel
    else:
        raise ValueError(f"Invalid coupling type: {config.coupling_type}")

    coupling = EMCoupling(geo)
    cellmodel = ep_model.setup_cell_model(
        cls=CellModel,
        coupling=coupling,
        cell_init_file=config.cell_init_file,
        drug_factors_file=config.drug_factors_file,
        popu_factors_file=config.popu_factors_file,
        disease_state=config.disease_state,
    )

    # Set-up solver and time it
    solver = ep_model.setup_solver(
        coupling=coupling,
        dt=config.dt,
        PCL=config.PCL,
        cellmodel=cellmodel,
    )

    mech_heart = mechanics_model.setup_solver(
        coupling=coupling,
        bnd_rigid=config.bnd_rigid,
        pre_stretch=config.pre_stretch,
        traction=config.traction,
        spring=config.spring,
        fix_right_plane=config.fix_right_plane,
        set_material=config.set_material,
        use_custom_newton_solver=config.mechanics_use_custom_newton_solver,
        debug_mode=config.debug_mode,
        ActiveModel=ActiveModel,
    )

    return EMState(
        coupling=coupling,
        solver=solver,
        mech_heart=mech_heart,
        geometry=geo,
        t0=0,
    )


class Runner:
    def __init__(
        self,
        conf: typing.Optional[config.Config] = None,
        empty: bool = False,
        **kwargs,
    ) -> None:

        if conf is None:
            conf = config.Config()

        self._config = conf

        from . import set_log_level

        set_log_level(conf.loglevel)

        if empty:
            return

        self._state_path = Path(self._config.outdir).joinpath("state.h5")
        reset = not self._config.load_state
        if not reset and self._state_path.is_file():
            # Load state
            logger.info("Load previously saved state")
            coupling, ep_solver, mech_heart, geo, t0 = io.load_state(
                self._state_path,
                self._config.drug_factors_file,
                self._config.popu_factors_file,
                self._config.disease_state,
                self._config.PCL,  # Set bcl from cli
            )
        else:
            logger.info("Create a new state")
            # Create a new state
            coupling, ep_solver, mech_heart, geo, t0 = setup_EM_model(self._config)
        self.coupling: em_model.BaseEMCoupling = coupling
        self.ep_solver: cbcbeat.SplittingSolver = ep_solver
        self.mech_heart: mechanics_model.MechanicsProblem = mech_heart
        self.geometry = geo
        self._t0: float = t0

        self._reset = reset

        self._setup_assigners()
        self.outdir = self._config.outdir

        logger.info(f"Starting at t0={self._t0}")

    @property
    def _dt(self):
        return self._config.dt

    @_dt.setter
    def _dt(self, value):
        self._config.dt = value

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

    @property
    def t0(self) -> float:
        return self._t0

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
        coupling: em_model.BaseEMCoupling,
        ep_solver: cbcbeat.SplittingSolver,
        mech_heart: mechanics_model.MechanicsProblem,
        geo: geometry.BaseGeometry,
        reset: bool = True,
        t0: float = 0,
    ):
        obj = cls(empty=True)
        obj.coupling = coupling
        obj.ep_solver = ep_solver
        obj.mech_heart = mech_heart
        obj.geometry = geo
        obj._t0 = t0

        obj._reset = reset
        obj._dt = ep_solver.parameters["MonodomainSolver"]["default_timestep"]
        obj._setup_assigners()
        return obj

    def _setup_assigners(self):
        self._time_stepper = None
        self.coupling.setup_assigners()

    def store(self):
        # Assign u, v and Ca for postprocessing
        self.coupling.assigners.assign()

        self.collector.store(TimeStepper.ns2ms(self.t))

    def _setup_datacollector(self):
        from .datacollector import DataCollector

        self.collector = DataCollector(
            outdir=self._outdir,
            geo=self.geometry,
            reset_state=self._reset,
        )
        self.coupling.register_datacollector(self.collector)
        self.mech_heart.solver.register_datacollector(self.collector)

    @property
    def dt_mechanics(self):
        return TimeStepper.ns2ms(float(self.t - self.mech_heart.material.active.t))

    def _solve_mechanics_now(self) -> bool:

        # Update these states that are needed in the Mechanics solver
        self.coupling.ep_to_coupling()
        norm = self.coupling.assigners.compute_pre_norm()
        return norm >= 0.05

    def _pre_mechanics_solve(self) -> None:
        self.coupling.assigners.assign_pre()
        self.coupling.coupling_to_mechanics()

    def _post_mechanics_solve(self) -> None:

        # Update previous lmbda
        self.coupling.update_prev()
        self.coupling.mechanics_to_coupling()
        self.coupling.coupling_to_ep()

    def _solve_mechanics(self):
        self._pre_mechanics_solve()
        # if self._config.mechanics_use_continuation:
        #     self.mech_heart.solve_for_control(self.coupling.XS_ep)
        # else:
        self.mech_heart.solve()
        self._post_mechanics_solve()

    def save_state(self, path):
        io.save_state(
            path,
            config=self._config,
            coupling=self.coupling,
            geo=self.geometry,
            dt=self._dt,
            t0=TimeStepper.ns2ms(self.t),
        )

    def solve(
        self,
        T: float = config.Config.T,
        save_freq: int = config.Config.save_freq,
        show_progress_bar: bool = config.Config.show_progress_bar,
        st_progress: typing.Any = None,
    ):
        if not hasattr(self, "_outdir"):
            raise RuntimeError("Please set the output directory")

        save_it = int(save_freq / self._dt)
        self.create_time_stepper(T, use_ns=True, st_progress=st_progress)
        pbar = create_progressbar(
            time_stepper=self._time_stepper,
            show_progress_bar=show_progress_bar,
        )

        # Store initial state
        # save_state_every_n_beat = 5  # Save state every fifth beat
        # five_beats = (
        #     TimeStepper.ms2ns(save_state_every_n_beat * 1000.0) / self._time_stepper.dt
        # )
        # beat_nr = 0
        # self.mech_heart.material.active.start_time(self.t)

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
                        # f"{TimeStepper.ns2ms(self.mech_heart.material.active.t):.2f} ms"
                        # f" to {TimeStepper.ns2ms(self.t):.2f} ms with timestep "
                        # f"{self.dt_mechanics:.5f} ms"
                    ),
                )
                self._solve_mechanics()

            self.ep_solver.vs_.assign(self.ep_solver.vs)

            # Store every 'save_freq' ms
            if i % save_it == 0:
                self.store()

            # Store state every 5 beats
            # if i > 0 and (i + 1) % five_beats == 0:
            #     self.save_state(
            #         self._state_path.parent.joinpath(
            #             f"state_{beat_nr}beat.h5",
            #         ),
            #     )
            #     beat_nr += save_state_every_n_beat

        self.save_state(self._state_path)


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
        if self.t0 > T:
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
        if math.isclose(self.T, self.t0):
            return 0
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
    show_progress_bar: bool = config.Config.show_progress_bar,
):
    if show_progress_bar:
        # Show progressbar
        pbar = tqdm(time_stepper, total=time_stepper.total_steps)
    else:
        # Hide progressbar
        pbar = _tqdm(time_stepper, total=time_stepper.total_steps)
    return pbar
