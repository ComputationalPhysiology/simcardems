from __future__ import annotations

import typing
from pathlib import Path

import cbcbeat
import dolfin
import pulse

from .. import ep_model
from .. import geometry as _geometry
from .. import mechanics_model
from .. import save_load_functions as io
from .. import utils
from ..config import Config
from ..time_stepper import TimeStepper

if typing.TYPE_CHECKING:
    from .. import datacollector

logger = utils.getLogger(__name__)


def setup_EM_model(
    cls_EMCoupling,
    cls_CellModel,
    cls_ActiveModel,
    geometry: _geometry.BaseGeometry,
    config: typing.Optional[Config] = None,
    cell_params: typing.Optional[typing.Dict[str, float]] = None,
    cell_inits: typing.Optional[dolfin.Function] = None,
    mech_state_init: typing.Optional[dolfin.Function] = None,
    state_params: typing.Optional[typing.Dict[str, float]] = None,
) -> BaseEMCoupling:
    if config is None:
        config = Config()

    if state_params is None:
        state_params = {}
    coupling = cls_EMCoupling(geometry, **state_params)

    cellmodel = ep_model.setup_cell_model(
        cls=cls_CellModel,
        coupling=coupling,
        cell_init_file=config.cell_init_file,
        drug_factors_file=config.drug_factors_file,
        popu_factors_file=config.popu_factors_file,
        disease_state=config.disease_state,
        cell_inits=cell_inits,
        cell_params=cell_params,
    )

    # Set-up solver and time it
    solver = ep_model.setup_solver(
        coupling=coupling,
        dt=config.dt,
        PCL=config.PCL,
        cellmodel=cellmodel,
    )
    coupling.register_ep_model(solver)

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
        ActiveModel=cls_ActiveModel,
    )
    if mech_state_init is not None:
        mech_heart.state.assign(mech_state_init)
    coupling.register_mech_model(mech_heart)
    coupling.setup_assigners()

    return coupling


def setup_EM_model_from_config(
    config: Config,
    geometry: typing.Optional[_geometry.BaseGeometry] = None,
    state_params: typing.Optional[typing.Dict[str, float]] = None,
) -> BaseEMCoupling:
    if geometry is None:
        geometry = _geometry.load_geometry(
            mesh_path=config.geometry_path,
            schema_path=config.geometry_schema_path,
        )

    if config.coupling_type == "explicit_ORdmm_Land":
        from .explicit_ORdmm_Land import EMCoupling, CellModel, ActiveModel
    elif config.coupling_type == "fully_coupled_ORdmm_Land":
        from .fully_coupled_ORdmm_Land import EMCoupling, CellModel, ActiveModel  # type: ignore
    elif config.coupling_type == "pureEP_ORdmm_Land":
        from .pureEP_ORdmm_Land import EMCoupling, CellModel, ActiveModel  # type: ignore
    else:
        raise ValueError(f"Invalid coupling type: {config.coupling_type}")

    return setup_EM_model(
        cls_EMCoupling=EMCoupling,
        cls_CellModel=CellModel,
        cls_ActiveModel=ActiveModel,
        geometry=geometry,
        config=config,
        state_params=state_params,
    )


class BaseEMCoupling:
    def __init__(
        self, geometry: _geometry.BaseGeometry, t: float = 0.0, **kwargs
    ) -> None:
        logger.debug("Create EM coupling")
        self.geometry = geometry
        self.t = t

    @property
    def state_params(self):
        return {"t": self.t}

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return NotImplemented

        if self.coupling_type != __o.coupling_type:
            return False

        if self.t != __o.t:
            return False

        return True

    def register_time_stepper(self, time_stepper: TimeStepper) -> None:
        self._time_stepper = time_stepper
        self._t_mechanics_prev = time_stepper.t

    @property
    def dt_mechanics(self) -> float:
        return TimeStepper.ns2ms(self._time_stepper.t - self._t_mechanics_prev)

    @property
    def coupling_type(self):
        "CustomType"

    def register_ep_model(self, solver: cbcbeat.SplittingSolver) -> None:
        pass

    def register_mech_model(self, solver: pulse.MechanicsProblem) -> None:
        pass

    def ep_to_coupling(self) -> None:
        pass

    def coupling_to_mechanics(self) -> None:
        pass

    def mechanics_to_coupling(self) -> None:
        pass

    def coupling_to_ep(self) -> None:
        pass

    def setup_assigners(self) -> None:
        from ..datacollector import Assigners

        self._assigners = Assigners()

    def update_prev_mechanics(self) -> None:
        self._t_mechanics_prev = self._time_stepper.t

    def update_prev_ep(self) -> None:
        pass

    def solve_mechanics(self) -> None:
        pass

    def solve_ep(self, interval: typing.Tuple[float, float]) -> None:
        pass

    @property
    def assigners(self) -> datacollector.Assigners:
        return self._assigners

    def save_state(
        self,
        path: typing.Union[str, Path],
        config: typing.Optional[Config] = None,
    ) -> None:
        if config is None:
            logger.warning("Saving state without configuration data")
            config = Config()
        config.coupling_type = self.coupling_type
        io.save_state(
            path=path,
            config=config,
            geo=self.geometry,
            state_params=self.state_params,
        )

    # Strictly speaking this could be a classmethod but we are
    # are not using the class
    @staticmethod
    def from_state(
        path: typing.Union[str, Path],
        drug_factors_file: typing.Union[str, Path] = "",
        popu_factors_file: typing.Union[str, Path] = "",
        disease_state="healthy",
        PCL: float = 1000,
    ):
        return io.load_state(
            path=path,
            drug_factors_file=drug_factors_file,
            popu_factors_file=popu_factors_file,
            disease_state=disease_state,
            PCL=PCL,
        )

    def register_datacollector(self, collector: datacollector.DataCollector) -> None:
        for group_name, group in self.assigners.functions.items():
            for func_name, func in group.items():
                collector.register(group=group_name, name=func_name, f=func)

    def print_mechanics_info(self):
        pass

    def print_ep_info(self):
        pass

    def cell_params(self):
        return {}
