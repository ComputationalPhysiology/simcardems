from __future__ import annotations

import abc
import typing

import cbcbeat
import dolfin
import pulse

from .. import utils
from ..geometry import BaseGeometry

if typing.TYPE_CHECKING:
    from ..datacollector import Assigners, DataCollector

logger = utils.getLogger(__name__)


class BaseEMCoupling(abc.ABC):
    def __init__(
        self,
        geometry: BaseGeometry,
        *args,
        **kwargs,
    ) -> None:

        logger.debug("Create EM coupling")

        self.geometry = geometry
        self._post_init(*args, **kwargs)

    def _post_init(self):
        pass

    @abc.abstractmethod
    def members(self) -> typing.Dict[str, dolfin.Function]:
        ...

    @abc.abstractmethod
    def register_ep_model(self, solver: cbcbeat.SplittingSolver) -> None:
        ...

    @abc.abstractmethod
    def register_mech_model(self, solver: pulse.MechanicsProblem) -> None:
        ...

    @abc.abstractmethod
    def ep_to_coupling(self) -> None:
        ...

    @abc.abstractmethod
    def coupling_to_mechanics(self) -> None:
        ...

    @abc.abstractmethod
    def mechanics_to_coupling(self) -> None:
        ...

    @abc.abstractmethod
    def coupling_to_ep(self) -> None:
        ...

    @abc.abstractmethod
    def setup_assigners(self) -> None:
        ...

    @abc.abstractmethod
    def update_prev(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def assigners(self) -> Assigners:
        ...

    def register_datacollector(self, collector: DataCollector) -> None:
        for group_name, group in self.assigners.functions.items():
            for func_name, func in group.items():
                collector.register(group=group_name, name=func_name, f=func)

    def print_mechanics_info(self):
        pass

    def print_ep_info(self):
        pass

    def cell_params(self):
        return {}
