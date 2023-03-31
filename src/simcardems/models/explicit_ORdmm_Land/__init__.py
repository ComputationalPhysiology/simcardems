import dolfin
import pulse

from . import cell_model
from . import em_model
from .cell_model import ORdmmLandExplicit as CellModel
from .em_model import EMCoupling

loggers = [
    "simcardems.explicit_ORdmm_Land.cell_model.logger",
    "simcardems.explicit_ORdmm_Land.em_model.logger",
]


class ActiveModel(pulse.ActiveModel):
    def __init__(self, coupling: EMCoupling, **kwargs) -> None:
        V = dolfin.FunctionSpace(coupling.geometry.mesh, "CG", 1)
        Ta = dolfin.Function(V)
        super().__init__(
            model="active_stress",
            activation=Ta,
            f0=coupling.geometry.f0,
            s0=coupling.geometry.s0,
            n0=coupling.geometry.n0,
        )


__all__ = [
    "EMCoupling",
    "CellModel",
    "ActiveModel",
    "loggers",
    "em_model",
    "cell_model",
]
