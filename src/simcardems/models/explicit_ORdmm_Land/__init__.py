import dolfin
import pulse

from ...geometry import BaseGeometry
from .cell_model import ORdmmLandExplicit as CellModel
from .em_model import EMCoupling


class ActiveModel(pulse.ActiveModel):
    def __init__(self, geometry: BaseGeometry, **kwargs) -> None:
        V = dolfin.FunctionSpace(geometry.mesh, "CG", 1)
        Ta = dolfin.Function(V)
        super().__init__(
            model="active_stress",
            activation=Ta,
            f0=geometry.f0,
            s0=geometry.s0,
            n0=geometry.n0,
        )


__all__ = ["EMCoupling", "CellModel", "ActiveModel"]
