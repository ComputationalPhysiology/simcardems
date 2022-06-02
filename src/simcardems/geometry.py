import abc
from typing import Dict
from typing import Optional

import dolfin
import numpy as np

from . import utils

logger = utils.getLogger(__name__)


def refine_mesh(
    mesh: dolfin.Mesh,
    num_refinements: int,
    redistribute: bool = False,
) -> dolfin.Mesh:

    for i in range(num_refinements):
        logger.info(f"Performing refinement {i+1}")
        mesh = dolfin.refine(mesh, redistribute=redistribute)

    return mesh


def create_boxmesh(Lx, Ly, Lz, dx=0.5, refinements=0):
    # Create computational domain [0, Lx] x [0, Ly] x [0, Lz]
    # with resolution prescribed by benchmark or more refinements

    N = lambda v: int(np.rint(v))
    mesh = dolfin.BoxMesh(
        dolfin.MPI.comm_world,
        dolfin.Point(0.0, 0.0, 0.0),
        dolfin.Point(Lx, Ly, Lz),
        N(Lx / dx),
        N(Ly / dx),
        N(Lz / dx),
    )

    for i in range(refinements):
        logger.info(f"Performing refinement {i + 1}")
        mesh = dolfin.refine(mesh, redistribute=False)

    return mesh


class BaseGeometry(abc.ABC):
    """Abstract geometry base class"""

    num_refinements: int = 0
    mechanics_mesh: dolfin.Mesh

    @property
    def ep_mesh(self) -> dolfin.Mesh:
        if not hasattr(self, "_ep_mesh"):
            self._ep_mesh = refine_mesh(self.mechanics_mesh, self.num_refinements)
        return self._ep_mesh

    @abc.abstractproperty
    def parameters(self) -> Dict[str, float]:
        ...


class Geometry(BaseGeometry):
    def __init__(self, mesh: dolfin.Mesh, num_refinements: int = 0) -> None:
        self.num_refinements = num_refinements
        self.mechanics_mesh = mesh

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mesh={self.mechanics_mesh}, "
            f"num_refinements={self.num_refinements})"
        )

    def parameters(self):
        return {"num_refinements": self.num_refinements}


class SlabGeometry(BaseGeometry):
    def __init__(
        self,
        lx: float,
        ly: float,
        lz: float,
        dx: float,
        num_refinements: int = 0,
        mechanics_mesh: Optional[dolfin.Mesh] = None,
        ep_mesh: Optional[dolfin.Mesh] = None,
    ) -> None:
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.dx = dx
        self.num_refinements = num_refinements

        if mechanics_mesh is None:
            mechanics_mesh = create_boxmesh(Lx=lx, Ly=ly, Lz=lz, dx=dx)
        else:
            pass
            # TODO: Should we do some validation?
            # Perhaps we should also make the other parameters optional

        self.mechanics_mesh = mechanics_mesh
        if ep_mesh is not None:
            # TODO: Do some validation.
            # For example we should make sure that num_refinements are correct
            # and the the ep mesh has the same partition as the mechanics mesh
            self._ep_mesh = ep_mesh

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lx={self.lx}, "
            f"ly={self.ly}, "
            f"lz={self.lz}, "
            f"dx={self.dx}, "
            f"num_refinements={self.num_refinements})"
        )

    @property
    def parameters(self):
        return {
            "lx": self.lx,
            "ly": self.ly,
            "lz": self.lz,
            "dx": self.dx,
            "num_refinements": self.num_refinements,
        }
