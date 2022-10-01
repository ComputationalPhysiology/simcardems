import abc
from typing import Dict
from typing import Optional
from typing import Union

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


def create_mesh_marking(mesh, marker_dict, filename=""):
    # Mark regions of mesh from a dictionary of markers defined as
    # marker_dict[marker_id] = {"x":(x1,x2), "y":(y1,y2), "z":(z1,z2)}

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    eps = dolfin.DOLFIN_EPS
    for marker_id in marker_dict:
        ranges = marker_dict[marker_id]
        for c in dolfin.cells(mesh):
            in_xrange = ranges["x"][0] - eps <= c.midpoint().x() <= ranges["x"][1] + eps
            in_yrange = ranges["y"][0] - eps <= c.midpoint().y() <= ranges["y"][1] + eps
            in_zrange = ranges["z"][0] - eps <= c.midpoint().z() <= ranges["z"][1] + eps
            if in_xrange and in_yrange and in_zrange:
                marker[c] = int(marker_id)

    if filename:
        marker.rename("cells", "cells")
        with dolfin.XDMFFile(mesh.mpi_comm(), filename) as file:
            file.write(marker)

    return marker


class BaseGeometry(abc.ABC):
    """Abstract geometry base class"""

    num_refinements: int = 0
    mechanics_mesh: dolfin.Mesh
    _mechanics_marking: dolfin.MeshFunction

    @property
    def ep_mesh(self) -> dolfin.Mesh:
        if not hasattr(self, "_ep_mesh"):
            self._ep_mesh = refine_mesh(self.mechanics_mesh, self.num_refinements)
            if hasattr(self, "_mechanics_marking"):
                self._ep_marking = dolfin.adapt(self._mechanics_marking, self._ep_mesh)

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
        marking: Optional[Union[dict, str]] = None,
        export_marking: Optional[Union[utils.PathLike, str]] = None,
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

        if marking is not None:
            self._mechanics_marking = dolfin.MeshFunction(
                "size_t",
                self.mechanics_mesh,
                self.mechanics_mesh.topology().dim(),
            )
            if isinstance(marking, str):
                with dolfin.XDMFFile(self.mechanics_mesh.mpi_comm(), marking) as xdmf:
                    xdmf.read(self._mechanics_marking, "cells")
            elif isinstance(marking, dict):
                self._mechanics_marking = create_mesh_marking(
                    self.mechanics_mesh,
                    marking,
                    str(export_marking),
                )

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
