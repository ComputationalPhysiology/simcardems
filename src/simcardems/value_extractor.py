import abc
import typing
from enum import Enum

import dolfin
import numpy as np

from . import utils
from .geometry import BaseGeometry
from .lvgeometry import LeftVentricularGeometry
from .slabgeometry import SlabGeometry


logger = utils.getLogger(__name__)


class ValueExtractor:
    def __init__(self, geo: BaseGeometry):
        self.geo = geo
        self.volume = dolfin.assemble(dolfin.Constant(1.0) * dolfin.dx(domain=geo.mesh))
        logger.debug("Creating ValueExtractor with geo: {geo!r}")

        if isinstance(self.geo, SlabGeometry):
            self.boundary: Boundary = SlabBoundary(geo.mesh)
        elif isinstance(self.geo, LeftVentricularGeometry):
            self.boundary = LVBoundary(geo.mesh)
        else:
            raise NotImplementedError

    def average(self, func: dolfin.Function) -> float:
        if func.value_rank() == 0:
            return dolfin.assemble(func * dolfin.dx) / self.volume
        else:
            # Take the magnitude
            return dolfin.assemble(dolfin.sqrt(func**2) * dolfin.dx) / self.volume

    def eval_at_node(
        self,
        func: dolfin.Function,
        point: np.ndarray,
        dofs: typing.Optional[np.ndarray] = None,
    ):
        try:
            return func(point)
        except RuntimeError as e:
            if dofs is None:
                msg = "Unable to evaluate function at Node"
                raise RuntimeError(msg) from e
            closest_dof = np.argmin(
                np.linalg.norm(dofs - np.array(point), axis=1),
            )
            return func.vector().get_local()[closest_dof]

    def eval(
        self,
        func: dolfin.Function,
        value: str,
        dofs: typing.Optional[np.ndarray] = None,
    ):
        if value == "average":
            return self.average(func)
        elif value in self.boundary.nodes():
            point = getattr(self.boundary, value)
            return self.eval_at_node(func=func, point=point, dofs=dofs)

        msg = f"Value {value} is not implemented"
        raise NotImplementedError(msg)


def center_func(fmin, fmax):
    return fmin + (fmax - fmin) / 2


class Boundary(abc.ABC):
    def __init__(self, mesh):
        self.mesh = mesh

    @staticmethod
    @abc.abstractmethod
    def nodes() -> typing.Sequence[str]:
        ...


class LVBoundary(Boundary):
    @staticmethod
    def nodes():
        return []


class SlabBoundaryNodes(Enum):
    center = "center"
    xmax = "xmax"
    xmin = "xmin"
    ymax = "ymax"
    ymin = "ymin"
    zmax = "zmax"
    zmin = "zmin"


class SlabBoundary(Boundary):
    @staticmethod
    def nodes():
        return SlabBoundaryNodes._member_names_

    @property
    def boundaries(self):
        coords = self.mesh.coordinates()
        return dict(
            max_x=coords.T[0].max(),
            min_x=coords.T[0].min(),
            max_y=coords.T[1].max(),
            min_y=coords.T[1].min(),
            max_z=coords.T[2].max(),
            min_z=coords.T[2].min(),
        )

    @property
    def xmin(self):
        return [
            self.boundaries["min_x"],
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def xmax(self):
        return [
            self.boundaries["max_x"],
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def ymin(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            self.boundaries["min_y"],
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def ymax(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            self.boundaries["max_y"],
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]

    @property
    def zmin(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            self.boundaries["min_z"],
        ]

    @property
    def zmax(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            self.boundaries["max_z"],
        ]

    @property
    def center(self):
        return [
            center_func(self.boundaries["min_x"], self.boundaries["max_x"]),
            center_func(self.boundaries["min_y"], self.boundaries["max_y"]),
            center_func(self.boundaries["min_z"], self.boundaries["max_z"]),
        ]
