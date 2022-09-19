import abc
from typing import Dict
from typing import Optional

import dolfin
import numpy as np

from . import utils

logger = utils.getLogger(__name__)


def box_marker(p0, p1, x, y, z):
    eps = dolfin.DOLFIN_EPS
    in_xrange = p0[0] - eps < x < p1[0] + eps
    in_yrange = p0[1] - eps < y < p1[1] + eps
    in_zrange = p0[2] - eps < z < p1[2] + eps
    return in_xrange and in_yrange and in_zrange


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
    mechanics_ht_mesh: dolfin.Mesh = None

    # ep_mesh : Return heart mesh (refined) used in EP model
    @property
    def ep_mesh(self) -> dolfin.Mesh:
        if not hasattr(self, "_ep_mesh"):
            self._ep_mesh = refine_mesh(self.mechanics_mesh, self.num_refinements)
        return self._ep_mesh

    # ep_mesh : Return full (heart + torso) mesh used in EP model
    @property
    def ep_ht_mesh(self) -> dolfin.Mesh:
        if self.mechanics_ht_mesh is None:
            return None
        if not hasattr(self, "_ep_ht_mesh"):
            self._ep_ht_mesh = self.mechanics_ht_mesh
            for i in range(self.num_refinements):
                self._ep_ht_mesh = dolfin.adapt(self._ep_ht_mesh)
        return self._ep_ht_mesh

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


class TorsoGeometry(BaseGeometry):
    def __init__(
        self,
        lx: float,
        ly: float,
        lz: float,
        dx: float,
        num_refinements: int = 0,
        mechanics_mesh: Optional[dolfin.Mesh] = None,
        mechanics_heart_marker: Optional[dolfin.MeshFunction] = None,
    ) -> None:
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.dx = dx
        self.num_refinements = num_refinements

        # Build torso mesh
        if mechanics_mesh is None:
            N = lambda v: int(np.rint(v))
            mechanics_mesh = dolfin.BoxMesh(
                dolfin.MPI.comm_world,
                dolfin.Point(-0.5 * lx, -0.5 * ly, -0.5 * lz),
                dolfin.Point(1.5 * lx, 1.5 * ly, 1.5 * lz),
                N(2 * lx / dx),
                N(2 * ly / dx),
                N(2 * lz / dx),
            )
        else:
            pass

        self.mechanics_ht_mesh = mechanics_mesh

        if mechanics_heart_marker is None:
            mechanics_heart_marker = dolfin.MeshFunction(
                "size_t",
                self.mechanics_ht_mesh,
                self.mechanics_ht_mesh.topology().dim(),
                0,
            )
            for c in dolfin.cells(self.mechanics_ht_mesh):
                mechanics_heart_marker[c] = box_marker(
                    [0.0, 0.0, 0.0],
                    [lx, ly, lz],
                    c.midpoint().x(),
                    c.midpoint().y(),
                    c.midpoint().z(),
                )
        else:
            pass

        self.mechanics_heart_marker = mechanics_heart_marker
        self.ep_heart_marker = dolfin.adapt(
            self.mechanics_heart_marker,
            self.ep_ht_mesh,
        )

        # Heart meshes : mechanics (coarse) and EP (refined)
        self.mechanics_mesh = dolfin.MeshView.create(self.mechanics_heart_marker, 1)
        self._ep_mesh = dolfin.MeshView.create(self.ep_heart_marker, 1)
        # Torso meshes : mechanics (coarse) and EP (refined)
        self.mechanics_torso_mesh = dolfin.MeshView.create(
            self.mechanics_heart_marker,
            0,
        )
        self._ep_torso_mesh = dolfin.MeshView.create(self.ep_heart_marker, 0)

        ## DEBUG - Export
        # marker_mech_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/marker_mech.xdmf")
        # marker_mech_file.write(self.mechanics_heart_marker)
        # marker_ep_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/marker_ep.xdmf")
        # marker_ep_file.write(self.ep_heart_marker)

        # meshH_mech_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/mesh_H_mech.xdmf")
        # meshT_mech_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/mesh_T_mech.xdmf")
        # meshH_ep_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/mesh_H_ep.xdmf")
        # meshT_ep_file = dolfin.XDMFFile(dolfin.MPI.comm_world, "./XDMF/HT/mesh_T_ep.xdmf")
        # meshH_mech_file.write(self.mechanics_mesh)
        # meshT_mech_file.write(self.mechanics_torso_mesh)
        # meshH_ep_file.write(self._ep_mesh)
        # meshT_ep_file.write(self._ep_torso_mesh)

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
