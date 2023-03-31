from typing import Dict
from typing import Optional
from typing import Tuple

import dolfin
import numpy as np
import pulse
from cardiac_geometries.geometry import MeshTypes

from . import utils
from .geometry import BaseGeometry

logger = utils.getLogger(__name__)


def create_boxmesh(lx, ly, lz, dx=0.5, refinements=0):
    # Create computational domain [0, lx] x [0, ly] x [0, lz]
    # with resolution prescribed by benchmark or more refinements

    N = lambda v: int(np.rint(v))
    mesh = dolfin.BoxMesh(
        dolfin.MPI.comm_world,
        dolfin.Point(0.0, 0.0, 0.0),
        dolfin.Point(lx, ly, lz),
        N(lx / dx),
        N(ly / dx),
        N(lz / dx),
    )

    for i in range(refinements):
        logger.info(f"Performing refinement {i + 1}")
        mesh = dolfin.refine(mesh, redistribute=False)

    return mesh


class SlabGeometry(BaseGeometry):
    @staticmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        return {
            "X0": (2, 1),
            "X1": (2, 2),
            "Y0": (2, 3),
            "Y1": (2, 4),
            "Z0": (2, 5),
            "Z1": (2, 6),
        }

    def _default_microstructure(
        self,
        mesh: dolfin.Mesh,
        ffun: dolfin.MeshFunction,
    ) -> pulse.Microstructure:
        from cardiac_geometries import slab_fibers

        return slab_fibers.create_microstructure(
            function_space=self.parameters["fiber_space"],
            mesh=mesh,
            ffun=ffun,
            markers=self.markers,
            alpha_endo=self.parameters["fibers_angle_endo"],
            alpha_epi=self.parameters["fibers_angle_epi"],
        )

    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        return create_slab_facet_function(
            mesh=mesh,
            lx=self.parameters["lx"],
            ly=self.parameters["ly"],
            lz=self.parameters["lz"],
            markers=self.markers,
        )

    def _default_mesh(self) -> dolfin.Mesh:
        return create_boxmesh(
            **{
                k: v
                for k, v in self.parameters.items()
                if k in ["lx", "ly", "lz", "dx"]
            },
        )

    @staticmethod
    def default_parameters():
        return dict(
            lx=2.0,
            ly=0.7,
            lz=0.3,
            dx=0.2,
            fibers_angle_endo=0,
            fibers_angle_epi=0,
            fiber_space="Quadrature_3",
            num_refinements=1,
            mesh_type=MeshTypes.slab.value,
        )

    def validate(self):
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lx={self.parameters['lx']}, "
            f"ly={self.parameters['ly']}, "
            f"lz={self.parameters['lz']}, "
            f"dx={self.parameters['dx']}, "
            f"num_refinements={self.parameters['num_refinements']})"
        )


def create_slab_facet_function(
    mesh: dolfin.Mesh,
    lx: float,
    ly: float,
    lz: float,
    markers: Optional[Dict[str, Tuple[int, int]]] = None,
) -> dolfin.MeshFunction:
    if markers is None:
        markers = SlabGeometry.default_markers()
    # Define domain to apply dirichlet boundary conditions
    x0 = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    x1 = dolfin.CompiledSubDomain("near(x[0], lx) && on_boundary", lx=lx)
    y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    y1 = dolfin.CompiledSubDomain("near(x[1], ly) && on_boundary", ly=ly)
    z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")
    z1 = dolfin.CompiledSubDomain("near(x[2], lz) && on_boundary", lz=lz)

    ffun = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ffun.set_all(0)

    x0.mark(ffun, markers["X0"][1])
    x1.mark(ffun, markers["X1"][1])

    y0.mark(ffun, markers["Y0"][1])
    y1.mark(ffun, markers["Y1"][1])

    z0.mark(ffun, markers["Z0"][1])
    z1.mark(ffun, markers["Z1"][1])
    return ffun


def create_slab_microstructure(fiber_space, mesh):
    family, degree = fiber_space.split("_")
    logger.debug("Set up microstructure")
    V_f = dolfin.VectorFunctionSpace(mesh, family, int(degree))
    f0 = dolfin.interpolate(
        dolfin.Constant((1, 0, 0)),
        V_f,
    )
    s0 = dolfin.interpolate(
        dolfin.Constant((0, 1, 0)),
        V_f,
    )
    n0 = dolfin.interpolate(
        dolfin.Constant((0, 0, 1)),
        V_f,
    )
    # Collect the microstructure
    return pulse.Microstructure(f0=f0, s0=s0, n0=n0)
