import abc
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import dolfin
import numpy as np
import pulse

from . import utils

logger = utils.getLogger(__name__)


def load_geometry(folder: utils.PathLike) -> "BaseGeometry":
    info_path = Path(folder) / "info.json"
    if not info_path.is_file():
        raise RuntimeError("Cannot find info.json in geometry folder")

    info = json.loads(info_path.read_text())
    mesh_type = info.get("mesh_type")
    if mesh_type is None:
        raise RuntimeError(f"{info_path} missing key 'mesh_type'")

    if mesh_type == "slab":
        return SlabGeometry.from_folder(folder)
    elif mesh_type == "lv_ellipsoid":
        raise NotImplementedError
    elif mesh_type == "biv_ellipsoid":
        raise NotImplementedError
    raise RuntimeError(f"Unknown mesh type {mesh_type!r}")


def refine_mesh(
    mesh: dolfin.Mesh,
    num_refinements: int,
    redistribute: bool = False,
) -> dolfin.Mesh:

    for i in range(num_refinements):
        logger.info(f"Performing refinement {i+1}")
        mesh = dolfin.refine(mesh, redistribute=redistribute)

    return mesh


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


class BaseGeometry(abc.ABC):
    """Abstract geometry base class"""

    num_refinements: int = 0
    mechanics_mesh: dolfin.Mesh
    ffun: dolfin.MeshFunction
    markers: Dict[str, Tuple[int, int]]
    microstructure: pulse.Microstructure

    @property
    @abc.abstractmethod
    def parameters(self) -> Dict[str, Any]:
        ...

    @property
    def mesh(self) -> dolfin.Mesh:
        return self.mechanics_mesh

    @property
    def dx(self):
        """Return the volume measure using self.mesh"""
        return dolfin.dx(domain=self.mesh)

    @property
    def ds(self):
        """Return the surface measure of exterior facets using
        self.mesh as domain and self.ffun as subdomain_data
        """
        return dolfin.ds(domain=self.mesh, subdomain_data=self.ffun)

    @property
    def ep_mesh(self) -> dolfin.Mesh:
        return self._ep_mesh

    @ep_mesh.setter
    def ep_mesh(self, mesh: Optional[dolfin.Mesh]) -> None:
        if mesh is None:
            self._ep_mesh = refine_mesh(self.mechanics_mesh, self.num_refinements)
            if self.outdir is not None:
                mesh_path = self.outdir / "ep_mesh.xdmf"
                with dolfin.XDMFFile(mesh_path.as_posix()) as f:
                    f.write(mesh)

        else:
            self._ep_mesh = mesh

    @property
    def outdir(self) -> Optional[Path]:
        return self._outdir

    @outdir.setter
    def outdir(self, folder: Optional[utils.PathLike]):
        if folder is None:
            self._outdir = None
        else:
            self._outdir = Path(folder)

    @property
    def marker_functions(self):
        return pulse.MarkerFunctions(ffun=self.ffun)

    @property
    def f0(self) -> dolfin.Function:
        return self.microstructure.f0

    @property
    def s0(self) -> dolfin.Function:
        return self.microstructure.s0

    @property
    def n0(self) -> dolfin.Function:
        return self.microstructure.n0


class Geometry(BaseGeometry):
    def __init__(
        self,
        mesh: dolfin.Mesh,
        ffun: dolfin.MeshFunction,
        microstructure: pulse.Microstructure,
        markers: Optional[Dict[str, Tuple[int, int]]] = None,
        num_refinements: int = 0,
    ) -> None:
        self.num_refinements = num_refinements
        self.mechanics_mesh = mesh
        self.ffun = ffun
        self.markers = markers or {}
        self.microstructure = microstructure

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mesh={self.mechanics_mesh}, "
            f"num_refinements={self.num_refinements})"
        )

    def parameters(self):
        return {"num_refinements": self.num_refinements}


class SlabGeometry(BaseGeometry):

    markers: Dict[str, Tuple[int, int]] = {
        "X0": (2, 1),
        "X1'": (2, 2),
        "Y0": (2, 3),
        "Z0": (2, 4),
    }

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        num_refinements: int = 0,
        mechanics_mesh: Optional[dolfin.Mesh] = None,
        ep_mesh: Optional[dolfin.Mesh] = None,
        microstructure: Optional[pulse.Microstructure] = None,
        ffun: Optional[dolfin.MeshFunction] = None,
        markers: Optional[Dict[str, Tuple[int, int]]] = None,
        outdir: Optional[utils.PathLike] = None,
    ) -> None:

        self.outdir = outdir  # type: ignore
        self._parameters = SlabGeometry.default_parameters()
        if parameters is not None:
            self._parameters.update(parameters)
        self.num_refinements = num_refinements

        self.mechanics_mesh = mechanics_mesh
        self.ep_mesh = ep_mesh

        self.ffun = ffun

        if markers is not None:
            self.markers = markers

        self.microstructure = microstructure

        self.validate()

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def ffun(self) -> dolfin.MeshFunction:
        return self._ffun  # type: ignore

    @ffun.setter
    def ffun(self, ffun: Optional[dolfin.MeshFunction]) -> None:
        if ffun is None:
            self._ffun = create_slab_facet_function(
                self.mechanics_mesh,
                self.parameters["lx"],
                self.markers,
            )
            if self.outdir is not None:
                ffun_path = self.outdir / "ffun.xdmf"
                with dolfin.XDMFFile(ffun_path.as_posix()) as f:
                    f.write(self.ffun)
        else:
            self._ffun = ffun

    @property
    def microstructure(self) -> pulse.Microstructure:
        return self._microstructure  # type: ignore

    @microstructure.setter
    def microstructure(self, microstructure: Optional[pulse.Microstructure]) -> None:
        if microstructure is None:
            microstructure = create_slab_microstructure(
                fiber_space=self.parameters["fiber_space"],
                mesh=self.mechanics_mesh,
            )

            if self.outdir is not None:
                path = self.outdir / "microstructure.h5"
                with dolfin.HDF5File(
                    self.mechanics_mesh.mpi_comm(),
                    path.as_posix(),
                    "w",
                ) as h5file:
                    h5file.write(microstructure.f0, "f0")
                    h5file.write(microstructure.s0, "s0")
                    h5file.write(microstructure.n0, "n0")
        self._microstructure = microstructure

    @property
    def mechanics_mesh(self) -> dolfin.Mesh:
        return self._mechanics_mesh

    @mechanics_mesh.setter
    def mechanics_mesh(self, mesh: Optional[dolfin.Mesh]) -> None:
        if mesh is None:
            mesh = create_boxmesh(
                **{
                    k: v
                    for k, v in self.parameters.items()
                    if k in ["lx", "ly", "lz", "dx"]
                },
            )
            if self.outdir is not None:
                mesh_path = self.outdir / "mesh.xdmf"
                with dolfin.XDMFFile(mesh_path.as_posix()) as f:
                    f.write(mesh)
        self._mechanics_mesh = mesh

    @staticmethod
    def default_parameters():
        return dict(lx=2.0, ly=0.7, lz=0.3, dx=0.2, fiber_space="DG_1")

    def validate(self):
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lx={self.parameters['lx']}, "
            f"ly={self.parameters['ly']}, "
            f"lz={self.parameters['lz']}, "
            f"dx={self.parameters['dx']}, "
            f"num_refinements={self.num_refinements})"
        )

    @classmethod
    def from_files(
        cls,
        mesh_path: utils.PathLike,
        ffun_path: utils.PathLike,
        marker_path: utils.PathLike,
        parameter_path: utils.PathLike,
        microstructure_path: utils.PathLike,
        **kwargs,
    ):

        markers = json.loads(Path(marker_path).read_text())
        parameters = json.loads(Path(parameter_path).read_text())
        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(Path(mesh_path).as_posix()) as f:
            f.read(mesh)

        ffun = dolfin.MeshFunction("size_t", mesh, 2)
        with dolfin.XDMFFile(Path(ffun_path).as_posix()) as f:
            f.read(ffun)

        fiber_space = parameters.get("fiber_space")
        if fiber_space is not None and microstructure_path is not None:
            family, degree = fiber_space.split("_")
            logger.debug("Set up microstructure")
            V_f = dolfin.VectorFunctionSpace(mesh, family, int(degree))
            f0 = dolfin.Function(V_f)
            s0 = dolfin.Function(V_f)
            n0 = dolfin.Function(V_f)
            with dolfin.HDF5File(
                mesh.mpi_comm(),
                Path(microstructure_path).as_posix(),
                "r",
            ) as h5file:
                h5file.read(f0, "f0")
                h5file.read(s0, "s0")
                h5file.read(n0, "n0")

            microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

        else:
            microstructure = None

        return cls(
            mechanics_mesh=mesh,
            ffun=ffun,
            markers=markers,
            parameters=parameters,
            microstructure=microstructure,
            **kwargs,
        )

    @classmethod
    def from_folder(cls, folder: utils.PathLike):
        folder = Path(folder)
        mesh_path = folder / "mesh.xdmf"
        ffun_path = folder / "ffun.xdmf"
        marker_path = folder / "markers.json"
        parameter_path = folder / "info.json"
        microstructure_path = folder / "microstructure.h5"
        return cls.from_files(
            mesh_path=mesh_path,
            ffun_path=ffun_path,
            marker_path=marker_path,
            parameter_path=parameter_path,
            microstructure_path=microstructure_path,
        )


def create_slab_facet_function(
    mesh: dolfin.Mesh,
    lx: float,
    markers: Dict[str, Tuple[int, int]] = SlabGeometry.markers,
) -> dolfin.MeshFunction:
    # Define domain to apply dirichlet boundary conditions
    left = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = dolfin.CompiledSubDomain("near(x[0], lx) && on_boundary", lx=lx)
    plane_y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    plane_z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")

    ffun = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ffun.set_all(0)

    left.mark(ffun, markers["left"])
    right.mark(ffun, markers["right"])
    plane_y0.mark(ffun, markers["plane_y0"])
    plane_z0.mark(ffun, markers["plane_z0"])
    return ffun


def create_slab_microstructure(fiber_space, mesh):

    family, degree = fiber_space.split("_")
    logger.debug("Set up microstructure")
    V_f = dolfin.VectorFunctionSpace(mesh, family, int(degree))
    f0 = dolfin.interpolate(
        dolfin.Expression(
            ("1.0", "0.0", "0.0"),
            degree=1,
            cell=mesh.ufl_cell(),
        ),
        V_f,
    )
    s0 = dolfin.interpolate(
        dolfin.Expression(
            ("0.0", "1.0", "0.0"),
            degree=1,
            cell=mesh.ufl_cell(),
        ),
        V_f,
    )
    n0 = dolfin.interpolate(
        dolfin.Expression(
            ("0.0", "0.0", "1.0"),
            degree=1,
            cell=mesh.ufl_cell(),
        ),
        V_f,
    )
    # Collect the microstructure
    return pulse.Microstructure(f0=f0, s0=s0, n0=n0)
