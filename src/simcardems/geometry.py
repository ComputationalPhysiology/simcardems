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
from cardiac_geometries.geometry import Geometry
from cardiac_geometries.geometry import H5Path
from cardiac_geometries.geometry import MeshTypes

from . import utils

logger = utils.getLogger(__name__)


def load_geometry(
    mesh_path: utils.PathLike,
    schema_path: Optional[utils.PathLike] = None,
) -> "BaseGeometry":

    if mesh_path == "":
        # Use default slab geometry
        return SlabGeometry()

    if schema_path is None:
        schema_path = Path(mesh_path).with_suffix(".json")

    geo = Geometry.from_file(
        fname=mesh_path,
        schema_path=schema_path,
        schema=BaseGeometry.default_schema(),
    )

    info = getattr(geo, "info", None)
    if info is None:
        raise RuntimeError("Unable to load info from geometry")

    mesh_type = info.get("mesh_type")
    if mesh_type is None:
        raise RuntimeError("Unable to get mesh type from info")

    if mesh_type == MeshTypes.slab.value:
        return SlabGeometry.from_geometry(geo)
    elif mesh_type == MeshTypes.lv_ellipsoid.value:
        return LeftVentricularGeometry.from_geometry(geo)

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

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        mechanics_mesh: Optional[dolfin.Mesh] = None,
        ep_mesh: Optional[dolfin.Mesh] = None,
        microstructure: Optional[pulse.Microstructure] = None,
        microstructure_ep: Optional[pulse.Microstructure] = None,
        ffun: Optional[dolfin.MeshFunction] = None,
        markers: Optional[Dict[str, Tuple[int, int]]] = None,
        outdir: Optional[utils.PathLike] = None,
    ) -> None:

        self.markers = type(self).default_markers()
        if markers is not None:
            self.markers.update(markers)

        self.outdir = outdir  # type: ignore
        self.parameters = type(self).default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        self.mechanics_mesh = mechanics_mesh
        self.ep_mesh = ep_mesh

        self.ffun = ffun
        self.microstructure = microstructure
        self.microstructure_ep = microstructure_ep
        self.validate()

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return NotImplemented
        # TODO: We might add more checks here
        return self.parameters == __o.parameters

    def validate(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        ...

    @staticmethod
    @abc.abstractmethod
    def default_parameters() -> Dict[str, Any]:
        ...

    @staticmethod
    def default_schema() -> Dict[str, H5Path]:
        return {
            "mesh": H5Path(
                h5group="/geometry/mesh/mechanics",
                is_mesh=True,
            ),
            "ep_mesh": H5Path(
                h5group="/geometry/mesh/ep",
                is_mesh=True,
            ),
            "ffun": H5Path(
                h5group="/geometry/meshfunctions/ffun",
                is_meshfunction=True,
                dim=2,
                mesh_key="mesh",
            ),
            "f0": H5Path(
                h5group="/geometry/microstructure/mechanics/f0",
                is_function=True,
                mesh_key="mesh",
            ),
            "s0": H5Path(
                h5group="/geometry/microstructure/mechanics/s0",
                is_function=True,
                mesh_key="mesh",
            ),
            "n0": H5Path(
                h5group="/geometry/microstructure/mechanics/n0",
                is_function=True,
                mesh_key="mesh",
            ),
            "f0_ep": H5Path(
                h5group="/geometry/microstructure/ep/f0",
                is_function=True,
                mesh_key="ep_mesh",
            ),
            "s0_ep": H5Path(
                h5group="/geometry/microstructure/ep/s0",
                is_function=True,
                mesh_key="ep_mesh",
            ),
            "n0_ep": H5Path(
                h5group="/geometry/microstructure/ep/n0",
                is_function=True,
                mesh_key="ep_mesh",
            ),
            "info": H5Path(
                h5group="/geometry/info",
                is_dolfin=False,
            ),
            "markers": H5Path(
                h5group="/geometry/markers",
                is_dolfin=False,
            ),
        }

    @abc.abstractmethod
    def _default_microstructure(
        self,
        fiber_space: str,
        mesh: dolfin.Mesh,
    ) -> pulse.Microstructure:
        ...

    @abc.abstractmethod
    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        ...

    @abc.abstractmethod
    def _default_mesh(self) -> dolfin.Mesh:
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
    def num_refinements(self) -> int:
        return self.parameters["num_refinements"]

    def dump(
        self,
        fname: utils.PathLike,
        schema_path: Optional[utils.PathLike] = None,
        unlink: bool = True,
    ):
        path = Path(fname)
        schema = type(self).default_schema()

        kwargs = {k: getattr(self, k) for k in schema if k != "info"}
        kwargs["info"] = self.parameters
        geo = Geometry(**kwargs, schema=schema)

        if schema_path is None:
            schema_path = path.with_suffix(".json")

        geo.save(path, schema_path=schema_path, unlink=unlink)
        logger.info(f"Saved geometry to {fname}")

    def _get_microstructure_if_None(
        self,
        mesh: dolfin.Mesh,
        label: str,
    ) -> pulse.Microstructure:
        microstructure = self._default_microstructure(
            fiber_space=self.parameters["fiber_space"],
            mesh=mesh,
        )

        if self.outdir is not None:
            path = self.outdir / f"microstructure{label}.h5"
            with dolfin.HDF5File(
                mesh.mpi_comm(),
                path.as_posix(),
                "w",
            ) as h5file:
                h5file.write(microstructure.f0, "f0")
                h5file.write(microstructure.s0, "s0")
                h5file.write(microstructure.n0, "n0")
        return microstructure

    @property
    def microstructure_ep(self) -> pulse.Microstructure:
        return self._microstructure_ep  # type: ignore

    @microstructure_ep.setter
    def microstructure_ep(self, microstructure: Optional[pulse.Microstructure]) -> None:
        if microstructure is None:
            microstructure = self._interpolate_microstructure()
        self._microstructure_ep = microstructure

    def _interpolate_microstructure(self) -> pulse.Microstructure:
        element = self.f0.ufl_element()
        V = dolfin.FunctionSpace(self.ep_mesh, element)
        f0 = dolfin.interpolate(self.f0, V)
        s0 = dolfin.interpolate(self.s0, V)
        n0 = dolfin.interpolate(self.n0, V)
        return pulse.Microstructure(f0=f0, s0=s0, n0=n0)

    @property
    def microstructure(self) -> pulse.Microstructure:
        return self._microstructure  # type: ignore

    @microstructure.setter
    def microstructure(self, microstructure: Optional[pulse.Microstructure]) -> None:
        if microstructure is None:
            microstructure = self._get_microstructure_if_None(
                mesh=self.mechanics_mesh,
                label="",
            )
        self._microstructure = microstructure

    @property
    def f0(self) -> dolfin.Function:
        return self.microstructure.f0

    @property
    def s0(self) -> dolfin.Function:
        return self.microstructure.s0

    @property
    def n0(self) -> dolfin.Function:
        return self.microstructure.n0

    @property
    def f0_ep(self) -> dolfin.Function:
        return self.microstructure_ep.f0

    @property
    def s0_ep(self) -> dolfin.Function:
        return self.microstructure_ep.s0

    @property
    def n0_ep(self) -> dolfin.Function:
        return self.microstructure_ep.n0

    @property
    def mechanics_mesh(self) -> dolfin.Mesh:
        return self._mechanics_mesh  # type: ignore

    @mechanics_mesh.setter
    def mechanics_mesh(self, mesh: Optional[dolfin.Mesh]) -> None:
        if mesh is None:
            mesh = self._default_mesh()
            if self.outdir is not None:
                mesh_path = self.outdir / "mesh.xdmf"
                with dolfin.XDMFFile(mesh_path.as_posix()) as f:
                    f.write(mesh)
        self._mechanics_mesh = mesh

    @property
    def ffun(self) -> dolfin.MeshFunction:
        return self._ffun  # type: ignore

    @ffun.setter
    def ffun(self, ffun: Optional[dolfin.MeshFunction]) -> None:
        if ffun is None:
            self._ffun = self._default_ffun(
                self.mechanics_mesh,
            )
            if self.outdir is not None:
                ffun_path = self.outdir / "ffun.xdmf"
                with dolfin.XDMFFile(ffun_path.as_posix()) as f:
                    f.write(self.ffun)
        else:
            self._ffun = ffun

    @property
    def ep_mesh(self) -> dolfin.Mesh:
        return self._ep_mesh  # type: ignore

    @ep_mesh.setter
    def ep_mesh(self, mesh: Optional[dolfin.Mesh]) -> None:
        if mesh is None:
            self._ep_mesh = refine_mesh(
                self.mechanics_mesh,
                self.parameters["num_refinements"],
            )
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

    @classmethod
    def from_files(
        cls,
        mesh_path: utils.PathLike,
        ffun_path: Optional[utils.PathLike] = None,
        marker_path: Optional[utils.PathLike] = None,
        parameter_path: Optional[utils.PathLike] = None,
        microstructure_path: Optional[utils.PathLike] = None,
        **kwargs,
    ):

        markers = cls.default_markers()
        if marker_path is not None:
            markers.update(json.loads(Path(marker_path).read_text()))

        parameters = cls.default_parameters()
        if parameter_path is not None:
            parameters.update(json.loads(Path(parameter_path).read_text()))

        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(Path(mesh_path).as_posix()) as f:
            f.read(mesh)

        if ffun_path is not None:
            ffun = dolfin.MeshFunction("size_t", mesh, 2)
            with dolfin.XDMFFile(Path(ffun_path).as_posix()) as f:
                f.read(ffun)
        else:
            ffun = create_slab_facet_function(
                mesh,
                lx=parameters["lx"],
                markers=markers,
            )

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
    def from_geometry(cls, geo: Geometry):
        kwargs = {}
        for attr_geo, attr_simcardems in [
            ("info", "parameters"),
            ("mesh", "mechanics_mesh"),
            ("ep_mesh", "ep_mesh"),
            ("markers", "markers"),
            ("ffun", "ffun"),
        ]:
            attr = getattr(geo, attr_geo, None)
            if attr is not None:
                kwargs[attr_simcardems] = attr

        micro_kwargs = {}
        for f in ["f0", "s0", "n0"]:
            attr = getattr(geo, f, None)
            if attr is not None:
                micro_kwargs[f] = attr
        if micro_kwargs:
            kwargs["microstructure"] = pulse.Microstructure(**micro_kwargs)

        micro_ep_kwargs = {}
        for f in ["f0_ep", "s0_ep", "n0_ep"]:
            attr = getattr(geo, f, None)
            if attr is not None:
                micro_ep_kwargs[f.rstrip("_ep")] = attr
        if micro_kwargs:
            kwargs["microstructure_ep"] = pulse.Microstructure(**micro_ep_kwargs)

        return cls(**kwargs)


class SlabGeometry(BaseGeometry):
    @staticmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        return {
            "X0": (2, 1),
            "X1": (2, 2),
            "Y0": (2, 3),
            "Z0": (2, 4),
        }

    def _default_microstructure(
        self,
        fiber_space: str,
        mesh: dolfin.Mesh,
    ) -> pulse.Microstructure:
        return create_slab_microstructure(fiber_space=fiber_space, mesh=mesh)

    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        return create_slab_facet_function(
            mesh,
            self.parameters["lx"],
            self.markers,
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
            fiber_space="DG_1",
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
    markers: Optional[Dict[str, Tuple[int, int]]] = None,
) -> dolfin.MeshFunction:
    if markers is None:
        markers = SlabGeometry.default_markers()
    # Define domain to apply dirichlet boundary conditions
    left = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
    right = dolfin.CompiledSubDomain("near(x[0], lx) && on_boundary", lx=lx)
    plane_y0 = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
    plane_z0 = dolfin.CompiledSubDomain("near(x[2], 0) && on_boundary")

    ffun = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ffun.set_all(0)

    left.mark(ffun, markers["X0"][1])
    right.mark(ffun, markers["X1"][1])
    plane_y0.mark(ffun, markers["Y0"][1])
    plane_z0.mark(ffun, markers["Z0"][1])
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


class LeftVentricularGeometry(BaseGeometry):
    @staticmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        return {
            "BASE": (5, 2),
            "ENDO": (6, 2),
            "EPI": (7, 2),
        }

    def _default_microstructure(
        self,
        fiber_space: str,
        mesh: dolfin.Mesh,
    ) -> pulse.Microstructure:
        raise NotImplementedError

    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        raise NotImplementedError

    def _default_mesh(self) -> dolfin.Mesh:
        raise NotImplementedError

    @staticmethod
    def default_parameters():
        return {
            "num_refinements": 1,
            "fiber_space": "P_1",
            "fibers_angle_endo": -60.0,
            "fibers_angle_epi": 60.0,
            "mesh_type": MeshTypes.lv_ellipsoid.value,
            "mu_apex_endo": -3.141592653589793,
            "mu_apex_epi": -3.141592653589793,
            "mu_base_endo": -1.2722641256100204,
            "mu_base_epi": -1.318116071652818,
            "psize_ref": 3.0,
            "r_long_endo": 17.0,
            "r_long_epi": 20.0,
            "r_short_endo": 7.0,
            "r_short_epi": 10.0,
        }

    def validate(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"
