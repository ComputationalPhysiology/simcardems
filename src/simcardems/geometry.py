import abc
import json
from typing import Any, Dict
from typing import Optional
from pathlib import Path
import json

import dolfin
import numpy as np
import pulse

from simcardems.config import Config


from . import utils

logger = utils.getLogger(__name__)


def load_geometry(folder: str) -> "BaseGeometry":
    info_path = Path(folder) / "info.json"
    if not info_path.is_file():
        raise RuntimeError("Cannot find info.json in geometry folder")

    info = json.loads(info_path.read_text())
    mesh_type = info.get("mesh_type")
    if mesh_type is None:
        raise RuntimeError(f"{info_path} missing key 'mesh_type'")


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
    markers: Dict[str, int]
    microstructure: pulse.Microstructure

    @property
    def ep_mesh(self) -> dolfin.Mesh:
        if not hasattr(self, "_ep_mesh"):
            self._ep_mesh = refine_mesh(self.mechanics_mesh, self.num_refinements)
        return self._ep_mesh

    @abc.abstractproperty
    def parameters(self) -> Dict[str, float]:
        ...

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
        markers: Optional[Dict[str, int]] = None,
        num_refinements: int = 0,
    ) -> None:
        self.num_refinements = num_refinements
        self.mechanics_mesh = mesh
        self.ffun = ffun
        self.markers = markers
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

    markers: Dict[str, int] = {"left": 1, "right": 2, "plane_y0": 3, "plane_z0": 4}

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        num_refinements: int = 0,
        mechanics_mesh: Optional[dolfin.Mesh] = None,
        ep_mesh: Optional[dolfin.Mesh] = None,
        microstructure: Optional[pulse.Microstructure] = None,
        fiber_space: str = "DG_1",
        ffun: Optional[dolfin.MeshFunction] = None,
        markers: Optional[Dict[str, int]] = None,
    ) -> None:

        self.parameters = SlabGeometry.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)
        self.num_refinements = num_refinements

        if mechanics_mesh is None:
            mechanics_mesh = create_boxmesh(
                {
                    k: v
                    for k, v in self.parameters.items()
                    if k in ["lx", "ly", "lz", "dx"]
                }
            )
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

        if ffun is None:
            self._create_ffun()
        else:
            self.ffun = ffun

        if markers is not None:
            self.markers = markers

        if microstructure is None:
            self._create_microstructure(fiber_space)
        else:
            self.microstructure = microstructure

        self.validate()

    @staticmethod
    def default_parameter():
        return dict(
            lx=2.0,
            ly=0.7,
            lz=0.3,
            dx=0.2,
        )

    def validate(self):
        pass

    def _create_microstructure(self, fiber_space):

        family, degree = fiber_space.split("_")
        logger.debug("Set up microstructure")
        V_f = dolfin.VectorFunctionSpace(self.mechanics_mesh, family, int(degree))
        f0 = dolfin.interpolate(
            dolfin.Expression(
                ("1.0", "0.0", "0.0"), degree=1, cell=self.mechanics_mesh.ufl_cell()
            ),
            V_f,
        )
        s0 = dolfin.interpolate(
            dolfin.Expression(
                ("0.0", "1.0", "0.0"), degree=1, cell=self.mechanics_mesh.ufl_cell()
            ),
            V_f,
        )
        n0 = dolfin.interpolate(
            dolfin.Expression(
                ("0.0", "0.0", "1.0"), degree=1, cell=self.mechanics_mesh.ufl_cell()
            ),
            V_f,
        )
        # Collect the microstructure
        self.microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

    def _create_ffun(self):
        self.ffun = create_slab_facet_function(
            self.mechanics_mesh, self.lx, self.markers
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

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            lx=config.lx,
            ly=config.ly,
            lz=config.lz,
            dx=config.dx,
            num_refinements=config.num_refinements,
        )

    @classmethod
    def from_files(cls, mesh_path, ffun_path, marker_path, parameter_path):

        markers = json.loads(Path(marker_path).read_text())
        parameters = json.loads(Path(parameter_path).read_text())
        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(Path(mesh_path).as_posix()) as infile:
            infile.read(mesh)

        # ffun_val = dolfin.MeshValueCollection("size_t", mesh, 2)
        # with dolfin.XDMFFile(Path(ffun_path).as_posix()) as f:
        #     f.read(ffun_val, "name_to_read")
        ffun = dolfin.MeshFunction("size_t", mesh, 2)
        with dolfin.XDMFFile(Path(ffun_path).as_posix()) as f:
            f.read(ffun)

        return cls(mechanics_mesh=mesh, ffun=ffun, markers=markers, **parameters)

    @property
    def parameters(self):
        return {
            "lx": self.lx,
            "ly": self.ly,
            "lz": self.lz,
            "dx": self.dx,
            "num_refinements": self.num_refinements,
        }


def create_slab_facet_function(
    mesh: dolfin.Mesh, lx: float, markers: Dict[str, int] = SlabGeometry.markers
):
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
