import contextlib
import warnings
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import cbcbeat
import dolfin
from dolfin import FiniteElement  # noqa: F401
from dolfin import MixedElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import geometry
from . import mechanics_model
from . import utils
from .config import Config


logger = utils.getLogger(__name__)


def vs_functions_to_dict(vs, state_names):
    return {
        name: utils.sub_function(vs, index) for index, name in enumerate(state_names)
    }


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r"):
    import h5py
    from mpi4py import MPI

    if h5py.h5.get_config().mpi and dolfin.MPI.size(dolfin.MPI.comm_world) > 1:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=MPI.COMM_WORLD)
    else:
        if dolfin.MPI.size(dolfin.MPI.comm_world) > 1:
            warnings.warn("h5py is not installed with MPI support")
        h5file = h5py.File(h5name, filemode)
    yield h5file

    h5file.close()


def dict_to_h5(data, h5name, h5group):
    with h5pyfile(h5name, "a") as h5file:
        if h5group == "":
            group = h5file
        else:
            group = h5file.create_group(h5group)
        for k, v in data.items():
            group.create_dataset(k, data=v)


def decode(x):
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, list):
        return [xi for xi in map(decode, x)]
    return x


def h5_to_dict(h5group):
    import h5py

    group = {}
    for key, value in h5group.items():
        if isinstance(value, h5py.Dataset):
            v = decode(value[...].tolist())
            group[key] = v

        elif isinstance(value, h5py.Group):
            group[key] = h5_to_dict(h5group[key])

        else:
            raise ValueError(f"Unknown value type {type(value)} for key {key}.")

    return group


def check_file_exists(h5_filename, raise_on_false=True):
    path = Path(h5_filename)
    if not path.is_file():
        if raise_on_false:
            raise FileNotFoundError(f"Path {h5_filename} does not exist")
        return False
    return True


def group_in_file(h5_filename, h5group):
    check_file_exists(h5_filename)
    exists = False

    with h5pyfile(h5_filename) as h5file:
        if h5group in h5file:
            exists = True

    return exists


def mech_heart_to_bnd_cond(mech_heart: mechanics_model.MechanicsProblem):
    if type(mech_heart).__name__ == "RigidMotionProblem":
        return "rigid"
    return "dirichlet"


def serialize_dict(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, Path):
            new_d[k] = v.as_posix()
        elif v is None:
            # Skip it
            continue
        elif isinstance(v, dict):
            new_d[k] = serialize_dict(v)
        elif isinstance(v, dolfin.Constant):
            try:
                new_d[k] = float(v)
            except Exception:
                continue
        else:
            new_d[k] = v
    return new_d


def save_state(
    path,
    config: Config,
    geo: geometry.BaseGeometry,
    state_params: Optional[Dict[str, float]] = None,
):
    path = Path(path)
    utils.remove_file(path)

    logger.info(f"Save state to {path}")
    geo.dump(path)
    logger.debug("Save using dolfin.HDF5File")

    logger.debug("Save using h5py")
    dict_to_h5(serialize_dict(config.as_dict()), path, "config")
    if state_params is None:
        state_params = {}
    dict_to_h5(serialize_dict(state_params), path, "state_params")


def load_state(
    path: Union[str, Path],
    drug_factors_file: Union[str, Path] = "",
    popu_factors_file: Union[str, Path] = "",
    disease_state="healthy",
    PCL: float = 1000,
):
    logger.debug(f"Load state from path {path}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")

    logger.debug("Open file with h5py")
    with h5pyfile(path) as h5file:
        config = Config(**h5_to_dict(h5file["config"]))

    if config.coupling_type == "explicit_ORdmm_Land":
        from .models.explicit_ORdmm_Land import EMCoupling
    elif config.coupling_type == "fully_coupled_ORdmm_Land":
        from .models.fully_coupled_ORdmm_Land import EMCoupling  # type: ignore
    elif config.coupling_type == "pureEP_ORdmm_Land":
        from .models.pureEP_ORdmm_Land import EMCoupling  # type: ignore
    else:
        raise ValueError(f"Invalid coupling type: {config.coupling_type}")

    return EMCoupling.from_state(
        path=path,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
        PCL=PCL,
    )


def load_initial_conditions_from_h5(
    path: Union[str, Path],
    CellModel: Type[cbcbeat.CardiacCellModel],
):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")
    logger.info(f"Loading initial conditions from {path}")

    with h5pyfile(path) as h5file:
        vs_signature = h5file["ep"]["vs"].attrs["signature"].decode()

    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), path.as_posix(), "r") as h5file:
        h5file.read(mesh, "/mesh", False)

    VS = dolfin.FunctionSpace(mesh, eval(vs_signature))
    vs = dolfin.Function(VS)

    with dolfin.HDF5File(mesh.mpi_comm(), path.as_posix(), "r") as h5file:
        h5file.read(vs, "/ep/vs")

    return vs_functions_to_dict(
        vs,
        state_names=CellModel.default_initial_conditions().keys(),
    )
