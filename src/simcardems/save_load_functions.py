import contextlib
import warnings
from pathlib import Path

import dolfin
from dolfin import FiniteElement  # noqa: F401
from dolfin import MixedElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import geometry
from . import mechanics_model
from . import setup_models
from . import utils
from .config import Config
from .models import em_model


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
        else:
            new_d[k] = v
    return new_d


def save_state(
    path,
    config: Config,
    coupling: em_model.BaseEMCoupling,
    geo: geometry.BaseGeometry,
    dt=0.02,
    t0=0,
):
    path = Path(path)
    utils.remove_file(path)

    logger.info(f"Save state to {path}")
    geo.dump(path)
    logger.debug("Save using dolfin.HDF5File")

    with dolfin.HDF5File(
        geo.comm(),
        path.as_posix(),
        "a",
    ) as h5file:
        for name, func in coupling.members().items():
            h5file.write(func, name)

    logger.debug("Save using h5py")
    dict_to_h5(serialize_dict(config.as_dict()), path, "config")
    dict_to_h5(
        coupling.cell_params(),
        path,
        "ep/cell_params",
    )
    dict_to_h5(
        dict(dt=dt, t0=t0),
        path,
        "state_params",
    )


def load_state(
    path,
    drug_factors_file="",
    popu_factors_file="",
    disease_state="healthy",
    PCL=1000,
):
    logger.debug(f"Load state from path {path}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")

    geo = geometry.load_geometry(path, schema_path=path.with_suffix(".json"))
    logger.debug("Open file with h5py")
    with h5pyfile(path) as h5file:
        state_params = h5_to_dict(h5file["state_params"])
        cell_params = h5_to_dict(h5file["ep"]["cell_params"])
        config = Config(**h5_to_dict(h5file["config"]))
        vs_signature = h5file["ep"]["vs"].attrs["signature"].decode()
        mech_signature = h5file["mechanics"]["state"].attrs["signature"].decode()

    VS = dolfin.FunctionSpace(geo.ep_mesh, eval(vs_signature))
    vs = dolfin.Function(VS)

    W = dolfin.FunctionSpace(geo.mechanics_mesh, eval(mech_signature))
    mech_state = dolfin.Function(W)

    V = dolfin.FunctionSpace(geo.ep_mesh, "CG", 1)
    lmbda = dolfin.Function(V, name="lambda")
    logger.debug("Load functions")
    with dolfin.HDF5File(geo.ep_mesh.mpi_comm(), path.as_posix(), "r") as h5file:
        h5file.read(vs, "/ep/vs")
        h5file.read(mech_state, "/mechanics/state")
        h5file.read(lmbda, "/em/lmbda_prev")

    if config.coupling_type == "explicit_ORdmm_Land":
        from .models.explicit_ORdmm_Land import CellModel, ActiveModel, EMCoupling

    else:
        raise ValueError(f"Invalid coupling type: {config.coupling_type}")

    cell_inits = vs_functions_to_dict(
        vs,
        state_names=CellModel.default_initial_conditions().keys(),
    )

    coupling = EMCoupling(
        geometry=geo,
        lmbda=lmbda,
    )

    solver = setup_models.setup_ep_solver(
        state_params["dt"],
        coupling,
        cell_params=cell_params,
        cell_inits=cell_inits,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
        disease_state=disease_state,
        PCL=PCL,
        CellModel=CellModel,
    )
    coupling.register_ep_model(solver)
    bnd_cond_dict = dict([(0, False), (1, True)])

    mech_heart = setup_models.setup_mechanics_solver(
        coupling=coupling,
        geo=geo,
        bnd_rigid=bnd_cond_dict[state_params["bnd_cond"]],
        state_prev=mech_state,
        cell_params=cell_params,
        ActiveModel=ActiveModel,
    )
    mech_heart.state.assign(mech_state)
    coupling.register_mech_model(mech_heart)
    coupling.coupling_to_mechanics()

    return setup_models.EMState(
        coupling=coupling,
        solver=solver,
        mech_heart=mech_heart,
        geometry=geo,
        t0=state_params["t0"],
    )


def load_initial_conditions_from_h5(path):
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

    return vs_functions_to_dict(vs)
