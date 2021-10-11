import contextlib
import os
import warnings
from collections import namedtuple
from pathlib import Path

import dolfin
from dolfin import FiniteElement  # noqa: F401
from dolfin import MixedElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import utils

EMState = namedtuple(
    "EMState",
    ["coupling", "solver", "mech_heart", "bnd_right_x", "mesh", "t0"],
)


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


def save_state(
    path,
    coupling,
    solver,
    mech_heart,
    dt=0.02,
    bnd_cond="dirichlet",
    Lx=2.0,
    Ly=0.7,
    Lz=0.3,
    t0=0,
):
    path = Path(path)
    utils.remove_file(path)

    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(f"Save state to {path}")

    with dolfin.HDF5File(coupling.mesh.mpi_comm(), path.as_posix(), "w") as h5file:
        h5file.write(coupling.mesh, "/mesh")
        for name in ["lmbda", "Zetas", "Zetaw"]:
            h5file.write(getattr(coupling, name), f"/em/{name}")

        h5file.write(solver.vs_, "/ep/vs")
        h5file.write(mech_heart.state, "mechanics/state")

    bnd_cond_dict = dict([("dirichlet", 0), ("rigid", 1)])

    dict_to_h5(solver.ode_solver._model.parameters(), path, "ep/cell_params")
    dict_to_h5(
        dict(
            dt=dt,
            bnd_cond=bnd_cond_dict[bnd_cond],
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            t0=t0,
        ),
        path,
        "state_params",
    )


def load_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")

    import ep_model
    import mechanics_model
    import em_model
    from ORdmm_Land_em_coupling_strong import vs_functions_to_dict

    with h5pyfile(path) as h5file:
        state_params = h5_to_dict(h5file["state_params"])
        cell_params = h5_to_dict(h5file["ep"]["cell_params"])
        vs_signature = h5file["ep"]["vs"].attrs["signature"].decode()
        mech_signature = h5file["mechanics"]["state"].attrs["signature"].decode()

    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), path.as_posix(), "r") as h5file:
        h5file.read(mesh, "/mesh", False)

    V = dolfin.FunctionSpace(mesh, "CG", 1)
    lmbda = dolfin.Function(V, name="lambda")
    Zetas = dolfin.Function(V, name="Zetas")
    Zetaw = dolfin.Function(V, name="Zetaw")

    VS = dolfin.FunctionSpace(mesh, eval(vs_signature))
    vs = dolfin.Function(VS)

    W = dolfin.FunctionSpace(mesh, eval(mech_signature))
    mech_state = dolfin.Function(W)

    with dolfin.HDF5File(mesh.mpi_comm(), path.as_posix(), "r") as h5file:
        h5file.read(lmbda, "/em/lmbda")
        h5file.read(Zetas, "/em/Zetas")
        h5file.read(Zetaw, "/em/Zetaw")
        h5file.read(vs, "/ep/vs")
        h5file.read(mech_state, "/mechanics/state")

    cell_inits = vs_functions_to_dict(vs)
    coupling = em_model.EMCoupling(mesh, lmbda=lmbda, Zetas=Zetas, Zetaw=Zetaw)

    # Set-up solver and time it
    solver = ep_model.setup_solver(
        mesh,
        state_params["dt"],
        coupling,
        cell_params=cell_params,
        cell_inits=cell_inits,
    )

    coupling.register_ep_model(solver)

    bnd_cond_dict = dict([(0, "dirichlet"), (1, "rigid")])

    mech_heart, bnd_right_x = mechanics_model.setup_mechanics_model(
        mesh=mesh,
        coupling=coupling,
        dt=state_params["dt"],
        bnd_cond=bnd_cond_dict[state_params["bnd_cond"]],
        cell_params=solver.ode_solver._model.parameters(),
        Lx=state_params["Lx"],
    )

    mech_heart.state.assign(mech_state)
    return EMState(
        coupling=coupling,
        solver=solver,
        mech_heart=mech_heart,
        bnd_right_x=bnd_right_x,
        mesh=mesh,
        t0=state_params["t0"],
    )


def load_initial_condions_from_h5(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(f"Loading initial conditions from {path}")

    from ORdmm_Land_em_coupling_strong import vs_functions_to_dict

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


def save_cell_params_to_h5(h5_filename, cell_params, params_list):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Saving cell params to : ", h5_filename)
    h5_filename = Path(h5_filename)
    mesh = cell_params[params_list[0]].function_space().mesh()

    with dolfin.HDF5File(mesh.mpi_comm(), h5_filename.as_posix(), "w") as h5_file:
        h5_base = h5_filename.stem
        if not h5_file.has_dataset("/mesh"):
            h5_file.write(mesh, "/mesh")
        for i, p in enumerate(params_list):
            h5_file.write(cell_params[p], "/function/param%d" % i)
            with dolfin.XDMFFile(
                mesh.mpi_comm(),
                h5_base + "_saved/param_%d.xdmf" % i,
            ) as xdmf:
                xdmf.write(cell_params[p])


def load_cell_params_from_h5(h5_filename, V, cell_params, params_list):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Loading cell params from : ", h5_filename)
    mesh = dolfin.Mesh()
    h5_file = dolfin.HDF5File(mesh.mpi_comm(), h5_filename, "r")
    h5_base = os.path.splitext(h5_filename)[0]
    param_f = dolfin.Function(V)
    for i, p in enumerate(params_list):
        if h5_file.has_dataset("/function/param%d/vector_0"):
            h5_file.read(param_f, "/function/param%d/vector_0" % i)
        else:
            if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
                print("Cannot load param[" + p + "]")
        cell_params[p] = param_f
        with dolfin.XDMFFile(
            mesh.mpi_comm(),
            h5_base + "_loaded/param_%d.xdmf" % i,
        ) as xdmf:
            xdmf.write(param_f)
    h5_file.close()


def save_state_variables_to_h5(h5_filename, cell_inits, vs, run_id, compare=False):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Saving state variables to : ", h5_filename)
    mesh = vs.function_space().sub(0).mesh()
    h5_file = dolfin.HDF5File(mesh.mpi_comm(), h5_filename, "w")
    h5_base = os.path.splitext(h5_filename)[0]

    if not h5_file.has_dataset("/mesh"):
        h5_file.write(mesh, "/mesh")

    for i, (k, vs_f_) in enumerate(cell_inits.items()):
        V = vs.function_space().sub(i)
        vs_space = V.collapse()
        vs_f = dolfin.Function(vs_space)
        vs_assigner = dolfin.FunctionAssigner(vs_space, V)
        vs_assigner.assign(vs_f, utils.sub_function(vs, i))
        h5_file.write(vs_f, "/function/state%d" % i)
        with dolfin.XDMFFile(
            mesh.mpi_comm(),
            h5_base + "_saved/state_%d.xdmf" % i,
        ) as xdmf:
            xdmf.write(vs_f)

        if compare:
            with dolfin.XDMFFile(
                mesh.mpi_comm(),
                h5_base + "_diff/state" + k + ".xdmf",
            ) as xdmf:
                vs_f_err = dolfin.project(vs_f - vs_f_, vs_f.function_space())
                vs_f_err.rename("s%d_d" % i, "state_" + k + "_diff")
                xdmf.write_checkpoint(
                    vs_f_err,
                    "s%d_d" % i,
                    run_id,
                    dolfin.XDMFFile.Encoding.HDF5,
                    append=True,
                )
            L2err_file = open(h5_base + "_diff/L2_err_state%d.dat" % i, "a+")
            if run_id <= 1:  # Clear file if this is the first run
                L2err_file.truncate(0)
            L2err_file.write(
                "%g %g\n" % (run_id, dolfin.norm(vs_f.vector() - vs_f_.vector())),
            )
            L2err_file.close()
    h5_file.close()


def save_state_variables_to_xml(filename, cell_inits, vs, run_id):
    file_base = os.path.splitext(filename)[0]
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Saving state variables to : ", file_base)
    mesh = vs.function_space().sub(0).mesh()

    dolfin.File(file_base + "/mesh.xml") << mesh

    for i, (k, vs_f_) in enumerate(cell_inits.items()):
        V = vs.function_space().sub(i)
        vs_space = V.collapse()
        vs_f = dolfin.Function(vs_space)
        vs_assigner = dolfin.FunctionAssigner(vs_space, V)
        vs_assigner.assign(vs_f, utils.sub_function(vs, i))
        dolfin.File(file_base + "/state%d.xml" % i) << vs_f


def save_cell_params_to_xml(filename, cell_params, params_list):
    file_base = os.path.splitext(filename)[0]
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print("Saving cell params to : ", file_base)

    for i, p in enumerate(params_list):
        dolfin.File(file_base + "/param%d.xml" % i) << cell_params[p]
