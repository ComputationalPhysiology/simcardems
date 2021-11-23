from pathlib import Path
from typing import Dict
from typing import List

import dolfin
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import utils
from .save_load_functions import h5pyfile

logger = utils.getLogger(__name__)


class DataCollector:
    def __init__(self, outdir, mesh, reset_state=True) -> None:
        self.outdir = Path(outdir)
        self._results_file = self.outdir.joinpath("results.h5")
        self.comm = mesh.mpi_comm()

        if reset_state:
            utils.remove_file(self._results_file)
        if not self._results_file.is_file():
            with dolfin.HDF5File(self.comm, self.results_file, "w") as h5file:
                h5file.write(mesh, "/mesh")

        self._xdmffiles: Dict[str, dolfin.XDMFFile] = {}
        self._functions: Dict[str, dolfin.Function] = {}

    @property
    def results_file(self):
        return self._results_file.as_posix()

    def register(self, name: str, f: dolfin.Function) -> None:
        if name in self.names:
            logger.info(f"Warning: {name} is allready registered - overwriting")
        self._xdmffiles[name] = dolfin.XDMFFile(
            self.comm,
            self.outdir.joinpath(f"{name}.xdmf").as_posix(),
        )
        self._functions[name] = f

    @property
    def names(self) -> List[str]:
        return list(self._functions.keys())

    def store(self, t):
        for name in self.names:
            f = self._functions[name]
            xdmf = self._xdmffiles[name]
            xdmf.write(f, t)

        with dolfin.HDF5File(self.comm, self.results_file, "a") as h5file:
            for name in self.names:
                f = self._functions[name]
                h5file.write(f, f"/{name}/{t:.2f}")


class DataLoader:
    def __init__(self, h5name) -> None:
        self._h5file = None
        self._h5name = Path(h5name)
        if not self._h5name.is_file():
            raise FileNotFoundError(f"File {h5name} does not exist")

        with h5pyfile(self._h5name) as h5file:

            # Check that we have mesh
            if "mesh" not in h5file:
                raise ValueError("No mesh in results file. Cannot load data")

            # Find the remining funcitons
            self.names = [name for name in h5file.keys() if name != "mesh"]
            if len(self.names) == 0:
                raise ValueError("No functions found in results file")
            # Get time stamps
            all_time_stamps = {
                name: sorted(list(h5file[name].keys()), key=lambda x: float(x))
                for name in self.names
            }
            # An verify that they are all the same
            self.time_stamps = all_time_stamps[self.names[0]]
            for name in self.names:
                assert self.time_stamps == all_time_stamps[name]

            if self.time_stamps is None or len(self.time_stamps) == 0:
                raise ValueError("No time stamps found")

            # Get the signatures - FIXME: Add a check that the signature exist.
            self._signatures = {
                name: h5file[name][self.time_stamps[0]].attrs["signature"].decode()
                for name in self.names
            }

        self.mesh = dolfin.Mesh()
        self._h5file = dolfin.HDF5File(
            self.mesh.mpi_comm(),
            self._h5name.as_posix(),
            "r",
        )

        self._create_functions()

    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

    def _create_functions(self):
        self._h5file.read(self.mesh, "/mesh", False)

        self._function_spaces = {
            signature: dolfin.FunctionSpace(self.mesh, eval(signature))
            for signature in set(self._signatures.values())
        }

        self._functions = {
            name: dolfin.Function(self._function_spaces[self._signatures[name]])
            for name in self.names
        }

    def get(self, name, t):
        if name not in self.names:
            raise KeyError(f"Invald name {name}")

        if t not in self.time_stamps:
            raise KeyError(f"Invalid time stamps {t}")

        func = self._functions[name]
        self._h5file.read(func, f"{name}/{t}/")
        return func
