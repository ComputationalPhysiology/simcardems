from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import dolfin
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import utils
from .save_load_functions import h5pyfile

logger = utils.getLogger(__name__)


class DataGroups(str, Enum):
    ep = "ep"
    mechanics = "mechanics"


class DataCollector:
    def __init__(self, outdir, mech_mesh, ep_mesh, reset_state=True) -> None:
        self.outdir = Path(outdir)
        self._results_file = self.outdir.joinpath("results.h5")
        self.comm = ep_mesh.mpi_comm()  # FIXME: Is this important?

        if reset_state:
            utils.remove_file(self._results_file)

        self._times_stamps = set()
        if not self._results_file.is_file():
            with dolfin.HDF5File(self.comm, self.results_file, "w") as h5file:
                h5file.write(mech_mesh, "/mechanics/mesh")
                h5file.write(ep_mesh, "/ep/mesh")

        else:

            try:
                with h5pyfile(self._results_file, "r") as f:
                    self._times_stamps = set(f["ep"]["V"].keys())
            except KeyError:
                pass

        self._functions: Dict[str, Dict[str, dolfin.Function]] = {
            "ep": {},
            "mechanics": {},
        }

    @property
    def results_file(self):
        return self._results_file.as_posix()

    def register(self, group: str, name: str, f: dolfin.Function) -> None:
        assert group in [
            "ep",
            "mechanics",
        ], f"Group has to be 'ep' or 'mechanics', got {group}"

        if name in self.names[group]:
            logger.warning(
                f"Warning: {name} in group {group} is already registered - overwriting",
            )
        self._functions[group][name] = f

    @property
    def names(self) -> Dict[str, List[str]]:
        return {k: list(v.keys()) for k, v in self._functions.items()}

    def store(self, t: float) -> None:

        t_str = f"{t:.2f}"
        logger.debug(f"Store results at time {t_str}")
        if f"{t_str}" in self._times_stamps:
            logger.info(f"Time stamp {t_str} already exist in file")
            return

        self._times_stamps.add(t_str)

        with dolfin.HDF5File(self.comm, self.results_file, "a") as h5file:
            for group, names in self.names.items():
                logger.debug(f"Save HDF5File {self.results_file} for group {group}")
                for name in names:
                    logger.debug(f"Save {name}")
                    f = self._functions[group][name]
                    h5file.write(f, f"{group}/{name}/{t_str}")


class DataLoader:
    def __init__(self, h5name) -> None:

        self._h5name = Path(h5name)
        if not self._h5name.is_file():
            raise FileNotFoundError(f"File {h5name} does not exist")

        with h5pyfile(self._h5name) as h5file:

            # Check that we have mesh

            if not ("ep" in h5file and "mesh" in h5file["ep"]):
                raise ValueError("No ep mesh in results file. Cannot load data")
            if not ("mechanics" in h5file and "mesh" in h5file["mechanics"]):
                raise ValueError("No mechanics mesh in results file. Cannot load data")

            # Find the remaining functions
            self.names = {
                group: [name for name in h5file[group].keys() if name != "mesh"]
                for group in ["ep", "mechanics"]
            }
            if len(self.names["ep"]) + len(self.names["mechanics"]) == 0:
                raise ValueError("No functions found in results file")

            # Get time stamps
            all_time_stamps = {
                "{group}:{name}": sorted(
                    list(h5file[group][name].keys()),
                    key=lambda x: float(x),
                )
                for group, names in self.names.items()
                for name in names
            }

            # An verify that they are all the same
            self.time_stamps = all_time_stamps[next(iter(all_time_stamps.keys()))]
            for name in all_time_stamps.keys():
                assert self.time_stamps == all_time_stamps[name], name

            if self.time_stamps is None or len(self.time_stamps) == 0:
                raise ValueError("No time stamps found")

            # Get the signatures - FIXME: Add a check that the signature exist.
            self._signatures = {
                group: {
                    name: h5file[group][name][self.time_stamps[0]]
                    .attrs["signature"]
                    .decode()
                    for name in names
                }
                for group, names in self.names.items()
            }

        self.ep_mesh = dolfin.Mesh()
        self.mech_mesh = dolfin.Mesh()
        self._h5file = dolfin.HDF5File(
            self.ep_mesh.mpi_comm(),
            self._h5name.as_posix(),
            "r",
        )

        self._create_functions()

    @property
    def size(self) -> int:
        return len(self.time_stamps)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self._h5name)})"

    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

    def _create_functions(self):
        self._h5file.read(self.ep_mesh, "/ep/mesh", True)
        self._h5file.read(self.mech_mesh, "/mechanics/mesh", True)

        self._function_spaces = {}

        for group, signature_dict in self._signatures.items():
            mesh = self.ep_mesh if group == "ep" else self.mech_mesh

            self._function_spaces.update(
                {
                    group: {
                        signature: dolfin.FunctionSpace(mesh, eval(signature))
                        for signature in set(signature_dict.values())
                    },
                },
            )

        self._functions = {
            group: {
                name: dolfin.Function(
                    self._function_spaces[group][self._signatures[group][name]],
                )
                for name in names
            }
            for group, names in self.names.items()
        }

    def get(
        self,
        group: DataGroups,
        name: str,
        t: Union[str, float],
    ) -> dolfin.Function:
        """Retrieve the function from the file

        Parameters
        ----------
        group : DataGroups
            The group where the function is stored, either
            'ep' or 'mechanics'
        name : str
            Name of the function
        t : Union[str, float]
            Time stamp you want to use. See `DataLoader.time_stamps`

        Returns
        -------
        dolfin.Function
            The function

        Raises
        ------
        KeyError
            If group does not exist in the file
        KeyError
            If name does not exist in group
        KeyError
            If 't' provided is not among the time stamps
        """
        if group not in self.names:
            raise KeyError(
                f"Cannot find group {group} in names, expected of of {self.names.keys()}",
            )

        names = self.names[group]
        if f"{name}" not in names:
            raise KeyError(
                f"Cannot find name {name} in group {group}. Possible options are {names}",
            )

        if isinstance(t, (int, float)):
            t = f"{t:.2f}"
        if t not in self.time_stamps:
            raise KeyError(f"Invalid time stamps {t}")

        func = self._functions[group][name]
        self._h5file.read(func, f"{group}/{name}/{t}/")
        return func
