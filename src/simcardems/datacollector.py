from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import cbcbeat
import dolfin
import numpy as np
import pulse
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from . import utils
from .geometry import BaseGeometry
from .geometry import load_geometry
from .save_load_functions import h5pyfile
from .value_extractor import ValueExtractor

logger = utils.getLogger(__name__)


class DataGroups(Enum):
    ep = "ep"
    mechanics = "mechanics"


class Assigners:
    """Helper class to assign subfunctions from EP and Mechanics state"""

    def __init__(
        self,
        *,
        vs: Optional[cbcbeat.SplittingSolver] = None,
        mech_state: Optional[pulse.MechanicsProblem] = None,
    ) -> None:
        self.vs = vs
        self.mech_state = mech_state

        self.functions: Dict[str, Dict[str, dolfin.Function]] = {
            "ep": {},
            "mechanics": {},
        }
        self.assigners: Dict[str, Dict[str, dolfin.FunctionAssigner]] = {
            "ep": {},
            "mechanics": {},
        }
        self.subspace_indices: Dict[str, Dict[str, int]] = {
            "ep": {},
            "mechanics": {},
        }

        self.pre_functions: Dict[str, Dict[str, dolfin.Function]] = {
            "ep": {},
            "mechanics": {},
        }
        self.pre_assigners: Dict[str, Dict[str, dolfin.FunctionAssigner]] = {
            "ep": {},
            "mechanics": {},
        }
        self.pre_subspace_indices: Dict[str, Dict[str, int]] = {
            "ep": {},
            "mechanics": {},
        }

    def assign(self) -> None:
        self.assign_mechanics()
        self.assign_ep()

    def assign_pre(self) -> None:
        self.assign_pre_mechanics()
        self.assign_ep()

    def _get(
        self,
        is_pre: bool,
    ) -> Tuple[
        Dict[str, Dict[str, dolfin.Function]],
        Dict[str, Dict[str, dolfin.FunctionAssigner]],
        Dict[str, Dict[str, int]],
    ]:
        if is_pre:
            return (self.pre_functions, self.pre_assigners, self.pre_subspace_indices)
        else:
            return (self.functions, self.assigners, self.subspace_indices)

    def _assign_mechanics(self, is_pre: bool) -> None:
        functions, assigners, subspace_indices = self._get(is_pre)

        keys = assigners["mechanics"].keys()
        if len(keys) == 0:
            return

        if self.mech_state is None:
            raise RuntimeError(
                "Unable to assign mechanics, no mechanics state registered",
            )
        for name in keys:
            assigner = assigners["mechanics"].get(name)
            index = subspace_indices["mechanics"].get(name)
            f = functions["mechanics"].get(name)
            assigner.assign(f, utils.sub_function(self.mech_state, index))

    def assign_pre_mechanics(self) -> None:
        self._assign_mechanics(is_pre=True)

    def assign_mechanics(self) -> None:
        self._assign_mechanics(is_pre=False)

    def _assign_ep(self, is_pre: bool) -> None:

        functions, assigners, subspace_indices = self._get(is_pre)

        keys = assigners["ep"].keys()
        if len(keys) == 0:
            return
        if self.vs is None:
            raise RuntimeError("Unable to assign EP, no EP state registered")

        for name in keys:
            assigner = assigners["ep"].get(name)
            index = subspace_indices["ep"].get(name)
            f = functions["ep"].get(name)

            assigner.assign(f, utils.sub_function(self.vs, index))

    def assign_pre_ep(self) -> None:
        self._assign_ep(is_pre=True)

    def assign_ep(self) -> None:
        self._assign_ep(is_pre=False)

    def compute_pre_norm(self):
        norm = 0.0
        hit = False
        for group in ["ep", "mechanics"]:
            for name, pre_func in self.pre_functions[group].items():
                if name in self.functions[group]:
                    hit = True
                    norm += utils.compute_norm(self.functions[group][name], pre_func)

        if not hit:
            # We never computed any norm so we should not interpret
            # the norm as zero, but rather assume that the norm is large
            # so that we actually solve in case we rely on this value as
            # a threshold
            return np.inf
        return norm

    def register_subfunction(
        self,
        name: str,
        group: Literal["ep", "mechanics"],
        subspace_index: int,
        is_pre: bool = False,
    ) -> None:
        func = self.vs if group == "ep" else self.mech_state
        f, assigner = utils.setup_assigner(func, subspace_index)
        functions, assigners, subspace_indices = self._get(is_pre)
        functions[group][name] = f
        assigners[group][name] = assigner
        subspace_indices[group][name] = subspace_index


class DataCollector:
    def __init__(self, outdir, geo: BaseGeometry, reset_state=True) -> None:
        self.outdir = Path(outdir)
        self._results_file = self.outdir.joinpath("results.h5")
        self.comm = geo.mesh.mpi_comm()  # FIXME: Is this important?

        if reset_state:
            utils.remove_file(self._results_file)

        self._times_stamps = set()
        if not self._results_file.is_file():
            geo.dump(self.results_file)

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

    def save_residual(self, residual, index):
        logger.debug("Save residual")
        with h5pyfile(self._results_file, "a") as h5file:
            if "residual" not in h5file:
                h5file.create_group("residual")
            h5file["residual"].create_dataset(str(index), data=residual)


class DataLoader:
    def __init__(self, h5name) -> None:

        self._h5file = None
        self._h5name = Path(h5name)
        if not self._h5name.is_file():
            raise FileNotFoundError(f"File {h5name} does not exist")

        self.geo = load_geometry(self._h5name)

        with h5pyfile(self._h5name) as h5file:

            # Find the remaining functions
            self.names = {
                group: [name for name in h5file.get(group, {}).keys()]
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

            if "residual" in h5file:
                self._residual = [
                    h5file["residual"][k][...] for k in h5file["residual"].keys()
                ]
            else:
                self._residual = []

        self._h5file = dolfin.HDF5File(
            self.ep_mesh.mpi_comm(),
            self._h5name.as_posix(),
            "r",
        )

        self._create_functions()
        self.value_extractor = ValueExtractor(self.geo)

    @property
    def residual(self) -> List[np.ndarray]:
        return self._residual

    @property
    def ep_mesh(self):
        return self.geo.ep_mesh

    @property
    def mech_mesh(self):
        return self.geo.mechanics_mesh

    @property
    def size(self) -> int:
        return len(self.time_stamps)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self._h5name)})"

    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

    def _create_functions(self):
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

        self._dofs = {
            group: {
                name: func.function_space().tabulate_dof_coordinates()
                for name, func in functions.items()
            }
            for group, functions in self._functions.items()
        }

    def extract_value(
        self,
        group: DataGroups,
        name: str,
        t: Union[str, float],
        reduction: str,
    ):
        func = self.get(group, name, t)
        dofs = self._dofs[self._group_to_str(group)][name]
        return self.value_extractor.eval(func, value=reduction, dofs=dofs)

    def _group_to_str(self, group):
        group_str = utils.enum2str(group, DataGroups)
        if group_str not in self.names:
            raise KeyError(
                f"Cannot find group {group} in names, expected of of {self.names.keys()}",
            )
        return group_str

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
        if self._h5file is None:
            raise RuntimeError(f"Unable to read from file {self._h5name}")

        group_str = self._group_to_str(group)
        names = self.names[group_str]
        if f"{name}" not in names:
            raise KeyError(
                f"Cannot find name {name} in group {group}. Possible options are {names}",
            )

        if isinstance(t, (int, float)):
            t = f"{t:.2f}"
        if t not in self.time_stamps:
            raise KeyError(f"Invalid time stamps {t}")

        func = self._functions[group_str][name]
        self._h5file.read(func, f"{group_str}/{name}/{t}/")
        return func
