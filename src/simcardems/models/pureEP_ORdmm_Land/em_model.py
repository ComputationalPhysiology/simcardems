from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import beat
import numpy as np
from dolfin import FiniteElement  # noqa: F401
from dolfin import MixedElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from ... import save_load_functions as io
from ... import utils
from ...config import Config
from ...geometry import load_geometry
from ..em_model import BaseEMCoupling
from ..em_model import setup_EM_model

if TYPE_CHECKING:
    from ...datacollector import Assigners

logger = utils.getLogger(__name__)


class EMCoupling(BaseEMCoupling):
    @property
    def coupling_type(self):
        return "pureEP_ORdmm_Land"

    @property
    def vs(self) -> np.ndarray:
        return self.ep_solver.ode.values

    def register_ep_model(self, solver: beat.MonodomainSplittingSolver):
        logger.debug("Registering EP model")
        self.ep_solver = solver
        logger.debug("Done registering EP model")

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return NotImplemented

        if not super().__eq__(__o):
            return False

        if not np.allclose(self.vs, __o.vs):
            return False
        return True

    @property
    def assigners(self) -> Assigners:
        return self._assigners

    @assigners.setter
    def assigners(self, assigners) -> None:
        self._assigners = assigners

    def setup_assigners(self) -> None:
        from ...datacollector import Assigners

        self.assigners = Assigners(vs=self.vs, V_ep=self.ep_solver.pde.V)
        for name, index in [
            ("V", 0),
            ("Ca", 45),
            ("CaTrpn", 42),
            ("TmB", 43),
            ("Cd", 44),
            ("XS", 40),
            ("XW", 41),
        ]:
            self.assigners.register_subfunction(
                name=name,
                group="ep",
                subspace_index=index,
            )

    def solve_ep(self, interval: Tuple[float, float]) -> None:
        self.ep_solver.step(interval)

    def print_ep_info(self):
        # Output some degrees of freedom

        total_dofs = self.ep_solver.pde.V.dim()
        logger.info("EP model")
        utils.print_mesh_info(self.ep_mesh, total_dofs)

    def cell_params(self):
        return self.ep_solver.ode.parameters

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    def update_prev_ep(self):
        ...
        # self.ep_solver.vs_.assign(self.ep_solver.vs)

    def save_state(
        self,
        path: Union[str, Path],
        config: Optional[Config] = None,
    ) -> None:
        super().save_state(path=path, config=config)
        with io.h5pyfile(path, "a") as h5file:
            h5file["ep/vs"] = self.vs
            h5file["ep/cell_params"] = self.cell_params()

    @classmethod
    def from_state(
        cls,
        path: Union[str, Path],
        drug_factors_file: Union[str, Path] = "",
        popu_factors_file: Union[str, Path] = "",
        disease_state="healthy",
        PCL: float = 1000,
    ) -> BaseEMCoupling:
        logger.debug(f"Load state from path {path}")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File {path} does not exist")

        geo = load_geometry(path, schema_path=path.with_suffix(".json"))
        logger.debug("Open file with h5py")
        with io.h5pyfile(path) as h5file:
            config = Config(**io.h5_to_dict(h5file["config"]))
            state_params = io.h5_to_dict(h5file["state_params"])
            cell_params = h5file["ep"]["cell_params"][:]
            vs = h5file["ep"]["vs"][:]

        config.drug_factors_file = drug_factors_file
        config.popu_factors_file = popu_factors_file
        config.disease_state = disease_state
        config.PCL = PCL

        logger.debug("Load functions")

        from . import CellModel, ActiveModel

        return setup_EM_model(
            cls_EMCoupling=cls,
            cls_CellModel=CellModel,
            cls_ActiveModel=ActiveModel,
            geometry=geo,
            config=config,
            cell_inits=vs,
            cell_params=cell_params,
            state_params=state_params,
        )
