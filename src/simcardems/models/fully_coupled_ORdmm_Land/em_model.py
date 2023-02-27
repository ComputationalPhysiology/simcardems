from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import dolfin
import numpy as np
from dolfin import FiniteElement  # noqa: F401
from dolfin import MixedElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from ... import save_load_functions as io
from ... import utils
from ...config import Config
from ...geometry import BaseGeometry
from ...geometry import load_geometry
from ...time_stepper import TimeStepper
from ..em_model import BaseEMCoupling
from ..em_model import setup_EM_model

if TYPE_CHECKING:
    from ... import mechanics_model
    from ... import datacollector

logger = utils.getLogger(__name__)


class EMCoupling(BaseEMCoupling):
    def __init__(
        self,
        geometry: BaseGeometry,
        **state_params,
    ) -> None:
        super().__init__(geometry=geometry, **state_params)

        self.V_mech = dolfin.FunctionSpace(self.mech_mesh, "CG", 1)
        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")

    @property
    def coupling_type(self):
        return "fully_coupled_ORdmm_Land"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return NotImplemented

        if not super().__eq__(__o):
            return False

        for attr in [
            "vs",
            "mech_state",
            "lmbda_mech",
            "Zetas_mech",
            "Zetaw_mech",
            "lmbda_ep",
            "Zetas_ep",
            "Zetaw_ep",
            "XS_mech",
            "XW_mech",
        ]:
            if not np.allclose(
                getattr(self, attr).vector().get_local(),
                getattr(__o, attr).vector().get_local(),
            ):
                logger.info(f"{attr} differs in equality")
                return False

        return True

    def register_time_stepper(self, time_stepper: TimeStepper) -> None:
        super().register_time_stepper(time_stepper)
        self.mech_solver.material.active.register_time_stepper(time_stepper)

    @property
    def dt_mechanics(self) -> float:
        return self.mech_solver.material.active.dt

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    @property
    def mech_state(self) -> dolfin.Function:
        return self.mech_solver.state

    @property
    def vs(self) -> dolfin.Function:
        return self.ep_solver.solution_fields()[0]

    @property
    def assigners(self) -> datacollector.Assigners:
        return self._assigners

    @assigners.setter
    def assigners(self, assigners) -> None:
        self._assigners = assigners

    def setup_assigners(self) -> None:
        from ...datacollector import Assigners

        self.assigners = Assigners(vs=self.vs, mech_state=self.mech_state)
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

        self.assigners.register_subfunction(
            name="u",
            group="mechanics",
            subspace_index=self.mech_solver.u_subspace_index,
        )

        for name, index in [
            ("XS", 40),
            ("XW", 41),
        ]:
            self.assigners.register_subfunction(
                name=name,
                group="ep",
                subspace_index=index,
                is_pre=True,
            )

        self.coupling_to_mechanics()

    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self.ep_solver = solver

        if hasattr(self, "mech_solver"):
            self.mechanics_to_coupling()
        self.coupling_to_mechanics()
        logger.debug("Done registering EP model")

    def register_mech_model(self, solver: mechanics_model.MechanicsProblem):
        logger.debug("Registering mech model")
        self.mech_solver = solver

        self.Zetas_mech = solver.material.active.Zetas_prev
        self.Zetaw_mech = solver.material.active.Zetaw_prev
        self.lmbda_mech = solver.material.active.lmbda_prev

        # Note sure why we need to do this for the LV?
        self.lmbda_mech.set_allow_extrapolation(True)
        self.Zetas_mech.set_allow_extrapolation(True)
        self.Zetaw_mech.set_allow_extrapolation(True)

        self.mechanics_to_coupling()
        if hasattr(self, "ep_solver"):
            self.coupling_to_mechanics()
        logger.debug("Done registering EP model")

    def update_prev_mechanics(self):
        self.mech_solver.material.active.update_prev()

    def update_prev_ep(self):
        self.ep_solver.vs_.assign(self.ep_solver.vs)

    def ep_to_coupling(self):
        logger.debug("Update mechanics")
        self.assigners.assign_ep()
        logger.debug("Done updating mechanics")

    def coupling_to_mechanics(self):
        logger.debug("Interpolate mechanics")
        if hasattr(self, "_assigners"):
            self.XS_mech.interpolate(self.assigners.functions["ep"]["XS"])
            self.XW_mech.interpolate(self.assigners.functions["ep"]["XW"])
        logger.debug("Done interpolating mechanics")

    def mechanics_to_coupling(self):
        logger.debug("Interpolate EP")
        self.lmbda_ep.interpolate(self.lmbda_mech)
        self.Zetas_ep.interpolate(self.Zetas_mech)
        self.Zetaw_ep.interpolate(self.Zetaw_mech)
        logger.debug("Done interpolating EP")

    def coupling_to_ep(self):
        logger.debug("Update EP")
        logger.debug("Done updating EP")

    def solve_mechanics(self) -> None:
        logger.debug("Solve mechanics")
        self.mech_solver.solve()

    def solve_ep(self, interval: Tuple[float, float]) -> None:
        logger.debug("Solve EP")
        self.ep_solver.step(interval)

    def print_mechanics_info(self):
        total_dofs = self.mech_state.function_space().dim()
        utils.print_mesh_info(self.mech_mesh, total_dofs)
        logger.info("Mechanics model")

    def print_ep_info(self):
        # Output some degrees of freedom
        total_dofs = self.vs.function_space().dim()
        logger.info("EP model")
        utils.print_mesh_info(self.ep_mesh, total_dofs)

    def cell_params(self):
        return self.ep_solver.ode_solver._model.parameters()

    def register_datacollector(self, collector: datacollector.DataCollector) -> None:
        super().register_datacollector(collector=collector)

        collector.register("ep", "Zetas", self.Zetas_ep)
        collector.register("ep", "Zetaw", self.Zetaw_ep)
        collector.register("ep", "lambda", self.lmbda_ep)
        collector.register("mechanics", "XS", self.XS_mech)
        collector.register("mechanics", "XW", self.XW_mech)
        collector.register("mechanics", "Zetas", self.Zetas_mech)
        collector.register("mechanics", "Zetaw", self.Zetaw_mech)
        collector.register("mechanics", "lambda", self.lmbda_mech)
        collector.register(
            "mechanics",
            "Ta",
            self.mech_solver.material.active.Ta_current,
        )
        self.mech_solver.solver.register_datacollector(collector)

    def save_state(
        self,
        path: Union[str, Path],
        config: Optional[Config] = None,
    ) -> None:
        super().save_state(path=path, config=config)

        with dolfin.HDF5File(
            self.geometry.comm(),
            Path(path).as_posix(),
            "a",
        ) as h5file:
            h5file.write(self.lmbda_mech, "/em/lmbda_prev")
            h5file.write(self.Zetas_mech, "/em/Zetas_prev")
            h5file.write(self.Zetaw_mech, "/em/Zetaw_prev")
            h5file.write(self.ep_solver.vs, "/ep/vs")
            h5file.write(self.mech_solver.state, "/mechanics/state")

        io.dict_to_h5(
            self.cell_params(),
            path,
            "ep/cell_params",
        )

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
            cell_params = io.h5_to_dict(h5file["ep"]["cell_params"])
            vs_signature = h5file["ep"]["vs"].attrs["signature"].decode()
            mech_signature = h5file["mechanics"]["state"].attrs["signature"].decode()

        config.drug_factors_file = drug_factors_file
        config.popu_factors_file = popu_factors_file
        config.disease_state = disease_state
        config.PCL = PCL

        VS = dolfin.FunctionSpace(geo.ep_mesh, eval(vs_signature))
        vs = dolfin.Function(VS)

        W = dolfin.FunctionSpace(geo.mechanics_mesh, eval(mech_signature))
        mech_state = dolfin.Function(W)

        # FIXME: load this signature from the file as well
        V = dolfin.FunctionSpace(geo.mechanics_mesh, "CG", 1)
        lmbda_prev = dolfin.Function(V, name="lambda")
        Zetas_prev = dolfin.Function(V, name="Zetas")
        Zetaw_prev = dolfin.Function(V, name="Zetaw")
        logger.debug("Load functions")
        with dolfin.HDF5File(geo.ep_mesh.mpi_comm(), path.as_posix(), "r") as h5file:
            h5file.read(vs, "/ep/vs")
            h5file.read(mech_state, "/mechanics/state")
            h5file.read(lmbda_prev, "/em/lmbda_prev")
            h5file.read(Zetas_prev, "/em/Zetas_prev")
            h5file.read(Zetaw_prev, "/em/Zetaw_prev")

        from . import CellModel, ActiveModel

        cell_inits = io.vs_functions_to_dict(
            vs,
            state_names=CellModel.default_initial_conditions().keys(),
        )

        cls_ActiveModel = partial(
            ActiveModel,
            Zetas=Zetas_prev,
            Zetaw=Zetaw_prev,
            lmbda=lmbda_prev,
        )

        return setup_EM_model(
            cls_EMCoupling=cls,
            cls_CellModel=CellModel,
            cls_ActiveModel=cls_ActiveModel,
            geometry=geo,
            config=config,
            cell_inits=cell_inits,
            cell_params=cell_params,
            mech_state_init=mech_state,
            state_params=state_params,
        )
