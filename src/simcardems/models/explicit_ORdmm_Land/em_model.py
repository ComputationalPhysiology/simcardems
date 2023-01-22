from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import cbcbeat
import dolfin
import ufl

from ... import utils
from ...config import Config
from ...geometry import BaseGeometry
from ..em_model import BaseEMCoupling

if TYPE_CHECKING:
    from ...datacollector import DataCollector, Assigners
    from ...mechanics_model import MechanicsProblem


logger = utils.getLogger(__name__)


def Ta(XS, XW, Zetas, Zetaw, lmbda, Tref, rs, Beta0):

    _min = ufl.min_value
    _max = ufl.max_value
    if isinstance(lmbda, (int, float)):
        _min = min
        _max = max
    lmbda = _min(1.2, lmbda)
    h_lambda_prima = 1 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
    h_lambda = _max(0, h_lambda_prima)

    return h_lambda * (Tref / rs) * (XS * (Zetas + 1) + XW * Zetaw)


class EMCoupling(BaseEMCoupling):
    def __init__(
        self,
        geometry: BaseGeometry,
        lmbda: Optional[dolfin.Function] = None,
    ) -> None:
        super().__init__(geometry=geometry)

        self.V_mech = dolfin.FunctionSpace(self.mech_mesh, "CG", 1)
        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")
        self._lmbda_mech_func = dolfin.Function(self.V_mech, name="Zetaw_mech")

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.XS_ep = dolfin.Function(self.V_ep, name="XS_ep")
        self.XW_ep = dolfin.Function(self.V_ep, name="XW_ep")
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.lmbda_ep_prev = dolfin.Function(self.V_ep, name="lambda_ep_prev")
        self._dLambda = dolfin.Function(self.V_ep, name="dLambda")
        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")

        # self.W_ep = dolfin.VectorFunctionSpace(self.ep_mesh, "CG", 2)
        # self.u_ep = dolfin.Function(self.W_ep, name="u_ep")

        self._projector_V_ep = utils.Projector(self.V_ep)
        self._projector_V_mech = utils.Projector(self.V_mech)

        if lmbda is not None:
            self.lmbda_ep.vector()[:] = lmbda.vector()
            self.lmbda_ep_prev.vector()[:] = lmbda.vector()

    def members(self) -> Dict[str, dolfin.Function]:
        return {
            "/em/lmbda_prev": self.lmbda_ep_prev,
            "/ep/vs": self.ep_solver.vs,
            "/mechanics/state": self.mech_solver.state,
        }

    @property
    def assigners(self) -> Assigners:
        return self._assigners

    @assigners.setter
    def assigners(self, assigners) -> None:
        self._assigners = assigners

    def setup_assigners(self) -> None:
        from ...datacollector import Assigners

        self.assigners = Assigners(vs=self.vs, mech_state=self.mech_state)
        for name, index in [
            ("v", 0),
            ("Ca", 45),
            ("XS", 45),
            ("XW", 45),
            ("CaTrpn", 45),
            ("TmB", 45),
            ("Cd", 45),
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
            ("XS", 45),
            ("XW", 45),
        ]:
            self.assigners.register_subfunction(
                name=name,
                group="ep",
                subspace_index=index,
                is_pre=True,
            )

        self.assigners.register_subfunction(
            name="u",
            group="mechanics",
            subspace_index=self._u_subspace_index,
        )

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    @property
    def mech_state(self):
        return self.mech_solver.state

    @property
    def dLambda_ep(self):
        self._dLambda.vector()[:] = self.lmbda_ep.vector() - self.lmbda_ep_prev.vector()
        return self._dLambda

    @property
    def lmbda_mech(self):
        F = dolfin.grad(self.u_mech) + dolfin.Identity(3)
        f = F * self.geometry.f0
        return dolfin.sqrt(f**2)

    @property
    def lmbda_mech_func(self):
        self._projector_V_mech(self._lmbda_mech_func, self.lmbda_mech)
        return self._lmbda_mech_func

    def update_prev(self):
        # Update previous lmbda
        self.lmbda_ep_prev.vector()[:] = self.lmbda_ep.vector()

    def _project_lmbda(self):
        F = dolfin.grad(self.u_ep) + dolfin.Identity(3)
        f = F * self.geometry.f0_ep
        self._projector_V_ep(self.lmbda_ep, dolfin.project(dolfin.sqrt(f**2)))

    def register_ep_model(self, solver: cbcbeat.SplittingSolver):
        logger.debug("Registering EP model")
        self.ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS_ep, self.XS_ep_assigner = utils.setup_assigner(self.vs, 40)
        self.XW_ep, self.XW_ep_assigner = utils.setup_assigner(self.vs, 41)
        self.Zetas_ep, self.Zetas_ep_assigner = utils.setup_assigner(self.vs, 46)
        self.Zetaw_ep, self.Zetaw_ep_assigner = utils.setup_assigner(self.vs, 47)

        params = self.ep_solver.ode_solver._model.parameters()
        self._Tref = params["Tref"]
        self._rs = params["rs"]
        self._Beta0 = params["Beta0"]
        if hasattr(self, "u_mech"):
            self.coupling_to_mechanics()
        logger.debug("Done registering EP model")

    def register_mech_model(self, solver: MechanicsProblem):
        logger.debug("Registering mech model")
        self.mech_solver = solver
        self.Ta_mech = self.mech_solver.material.activation

        self._u_subspace_index = (
            1 if type(solver).__name__ == "RigidMotionProblem" else 0
        )
        self.u_mech, self.u_mech_assigner = utils.setup_assigner(
            solver.state,
            self._u_subspace_index,
        )
        self.mechanics_to_coupling()
        logger.debug("Done registering EP model")

    def register_datacollector(self, collector: "DataCollector") -> None:
        super().register_datacollector(collector=collector)

        collector.register("ep", "lmbda", self.lmbda_ep)
        collector.register("ep", "dLmbda", self.dLambda_ep)
        collector.register("mechanics", "Ta", self.Ta_mech)

    def ep_to_coupling(self):
        logger.debug("Transfer variables from EP to coupling")
        self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))
        self.Zetas_ep_assigner.assign(self.Zetas_ep, utils.sub_function(self.vs, 46))
        self.Zetaw_ep_assigner.assign(self.Zetaw_ep, utils.sub_function(self.vs, 47))

        logger.debug("Done transferring variables from EP to coupling")

    def coupling_to_mechanics(self):
        logger.debug("Transfer variables from coupling to mechanics")
        self.XS_mech.interpolate(self.XS_ep)
        self.XW_mech.interpolate(self.XW_ep)
        self.Zetas_mech.interpolate(self.Zetas_ep)
        self.Zetaw_mech.interpolate(self.Zetaw_ep)
        self._projector_V_mech(
            self.Ta_mech,
            Ta(
                XS=self.XS_mech,
                XW=self.XW_mech,
                Zetas=self.Zetas_mech,
                Zetaw=self.Zetaw_mech,
                lmbda=self.lmbda_mech,
                Tref=self._Tref,
                rs=self._rs,
                Beta0=self._Beta0,
            ),
        )
        logger.debug("Done transferring variables from coupling to mechanics")

    def mechanics_to_coupling(self):
        logger.debug("Transfer variables from mechanics to coupling")
        self.u_mech_assigner.assign(
            self.u_mech,
            utils.sub_function(self.mech_state, self._u_subspace_index),
        )
        self.lmbda_ep.interpolate(self.lmbda_mech_func)
        # self.u_ep.interpolate(self.u_mech)
        # self._project_lmbda()

        logger.debug("Done transferring variables from mechanics to coupling")

    def coupling_to_ep(self):
        logger.debug("Update EP")
        logger.debug("Done updating EP")

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

    def save_state(
        self,
        path: Union[str, Path],
        config: Optional[Config] = None,
        **state_params,
    ) -> None:
        ...

    @classmethod
    def from_state(cls, path: Union[str, Path]) -> EMCoupling:
        ...
