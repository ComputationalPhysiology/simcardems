import logging

import dolfin

from . import utils

logger = logging.getLogger(__name__)


class EMCoupling:
    def __init__(
        self,
        mech_mesh,
        ep_mesh,
        lmbda=dolfin.Constant(1.0),
        Zetas=dolfin.Constant(0.0),
        Zetaw=dolfin.Constant(0.0),
    ) -> None:

        self.mesh = mech_mesh

        self.V_mech = dolfin.FunctionSpace(mech_mesh, "CG", 1)
        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        self.lmbda_mech = dolfin.Function(self.V_mech, name="lambda_mech")
        self.lmbda_mech.assign(lmbda)
        self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        self.Zetas_mech.assign(Zetas)
        self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")
        self.Zetaw_mech.assign(Zetaw)

        self.V_ep = dolfin.FunctionSpace(ep_mesh, "CG", 1)
        self.XS_ep = dolfin.Function(self.V_ep, name="XS_ep")
        self.XW_ep = dolfin.Function(self.V_ep, name="XW_ep")
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.lmbda_ep.assign(lmbda)
        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetas_ep.assign(Zetas)
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")
        self.Zetaw_ep.assign(Zetaw)

    def register_ep_model(self, solver):
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS_ep, self.XS_ep_assigner = utils.setup_assigner(self.vs, 40)
        self.XW_ep, self.XW_ep_assigner = utils.setup_assigner(self.vs, 41)

    def update_mechanics(self):
        self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))

    def interpolate_mechanics(self):
        self.XS_mech = dolfin.interpolate(self.XS_ep, self.V_mech)
        self.XW_mech = dolfin.interpolate(self.XW_ep, self.V_mech)

    def update_ep(self):
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.lmbda_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetas_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.Zetaw_ep)

    def interpolate_ep(self):
        self.lmbda_ep = dolfin.interpolate(self.lmbda_mech, self.V_ep)
        self.Zetas_ep = dolfin.interpolate(self.Zetas_mech, self.V_ep)
        self.Zetaw_ep = dolfin.interpolate(self.Zetaw_mech, self.V_ep)
