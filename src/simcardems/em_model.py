import dolfin

from . import geometry as _geometry
from . import utils


logger = utils.getLogger(__name__)


class EMCoupling:
    def __init__(
        self,
        geometry: _geometry.BaseGeometry,
        lmbda_mech=dolfin.Constant(1.0),
        Zetas_mech=dolfin.Constant(0.0),
        Zetaw_mech=dolfin.Constant(0.0),
    ) -> None:
        logger.debug("Create EM coupling")
        self.geometry = geometry
        self.V_mech = dolfin.FunctionSpace(self.mech_mesh, "CG", 1)
        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        self.lmbda_mech = dolfin.Function(self.V_mech, name="lambda_mech")
        self.lmbda_mech.assign(lmbda_mech)
        self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        self.Zetas_mech.assign(Zetas_mech)
        self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")
        self.Zetaw_mech.assign(Zetaw_mech)

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.XS_ep = dolfin.Function(self.V_ep, name="XS_ep")
        self.XW_ep = dolfin.Function(self.V_ep, name="XW_ep")
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")
        self.mechanics_to_coupling()

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS_ep, self.XS_ep_assigner = utils.setup_assigner(self.vs, 40)
        self.XW_ep, self.XW_ep_assigner = utils.setup_assigner(self.vs, 41)
        self.coupling_to_mechanics
        logger.debug("Done registering EP model")

    def ep_to_coupling(self):
        logger.debug("Update mechanics")
        self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))
        logger.debug("Done updating mechanics")

    def coupling_to_mechanics(self):
        logger.debug("Interpolate mechanics")
        self.XS_mech.assign(dolfin.interpolate(self.XS_ep, self.V_mech))
        self.XW_mech.assign(dolfin.interpolate(self.XW_ep, self.V_mech))
        logger.debug("Done interpolating mechanics")

    def mechanics_to_coupling(self):
        logger.debug("Interpolate EP")
        self.lmbda_ep.assign(dolfin.interpolate(self.lmbda_mech, self.V_ep))
        self.Zetas_ep.assign(dolfin.interpolate(self.Zetas_mech, self.V_ep))
        self.Zetaw_ep.assign(dolfin.interpolate(self.Zetaw_mech, self.V_ep))
        logger.debug("Done interpolating EP")

    def coupling_to_ep(self):
        logger.debug("Update EP")
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.lmbda_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetas_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.Zetaw_ep)
        logger.debug("Done updating EP")
