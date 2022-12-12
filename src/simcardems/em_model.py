import dolfin

from . import geometry as _geometry
from . import utils
from .mechanics_model import MechanicsProblem


logger = utils.getLogger(__name__)


class EMCoupling:
    def __init__(
        self,
        geometry: _geometry.BaseGeometry,
    ) -> None:
        logger.debug("Create EM coupling")
        self.geometry = geometry
        # self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        # self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        # self.lmbda_mech = dolfin.Function(self.V_mech, name="lambda_mech")
        # self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        # self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.W_ep = dolfin.VectorFunctionSpace(self.ep_mesh, "CG", 2)
        # self.XS_ep = dolfin.Function(self.V_ep, name="XS_ep")
        # self.XW_ep = dolfin.Function(self.V_ep, name="XW_ep")
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.lmbda_ep.vector()[:] = 1.0
        self.lmbda_ep_prev = dolfin.Function(self.V_ep, name="lambda_ep_prev")
        self.lmbda_ep_prev.vector()[:] = 1.0
        self._dLambda = dolfin.Function(self.V_ep, name="dLambda")
        self.Ta_ep = dolfin.Function(self.V_ep, name="Ta_ep")

        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")

        self.u_ep = dolfin.Function(self.W_ep, name="u_ep")

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    @property
    def mech_state(self):
        return self._mech_solver.state

    @property
    def dLambda_ep(self):
        self._dLambda.vector()[:] = self.lmbda_ep.vector() - self.lmbda_ep_prev.vector()
        return self._dLambda

    def _project_lmbda(self):

        F = dolfin.grad(self.u_ep) + dolfin.Identity(3)
        f = F * self.geometry.f0_ep
        self.lmbda_ep.assign(dolfin.project(dolfin.sqrt(f**2), self.V_ep))

    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        # self.XS_ep, self.XS_ep_assigner = utils.setup_assigner(self.vs, 40)
        # self.XW_ep, self.XW_ep_assigner = utils.setup_assigner(self.vs, 41)
        # self.coupling_to_mechanics()
        logger.debug("Done registering EP model")

    def register_mech_model(self, solver: MechanicsProblem):
        logger.debug("Registering mech model")
        self._mech_solver = solver
        self.Ta = self._mech_solver.material.activation

        self._u_subspace_index = (
            1 if type(solver).__name__ == "RigidMotionProblem" else 0
        )
        self.u_mech, self.u_mech_assigner = utils.setup_assigner(
            solver.state,
            self._u_subspace_index,
        )
        # Don't know why we need to set this one
        # self.u_mech.set_allow_extrapolation(True)
        self.f0 = solver.material.f0

        # self.Zetas_mech = solver.material.active.Zetas_prev
        # self.Zetaw_mech = solver.material.active.Zetaw_prev
        # self.lmbda_mech = solver.material.active.lmbda_prev

        # # Note sure why we need to do this for the LV?
        # self.lmbda_mech.set_allow_extrapolation(True)
        # self.Zetas_mech.set_allow_extrapolation(True)
        # self.Zetaw_mech.set_allow_extrapolation(True)

        # self.mechanics_to_coupling()
        logger.debug("Done registering EP model")

    def ep_to_coupling(self):
        logger.debug("Update mechanics")
        # Save the last lambda to be used for computing next dLambda
        self.lmbda_ep_prev.vector()[:] = self.lmbda_ep.vector()
        self.Ta_ep.assign(
            dolfin.project(self._ep_solver.ode_solver._model.Ta(self.vs), self.V_ep),
        )

        # self.Ta_ep.assign(dolfin.project(self._ep_solver))
        # self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        # self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))
        logger.debug("Done updating mechanics")

    def coupling_to_mechanics(self):
        logger.debug("Interpolate mechanics")
        # self.XS_mech.interpolate(self.XS_ep)
        # self.XW_mech.interpolate(self.XW_ep)
        self.Ta.interpolate(self.Ta_ep)
        logger.debug("Done interpolating mechanics")

    def mechanics_to_coupling(self):
        logger.debug("Interpolate EP")
        # self.lmbda_ep.interpolate(self.lmbda_mech)
        # self.Zetas_ep.interpolate(self.Zetas_mech)
        # self.Zetaw_ep.interpolate(self.Zetaw_mech)

        self.u_mech_assigner.assign(
            self.u_mech,
            utils.sub_function(self.mech_state, self._u_subspace_index),
        )
        self.u_ep.assign(dolfin.interpolate(self.u_mech, self.W_ep))
        self._project_lmbda()

        logger.debug("Done interpolating EP")

    def coupling_to_ep(self):
        logger.debug("Update EP")
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.Zetas_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetaw_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.lmbda_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 49), self.dLambda_ep)
        logger.debug("Done updating EP")
