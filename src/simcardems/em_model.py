import dolfin

from . import geometry as _geometry
from . import land_model
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
        self.V_mech = dolfin.FunctionSpace(self.mech_mesh, "CG", 1)
        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        self.lmbda_mech = dolfin.Function(self.V_mech, name="lambda_mech")
        # self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        # self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.XS_ep = dolfin.Function(self.V_ep, name="XS_ep")
        self.XW_ep = dolfin.Function(self.V_ep, name="XW_ep")

        # self.Q_ep = pulse.QuadratureSpace(self.ep_mesh, degree=3, dim=1)
        self.Q_ep = self.V_ep  # This should eventually be a quadrature space
        self.lmbda_ep = dolfin.Function(self.Q_ep, name="lambda_ep")
        self.lmbda_ep_prev = dolfin.Function(self.Q_ep, name="lambda_ep_prev")
        self.lmbda_ep_prev.vector()[:] = 1.0
        self._dLambda_ep = dolfin.Function(self.Q_ep, name="lambda_ep")
        self.Zetas_ep = dolfin.Function(self.Q_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.Q_ep, name="Zetaw_ep")
        self.Zetas_ep_prev = dolfin.Function(self.Q_ep, name="Zetas_ep_prev")
        self.Zetaw_ep_prev = dolfin.Function(self.Q_ep, name="Zetaw_ep_prev")

        self.W_ep = dolfin.VectorFunctionSpace(self.ep_mesh, "CG", 2)
        self.u_ep = dolfin.Function(self.W_ep)

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    @property
    def dLambda(self):
        self._dLambda_ep.vector()[:] = (
            self.lmbda_ep.vector() - self.lmbda_ep_prev.vector()
        )
        return self._dLambda_ep

    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS_ep, self.XS_ep_assigner = utils.setup_assigner(self.vs, 40)
        self.XW_ep, self.XW_ep_assigner = utils.setup_assigner(self.vs, 41)
        self.coupling_to_mechanics()
        logger.debug("Done registering EP model")

    def register_mech_model(self, solver: MechanicsProblem):
        logger.debug("Registering mech model")
        self._mech_solver = solver

        self._u_subspace_index = 1 if solver.boundary_condition == "rigid" else 0
        self.u_mech, self.u_mech_assigner = utils.setup_assigner(
            solver.state,
            self._u_subspace_index,
        )
        self.f0 = solver.material.f0

        # self.Zetas_mech = solver.material.active.Zetas_prev
        # self.Zetaw_mech = solver.material.active.Zetaw_prev
        # self.lmbda_mech = solver.material.active.lmbda_prev
        self.mech_solver = solver
        self.mechanics_to_coupling()
        logger.debug("Done registering EP model")

    @property
    def mech_state(self):
        return self.mech_solver.state

    def ep_to_coupling(self):
        logger.debug("Update mechanics")
        self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))
        logger.debug("Done updating mechanics")

    def coupling_to_mechanics(self):
        logger.debug("Interpolate mechanics")
        # print("XS = ", self.XS_mech.vector().get_local())
        # print("XW = ", self.XW_mech.vector().get_local())
        self.XS_mech.assign(dolfin.interpolate(self.XS_ep, self.V_mech))
        self.XW_mech.assign(dolfin.interpolate(self.XW_ep, self.V_mech))
        logger.debug("Done interpolating mechanics")

    def _project_lmbda(self):
        F = dolfin.grad(self.u_mech) + dolfin.Identity(3)
        f = F * self.f0
        self.lmbda_ep.assign(dolfin.project(dolfin.sqrt(f**2), self.Q_ep))

    def _compute_zeta(self):
        self.Zetaw_ep.vector()[:] = land_model.advance_Zeta(
            self.Zetaw_ep_prev.vector(),
            self.mech_solver.material.active.Aw,
            self.mech_solver.material.active.cw,
            self.dLambda.vector(),
            self.mech_solver.material.active.dt,
            self.mech_solver.material.active._scheme,
        )
        self.Zetas_ep.vector()[:] = land_model.advance_Zeta(
            self.Zetas_ep_prev.vector(),
            self.mech_solver.material.active.Aw,
            self.mech_solver.material.active.cw,
            self.dLambda.vector(),
            self.mech_solver.material.active.dt,
            self.mech_solver.material.active._scheme,
        )

    def update_prev(self):
        self.Zetas_ep_prev.vector()[:] = self.Zetas_ep.vector()
        self.Zetaw_ep_prev.vector()[:] = self.Zetaw_ep.vector()
        self.lmbda_ep_prev.vector()[:] = self.lmbda_ep.vector()

    def mechanics_to_coupling(self):
        logger.debug("Interpolate EP")
        # print("Zetas = ", self.Zetas_mech.vector().get_local())
        # print("Zetaw = ", self.Zetaw_mech.vector().get_local())
        self.u_mech_assigner.assign(
            self.u_mech,
            utils.sub_function(self.mech_state, self._u_subspace_index),
        )
        self.u_ep.assign(dolfin.interpolate(self.u_mech, self.W_ep))
        self._project_lmbda()
        self._compute_zeta()

        # self.lmbda_ep.assign(dolfin.interpolate(self.lmbda_mech, self.V_ep))

        # utils.local_project(self.Zetas_mech, self.V_ep, self.Zetas_ep)
        # self.Zetas_ep.assign(
        #     dolfin.project(
        #         self.Zetas_mech,
        #         self.V_ep,
        #         form_compiler_parameters={"representation": "quadrature"},
        #     ),
        # )
        # # utils.local_project(self.Zetaw_mech, self.V_ep, self.Zetaw_ep)
        # self.Zetaw_ep.assign(
        #     dolfin.project(
        #         self.Zetaw_mech,
        #         self.V_ep,
        #         form_compiler_parameters={"representation": "quadrature"},
        #     ),
        # )
        # self.Zetas_ep.assign(dolfin.interpolate(self.Zetas_mech, self.V_ep))
        # self.Zetaw_ep.assign(dolfin.interpolate(self.Zetaw_mech, self.V_ep))
        logger.debug("Done interpolating EP")

    def coupling_to_ep(self):
        logger.debug("Update EP")
        # breakpoint()
        # dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.lmbda_ep)
        # dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetas_ep)
        # dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.Zetaw_ep)
        logger.debug("Done updating EP")
