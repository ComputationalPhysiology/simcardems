import dolfin

from . import geometry as _geometry
from . import utils


logger = utils.getLogger(__name__)


class EMCoupling:
    def __init__(
        self,
        geometry: _geometry.BaseGeometry,
        lmbda_mech=dolfin.Constant(1.0),
    ) -> None:
        logger.debug("Create EM coupling")
        self.geometry = geometry
        self.V_mech = dolfin.FunctionSpace(self.mech_mesh, "CG", 1)

        self.XS_mech = dolfin.Function(self.V_mech, name="XS_mech")
        self.XW_mech = dolfin.Function(self.V_mech, name="XW_mech")
        self.TmB_mech = dolfin.Function(self.V_mech, name="TmB_mech")
        self.Zetas_mech = dolfin.Function(self.V_mech, name="Zetas_mech")
        self.Zetaw_mech = dolfin.Function(self.V_mech, name="Zetaw_mech")

        self.CaTrpn_mech = dolfin.Function(self.V_mech, name="CaTrpn_mech")

        self.lmbda_mech = dolfin.Function(self.V_mech, name="lambda_mech")
        self.lmbda_mech.assign(lmbda_mech)

        self.V_ep = dolfin.FunctionSpace(self.ep_mesh, "CG", 1)
        self.lmbda_ep = dolfin.Function(self.V_ep, name="lambda_ep")
        self.Zetas_ep = dolfin.Function(self.V_ep, name="Zetas_ep")
        self.Zetaw_ep = dolfin.Function(self.V_ep, name="Zetaw_ep")
        self._mechanics_to_coupling()

    @property
    def mech_mesh(self):
        return self.geometry.mechanics_mesh

    @property
    def ep_mesh(self):
        return self.geometry.ep_mesh

    def register_mech_model(self, mech_heart):
        self._mech_heart = mech_heart

        # Variables for assigning mech_heart state to s
        self.s, self.s_assigner = utils.setup_assigner(mech_heart.state, 2)
        self.s_assigner.assign(self.s, utils.sub_function(mech_heart.state, 2))

        # Variables for assigning s to mech_heart state
        V = self.s.function_space()
        VS0 = mech_heart.state.function_space().sub(2)
        self.mech_heart_assigner = dolfin.FunctionAssigner(VS0, V)

        self.XS_mech, self.XS_mech_assigner, self.V_mech_XS = utils.setup_assigner(
            self.s,
            0,
            retrun_space=True,
        )
        self.XW_mech, self.XW_mech_assigner, self.V_mech_XW = utils.setup_assigner(
            self.s,
            1,
            retrun_space=True,
        )
        self.TmB_mech, self.TmB_mech_assigner, self.V_mech_TmB = utils.setup_assigner(
            self.s,
            2,
            retrun_space=True,
        )
        (
            self.Zetas_mech,
            self.Zetas_mech_assigner,
            self.V_mech_Zetas,
        ) = utils.setup_assigner(
            self.s,
            3,
            retrun_space=True,
        )
        (
            self.Zetaw_mech,
            self.Zetaw_mech_assigner,
            self.V_mech_Zetaw,
        ) = utils.setup_assigner(
            self.s,
            4,
            retrun_space=True,
        )

    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self._ep_solver = solver
        self.vs = solver.solution_fields()[0]
        self.XS_ep, self.XS_ep_assigner, self.V_XS = utils.setup_assigner(
            self.vs,
            40,
            retrun_space=True,
        )
        self.XW_ep, self.XW_ep_assigner, self.V_XW = utils.setup_assigner(
            self.vs,
            41,
            retrun_space=True,
        )
        self.TmB_ep, self.TmB_ep_assigner, self.V_TmB = utils.setup_assigner(
            self.vs,
            43,
            retrun_space=True,
        )
        self.CaTrpn_ep, self.CaTrpn_ep_assigner = utils.setup_assigner(self.vs, 42)

        logger.debug("Done registering EP model")

    def ep_to_coupling(self):
        """Stuff sent from the EP model to the coupling"""
        logger.debug("EP to coupling")
        self.XS_ep_assigner.assign(self.XS_ep, utils.sub_function(self.vs, 40))
        self.XW_ep_assigner.assign(self.XW_ep, utils.sub_function(self.vs, 41))
        self.TmB_ep_assigner.assign(self.TmB_ep, utils.sub_function(self.vs, 43))
        self.CaTrpn_ep_assigner.assign(self.CaTrpn_ep, utils.sub_function(self.vs, 42))
        logger.debug("Done EP to coupling")

    def coupling_to_mechanics(self):
        """Stuff sent from the coupling to the mechanics model"""
        logger.debug("Coupling to mechanics")
        dolfin.assign(
            utils.sub_function(self.s, 0),
            dolfin.interpolate(self.XS_ep, self.V_mech),
        )
        dolfin.assign(
            utils.sub_function(self.s, 1),
            dolfin.interpolate(self.XW_ep, self.V_mech),
        )
        dolfin.assign(
            utils.sub_function(self.s, 2),
            dolfin.interpolate(self.TmB_ep, self.V_mech),
        )
        self.mech_heart_assigner.assign(self._mech_heart.state.sub(2), self.s)
        self.CaTrpn_mech.assign(dolfin.interpolate(self.CaTrpn_ep, self.V_mech))
        logger.debug("Done coupling to mechanics")

    def mechanics_to_coupling(self):
        """Stuff sent from the mechanics model to the coupling"""
        logger.debug("Mechanics to coupling")
        # Assign to mechanics variables in the coupling
        self.XS_mech_assigner.assign(self.XS_mech, utils.sub_function(self.s, 0))
        self.XW_mech_assigner.assign(self.XW_mech, utils.sub_function(self.s, 1))
        self.TmB_mech_assigner.assign(self.TmB_mech, utils.sub_function(self.s, 2))
        self.Zetas_mech_assigner.assign(self.Zetas_mech, utils.sub_function(self.s, 3))
        self.Zetaw_mech_assigner.assign(self.Zetaw_mech, utils.sub_function(self.s, 4))

        # Interpolate the mechanics variables on the coupling to the
        # EP variables on the coupling
        logger.debug("Interpolate")
        self.TmB_ep.assign(dolfin.interpolate(self.TmB_mech, self.V_TmB))
        self.XS_ep.assign(dolfin.interpolate(self.XS_mech, self.V_XS))
        self.XW_ep.assign(dolfin.interpolate(self.XW_mech, self.V_XW))
        self._mechanics_to_coupling()
        logger.debug("Done mechanics to coupling")

    def _mechanics_to_coupling(self):
        """Interpolate the the stuff that is sent from the mechanics to the coupling
        that is not sent back to the mechanics model later
        """
        self.lmbda_ep.assign(dolfin.interpolate(self.lmbda_mech, self.V_ep))
        self.Zetas_ep.assign(dolfin.interpolate(self.Zetas_mech, self.V_ep))
        self.Zetaw_ep.assign(dolfin.interpolate(self.Zetaw_mech, self.V_ep))

    def coupling_to_ep(self):
        """Stuff sent from the coupling to the EP model"""
        logger.debug("Coupling to EP")
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 40), self.XS_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 41), self.XW_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 42), self.TmB_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 46), self.lmbda_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 47), self.Zetas_ep)
        dolfin.assign(utils.sub_function(self._ep_solver.vs, 48), self.Zetaw_ep)
        logger.debug("Done coupling to EP")
