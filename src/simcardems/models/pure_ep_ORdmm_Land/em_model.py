from ... import utils

logger = utils.getLogger(__name__)


class EMCoupling:
    def register_ep_model(self, solver):
        logger.debug("Registering EP model")
        self.ep_solver = solver
        self.vs = solver.solution_fields()[0]
        logger.debug("Done registering EP model")

    def register_mech_model(self, *args, **kwargs):
        pass

    def ep_to_coupling(self):
        pass

    def coupling_to_mechanics(self):
        pass

    def mechanics_to_coupling(self):
        pass

    def coupling_to_ep(self):
        pass
