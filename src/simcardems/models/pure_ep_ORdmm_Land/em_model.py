from typing import Tuple

import cbcbeat

from ... import utils

logger = utils.getLogger(__name__)


class EMCoupling:
    @property
    def coupling_type(self):
        return "pureEP_ORdmm_Land"

    def register_ep_model(self, solver: cbcbeat.SplittingSolver):
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

    def solve_mechanics(self) -> None:
        pass

    def solve_ep(self, interval: Tuple[float, float]) -> None:
        self.ep_solver.step(interval)

    def update_prev_mechanics(self):
        pass

    def update_prev_ep(self):
        self.ep_solver.vs_.assign(self.ep_solver.vs)
