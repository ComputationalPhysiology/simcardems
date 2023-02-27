from enum import Enum
from typing import Dict
from typing import Optional

import dolfin
import numpy as np
import pulse
import ufl

from ... import utils
from ...time_stepper import TimeStepper
from .cell_model import ORdmmLandFull
from .em_model import EMCoupling


logger = utils.getLogger(__name__)


class Scheme(str, Enum):
    fd = "fd"
    bd = "bd"
    analytic = "analytic"


def _Zeta(Zeta_prev, A, c, dLambda, dt, scheme: Scheme):
    if scheme == Scheme.analytic:
        return Zeta_prev * dolfin.exp(-c * dt) + (A * dLambda / c * dt) * (
            1.0 - dolfin.exp(-c * dt)
        )

    elif scheme == Scheme.bd:
        return Zeta_prev + A * dLambda / (1.0 + c * dt)
    else:
        return Zeta_prev * (1.0 - c * dt) + A * dLambda


class LandModel(pulse.ActiveModel):
    def __init__(
        self,
        coupling: EMCoupling,
        parameters: Optional[Dict[str, float]] = None,
        Zetas=None,
        Zetaw=None,
        lmbda=None,
        eta=0,
        scheme: Scheme = Scheme.analytic,
        dLambda_tol: float = 1e-12,
        **kwargs,
    ):
        logger.debug("Initialize Land Model")
        super().__init__(
            f0=coupling.geometry.f0,
            s0=coupling.geometry.s0,
            n0=coupling.geometry.n0,
        )

        self._eta = eta
        self.function_space = dolfin.FunctionSpace(coupling.mech_mesh, "CG", 1)

        self.XS = coupling.XS_mech
        self.XW = coupling.XW_mech
        if parameters is None:
            parameters = ORdmmLandFull.default_parameters()

        self._parameters = parameters

        self._scheme = scheme

        self._dLambda = dolfin.Function(self.function_space)
        self.lmbda_prev = dolfin.Function(self.function_space)
        self.lmbda_prev.vector()[:] = 1.0
        if lmbda is not None:
            self.lmbda_prev.assign(lmbda)
        self.lmbda = dolfin.Function(self.function_space)

        self._Zetas = dolfin.Function(self.function_space)
        self.Zetas_prev = dolfin.Function(self.function_space)
        if Zetas is not None:
            self.Zetas_prev.assign(Zetas)

        self._Zetaw = dolfin.Function(self.function_space)
        self.Zetaw_prev = dolfin.Function(self.function_space)
        if Zetaw is not None:
            self.Zetaw_prev.assign(Zetaw)

        self.Ta_current = dolfin.Function(self.function_space, name="Ta")
        self._projector = utils.Projector(self.function_space)
        self._dLambda_tol = dLambda_tol
        self._t_prev = 0.0

    @property
    def dLambda(self):
        logger.debug("Evaluate dLambda")
        self._dLambda.vector()[:] = self.lmbda.vector() - self.lmbda_prev.vector()
        self._dLambda.vector()[
            np.where(np.abs(self._dLambda.vector().get_local()) < self._dLambda_tol)[0]
        ] = 0.0
        return self._dLambda

    @property
    def Aw(self):
        Tot_A = self._parameters["Tot_A"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        return (
            Tot_A
            * rs
            * scale_popu_rs
            / (rs * scale_popu_rs + rw * scale_popu_rw * (1.0 - (rs * scale_popu_rs)))
        )

    @property
    def As(self):
        return self.Aw

    @property
    def cw(self):
        phi = self._parameters["phi"]
        kuw = self._parameters["kuw"]
        rw = self._parameters["rw"]

        scale_popu_kuw = self._parameters["scale_popu_kuw"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        return (
            kuw
            * scale_popu_kuw
            * phi
            * (1.0 - (rw * scale_popu_rw))
            / (rw * scale_popu_rw)
        )

    @property
    def cs(self):
        phi = self._parameters["phi"]
        kws = self._parameters["kws"]
        rs = self._parameters["rs"]
        rw = self._parameters["rw"]
        scale_popu_kws = self._parameters["scale_popu_kws"]
        scale_popu_rw = self._parameters["scale_popu_rw"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        return (
            kws
            * scale_popu_kws
            * phi
            * rw
            * scale_popu_rw
            * (1.0 - (rs * scale_popu_rs))
            / (rs * scale_popu_rs)
        )

    def update_Zetas(self):
        logger.debug("update Zetas")
        self._Zetas.vector()[:] = _Zeta(
            self.Zetas_prev.vector(),
            self.As,
            self.cs,
            self.dLambda.vector(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetas(self):
        return self._Zetas

    def update_Zetaw(self):
        logger.debug("update Zetaw")
        self._Zetaw.vector()[:] = _Zeta(
            self.Zetaw_prev.vector(),
            self.Aw,
            self.cw,
            self.dLambda.vector(),
            self.dt,
            self._scheme,
        )

    @property
    def Zetaw(self):
        return self._Zetaw

    def register_time_stepper(self, time_stepper: TimeStepper) -> None:
        self.time_stepper = time_stepper
        self._t_prev = time_stepper.t

    @property
    def dt(self) -> float:
        return TimeStepper.ns2ms(self.t - self._t_prev)

    @property
    def t(self) -> float:
        if not hasattr(self, "time_stepper"):
            return 0.0
        return self.time_stepper.t

    def update_prev(self):
        logger.debug("update previous")
        self.Zetas_prev.vector()[:] = self.Zetas.vector()
        self.Zetaw_prev.vector()[:] = self.Zetaw.vector()
        self.lmbda_prev.vector()[:] = self.lmbda.vector()
        self._projector.project(self.Ta_current, self.Ta)
        self._t_prev = self.t

    @property
    def Ta(self):
        logger.debug("Evaluate Ta")
        Tref = self._parameters["Tref"]
        rs = self._parameters["rs"]
        scale_popu_Tref = self._parameters["scale_popu_Tref"]
        scale_popu_rs = self._parameters["scale_popu_rs"]
        Beta0 = self._parameters["Beta0"]

        _min = ufl.min_value
        _max = ufl.max_value
        if isinstance(self.lmbda, (int, float)):
            _min = min
            _max = max
        lmbda = _min(1.2, self.lmbda)
        h_lambda_prima = 1.0 + Beta0 * (lmbda + _min(lmbda, 0.87) - 1.87)
        h_lambda = _max(0, h_lambda_prima)

        return (
            h_lambda
            * (Tref * scale_popu_Tref / (rs * scale_popu_rs))
            * (self.XS * (self.Zetas + 1.0) + self.XW * self.Zetaw)
        )

    def Wactive(self, F, **kwargs):
        """Active stress energy"""
        logger.debug("Compute active stress energy")
        C = F.T * F
        f = F * self.f0
        self._projector.project(self.lmbda, dolfin.sqrt(f**2))
        self.update_Zetas()
        self.update_Zetaw()
        return pulse.material.active_model.Wactive_transversally(
            Ta=self.Ta,
            C=C,
            f0=self.f0,
            eta=self.eta,
        )
