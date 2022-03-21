import functools
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypedDict

import dolfin
import numpy as np
import ufl
from cbcbeat.cellmodels import CardiacCellModel

from . import utils

logger = utils.getLogger(__name__)


def Max(a, b):
    return (a + b + abs(a - b)) / dolfin.Constant(2)


class CustomParameterSchema(TypedDict):
    value: float
    factors: Dict[str, float]


class Parameter:
    def __init__(
        self,
        name: str,
        value: float,
        factors: Optional[Dict[str, float]] = None,
    ) -> None:
        self.name = name
        self._value = value

        factors = factors or {}
        self._factors = factors

    def factors(self) -> Dict[str, float]:
        return self._factors.copy()

    def __deepcopy__(self):
        return self.__copy__()

    def __copy__(self):
        return Parameter(self.name, self._value, self.factors())

    def copy(self):
        return self.__copy__()

    @property
    def unscaled_value(self) -> float:
        return self._value

    @unscaled_value.setter
    def unscaled_value(self, value) -> None:
        self._value = value

    @property
    def factor(self) -> float:
        return functools.reduce(lambda x, y: x * y, self._factors.values(), 1.0)

    @property
    def value(self) -> float:
        return self.factor * self._value

    def __eq__(self, other) -> bool:
        return np.isclose(self.value, float(other))

    def __mul__(self, other) -> float:
        return self.value * other

    def __rmul__(self, other) -> float:
        return self.value * other

    def __add__(self, other) -> float:
        return self.value + other

    def __radd__(self, other) -> float:
        return self.value + other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self._value}, factors={self._factors})"

    def add_factor(self, name, value) -> None:
        if name in self._factors:
            logger.warning(
                f"Overwriting existing scaling factor {name} for parameter {self.name}",
            )
        self._factors[name] = value

    def __float__(self) -> float:
        return self.value


def tuplize(p: Parameter) -> Tuple[str, Parameter]:
    """Convert parameter into a tuple containing name
    as the first item and the parameter as the second.
    This is useful if you want to construct a dictionary
    of parameters who's keys are the parameter names

    Parameters
    ----------
    p : Parameter
        The parameter

    Returns
    -------
    Tuple[str, Parameter]
        (name, Parameter)
    """
    return (p.name, p)


class InvalidParameterError(KeyError):
    """Exception raised if parameter name is
    not a parameter in the model
    """


def apply_custom_parameters(
    parameters: Dict[str, Parameter],
    custom_parameters: Dict[str, CustomParameterSchema],
) -> Dict[str, Parameter]:

    # Make a copy so that we don't change the original parameters
    new_parameters = parameters.copy()

    for name, d in custom_parameters.items():
        p_ref = new_parameters.get(name)
        if p_ref is None:
            raise InvalidParameterError(f"Unknown parameter {name}")
        # Make a copy so that we don't change the original parameters
        p = p_ref.copy()

        value = d.get("value")
        if value is not None:
            p.unscaled_value = value

        factors = d.get("factors", {})
        for factor_name, factor_value in factors.items():
            p.add_factor(factor_name, factor_value)

        new_parameters[name] = p

    return new_parameters


def apply_HF_scaling(parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
    """Apply default scaling factor for heart failure

    Parameters
    ----------
    parameters : Dict[str, Parameter]
        The input parameters

    Returns
    -------
    Dict[str, Parameter]
        Parameters with heart failure scaling factors applied

    Raises
    ------
    InvalidParameterError
        If a parameter that should be scaled cannot be found
        in the model
    """
    scaling = {
        "CaMKa_ref": 1.50,
        "scale_Jrel_inf": pow(0.8, 8.0),
        "scale_Jleak": 1.3,
        "scale_Jup": 0.45,
        "GNaL": 1.3,
        "GK1": 0.68,
        "thL": 1.8,
        "Gto": 0.4,
        "Gncx": 1.6,
        "Pnak": 0.7,
        "cat50_ref": 0.6,
    }
    new_parameters = parameters.copy()
    for name, factor in scaling.items():
        p_ref = parameters.get(name)
        if p_ref is None:
            raise InvalidParameterError(f"Unknown parameter {name}")

        p = p_ref.copy()
        p.add_factor("HF", factor)
        new_parameters[name] = p

    return new_parameters


class ORdmm_Land(CardiacCellModel):
    @staticmethod
    def default_parameters(disease_state: str = "healthy") -> Dict[str, Parameter]:
        """Set-up and return default parameters.

        Parameters
        ----------
        disease_state : str, optional
            String with "hf" or "healthy", by default "healthy".
            If "hf", then parameters representing heart failure
            will be used.

        Returns
        -------
        OrderedDict
            Dictionary with default values
        """
        params: Dict[str, Parameter] = dict(
            map(
                tuplize,
                [
                    # Parameter("scale_ICaL", 1.018),
                    # Parameter("scale_IK1", 1.414),
                    # Parameter("scale_IKr", 1.119),
                    # Parameter("scale_IKs", 1.648),
                    # Parameter("scale_INaL", 2.274),
                    Parameter("celltype", 0),
                    Parameter("cao", 1.8),
                    Parameter("ko", 5.4),
                    Parameter("nao", 140.0),
                    Parameter("F", 96485.0),
                    Parameter("R", 8314.0),
                    Parameter("T", 310.0),
                    Parameter("L", 0.01),
                    Parameter("rad", 0.0011),
                    Parameter("Ahf", 0.99),
                    Parameter("GNa", 31),
                    Parameter("thL", 200.0),
                    Parameter("Gto", 0.02),
                    Parameter("delta_epi", 1.0),
                    Parameter("Aff", 0.6),
                    Parameter("Kmn", 0.002),
                    Parameter("k2n", 1000.0),
                    Parameter("tjca", 75.0),
                    Parameter("zca", 2.0),
                    Parameter("bt", 4.75),
                    Parameter("Beta0", 2.3),
                    Parameter("Beta1", -2.4),
                    Parameter("Tot_A", 25),
                    Parameter("Tref", 120),
                    Parameter("Trpn50", 0.35),
                    Parameter("calib", 1),
                    Parameter("cat50_ref", 0.805),
                    Parameter("emcoupling", 1),
                    Parameter("etal", 200),
                    Parameter("etas", 20),
                    Parameter("gammas", 0.0085),
                    Parameter("gammaw", 0.615),
                    Parameter("isacs", 0),
                    Parameter("ktrpn", 0.1),
                    Parameter("ku", 0.04),
                    Parameter("kuw", 0.182),
                    Parameter("kws", 0.012),
                    Parameter("mode", 1),
                    Parameter("ntm", 2.4),
                    Parameter("ntrpn", 2),
                    Parameter("p_a", 2.1),
                    Parameter("p_b", 9.1),
                    Parameter("p_k", 7),
                    Parameter("phi", 2.23),
                    Parameter("rs", 0.25),
                    Parameter("rw", 0.5),
                    Parameter("CaMKo", 0.05),
                    Parameter("KmCaM", 0.0015),
                    Parameter("KmCaMK", 0.15),
                    Parameter("aCaMK", 0.05),
                    Parameter("bCaMK", 0.00068),
                    Parameter("PKNa", 0.01833),
                    Parameter("Gncx", 0.0008),
                    Parameter("KmCaAct", 0.00015),
                    Parameter("kasymm", 12.5),
                    Parameter("kcaoff", 5000.0),
                    Parameter("kcaon", 1500000.0),
                    Parameter("kna1", 15.0),
                    Parameter("kna2", 5.0),
                    Parameter("kna3", 88.12),
                    Parameter("qca", 0.167),
                    Parameter("qna", 0.5224),
                    Parameter("wca", 60000.0),
                    Parameter("wna", 60000.0),
                    Parameter("wnaca", 5000.0),
                    Parameter("H", 1e-07),
                    Parameter("Khp", 1.698e-07),
                    Parameter("Kki", 0.5),
                    Parameter("Kko", 0.3582),
                    Parameter("Kmgatp", 1.698e-07),
                    Parameter("Knai0", 9.073),
                    Parameter("Knao0", 27.78),
                    Parameter("Knap", 224.0),
                    Parameter("Kxkur", 292.0),
                    Parameter("MgADP", 0.05),
                    Parameter("MgATP", 9.8),
                    Parameter("Pnak", 30),
                    Parameter("delta", -0.155),
                    Parameter("eP", 4.2),
                    Parameter("k1m", 182.4),
                    Parameter("k1p", 949.5),
                    Parameter("k2m", 39.4),
                    Parameter("k2p", 687.2),
                    Parameter("k3m", 79300.0),
                    Parameter("k3p", 1899.0),
                    Parameter("k4m", 40.0),
                    Parameter("k4p", 639.0),
                    Parameter("zk", 1.0),
                    Parameter("GKb", 0.003),
                    Parameter("PNab", 3.75e-10),
                    Parameter("PCab", 2.5e-08),
                    Parameter("GpCa", 0.0005),
                    Parameter("Esac_ns", -10),
                    Parameter("Gsac_k", 1.097904761904762),
                    Parameter("Gsac_ns", 0.006),
                    Parameter("lambda_max", 1.1),
                    Parameter("amp", -80.0),
                    Parameter("duration", 0.5),
                    Parameter("BSLmax", 1.124),
                    Parameter("BSRmax", 0.047),
                    Parameter("KmBSL", 0.0087),
                    Parameter("KmBSR", 0.00087),
                    Parameter("cmdnmax", 0.05),
                    Parameter("csqnmax", 10.0),
                    Parameter("kmcmdn", 0.00238),
                    Parameter("kmcsqn", 0.8),
                    Parameter("kmtrpn", 0.0005),
                    Parameter("trpnmax", 0.07),
                    Parameter("GNaL", 0.0075 * 2.274),  # multiply with scale_NaL
                    Parameter("CaMKa_ref", 1.0),
                    Parameter("PCa", 0.0001 * 1.018),  # multiply with scale_ICaL
                    Parameter("GKr", 0.046 * 1.119),  # multiply with scale_IKr
                    Parameter("GKs", 0.0034 * 1.648),  # multiply with scale_IKs
                    Parameter("GK1", 0.1908 * 1.414),  # multiply with scale_IK1
                    Parameter("Gsac_ns", 0.006),
                    Parameter("Gsac_k", (0.2882 * 800 / 210)),
                    Parameter("scale_Jrel_inf", 25.62890625),
                    Parameter("KRyR", 1.0),
                    Parameter("scale_Jleak", 1.0),
                    Parameter("scale_Jup", 1.0),
                ],
            ),
        )

        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = dict(
            [
                ("v", -87),
                ("CaMKt", 0),
                ("m", 0),
                ("hf", 1),
                ("hs", 1),
                ("j", 1),
                ("hsp", 1),
                ("jp", 1),
                ("mL", 0),
                ("hL", 1),
                ("hLp", 1),
                ("a", 0),
                ("iF", 1),
                ("iS", 1),
                ("ap", 0),
                ("iFp", 1),
                ("iSp", 1),
                ("d", 0),
                ("ff", 1),
                ("fs", 1),
                ("fcaf", 1),
                ("fcas", 1),
                ("jca", 1),
                ("ffp", 1),
                ("fcafp", 1),
                ("nca", 0),
                ("xrf", 0),
                ("xrs", 0),
                ("xs1", 0),
                ("xs2", 0),
                ("xk1", 1),
                ("Jrelnp", 0),
                ("Jrelp", 0),
                ("nai", 7),
                ("nass", 7),
                ("ki", 145),
                ("kss", 145),
                ("cass", 0.0001),
                ("cansr", 1.2),
                ("cajsr", 1.2),
                ("XS", 0),
                ("XW", 0),
                ("CaTrpn", 0),
                ("TmB", 1),
                ("Cd", 0),
                ("cai", 0.0001),
                ("lmbda", 1),
                ("Zetas", 0),
                ("Zetaw", 0),
            ],
        )
        return ic

    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else dolfin.Constant(0.0)

        # Assign states
        assert len(s) == 48
        (
            CaMKt,
            m,
            hf,
            hs,
            j,
            hsp,
            jp,
            mL,
            hL,
            hLp,
            a,
            iF,
            iS,
            ap,
            iFp,
            iSp,
            d,
            ff,
            fs,
            fcaf,
            fcas,
            jca,
            ffp,
            fcafp,
            nca,
            xrf,
            xrs,
            xs1,
            xs2,
            xk1,
            Jrelnp,
            Jrelp,
            nai,
            nass,
            ki,
            kss,
            cass,
            cansr,
            cajsr,
            XS,
            XW,
            CaTrpn,
            TmB,
            Cd,
            cai,
            lmbda,
            Zetas,
            Zetaw,
        ) = s

        # Assign parameters

        # I think we should multiply these parameter with the respective conductances
        # scale_ICaL = self._parameters["scale_ICaL"]
        # scale_IK1 = self._parameters["scale_IK1"]
        # scale_IKr = self._parameters["scale_IKr"]
        # scale_IKs = self._parameters["scale_IKs"]
        # scale_INaL = self._parameters["scale_INaL"]
        cao = self._parameters["cao"]
        ko = self._parameters["ko"]
        nao = self._parameters["nao"]
        F = self._parameters["F"]
        R = self._parameters["R"]
        T = self._parameters["T"]
        CaMKo = self._parameters["CaMKo"]
        KmCaM = self._parameters["KmCaM"]
        KmCaMK = self._parameters["KmCaMK"]
        PKNa = self._parameters["PKNa"]
        Ahf = self._parameters["Ahf"]
        GNa = self._parameters["GNa"]
        Gto = self._parameters["Gto"]
        Aff = self._parameters["Aff"]
        zca = self._parameters["zca"]
        Gncx = self._parameters["Gncx"]
        KmCaAct = self._parameters["KmCaAct"]
        kasymm = self._parameters["kasymm"]
        kcaoff = self._parameters["kcaoff"]
        kcaon = self._parameters["kcaon"]
        kna1 = self._parameters["kna1"]
        kna2 = self._parameters["kna2"]
        kna3 = self._parameters["kna3"]
        qca = self._parameters["qca"]
        qna = self._parameters["qna"]
        wca = self._parameters["wca"]
        wna = self._parameters["wna"]
        wnaca = self._parameters["wnaca"]
        H = self._parameters["H"]
        Khp = self._parameters["Khp"]
        Kki = self._parameters["Kki"]
        Kko = self._parameters["Kko"]
        Kmgatp = self._parameters["Kmgatp"]
        Knai0 = self._parameters["Knai0"]
        Knao0 = self._parameters["Knao0"]
        Knap = self._parameters["Knap"]
        Kxkur = self._parameters["Kxkur"]
        MgADP = self._parameters["MgADP"]
        MgATP = self._parameters["MgATP"]
        Pnak = self._parameters["Pnak"]
        delta = self._parameters["delta"]
        eP = self._parameters["eP"]
        k1m = self._parameters["k1m"]
        k1p = self._parameters["k1p"]
        k2m = self._parameters["k2m"]
        k2p = self._parameters["k2p"]
        k3m = self._parameters["k3m"]
        k3p = self._parameters["k3p"]
        k4m = self._parameters["k4m"]
        k4p = self._parameters["k4p"]
        zk = self._parameters["zk"]
        GKb = self._parameters["GKb"]
        PNab = self._parameters["PNab"]
        PCab = self._parameters["PCab"]
        GpCa = self._parameters["GpCa"]

        GNaL = self._parameters["GNaL"]
        CaMKa_ref = self._parameters["CaMKa_ref"]
        PCa = self._parameters["PCa"]
        GKr = self._parameters["GKr"]
        GKs = self._parameters["GKs"]
        GK1 = self._parameters["GK1"]
        Gsac_ns = self._parameters["Gsac_ns"]
        Gsac_k = self._parameters["Gsac_k"]  # Pueyo endo

        # Init return args
        current = [ufl.zero()] * 1

        # Expressions for the CaMKt component
        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
        CaMKa = (CaMKb + CaMKt) * CaMKa_ref

        # Expressions for the reversal potentials component
        ENa = R * T * ufl.ln(nao / nai) / F
        EK = R * T * ufl.ln(ko / ki) / F
        EKs = R * T * ufl.ln((ko + PKNa * nao) / (PKNa * nai + ki)) / F
        vffrt = (F * F) * v / (R * T)
        vfrt = F * v / (R * T)

        # Expressions for the I_Na component
        Ahs = 1.0 - Ahf
        h = Ahf * hf + Ahs * hs
        hp = Ahf * hf + Ahs * hsp
        fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
        INa = (
            GNa
            * ufl.elem_pow(m, 3.0)
            * (-ENa + v)
            * ((1.0 - fINap) * h * j + fINap * hp * jp)
        )

        # Expressions for the INaL component

        fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        INaL = (-ENa + v) * ((1.0 - fINaLp) * hL + fINaLp * hLp) * GNaL * mL

        # Expressions for the Ito component
        AiF = 1.0 / (1.0 + 0.24348537187522867 * ufl.exp(0.006613756613756614 * v))
        AiS = 1.0 - AiF
        i = AiF * iF + AiS * iS
        ip = AiF * iFp + AiS * iSp
        fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
        Ito = Gto * (-EK + v) * ((1.0 - fItop) * a * i + ap * fItop * ip)

        # Expressions for the ICaL ICaNa ICaK component
        Afs = 1.0 - Aff
        f = Aff * ff + Afs * fs
        Afcaf = 0.3 + 0.6 / (1.0 + 0.36787944117144233 * ufl.exp(0.1 * v))
        Afcas = 1.0 - Afcaf
        fca = Afcaf * fcaf + Afcas * fcas
        fp = Aff * ffp + Afs * fs
        fcap = Afcaf * fcafp + Afcas * fcas
        PhiCaL = (
            4.0
            * (-0.341 * cao + cass * ufl.exp(2.0 * vfrt))
            * vffrt
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )
        PhiCaNa = (
            1.0
            * (-0.75 * nao + 0.75 * ufl.exp(1.0 * vfrt) * nass)
            * vffrt
            / (-1.0 + ufl.exp(1.0 * vfrt))
        )
        PhiCaK = (
            1.0
            * (-0.75 * ko + 0.75 * ufl.exp(1.0 * vfrt) * kss)
            * vffrt
            / (-1.0 + ufl.exp(1.0 * vfrt))
        )

        PCap = 1.1 * PCa
        PCaNa = 0.00125 * PCa
        PCaK = 0.0003574 * PCa
        PCaNap = 0.00125 * PCap
        PCaKp = 0.0003574 * PCap
        fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        ICaL = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCa * PhiCaL * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCap * PhiCaL * d * fICaLp
        ICaNa = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCaNa * PhiCaNa * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCaNap * PhiCaNa * d * fICaLp
        ICaK = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCaK * PhiCaK * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCaKp * PhiCaK * d * fICaLp

        # Expressions for the IKr component
        Axrf = 1.0 / (1.0 + 4.197299094734718 * ufl.exp(0.02617115938236064 * v))
        Axrs = 1.0 - Axrf
        xr = Axrf * xrf + Axrs * xrs
        rkr = 1.0 / (
            (1.0 + 2.0820090840784555 * ufl.exp(0.013333333333333334 * v))
            * (1.0 + 0.7165313105737893 * ufl.exp(0.03333333333333333 * v))
        )
        IKr = 0.4303314829119352 * ufl.sqrt(ko) * (-EK + v) * GKr * rkr * xr

        # Expressions for the IKs component
        KsCa = 1.0 + 0.6 / (1.0 + 6.481821026062645e-07 * ufl.elem_pow(1.0 / cai, 1.4))

        IKs = (-EKs + v) * GKs * KsCa * xs1 * xs2
        rk1 = 1.0 / (
            1.0
            + 69220.6322106767
            * ufl.exp(0.10534077741493732 * v - 0.27388602127883704 * ko)
        )

        IK1 = ufl.sqrt(ko) * (-EK + v) * GK1 * rk1 * xk1

        # Expressions for the INaCa_i component
        hca = ufl.exp(F * qca * v / (R * T))
        hna = ufl.exp(F * qna * v / (R * T))
        h1_i = 1 + (1 + hna) * nai / kna3
        h2_i = hna * nai / (kna3 * h1_i)
        h3_i = 1.0 / h1_i
        h4_i = 1.0 + (1 + nai / kna2) * nai / kna1
        h5_i = (nai * nai) / (kna1 * kna2 * h4_i)
        h6_i = 1.0 / h4_i
        h7_i = 1.0 + nao * (1.0 + 1.0 / hna) / kna3
        h8_i = nao / (kna3 * h7_i * hna)
        h9_i = 1.0 / h7_i
        h10_i = 1.0 + kasymm + nao * (1.0 + nao / kna2) / kna1
        h11_i = (nao * nao) / (kna1 * kna2 * h10_i)
        h12_i = 1.0 / h10_i
        k1_i = cao * kcaon * h12_i
        k2_i = kcaoff
        k3p_i = wca * h9_i
        k3pp_i = wnaca * h8_i
        k3_i = k3p_i + k3pp_i
        k4p_i = wca * h3_i / hca
        k4pp_i = wnaca * h2_i
        k4_i = k4p_i + k4pp_i
        k5_i = kcaoff
        k6_i = kcaon * cai * h6_i
        k7_i = wna * h2_i * h5_i
        k8_i = wna * h11_i * h8_i
        x1_i = (k2_i + k3_i) * k5_i * k7_i + (k6_i + k7_i) * k2_i * k4_i
        x2_i = (k1_i + k8_i) * k4_i * k6_i + (k4_i + k5_i) * k1_i * k7_i
        x3_i = (k2_i + k3_i) * k6_i * k8_i + (k6_i + k7_i) * k1_i * k3_i
        x4_i = (k1_i + k8_i) * k3_i * k5_i + (k4_i + k5_i) * k2_i * k8_i
        E1_i = x1_i / (x1_i + x2_i + x3_i + x4_i)
        E2_i = x2_i / (x1_i + x2_i + x3_i + x4_i)
        E3_i = x3_i / (x1_i + x2_i + x3_i + x4_i)
        E4_i = x4_i / (x1_i + x2_i + x3_i + x4_i)
        allo_i = 1.0 / (1.0 + ufl.elem_pow(KmCaAct / cai, 2.0))
        zna = 1.0
        JncxNa_i = E3_i * k4pp_i - E2_i * k3pp_i + 3.0 * E4_i * k7_i - 3.0 * E1_i * k8_i
        JncxCa_i = E2_i * k2_i - E1_i * k1_i
        INaCa_i = 0.8 * Gncx * (zca * JncxCa_i + zna * JncxNa_i) * allo_i

        # Expressions for the INaCa_ss component
        h1 = 1 + (1 + hna) * nass / kna3
        h2 = hna * nass / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + (1 + nass / kna2) * nass / kna1
        h5 = (nass * nass) / (kna1 * kna2 * h4)
        h6 = 1.0 / h4
        h7 = 1.0 + nao * (1.0 + 1.0 / hna) / kna3
        h8 = nao / (kna3 * h7 * hna)
        h9 = 1.0 / h7
        h10 = 1.0 + kasymm + nao * (1 + nao / kna2) / kna1
        h11 = (nao * nao) / (kna1 * kna2 * h10)
        h12 = 1.0 / h10
        k1 = cao * kcaon * h12
        k2 = kcaoff
        k3p_ss = wca * h9
        k3pp = wnaca * h8
        k3 = k3p_ss + k3pp
        k4p_ss = wca * h3 / hca
        k4pp = wnaca * h2
        k4 = k4p_ss + k4pp
        k5 = kcaoff
        k6 = kcaon * cass * h6
        k7 = wna * h2 * h5
        k8 = wna * h11 * h8
        x1_ss = (k2 + k3) * k5 * k7 + (k6 + k7) * k2 * k4
        x2_ss = (k1 + k8) * k4 * k6 + (k4 + k5) * k1 * k7
        x3_ss = (k2 + k3) * k6 * k8 + (k6 + k7) * k1 * k3
        x4_ss = (k1 + k8) * k3 * k5 + (k4 + k5) * k2 * k8
        E1_ss = x1_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E2_ss = x2_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E3_ss = x3_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E4_ss = x4_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        allo_ss = 1.0 / (1.0 + ufl.elem_pow(KmCaAct / cass, 2.0))
        JncxNa_ss = E3_ss * k4pp - E2_ss * k3pp + 3.0 * E4_ss * k7 - 3.0 * E1_ss * k8
        JncxCa_ss = E2_ss * k2 - E1_ss * k1
        INaCa_ss = 0.2 * Gncx * (zca * JncxCa_ss + zna * JncxNa_ss) * allo_ss

        # Expressions for the INaK component
        Knai = Knai0 * ufl.exp(0.3333333333333333 * F * delta * v / (R * T))
        Knao = Knao0 * ufl.exp(0.3333333333333333 * F * (1.0 - delta) * v / (R * T))
        P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)
        a1 = (
            k1p
            * ufl.elem_pow(nai / Knai, 3.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ki / Kki, 2.0)
                + ufl.elem_pow(1.0 + nai / Knai, 3.0)
            )
        )
        b1 = MgADP * k1m
        a2 = k2p
        b2 = (
            k2m
            * ufl.elem_pow(nao / Knao, 3.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ko / Kko, 2.0)
                + ufl.elem_pow(1.0 + nao / Knao, 3.0)
            )
        )
        a3 = (
            k3p
            * ufl.elem_pow(ko / Kko, 2.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ko / Kko, 2.0)
                + ufl.elem_pow(1.0 + nao / Knao, 3.0)
            )
        )
        b3 = H * k3m * P / (1.0 + MgATP / Kmgatp)
        a4 = MgATP * k4p / (Kmgatp * (1.0 + MgATP / Kmgatp))
        b4 = (
            k4m
            * ufl.elem_pow(ki / Kki, 2.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ki / Kki, 2.0)
                + ufl.elem_pow(1.0 + nai / Knai, 3.0)
            )
        )
        x1 = a1 * a2 * a4 + a1 * a2 * b3 + a2 * b3 * b4 + b2 * b3 * b4
        x2 = a1 * a2 * a3 + a2 * a3 * b4 + a3 * b1 * b4 + b1 * b2 * b4
        x3 = a2 * a3 * a4 + a3 * a4 * b1 + a4 * b1 * b2 + b1 * b2 * b3
        x4 = a1 * a3 * a4 + a1 * a4 * b2 + a1 * b2 * b3 + b2 * b3 * b4
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        JnakNa = 3.0 * E1 * a3 - 3.0 * E2 * b3
        JnakK = 2.0 * E4 * b1 - 2.0 * E3 * a1
        INaK = Pnak * (zk * JnakK + zna * JnakNa)

        # Expressions for the IKb component
        xkb = 1.0 / (1.0 + 2.202363450949239 * ufl.exp(-0.05452562704471101 * v))
        IKb = GKb * (-EK + v) * xkb

        # Expressions for the INab component
        INab = PNab * (-nao + ufl.exp(vfrt) * nai) * vffrt / (-1.0 + ufl.exp(vfrt))

        # Expressions for the ICab component
        ICab = (
            4.0
            * PCab
            * (-0.341 * cao + cai * ufl.exp(2.0 * vfrt))
            * vffrt
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )

        # Expressions for the IpCa component
        IpCa = GpCa * cai / (0.0005 + cai)

        # Expressions for the Isac (Pueyo)--> ns + k component
        Esac_ns = -10
        lambda_max = 1.1

        Isac_P_ns = ufl.conditional(
            ufl.lt(lmbda, 1.0),
            0,
            Gsac_ns * ((lmbda - 1.0) / (lambda_max - 1.0)) * (v - Esac_ns),
        )
        Isac_P_k = ufl.conditional(
            ufl.lt(lmbda, 1.0),
            0,
            Gsac_k
            * ((lmbda - 1.0) / (lambda_max - 1.0))
            * (1.0 / (1.0 + ufl.exp((19.05 - v) / (29.98)))),
        )

        # Expressions for the Istim component
        Istim = 0  # amp*(ufl.le(time, duration))

        # Expressions for the membrane potential component
        current[0] = (
            -Isac_P_k
            - Isac_P_ns
            - ICaK
            - ICaL
            - ICaNa
            - ICab
            - IK1
            - IKb
            - IKr
            - IKs
            - INa
            - INaCa_i
            - INaCa_ss
            - INaK
            - INaL
            - INab
            - IpCa
            - Istim
            - Ito
        )

        # Return results
        return current[0]

    def I(self, v, s, time=None):  # noqa: E741, E743
        """
        Transmembrane current

           I = -dV/dt

        """
        return -self._I(v, s, time)

    def F(self, v, s, time=None):
        """
        Right hand side for ODE system
        """
        time = time if time else dolfin.Constant(0.0)

        # Assign states
        assert len(s) == 48
        (
            CaMKt,
            m,
            hf,
            hs,
            j,
            hsp,
            jp,
            mL,
            hL,
            hLp,
            a,
            iF,
            iS,
            ap,
            iFp,
            iSp,
            d,
            ff,
            fs,
            fcaf,
            fcas,
            jca,
            ffp,
            fcafp,
            nca,
            xrf,
            xrs,
            xs1,
            xs2,
            xk1,
            Jrelnp,
            Jrelp,
            nai,
            nass,
            ki,
            kss,
            cass,
            cansr,
            cajsr,
            XS,
            XW,
            CaTrpn,
            TmB,
            Cd,
            cai,
            lmbda,
            Zetas,
            Zetaw,
        ) = s

        # Assign parameters
        cao = self._parameters["cao"]
        ko = self._parameters["ko"]
        nao = self._parameters["nao"]
        F = self._parameters["F"]
        R = self._parameters["R"]
        T = self._parameters["T"]
        L = self._parameters["L"]
        rad = self._parameters["rad"]
        CaMKo = self._parameters["CaMKo"]
        KmCaM = self._parameters["KmCaM"]
        KmCaMK = self._parameters["KmCaMK"]
        aCaMK = self._parameters["aCaMK"]
        bCaMK = self._parameters["bCaMK"]
        PKNa = self._parameters["PKNa"]
        Ahf = self._parameters["Ahf"]
        GNa = self._parameters["GNa"]
        thL = self._parameters["thL"]
        Gto = self._parameters["Gto"]
        delta_epi = self._parameters["delta_epi"]
        Aff = self._parameters["Aff"]
        Kmn = self._parameters["Kmn"]
        k2n = self._parameters["k2n"]
        tjca = self._parameters["tjca"]
        zca = self._parameters["zca"]
        Gncx = self._parameters["Gncx"]
        KmCaAct = self._parameters["KmCaAct"]
        kasymm = self._parameters["kasymm"]
        kcaoff = self._parameters["kcaoff"]
        kcaon = self._parameters["kcaon"]
        kna1 = self._parameters["kna1"]
        kna2 = self._parameters["kna2"]
        kna3 = self._parameters["kna3"]
        qca = self._parameters["qca"]
        qna = self._parameters["qna"]
        wca = self._parameters["wca"]
        wna = self._parameters["wna"]
        wnaca = self._parameters["wnaca"]
        H = self._parameters["H"]
        Khp = self._parameters["Khp"]
        Kki = self._parameters["Kki"]
        Kko = self._parameters["Kko"]
        Kmgatp = self._parameters["Kmgatp"]
        Knai0 = self._parameters["Knai0"]
        Knao0 = self._parameters["Knao0"]
        Knap = self._parameters["Knap"]
        Kxkur = self._parameters["Kxkur"]
        MgADP = self._parameters["MgADP"]
        MgATP = self._parameters["MgATP"]
        Pnak = self._parameters["Pnak"]
        delta = self._parameters["delta"]
        eP = self._parameters["eP"]
        k1m = self._parameters["k1m"]
        k1p = self._parameters["k1p"]
        k2m = self._parameters["k2m"]
        k2p = self._parameters["k2p"]
        k3m = self._parameters["k3m"]
        k3p = self._parameters["k3p"]
        k4m = self._parameters["k4m"]
        k4p = self._parameters["k4p"]
        zk = self._parameters["zk"]
        GKb = self._parameters["GKb"]
        PNab = self._parameters["PNab"]
        PCab = self._parameters["PCab"]
        GpCa = self._parameters["GpCa"]
        bt = self._parameters["bt"]
        BSLmax = self._parameters["BSLmax"]
        BSRmax = self._parameters["BSRmax"]
        KmBSL = self._parameters["KmBSL"]
        KmBSR = self._parameters["KmBSR"]
        cmdnmax = self._parameters["cmdnmax"]
        csqnmax = self._parameters["csqnmax"]
        kmcmdn = self._parameters["kmcmdn"]
        kmcsqn = self._parameters["kmcsqn"]
        trpnmax = self._parameters["trpnmax"]
        Beta1 = self._parameters["Beta1"]

        Trpn50 = self._parameters["Trpn50"]
        cat50_ref = self._parameters["cat50_ref"]

        etal = self._parameters["etal"]
        etas = self._parameters["etas"]
        gammas = self._parameters["gammas"]
        gammaw = self._parameters["gammaw"]
        ktrpn = self._parameters["ktrpn"]
        ku = self._parameters["ku"]
        kuw = self._parameters["kuw"]
        kws = self._parameters["kws"]
        ntm = self._parameters["ntm"]
        ntrpn = self._parameters["ntrpn"]
        p_k = self._parameters["p_k"]

        rs = self._parameters["rs"]
        rw = self._parameters["rw"]

        GNaL = self._parameters["GNaL"]
        CaMKa_ref = self._parameters["CaMKa_ref"]
        PCa = self._parameters["PCa"]
        GKr = self._parameters["GKr"]
        GKs = self._parameters["GKs"]
        GK1 = self._parameters["GK1"]
        Gsac_ns = self._parameters["Gsac_ns"]
        Gsac_k = self._parameters["Gsac_k"]  # Pueyo endo
        scale_Jrel_inf = self._parameters["scale_Jrel_inf"]
        KRyR = self._parameters["KRyR"]
        scale_Jleak = self._parameters["scale_Jleak"]
        scale_Jup = self._parameters["scale_Jup"]

        # Init return args
        F_expressions = [dolfin.Constant(0.0)] * 48

        # Expressions for the cell geometry component
        vcell = 3140.0 * L * (rad * rad)
        Ageo = 6.28 * (rad * rad) + 6.28 * L * rad
        Acap = 2 * Ageo
        vmyo = 0.68 * vcell
        vnsr = 0.0552 * vcell
        vjsr = 0.0048 * vcell
        vss = 0.02 * vcell

        # Expressions for the CaMKt component
        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / cass)
        CaMKa = (CaMKb + CaMKt) * CaMKa_ref
        F_expressions[0] = -bCaMK * CaMKt + aCaMK * (CaMKb + CaMKt) * CaMKb

        # Expressions for the reversal potentials component
        ENa = R * T * ufl.ln(nao / nai) / F
        EK = R * T * ufl.ln(ko / ki) / F
        EKs = R * T * ufl.ln((ko + PKNa * nao) / (PKNa * nai + ki)) / F
        vffrt = (F * F) * v / (R * T)
        vfrt = F * v / (R * T)

        # Expressions for the I_Na component
        mss = 1.0 / (1.0 + 0.0014599788446489682 * ufl.exp(-0.13333333333333333 * v))
        tm = 1.0 / (
            9.454904638564724 * ufl.exp(0.02876042565429968 * v)
            + 1.9314113558536928e-05 * ufl.exp(-0.16792611251049538 * v)
        )
        F_expressions[1] = (-m + mss) / tm
        hss = 1.0 / (1 + 302724.605401998 * ufl.exp(0.1607717041800643 * v))
        thf = 1.0 / (
            1.183856958289087e-05 * ufl.exp(-0.15910898965791567 * v)
            + 6.305549185817275 * ufl.exp(0.0493339911198816 * v)
        )
        ths = 1.0 / (
            0.005164670235381792 * ufl.exp(-0.035650623885918005 * v)
            + 0.36987619372096325 * ufl.exp(0.017649135192375574 * v)
        )
        Ahs = 1.0 - Ahf
        F_expressions[2] = (-hf + hss) / thf
        F_expressions[3] = (-hs + hss) / ths
        h = Ahf * hf + Ahs * hs
        jss = hss
        tj = 2.038 + 1.0 / (
            0.3131936394738773 * ufl.exp(0.02600780234070221 * v)
            + 1.1315282095590072e-07 * ufl.exp(-0.12075836251660427 * v)
        )
        F_expressions[4] = (-j + jss) / tj
        hssp = 1.0 / (1 + 820249.0921708513 * ufl.exp(0.1607717041800643 * v))
        thsp = 3.0 * ths
        F_expressions[5] = (-hsp + hssp) / thsp
        hp = Ahf * hf + Ahs * hsp
        tjp = 1.46 * tj
        F_expressions[6] = (-jp + jss) / tjp
        fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
        INa = (
            GNa
            * ufl.elem_pow(m, 3.0)
            * (-ENa + v)
            * ((1.0 - fINap) * h * j + fINap * hp * jp)
        )

        # Expressions for the INaL component
        mLss = 1.0 / (1.0 + 0.000291579585635531 * ufl.exp(-0.18996960486322187 * v))
        tmL = tm
        F_expressions[7] = (-mL + mLss) / tmL
        hLss = 1.0 / (1.0 + 120578.15595522427 * ufl.exp(0.13354700854700854 * v))
        F_expressions[8] = (-hL + hLss) / (thL)
        hLssp = 1.0 / (1.0 + 275969.2903869871 * ufl.exp(0.13354700854700854 * v))
        thLp = 3.0 * thL
        F_expressions[9] = (-hLp + hLssp) / thLp

        fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        INaL = (-ENa + v) * ((1.0 - fINaLp) * hL + fINaLp * hLp) * GNaL * mL

        # Expressions for the Ito component
        ass = 1.0 / (1.0 + 2.6316508161673635 * ufl.exp(-0.06747638326585695 * v))
        ta = 1.0515 / (
            1.0 / (1.2089 + 2.2621017070578837 * ufl.exp(-0.03403513787634354 * v))
            + 3.5 / (1.0 + 30.069572727397507 * ufl.exp(0.03403513787634354 * v))
        )
        F_expressions[10] = (-a + ass) / ta
        iss = 1.0 / (1.0 + 2194.970764538301 * ufl.exp(0.17510068289266328 * v))
        tiF = 4.562 + delta_epi / (
            0.14468698421272827 * ufl.exp(-0.01 * v)
            + 1.6300896349780942 * ufl.exp(0.06027727546714889 * v)
        )
        tiS = 23.62 + delta_epi / (
            0.00027617763953377436 * ufl.exp(-0.01693480101608806 * v)
            + 0.024208962804604526 * ufl.exp(0.12377769525931426 * v)
        )
        AiF = 1.0 / (1.0 + 0.24348537187522867 * ufl.exp(0.006613756613756614 * v))
        AiS = 1.0 - AiF
        F_expressions[11] = (-iF + iss) / tiF
        F_expressions[12] = (-iS + iss) / tiS
        i = AiF * iF + AiS * iS
        assp = 1.0 / (1.0 + 5.167428462230666 * ufl.exp(-0.06747638326585695 * v))
        F_expressions[13] = (-ap + assp) / ta
        dti_develop = 1.354 + 0.0001 / (
            2.6591269045230603e-05 * ufl.exp(0.06293266205160478 * v)
            + 4.5541779737128264e24 * ufl.exp(-4.642525533890436 * v)
        )
        dti_recover = 1.0 - 0.5 / (1.0 + 33.11545195869231 * ufl.exp(0.05 * v))
        tiFp = dti_develop * dti_recover * tiF
        tiSp = dti_develop * dti_recover * tiS
        F_expressions[14] = (-iFp + iss) / tiFp
        F_expressions[15] = (-iSp + iss) / tiSp
        ip = AiF * iFp + AiS * iSp
        fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
        Ito = Gto * (-EK + v) * ((1.0 - fItop) * a * i + ap * fItop * ip)

        # Expressions for the ICaL ICaNa ICaK component
        dss = 1.0 / (1.0 + 0.39398514226669484 * ufl.exp(-0.23640661938534277 * v))
        td = 0.6 + 1.0 / (
            3.5254214873653824 * ufl.exp(0.09 * v)
            + 0.7408182206817179 * ufl.exp(-0.05 * v)
        )
        F_expressions[16] = (-d + dss) / td
        fss = 1.0 / (1.0 + 199.86038496778565 * ufl.exp(0.27056277056277056 * v))
        tff = 7.0 + 1.0 / (
            0.03325075244518792 * ufl.exp(0.1 * v)
            + 0.0006090087745647571 * ufl.exp(-0.1 * v)
        )
        tfs = 1000.0 + 1.0 / (
            1.0027667890106652e-05 * ufl.exp(-0.25 * v)
            + 8.053415618124885e-05 * ufl.exp(0.16666666666666666 * v)
        )
        Afs = 1.0 - Aff
        F_expressions[17] = (-ff + fss) / tff
        F_expressions[18] = (-fs + fss) / tfs
        f = Aff * ff + Afs * fs
        fcass = fss
        tfcaf = 7.0 + 1.0 / (
            0.0708317980974062 * ufl.exp(-0.14285714285714285 * v)
            + 0.02258872488031037 * ufl.exp(0.14285714285714285 * v)
        )
        tfcas = 100.0 + 1.0 / (
            0.00012 * ufl.exp(0.14285714285714285 * v)
            + 0.00012 * ufl.exp(-0.3333333333333333 * v)
        )
        Afcaf = 0.3 + 0.6 / (1.0 + 0.36787944117144233 * ufl.exp(0.1 * v))
        Afcas = 1.0 - Afcaf
        F_expressions[19] = (-fcaf + fcass) / tfcaf
        F_expressions[20] = (-fcas + fcass) / tfcas
        fca = Afcaf * fcaf + Afcas * fcas
        F_expressions[21] = (-jca + fcass) / tjca
        tffp = 2.5 * tff
        F_expressions[22] = (-ffp + fss) / tffp
        fp = Aff * ffp + Afs * fs
        tfcafp = 2.5 * tfcaf
        F_expressions[23] = (-fcafp + fcass) / tfcafp
        fcap = Afcaf * fcafp + Afcas * fcas
        km2n = 1.0 * jca
        anca = 1.0 / (ufl.elem_pow(1.0 + Kmn / cass, 4.0) + k2n / km2n)
        F_expressions[24] = k2n * anca - km2n * nca
        PhiCaL = (
            4.0
            * (-0.341 * cao + cass * ufl.exp(2.0 * vfrt))
            * vffrt
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )
        PhiCaNa = (
            1.0
            * (-0.75 * nao + 0.75 * ufl.exp(1.0 * vfrt) * nass)
            * vffrt
            / (-1.0 + ufl.exp(1.0 * vfrt))
        )
        PhiCaK = (
            1.0
            * (-0.75 * ko + 0.75 * ufl.exp(1.0 * vfrt) * kss)
            * vffrt
            / (-1.0 + ufl.exp(1.0 * vfrt))
        )
        PCap = 1.1 * PCa
        PCaNa = 0.00125 * PCa
        PCaK = 0.0003574 * PCa
        PCaNap = 0.00125 * PCap
        PCaKp = 0.0003574 * PCap
        fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        ICaL = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCa * PhiCaL * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCap * PhiCaL * d * fICaLp
        ICaNa = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCaNa * PhiCaNa * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCaNap * PhiCaNa * d * fICaLp
        ICaK = (1.0 - fICaLp) * (
            (1.0 - nca) * f + fca * jca * nca
        ) * PCaK * PhiCaK * d + (
            (1.0 - nca) * fp + fcap * jca * nca
        ) * PCaKp * PhiCaK * d * fICaLp

        # Expressions for the IKr component
        xrss = 1.0 / (1.0 + 0.29287308872377504 * ufl.exp(-0.14729709824716453 * v))
        txrf = 12.98 + 1.0 / (
            0.0001020239312894894 * ufl.exp(0.25846471956577927 * v)
            + 0.00042992960891929087 * ufl.exp(-0.04906771344455348 * v)
        )
        txrs = 1.865 + 1.0 / (
            0.0005922420036809394 * ufl.exp(0.13596193065941536 * v)
            + 3.549966111802463e-05 * ufl.exp(-0.03855050115651503 * v)
        )
        Axrf = 1.0 / (1.0 + 4.197299094734718 * ufl.exp(0.02617115938236064 * v))
        Axrs = 1.0 - Axrf
        F_expressions[25] = (-xrf + xrss) / txrf
        F_expressions[26] = (-xrs + xrss) / txrs
        xr = Axrf * xrf + Axrs * xrs
        rkr = 1.0 / (
            (1.0 + 2.0820090840784555 * ufl.exp(0.013333333333333334 * v))
            * (1.0 + 0.7165313105737893 * ufl.exp(0.03333333333333333 * v))
        )
        IKr = 0.4303314829119352 * ufl.sqrt(ko) * (-EK + v) * GKr * rkr * xr

        # Expressions for the IKs component
        xs1ss = 1.0 / (1.0 + 0.27288596035656526 * ufl.exp(-0.11195700850873264 * v))
        txs1 = 817.3 + 1.0 / (
            0.003504067763074858 * ufl.exp(0.056179775280898875 * v)
            + 0.0005184809083581659 * ufl.exp(-0.004347826086956522 * v)
        )
        F_expressions[27] = (-xs1 + xs1ss) / txs1
        xs2ss = xs1ss
        txs2 = 1.0 / (
            0.0022561357010639103 * ufl.exp(-0.03225806451612903 * v)
            + 0.0008208499862389881 * ufl.exp(0.05 * v)
        )
        F_expressions[28] = (-xs2 + xs2ss) / txs2
        KsCa = 1.0 + 0.6 / (1.0 + 6.481821026062645e-07 * ufl.elem_pow(1.0 / cai, 1.4))
        IKs = (-EKs + v) * GKs * KsCa * xs1 * xs2
        xk1ss = 1.0 / (
            1.0 + ufl.exp((-144.59 - v - 2.5538 * ko) / (3.8115 + 1.5692 * ko))
        )
        txk1 = 122.2 / (
            0.0019352007631390235 * ufl.exp(-0.049115913555992145 * v)
            + 30.43364757524903 * ufl.exp(0.014423770373575654 * v)
        )
        F_expressions[29] = (-xk1 + xk1ss) / txk1
        rk1 = 1.0 / (
            1.0
            + 69220.6322106767
            * ufl.exp(0.10534077741493732 * v - 0.27388602127883704 * ko)
        )

        IK1 = ufl.sqrt(ko) * (-EK + v) * GK1 * rk1 * xk1

        # Expressions for the INaCa_i component
        hca = ufl.exp(F * qca * v / (R * T))
        hna = ufl.exp(F * qna * v / (R * T))
        h1_i = 1 + (1 + hna) * nai / kna3
        h2_i = hna * nai / (kna3 * h1_i)
        h3_i = 1.0 / h1_i
        h4_i = 1.0 + (1 + nai / kna2) * nai / kna1
        h5_i = (nai * nai) / (kna1 * kna2 * h4_i)
        h6_i = 1.0 / h4_i
        h7_i = 1.0 + nao * (1.0 + 1.0 / hna) / kna3
        h8_i = nao / (kna3 * h7_i * hna)
        h9_i = 1.0 / h7_i
        h10_i = 1.0 + kasymm + nao * (1.0 + nao / kna2) / kna1
        h11_i = (nao * nao) / (kna1 * kna2 * h10_i)
        h12_i = 1.0 / h10_i
        k1_i = cao * kcaon * h12_i
        k2_i = kcaoff
        k3p_i = wca * h9_i
        k3pp_i = wnaca * h8_i
        k3_i = k3p_i + k3pp_i
        k4p_i = wca * h3_i / hca
        k4pp_i = wnaca * h2_i
        k4_i = k4p_i + k4pp_i
        k5_i = kcaoff
        k6_i = kcaon * cai * h6_i
        k7_i = wna * h2_i * h5_i
        k8_i = wna * h11_i * h8_i
        x1_i = (k2_i + k3_i) * k5_i * k7_i + (k6_i + k7_i) * k2_i * k4_i
        x2_i = (k1_i + k8_i) * k4_i * k6_i + (k4_i + k5_i) * k1_i * k7_i
        x3_i = (k2_i + k3_i) * k6_i * k8_i + (k6_i + k7_i) * k1_i * k3_i
        x4_i = (k1_i + k8_i) * k3_i * k5_i + (k4_i + k5_i) * k2_i * k8_i
        E1_i = x1_i / (x1_i + x2_i + x3_i + x4_i)
        E2_i = x2_i / (x1_i + x2_i + x3_i + x4_i)
        E3_i = x3_i / (x1_i + x2_i + x3_i + x4_i)
        E4_i = x4_i / (x1_i + x2_i + x3_i + x4_i)
        allo_i = 1.0 / (1.0 + ufl.elem_pow(KmCaAct / cai, 2.0))
        zna = 1.0
        JncxNa_i = E3_i * k4pp_i - E2_i * k3pp_i + 3.0 * E4_i * k7_i - 3.0 * E1_i * k8_i
        JncxCa_i = E2_i * k2_i - E1_i * k1_i
        INaCa_i = 0.8 * Gncx * (zca * JncxCa_i + zna * JncxNa_i) * allo_i

        # Expressions for the INaCa_ss component
        h1 = 1 + (1 + hna) * nass / kna3
        h2 = hna * nass / (kna3 * h1)
        h3 = 1.0 / h1
        h4 = 1.0 + (1 + nass / kna2) * nass / kna1
        h5 = (nass * nass) / (kna1 * kna2 * h4)
        h6 = 1.0 / h4
        h7 = 1.0 + nao * (1.0 + 1.0 / hna) / kna3
        h8 = nao / (kna3 * h7 * hna)
        h9 = 1.0 / h7
        h10 = 1.0 + kasymm + nao * (1 + nao / kna2) / kna1
        h11 = (nao * nao) / (kna1 * kna2 * h10)
        h12 = 1.0 / h10
        k1 = cao * kcaon * h12
        k2 = kcaoff
        k3p_ss = wca * h9
        k3pp = wnaca * h8
        k3 = k3p_ss + k3pp
        k4p_ss = wca * h3 / hca
        k4pp = wnaca * h2
        k4 = k4p_ss + k4pp
        k5 = kcaoff
        k6 = kcaon * cass * h6
        k7 = wna * h2 * h5
        k8 = wna * h11 * h8
        x1_ss = (k2 + k3) * k5 * k7 + (k6 + k7) * k2 * k4
        x2_ss = (k1 + k8) * k4 * k6 + (k4 + k5) * k1 * k7
        x3_ss = (k2 + k3) * k6 * k8 + (k6 + k7) * k1 * k3
        x4_ss = (k1 + k8) * k3 * k5 + (k4 + k5) * k2 * k8
        E1_ss = x1_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E2_ss = x2_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E3_ss = x3_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        E4_ss = x4_ss / (x1_ss + x2_ss + x3_ss + x4_ss)
        allo_ss = 1.0 / (1.0 + ufl.elem_pow(KmCaAct / cass, 2.0))
        JncxNa_ss = E3_ss * k4pp - E2_ss * k3pp + 3.0 * E4_ss * k7 - 3.0 * E1_ss * k8
        JncxCa_ss = E2_ss * k2 - E1_ss * k1
        INaCa_ss = 0.2 * Gncx * (zca * JncxCa_ss + zna * JncxNa_ss) * allo_ss

        # Expressions for the INaK component
        Knai = Knai0 * ufl.exp(0.3333333333333333 * F * delta * v / (R * T))
        Knao = Knao0 * ufl.exp(0.3333333333333333 * F * (1.0 - delta) * v / (R * T))
        P = eP / (1.0 + H / Khp + nai / Knap + ki / Kxkur)
        a1 = (
            k1p
            * ufl.elem_pow(nai / Knai, 3.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ki / Kki, 2.0)
                + ufl.elem_pow(1.0 + nai / Knai, 3.0)
            )
        )
        b1 = MgADP * k1m
        a2 = k2p
        b2 = (
            k2m
            * ufl.elem_pow(nao / Knao, 3.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ko / Kko, 2.0)
                + ufl.elem_pow(1.0 + nao / Knao, 3.0)
            )
        )
        a3 = (
            k3p
            * ufl.elem_pow(ko / Kko, 2.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ko / Kko, 2.0)
                + ufl.elem_pow(1.0 + nao / Knao, 3.0)
            )
        )
        b3 = H * k3m * P / (1.0 + MgATP / Kmgatp)
        a4 = MgATP * k4p / (Kmgatp * (1.0 + MgATP / Kmgatp))
        b4 = (
            k4m
            * ufl.elem_pow(ki / Kki, 2.0)
            / (
                -1.0
                + ufl.elem_pow(1.0 + ki / Kki, 2.0)
                + ufl.elem_pow(1.0 + nai / Knai, 3.0)
            )
        )
        x1 = a1 * a2 * a4 + a1 * a2 * b3 + a2 * b3 * b4 + b2 * b3 * b4
        x2 = a1 * a2 * a3 + a2 * a3 * b4 + a3 * b1 * b4 + b1 * b2 * b4
        x3 = a2 * a3 * a4 + a3 * a4 * b1 + a4 * b1 * b2 + b1 * b2 * b3
        x4 = a1 * a3 * a4 + a1 * a4 * b2 + a1 * b2 * b3 + b2 * b3 * b4
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        JnakNa = 3.0 * E1 * a3 - 3.0 * E2 * b3
        JnakK = 2.0 * E4 * b1 - 2.0 * E3 * a1
        INaK = Pnak * (zk * JnakK + zna * JnakNa)

        # Expressions for the IKb component
        xkb = 1.0 / (1.0 + 2.202363450949239 * ufl.exp(-0.05452562704471101 * v))
        IKb = GKb * (-EK + v) * xkb

        # Expressions for the INab component
        INab = PNab * (-nao + ufl.exp(vfrt) * nai) * vffrt / (-1.0 + ufl.exp(vfrt))

        # Expressions for the ICab component
        ICab = (
            4.0
            * PCab
            * (-0.341 * cao + cai * ufl.exp(2.0 * vfrt))
            * vffrt
            / (-1.0 + ufl.exp(2.0 * vfrt))
        )

        # Expressions for the IpCa component
        IpCa = GpCa * cai / (0.0005 + cai)

        # Expressions for the Isac (Pueyo)--> ns + k component
        Esac_ns = -10
        lambda_max = 1.1
        Isac_P_ns = ufl.conditional(
            ufl.lt(lmbda, 1.0),
            0,
            Gsac_ns * ((lmbda - 1.0) / (lambda_max - 1.0)) * (v - Esac_ns),
        )
        Isac_P_k = ufl.conditional(
            ufl.lt(lmbda, 1.0),
            0,
            Gsac_k
            * ((lmbda - 1.0) / (lambda_max - 1.0))
            * (1.0 / (1.0 + ufl.exp((19.05 - v) / (29.98)))),
        )

        # Expressions for the Istim component
        Istim = 0  # amp*(ufl.le(time, duration))

        # Expressions for the diffusion fluxes component
        JdiffNa = 0.5 * nass - 0.5 * nai
        JdiffK = 0.5 * kss - 0.5 * ki
        Jdiff = 5.0 * cass - 5.0 * cai

        # Expressions for the ryanodione receptor component
        a_rel = 0.5 * bt
        Jrel_inf = (
            -ICaL * a_rel / (1.0 + scale_Jrel_inf * ufl.elem_pow(1.0 / cajsr, 8.0))
        )
        tau_rel_tmp = bt / (1.0 + 0.0123 / cajsr)
        tau_rel = ufl.conditional(ufl.lt(tau_rel_tmp, 0.001), 0.001, tau_rel_tmp)
        F_expressions[30] = (-Jrelnp + Jrel_inf) / tau_rel
        btp = 1.25 * bt
        a_relp = 0.5 * btp
        Jrel_infp = (
            -ICaL * a_relp / (1.0 + scale_Jrel_inf * ufl.elem_pow(1.0 / cajsr, 8.0))
        )
        tau_relp_tmp = btp / (1.0 + 0.0123 / cajsr)
        tau_relp = ufl.conditional(ufl.lt(tau_relp_tmp, 0.001), 0.001, tau_relp_tmp)
        F_expressions[31] = (-Jrelp + Jrel_infp) / tau_relp
        fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
        Jrel = ((1.0 - fJrelp) * Jrelnp + Jrelp * fJrelp) * KRyR

        # Expressions for the calcium buffers component
        Jupnp = 0.004375 * cai / (0.00092 + cai)
        Jupp = 0.01203125 * cai / (0.00075 + cai)
        fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
        Jleak = 0.0002625 * cansr * scale_Jleak
        Jup = (-Jleak + (1.0 - fJupp) * Jupnp + Jupp * fJupp) * scale_Jup
        Jtr = 0.01 * cansr - 0.01 * cajsr

        # Expressions for the intracellular concentrations component
        F_expressions[32] = JdiffNa * vss / vmyo + (
            -INa - INaL - INab - Isac_P_ns / 3 - 3.0 * INaCa_i - 3.0 * INaK
        ) * Acap / (F * vmyo)
        F_expressions[33] = -JdiffNa + (-ICaNa - 3.0 * INaCa_ss) * Acap / (F * vss)
        F_expressions[34] = JdiffK * vss / vmyo + (
            -Isac_P_k - IK1 - IKb - IKr - IKs - Istim - Ito - Isac_P_ns / 3 + 2.0 * INaK
        ) * Acap / (F * vmyo)
        F_expressions[35] = -JdiffK - Acap * ICaK / (F * vss)
        Bcass = 1.0 / (
            1.0
            + BSLmax * KmBSL * ufl.elem_pow(KmBSL + cass, -2.0)
            + BSRmax * KmBSR * ufl.elem_pow(KmBSR + cass, -2.0)
        )
        F_expressions[36] = (
            -Jdiff
            + Jrel * vjsr / vss
            + 0.5 * (-ICaL + 2.0 * INaCa_ss) * Acap / (F * vss)
        ) * Bcass
        F_expressions[37] = -Jtr * vjsr / vnsr + Jup
        Bcajsr = 1.0 / (1.0 + csqnmax * kmcsqn * ufl.elem_pow(kmcsqn + cajsr, -2.0))
        F_expressions[38] = (-Jrel + Jtr) * Bcajsr

        # Expressions for the mechanics component
        kwu = -kws + kuw * (-1 + 1.0 / rw)
        ksu = kws * rw * (-1 + 1.0 / rs)

        lambda_min12 = ufl.conditional(ufl.lt(lmbda, 1.2), lmbda, 1.2)
        XS = ufl.conditional(ufl.lt(XS, 0), 0, XS)
        XW = ufl.conditional(ufl.lt(XW, 0), 0, XW)

        XU = 1 - TmB - XS - XW
        gammawu = gammaw * abs(Zetaw)
        # gammasu = gammas*ufl.conditional(ufl.gt(Zetas*(ufl.gt(Zetas, 0)), (-1 -\
        #     Zetas)*(ufl.lt(Zetas, -1))), Zetas*(ufl.gt(Zetas, 0)), (-1 -\
        #     Zetas)*(ufl.lt(Zetas, -1)))
        zetas1 = Zetas * ufl.conditional(ufl.gt(Zetas, 0), 1, 0)
        zetas2 = (-1 - Zetas) * ufl.conditional(ufl.lt(Zetas, -1), 1, 0)
        gammasu = gammas * Max(zetas1, zetas2)

        F_expressions[39] = kws * XW - XS * gammasu - XS * ksu
        F_expressions[40] = kuw * XU - kws * XW - XW * gammawu - XW * kwu
        cat50 = cat50_ref + Beta1 * (-1 + lambda_min12)
        CaTrpn = ufl.conditional(ufl.lt(CaTrpn, 0), 0, CaTrpn)
        F_expressions[41] = ktrpn * (
            -CaTrpn + ufl.elem_pow(1000 * cai / cat50, ntrpn) * (1 - CaTrpn)
        )
        kb = ku * ufl.elem_pow(Trpn50, ntm) / (1 - rs - rw * (1 - rs))
        F_expressions[42] = (
            ufl.conditional(
                ufl.lt(ufl.elem_pow(CaTrpn, -ntm / 2), 100),
                ufl.elem_pow(CaTrpn, -ntm / 2),
                100,
            )
            * XU
            * kb
            - ku * ufl.elem_pow(CaTrpn, ntm / 2) * TmB
        )

        C = -1 + lambda_min12
        dCd = -Cd + C
        eta = ufl.conditional(ufl.lt(dCd, 0), etas, etal)
        F_expressions[43] = p_k * (-Cd + C) / eta
        Bcai = 1.0 / (1.0 + cmdnmax * kmcmdn * ufl.elem_pow(kmcmdn + cai, -2.0))
        J_TRPN = trpnmax * F_expressions[41]
        F_expressions[44] = (
            -J_TRPN
            + Jdiff * vss / vmyo
            - Jup * vnsr / vmyo
            + 0.5 * (-ICab - IpCa - Isac_P_ns / 3 + 2.0 * INaCa_i) * Acap / (F * vmyo)
        ) * Bcai

        # Return results
        return dolfin.as_vector(F_expressions)

    def num_states(self):
        return 48

    def __str__(self):
        return "ORdmm_Land_em_coupling cardiac cell model"
