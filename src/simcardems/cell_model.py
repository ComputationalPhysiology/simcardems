import functools
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypedDict

import numpy as np
from cbcbeat.cellmodels import CardiacCellModel

from . import utils

logger = utils.getLogger(__name__)


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
        "CaMKa": 1.50,
        "Jrel_inf": pow(0.8, 8.0),
        "Jleak": 1.3,
        "Jup": 0.45,
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
                    Parameter("scale_ICaL", 1.018),
                    Parameter("scale_IK1", 1.414),
                    Parameter("scale_IKr", 1.119),
                    Parameter("scale_IKs", 1.648),
                    Parameter("scale_INaL", 2.274),
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
                ],
            ),
        )

        return params
