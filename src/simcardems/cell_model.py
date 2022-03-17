import functools
from typing import Dict
from typing import Optional

from . import utils

logger = utils.getLogger(__name__)


class Parameter:
    def __init__(self, name, value, factors: Optional[Dict[str, float]] = None) -> None:
        self.name = name
        self._value = value

        factors = factors or {}
        self._factors = factors

    def factors(self) -> Dict[str, float]:
        return self._factors.copy()

    @property
    def factor(self) -> float:
        return functools.reduce(lambda x, y: x * y, self._factors.values(), 1.0)

    @property
    def value(self) -> float:
        return self.factor * self._value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self._value}, factors={self._factors})"

    def add_factor(self, name, value):
        if name in self._factors:
            logger.warning(
                f"Overwriting existing scaling factor {name} for parameter {self.name}",
            )
        self._factors[name] = value
