from abc import ABC
from abc import abstractmethod
from typing import Dict

from cbcbeat import CardiacCellModel


class BaseCellModel(CardiacCellModel, ABC):
    @abstractmethod
    @staticmethod
    def update_disease_parameters(
        params: Dict[str, float],
        disease_state: str = "healthy",
    ) -> None:
        ...
