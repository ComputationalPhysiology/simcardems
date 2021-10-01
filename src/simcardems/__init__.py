import logging as _logging

from . import cli
from . import datacollector
from . import em_model
from . import ep_model
from . import mechanics_model
from . import ORdmm_Land
from . import postprocess
from . import save_load_functions
from . import utils
from .datacollector import DataCollector
from .datacollector import DataLoader
from .em_model import EMCoupling
from .mechanics_model import LandModel
from .mechanics_model import MechanicsProblem
from .mechanics_model import RigidMotionProblem

for module in ["matplotlib", "h5py"]:
    _logger = _logging.getLogger(module)
    _logger.setLevel(_logging.WARNING)

__all__ = [
    "datacollector",
    "em_model",
    "ep_model",
    "mechanics_model",
    "ORdmm_Land",
    "postprocess",
    "save_load_functions",
    "utils",
    "cli",
    "LandModel",
    "MechanicsProblem",
    "RigidMotionProblem",
    "EMCoupling",
    "DataCollector",
    "DataLoader",
]
