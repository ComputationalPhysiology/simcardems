import logging as _logging
import warnings as _warnings

import daiquiri as _daiquiri
import dolfin as _dolfin
import pulse as _pulse
from ffc.quadrature.deprecation import (
    QuadratureRepresentationDeprecationWarning as _QuadratureRepresentationDeprecationWarning,
)

from . import boundary_conditions
from . import cli
from . import config
from . import datacollector
from . import em_model
from . import ep_model
from . import geometry
from . import land_model
from . import mechanics_model
from . import newton_solver
from . import ORdmm_Land
from . import postprocess
from . import save_load_functions
from . import setup_models
from . import utils
from .config import Config
from .config import default_parameters
from .datacollector import DataCollector
from .datacollector import DataLoader
from .em_model import EMCoupling
from .land_model import LandModel
from .mechanics_model import MechanicsProblem
from .mechanics_model import RigidMotionProblem
from .newton_solver import MechanicsNewtonSolver
from .newton_solver import MechanicsNewtonSolver_ODE
from .setup_models import Runner
from .setup_models import TimeStepper
from .version import __version__


def set_log_level(level):
    from daiquiri import set_default_log_levels

    loggers = [
        "simcardems.benchmark.logger"
        "simcardems.boundary_conditions.logger"
        "simcardems.cli.logger"
        "simcardems.config.logger"
        "simcardems.datacollector.logger"
        "simcardems.em_model.logger"
        "simcardems.ep_model.logger"
        "simcardems.geometry.logger"
        "simcardems.gui.logger"
        "simcardems.land_model.logger"
        "simcardems.mechanics_model.logger"
        "simcardems.newton_solver.logger"
        "simcardems.postprocess.logger"
        "simcardems.save_load_functions.logger"
        "simcardems.setup_models.logger"
        "simcardems.utils.logger"
        "simcardems.value_extractor.logger",
    ]
    _pulse.set_log_level(level)
    _daiquiri.setup(level=level)

    if level < _logging.INFO:
        # Turn on INFO for dolfin
        _dolfin.set_log_level(_logging.INFO)

    # If debug level turn on more logging
    if level < _logging.DEBUG:
        _dolfin.set_log_level(_logging.DEBUG)
        for module in ["FFC", "UFL"]:
            _logger = _logging.getLogger(module)
            _logger.setLevel(_logging.INFO)

    set_default_log_levels((logger, level) for logger in loggers)


set_log_level(_logging.INFO)

_dolfin.set_log_level(_logging.WARNING)
for module in ["matplotlib", "h5py", "FFC", "UFL"]:
    _logger = _logging.getLogger(module)
    _logger.setLevel(_logging.WARNING)


_dolfin.parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math"]
_dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
_dolfin.parameters["form_compiler"]["quadrature_degree"] = 3
_dolfin.parameters["form_compiler"]["representation"] = "uflacs"

_warnings.simplefilter("once", _QuadratureRepresentationDeprecationWarning)

__all__ = [
    "datacollector",
    "boundary_conditions",
    "em_model",
    "ep_model",
    "mechanics_model",
    "ORdmm_Land",
    "postprocess",
    "geometry",
    "save_load_functions",
    "utils",
    "cli",
    "LandModel",
    "MechanicsProblem",
    "RigidMotionProblem",
    "EMCoupling",
    "DataCollector",
    "DataLoader",
    "setup_models",
    "Runner",
    "default_parameters",
    "__version__",
    "TimeStepper",
    "land_model",
    "newton_solver",
    "MechanicsNewtonSolver_ODE",
    "MechanicsNewtonSolver",
    "config",
    "Config",
]
