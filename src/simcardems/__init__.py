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
from . import ep_model
from . import geometry
from . import lvgeometry
from . import mechanics_model
from . import models
from . import newton_solver
from . import postprocess
from . import runner
from . import save_load_functions
from . import slabgeometry
from . import utils
from .config import Config
from .config import default_parameters
from .datacollector import DataCollector
from .datacollector import DataLoader
from .mechanics_model import MechanicsProblem
from .mechanics_model import RigidMotionProblem
from .newton_solver import MechanicsNewtonSolver
from .newton_solver import MechanicsNewtonSolver_ODE
from .runner import Runner
from .runner import TimeStepper
from .version import __version__


def set_log_level(level):
    from daiquiri import set_default_log_levels

    _pulse.set_log_level(level)
    _daiquiri.setup(level=level)

    if level < _logging.INFO:
        # Turn on INFO for dolfin
        _dolfin.set_log_level(_logging.INFO)

    # If debug level turn on more logging
    if level < _logging.DEBUG:
        _dolfin.set_log_level(_logging.DEBUG)
        for module in ["FFC", "UFL", "UFL_LEGACY"]:
            _logger = _logging.getLogger(module)
            _logger.setLevel(_logging.INFO)

    set_default_log_levels([("simcardems", level)])


set_log_level(_logging.INFO)

_dolfin.set_log_level(_logging.WARNING)
for module in ["matplotlib", "h5py", "FFC", "UFL", "UFL_LEGACY"]:
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
    "ep_model",
    "mechanics_model",
    "postprocess",
    "geometry",
    "save_load_functions",
    "utils",
    "cli",
    "slabgeometry",
    "LandModel",
    "MechanicsProblem",
    "RigidMotionProblem",
    "EMCoupling",
    "DataCollector",
    "DataLoader",
    "runner",
    "Runner",
    "default_parameters",
    "__version__",
    "TimeStepper",
    "lvgeometry",
    "newton_solver",
    "MechanicsNewtonSolver_ODE",
    "MechanicsNewtonSolver",
    "config",
    "Config",
    "models",
]
