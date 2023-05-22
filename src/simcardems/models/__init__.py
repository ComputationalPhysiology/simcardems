from . import cell_model
from . import em_model
from . import explicit_ORdmm_Land
from . import fully_coupled_ORdmm_Land
from . import fully_coupled_Tor_Land
from . import pureEP_ORdmm_Land


def list_coupling_types():
    return [
        "explicit_ORdmm_Land",
        "fully_coupled_ORdmm_Land",
        "fully_coupled_Tor_Land",
        "pureEP_ORdmm_Land",
    ]


loggers = (
    explicit_ORdmm_Land.loggers
    + fully_coupled_ORdmm_Land.loggers
    + pureEP_ORdmm_Land.loggers
    + fully_coupled_Tor_Land.loggers
)


__all__ = [
    "explicit_ORdmm_Land",
    "fully_coupled_ORdmm_Land",
    "fully_coupled_Tor_Land",
    "em_model",
    "cell_model",
    "pureEP_ORdmm_Land",
    "loggers",
]
