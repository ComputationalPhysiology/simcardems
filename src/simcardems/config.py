import logging
import typing
from dataclasses import dataclass

import dolfin

from . import land_model
from . import mechanics_model
from . import utils


@dataclass
class Config:
    outdir: utils.PathLike = "results"
    T: float = 1000
    dx: float = 0.2
    dt: float = 0.05
    bnd_cond: mechanics_model.BoundaryConditions = (
        mechanics_model.BoundaryConditions.dirichlet
    )
    load_state: bool = False
    cell_init_file: utils.PathLike = ""
    hpc: bool = False
    lx: float = 2.0
    ly: float = 0.7
    lz: float = 0.3
    save_freq: int = 1
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None
    traction: typing.Union[dolfin.Constant, float] = None
    spring: typing.Union[dolfin.Constant, float] = None
    fix_right_plane: bool = False
    loglevel: int = logging.INFO
    num_refinements: int = 1
    set_material: str = ""
    drug_factors_file: str = ""
    popu_factors_file: str = ""
    disease_state: str = "healthy"
    mechanics_ode_scheme: land_model.Scheme = land_model.Scheme.analytic
    ep_ode_scheme: str = "GRL1"
    ep_preconditioner: str = "sor"
    ep_theta: float = 0.5
    linear_mechanics_solver: str = "mumps"
    mechanics_use_continuation: bool = False
    mechanics_use_custom_newton_solver: bool = False
    PCL: float = 1000

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def default_parameters():
    return {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
