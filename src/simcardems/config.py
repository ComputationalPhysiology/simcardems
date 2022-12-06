import logging
import typing
from dataclasses import dataclass

import dolfin

from . import land_model
from . import utils


@dataclass
class Config:
    outdir: utils.PathLike = "results"
    geometry_path: utils.PathLike = ""
    geometry_schema_path: typing.Optional[utils.PathLike] = None
    T: float = 1000
    dt: float = 0.05
    bnd_rigid: bool = False
    load_state: bool = False
    cell_init_file: utils.PathLike = ""
    show_progress_bar: bool = True
    save_freq: int = 1
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None
    traction: typing.Union[dolfin.Constant, float] = None
    spring: typing.Union[dolfin.Constant, float] = None
    fix_right_plane: bool = False
    loglevel: int = logging.INFO
    num_refinements: int = 1
    set_material: str = ""
    debug_mode: bool = False
    drug_factors_file: utils.PathLike = ""
    popu_factors_file: utils.PathLike = ""
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
