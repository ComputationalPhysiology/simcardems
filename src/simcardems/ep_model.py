import json
from pathlib import Path
from typing import Dict
from typing import Optional

import cbcbeat
import dolfin

from . import utils
from .ORdmm_Land import ORdmm_Land as CellModel

logger = utils.getLogger(__name__)


def define_conductivity_tensor(chi, C_m):

    # Conductivities as defined by page 4339 of Niederer benchmark
    sigma_il = 0.17  # mS / mm
    sigma_it = 0.019  # mS / mm
    sigma_el = 0.62  # mS / mm
    sigma_et = 0.24  # mS / mm

    # Compute monodomain approximation by taking harmonic mean in each
    # direction of intracellular and extracellular part
    def harmonic_mean(a, b):
        return a * b / (a + b)

    sigma_l = harmonic_mean(sigma_il, sigma_el)
    sigma_t = harmonic_mean(sigma_it, sigma_et)

    # Scale conductivities by 1/(C_m * chi)
    s_l = sigma_l / (C_m * chi)  # mm^2 / ms
    s_t = sigma_t / (C_m * chi)  # mm^2 / ms

    # Define conductivity tensor
    M = dolfin.as_tensor(((s_l, 0, 0), (0, s_t, 0), (0, 0, s_t)))

    return M


def setup_ep_model(cellmodel, mesh, PCL=1000):
    """Set-up cardiac model based on benchmark parameters."""

    # Define time
    time = dolfin.Constant(0.0)

    # Surface to volume ratio
    chi = 140.0  # mm^{-1}
    # Membrane capacitance
    C_m = 0.01  # mu F / mm^2

    # Define conductivity tensor
    M = define_conductivity_tensor(chi, C_m)

    # Mark stimulation region defined as [0, L]^3
    S1_marker = 1
    S1_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    S1_markers.set_all(S1_marker)  # Mark the whole mesh

    # Define stimulation (NB: region of interest carried by the mesh
    # and assumptions in cbcbeat)
    duration = 2.0  # ms
    A = 50000.0  # mu A/cm^3
    cm2mm = 10.0
    factor = 1.0 / (chi * C_m)  # NB: cbcbeat convention
    amplitude = factor * A * (1.0 / cm2mm) ** 3  # mV/ms

    s = "((std::fmod(time,PCL) >= start) & (std::fmod(time,PCL) <= duration + start)) ? amplitude : 0.0"

    I_s = dolfin.Expression(
        s,
        time=time,
        start=0.0,
        duration=duration,
        amplitude=amplitude,
        PCL=PCL,
        degree=0,
    )
    # Store input parameters in cardiac model
    stimulus = cbcbeat.Markerwise((I_s,), (S1_marker,), S1_markers)

    petsc_options = [
        ["ksp_type", "cg"],
        ["pc_type", "gamg"],
        ["pc_gamg_verbose", "10"],
        ["pc_gamg_square_graph", "0"],
        ["pc_gamg_coarse_eq_limit", "3000"],
        ["mg_coarse_pc_type", "redundant"],
        ["mg_coarse_sub_pc_type", "lu"],
        ["mg_levels_ksp_type", "richardson"],
        ["mg_levels_ksp_max_it", "3"],
        ["mg_levels_pc_type", "sor"],
    ]
    for opt in petsc_options:
        dolfin.PETScOptions.set(*opt)

    heart = cbcbeat.CardiacModel(
        domain=mesh,
        time=time,
        M_i=M,
        M_e=None,
        cell_models=cellmodel,
        stimulus=stimulus,
        applied_current=None,
    )

    return heart


def setup_splitting_solver_parameters(
    dt,
    theta=0.5,
    preconditioner="sor",
    scheme="GRL1",
):
    ps = cbcbeat.SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["MonodomainSolver"]["linear_solver_type"] = "iterative"
    ps["MonodomainSolver"]["theta"] = theta
    ps["MonodomainSolver"]["preconditioner"] = preconditioner
    ps["MonodomainSolver"]["default_timestep"] = dt
    ps["MonodomainSolver"]["use_custom_preconditioner"] = False
    ps["theta"] = theta
    ps["enable_adjoint"] = False
    ps["apply_stimulus_current_to_pde"] = True
    # ps["BasicCardiacODESolver"]["scheme"] = scheme
    ps["CardiacODESolver"]["scheme"] = scheme
    # ps["ode_solver_choice"] = "BasicCardiacODESolver"
    # ps["BasicCardiacODESolver"]["V_polynomial_family"] = "CG"
    # ps["BasicCardiacODESolver"]["V_polynomial_degree"] = 1
    # ps["BasicCardiacODESolver"]["S_polynomial_family"] = "CG"
    # ps["BasicCardiacODESolver"]["S_polynomial_degree"] = 1
    return ps


def file_exist(filename: Optional[str], suffix: str) -> bool:
    return (
        filename is not None
        and filename != ""
        and Path(filename).is_file()
        and Path(filename).suffix == suffix
    )


def load_json(filename: str):
    with open(filename, "r") as fid:
        d = json.load(fid)
    return d


def handle_cell_params(
    cell_params: Optional[Dict[str, float]] = None,
    disease_state: str = "healthy",
    drug_factors_file: str = "",
    popu_factors_file: str = "",
):
    cell_params_tmp = CellModel.default_parameters(disease_state)
    # FIXME: In this case we update the parameters first, while in the
    # initial condition case we do that last. We need to be consistent
    # about this.
    if cell_params is not None:
        cell_params_tmp.update(cell_params)
    # Adding optional drug factors to parameters (if drug_factors_file exists)
    if file_exist(drug_factors_file, ".json"):
        logger.info(f"Drug scaling factors loaded from {drug_factors_file}")
        cell_params_tmp.update(load_json(drug_factors_file))
    else:
        if drug_factors_file != "":
            logger.warning(f"Unable to load drug factors file {drug_factors_file}")
    # FIXME: A problem here is that popu_factors_file will overwrite the
    # drug_factors_file. Is it possible to only have one file?
    if file_exist(popu_factors_file, ".json"):
        logger.info(f"Population scaling factors loaded from {popu_factors_file}")
        cell_params_tmp.update(load_json(popu_factors_file))
    else:
        if popu_factors_file != "":
            logger.warning(f"Unable to load popu factors file {popu_factors_file}")

    return cell_params_tmp


def handle_cell_inits(
    cell_inits: Optional[Dict[str, float]] = None,
    cell_init_file: str = "",
) -> Dict[str, float]:
    cell_inits_tmp = CellModel.default_initial_conditions()
    if file_exist(cell_init_file, ".json"):
        cell_inits_tmp.update(load_json(cell_init_file))

    if file_exist(cell_init_file, ".h5"):
        cell_inits_tmp.update(load_json(cell_init_file))
        from .save_load_functions import load_initial_conditions_from_h5

        cell_inits = load_initial_conditions_from_h5(cell_init_file)

    # FIXME: This is a bit confusing, since it will overwrite the
    # inputs from the cell_init_file. There should be only one way to
    # do this IMO. I think this might be difficult for the user to reason
    # about. I think in general we should handle the loading from files
    # at higher level.
    if cell_inits is not None:
        cell_inits_tmp.update(cell_inits)
    return cell_inits_tmp
