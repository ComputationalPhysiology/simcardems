from unittest import mock

import pytest
import simcardems


@pytest.mark.slow
@pytest.mark.parametrize(
    "mech_model_type",
    simcardems.mechanics_model.MechanicsModelType._member_names_,
)
def test_em_model(coupling, ep_solver, cell_params, mech_model_type):
    coupling.register_ep_model(ep_solver)
    with mock.patch(
        "simcardems.setup_models.pulse.mechanicsproblem.MechanicsProblem.solve",
    ) as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        simcardems.setup_models.setup_mechanics_solver(
            coupling=coupling,
            mech_model_type=mech_model_type,
            cell_params=cell_params,
        )
