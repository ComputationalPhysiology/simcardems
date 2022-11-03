from unittest import mock

import pytest
import simcardems


@pytest.mark.slow
@pytest.mark.parametrize("bnd_rigid", [False, True])
def test_em_model(coupling, ep_solver, cell_params, geo, bnd_rigid):
    coupling.register_ep_model(ep_solver)
    with mock.patch(
        "simcardems.setup_models.pulse.mechanicsproblem.MechanicsProblem.solve",
    ) as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        simcardems.setup_models.setup_mechanics_solver(
            coupling=coupling,
            bnd_rigid=bnd_rigid,
            cell_params=cell_params,
            geo=geo,
        )
