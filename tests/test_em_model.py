from unittest import mock

import pytest
import simcardems


@pytest.mark.slow
@pytest.mark.parametrize("bnd_cond", ["dirichlet", "rigid"])
def test_em_model(mesh, coupling, ep_solver, cell_params, bnd_cond):
    coupling.register_ep_model(ep_solver)
    with mock.patch(
        "simcardems.mechanics_model.pulse.mechanicsproblem.MechanicsProblem.solve",
    ) as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        mech_heart, bnd_right_x = simcardems.mechanics_model.setup_mechanics_model(
            mesh=mesh,
            coupling=coupling,
            dt=0.01,
            bnd_cond=bnd_cond,
            cell_params=cell_params,
            Lx=1,
        )
