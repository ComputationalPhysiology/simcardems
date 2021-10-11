from pathlib import Path
from unittest import mock

import dolfin
import pytest
import simcardems

_here = Path(__file__).absolute().parent


@pytest.fixture(scope="session")
def mesh():
    return dolfin.UnitCubeMesh(1, 1, 1)


@pytest.fixture
def coupling(mesh):
    return simcardems.EMCoupling(mesh)


@pytest.fixture
def cell_params():
    return simcardems.ORdmm_Land.ORdmm_Land.default_parameters()


@pytest.fixture(
    params=[
        "",
        _here.parent.joinpath("demos")
        .joinpath("initial_conditions")
        .joinpath("init_5000beats.json"),
    ],
)
def ep_solver(request, mesh, coupling):
    params = dolfin.Parameters("CardiacODESolver")
    params.add("scheme", "BackwardEuler")
    states = simcardems.ORdmm_Land.ORdmm_Land.default_initial_conditions()
    modelparams = simcardems.ORdmm_Land.ORdmm_Land.default_parameters()
    VS = dolfin.VectorFunctionSpace(mesh, "CG", 1, dim=len(states))
    vs = dolfin.Function(VS)
    vs.assign(dolfin.Constant(list(states.values())))
    vs_ = vs.copy()

    config = {
        "default_parameters.return_value": params,
        "__name__": "CardiacODESolver",
    }

    with mock.patch(
        "simcardems.ep_model.cbcbeat.splittingsolver.CardiacODESolver", **config
    ) as m:
        instance = m.return_value
        instance.solution_fields.return_value = (vs_, vs)
        instance._model.parameters.return_value = modelparams
        solver = simcardems.ep_model.setup_solver(
            mesh=mesh,
            dt=0.01,
            coupling=coupling,
            cell_init_file=request.param,
            scheme="ForwardEuler",
        )
    yield solver
