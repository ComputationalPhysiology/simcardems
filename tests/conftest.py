from pathlib import Path
from unittest import mock

import dolfin
import pytest
import simcardems

_here = Path(__file__).absolute().parent


@pytest.fixture(scope="session")
def mesh():
    return dolfin.UnitCubeMesh(1, 1, 1)


@pytest.fixture(scope="session")
def geometry(mesh):
    return simcardems.geometry.SlabGeometry(
        lx=1,
        ly=1,
        lz=1,
        dx=1,
        num_refinements=1,
        mechanics_mesh=mesh,
    )


@pytest.fixture
def coupling(geometry):
    return simcardems.EMCoupling(geometry)


@pytest.fixture
def cell_params():
    return simcardems.cell_model.ORdmm_Land.default_parameters()


@pytest.fixture(
    params=[
        "",
        _here.parent.joinpath("demos")
        .joinpath("initial_conditions")
        .joinpath("init_5000beats.json"),
    ],
)
def ep_solver(request, coupling):
    params = dolfin.Parameters("CardiacODESolver")
    params.add("scheme", "BackwardEuler")
    states = simcardems.cell_model.ORdmm_Land.default_initial_conditions()
    modelparams = simcardems.cell_model.ORdmm_Land.default_parameters()
    VS = dolfin.VectorFunctionSpace(coupling.ep_mesh, "CG", 1, dim=len(states))
    vs = dolfin.Function(VS)
    vs.assign(dolfin.Constant(list(states.values())))
    vs_ = vs.copy()

    config = {
        "default_parameters.return_value": params,
        "__name__": "CardiacODESolver",
    }

    with mock.patch(
        "simcardems.setup_models.cbcbeat.splittingsolver.CardiacODESolver", **config
    ) as m:
        instance = m.return_value
        instance.solution_fields.return_value = (vs_, vs)
        instance._model.parameters.return_value = modelparams
        solver = simcardems.setup_models.setup_ep_solver(
            dt=0.01,
            coupling=coupling,
            cell_init_file=request.param,
            scheme="ForwardEuler",
        )
    yield solver
