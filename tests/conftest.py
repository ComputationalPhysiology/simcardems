from pathlib import Path

import dolfin
import pytest
import simcardems

_here = Path(__file__).absolute().parent


@pytest.fixture(scope="session")
def mesh():
    return dolfin.UnitCubeMesh(2, 2, 2)


@pytest.fixture
def coupling(mesh):
    return simcardems.EMCoupling(mesh)


@pytest.fixture
def cell_params():
    return simcardems.ORdmm_Land.ORdmm_Land.default_parameters()


@pytest.fixture(
    params=[
        None,
        _here.parent.joinpath("demos")
        .joinpath("initial_conditions")
        .joinpath("init_5000beats.json"),
    ],
)
def ep_solver(request, mesh, coupling):
    return simcardems.ep_model.setup_solver(
        mesh=mesh,
        dt=0.01,
        coupling=coupling,
        cell_init_file=request.param,
    )
