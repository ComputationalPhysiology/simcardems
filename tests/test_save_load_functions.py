from unittest import mock

import numpy as np
import pytest
import simcardems
from simcardems import save_load_functions as slf


@pytest.fixture
def dummyfile():
    path = "tmp_h5pyfile.h5"
    simcardems.utils.remove_file(path)
    yield path
    simcardems.utils.remove_file(path)


def tests_h5pyfile(dummyfile):

    h5group = "test"
    data1 = [1, 2, 3, 4]
    data2 = [4, 5, 6]

    with slf.h5pyfile(dummyfile, "w") as h5file:
        group = h5file.create_group(h5group)
        group.create_dataset("data1", data=data1)

    with slf.h5pyfile(dummyfile, "a") as h5file:
        h5file[h5group].create_dataset("data2", data=data2)

    with slf.h5pyfile(dummyfile, "r") as h5file:
        assert h5group in h5file

        assert "data1" in h5file[h5group]
        assert np.isclose(h5file[h5group]["data1"][:], data1).all()

        assert "data2" in h5file[h5group]
        assert np.isclose(h5file[h5group]["data2"][:], data2).all()


@pytest.mark.parametrize(
    "data",
    (
        {},
        {"a": 1, "b": 2},
        {"a": 1.0, "b": 2},
        {"a": 1.0, "b": 2, "c": "three"},
        {"a": 1.0, "b": 2, "c": "three", "d": [1, 2]},
    ),
)
def test_dict_to_h5(data, dummyfile):

    h5group = "testgroup"
    slf.dict_to_h5(data, dummyfile, h5group)

    with slf.h5pyfile(dummyfile, "r") as h5file:
        loaded_data = slf.h5_to_dict(h5file[h5group])

    assert loaded_data == data


@pytest.mark.slow
@mock.patch("simcardems.mechanics_model.pulse.mechanicsproblem.MechanicsProblem.solve")
def test_save_and_load_state(
    solve_mock,
    dummyfile,
    mesh,
    coupling,
    ep_solver,
    cell_params,
):

    solve_mock.return_value = (1, True)  # (niter, nconv)
    coupling.register_ep_model(ep_solver)

    dt = 0.01

    bnd_cond = "dirichlet"

    mech_heart = simcardems.mechanics_model.setup_mechanics_model(
        coupling=coupling,
        dt=0.01,
        bnd_cond=bnd_cond,
        cell_params=cell_params,
    )

    # Save some non-zero values
    t0 = 1.0
    ep_solver.vs.vector()[:] = 1.0
    ep_solver.vs_.vector()[:] = 1.0
    coupling.XS_mech.vector()[:] = 1.0
    coupling.XW_mech.vector()[:] = 1.0
    mech_heart.state.vector()[:] = 1.0

    slf.save_state(
        dummyfile,
        solver=ep_solver,
        mech_heart=mech_heart,
        coupling=coupling,
        dt=dt,
        bnd_cond=bnd_cond,
        t0=t0,
    )

    with mock.patch("simcardems.ep_model.cbcbeat.SplittingSolver") as m:
        m.return_value = ep_solver

        coupling_, ep_solver_, mech_heart_, t0_ = slf.load_state(
            dummyfile,
        )

    assert t0_ == t0
    assert (mesh.coordinates() == coupling_.mech_mesh.coordinates()).all()
    assert simcardems.utils.compute_norm(ep_solver.vs_, ep_solver_.vs_) < 1e-12
    assert simcardems.utils.compute_norm(mech_heart.state, mech_heart_.state) < 1e-12
    assert simcardems.utils.compute_norm(coupling.vs, coupling_.vs) < 1e-12
    assert simcardems.utils.compute_norm(coupling.XS_mech, coupling_.XS_mech) < 1e-12
    assert simcardems.utils.compute_norm(coupling.XW_mech, coupling_.XW_mech) < 1e-12
    assert (
        simcardems.utils.compute_norm(coupling.lmbda_mech, coupling_.lmbda_mech) < 1e-12
    )
    assert (
        simcardems.utils.compute_norm(coupling.Zetas_mech, coupling_.Zetas_mech) < 1e-12
    )
    assert (
        simcardems.utils.compute_norm(coupling.Zetaw_mech, coupling_.Zetaw_mech) < 1e-12
    )
