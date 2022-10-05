import json
from pathlib import Path
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

    mech_heart = simcardems.setup_models.setup_mechanics_solver(
        coupling=coupling,
        bnd_cond=bnd_cond,
        cell_params=cell_params,
    )

    runner = simcardems.setup_models.Runner.from_models(
        coupling=coupling,
        ep_solver=ep_solver,
        mech_heart=mech_heart,
    )
    runner.outdir = "dummy_folder"

    t0 = 0.0
    t1 = 0.2
    runner.solve(t1)

    # Translate these value to the coupling object
    coupling.ep_to_coupling()
    # Translate to the mechanics model
    coupling.coupling_to_mechanics()

    slf.save_state(
        dummyfile,
        solver=ep_solver,
        mech_heart=mech_heart,
        coupling=coupling,
        dt=dt,
        bnd_cond=bnd_cond,
        t0=t0,
    )

    with mock.patch("simcardems.setup_models.cbcbeat.SplittingSolver") as m:
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

    active = mech_heart.material.active
    active_ = mech_heart.material.active

    assert simcardems.utils.compute_norm(active.lmbda, active_.lmbda) < 1e-12
    assert simcardems.utils.compute_norm(active.Zetas, active_.Zetas) < 1e-12
    assert simcardems.utils.compute_norm(active.Zetaw, active_.Zetaw) < 1e-12


@mock.patch("simcardems.mechanics_model.pulse.mechanicsproblem.MechanicsProblem.solve")
def test_load_state_with_new_parameters_uses_new_parameters(
    solve_mock,
    dummyfile,
    coupling,
    cell_params,
):

    ep_solver = simcardems.setup_models.setup_ep_solver(
        dt=0.01,
        coupling=coupling,
        cell_params=cell_params,
        scheme="ForwardEuler",
    )
    solve_mock.return_value = (1, True)  # (niter, nconv)
    coupling.register_ep_model(ep_solver)

    dt = 0.01

    bnd_cond = "dirichlet"

    mech_heart = simcardems.setup_models.setup_mechanics_solver(
        coupling=coupling,
        bnd_cond=bnd_cond,
        cell_params=cell_params,
    )

    # Save some non-zero values - just run the model
    runner = simcardems.setup_models.Runner.from_models(
        coupling=coupling,
        ep_solver=ep_solver,
        mech_heart=mech_heart,
    )
    runner.outdir = "dummy_folder"

    t0 = 0.2
    runner.solve(t0)

    slf.save_state(
        dummyfile,
        solver=ep_solver,
        mech_heart=mech_heart,
        coupling=coupling,
        dt=dt,
        bnd_cond=bnd_cond,
        t0=t0,
    )

    drug_factors_file = Path("drug_factors_file.json")
    drug_factors = {"scale_drug_INa": 42.43}
    drug_factors_file.write_text(json.dumps(drug_factors))

    popu_factors_file = Path("popu_factors_file.json")
    popu_factors = {"scale_popu_GNa": 13.13}
    popu_factors_file.write_text(json.dumps(popu_factors))

    coupling_, ep_solver_, mech_heart_, t0_ = slf.load_state(
        dummyfile,
        drug_factors_file=drug_factors_file,
        popu_factors_file=popu_factors_file,
    )

    assert np.isclose(
        ep_solver_.ode_solver._model.parameters()["scale_drug_INa"],
        drug_factors["scale_drug_INa"],
    )
    assert np.isclose(
        ep_solver_.ode_solver._model.parameters()["scale_popu_GNa"],
        popu_factors["scale_popu_GNa"],
    )

    drug_factors_file.unlink()
    popu_factors_file.unlink()
