from pathlib import Path

import numpy as np
import pytest
import simcardems
from simcardems import save_load_functions as slf


def tests_h5pyfile():
    dummyfile = Path("tmp_h5pyfile.h5")
    if dummyfile.is_file():
        dummyfile.unlink()

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

    dummyfile.unlink()


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
def test_dict_to_h5(data):

    dummyfile = Path("tmp_h5pyfile.h5")
    if dummyfile.is_file():
        dummyfile.unlink()

    h5group = "testgroup"
    slf.dict_to_h5(data, dummyfile, h5group)

    with slf.h5pyfile(dummyfile, "r") as h5file:
        loaded_data = slf.h5_to_dict(h5file[h5group])

    assert loaded_data == data


@pytest.mark.slow
def test_save_and_load_state(mesh, coupling, ep_solver, cell_params):

    coupling.register_ep_model(ep_solver)

    dt = 0.01
    Lx = 1
    Ly = 1
    Lz = 1

    bnd_cond = "dirichlet"

    mech_heart, bnd_right_x = simcardems.mechanics_model.setup_mechanics_model(
        mesh=mesh,
        coupling=coupling,
        dt=0.01,
        bnd_cond=bnd_cond,
        cell_params=cell_params,
        Lx=1,
    )

    nsteps = 10
    for i in range(nsteps):
        t0 = i * dt
        t1 = (i + 1) * dt
        ep_solver.step((t0, t1))
        coupling.update_mechanics()
        mech_heart.solve()
        mech_heart.material.active.update_prev()
        coupling.update_ep()
        ep_solver.vs_.assign(ep_solver.vs)

    state_path = Path("tmp_state.h5")

    slf.save_state(
        state_path,
        solver=ep_solver,
        mech_heart=mech_heart,
        dt=dt,
        bnd_cond=bnd_cond,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        t0=t0,
    )

    coupling_, ep_solver_, mech_heart_, bnd_right_x, mesh_, t0_ = slf.load_state(
        state_path,
    )

    assert t0_ == t0
    assert (mesh.coordinates() == mesh_.coordinates()).all()
    assert simcardems.utils.compute_norm(ep_solver.vs_, ep_solver_.vs_) < 1e-12
    assert simcardems.utils.compute_norm(mech_heart.state, mech_heart_.state) < 1e-12
    assert simcardems.utils.compute_norm(coupling.vs, coupling_.vs) < 1e-12
    assert simcardems.utils.compute_norm(coupling.XS, coupling_.XS) < 1e-12
    assert simcardems.utils.compute_norm(coupling.XW, coupling_.XW) < 1e-12
    assert simcardems.utils.compute_norm(coupling.lmbda, coupling_.lmbda) < 1e-12
    assert simcardems.utils.compute_norm(coupling.Zetas, coupling_.Zetas) < 1e-12
    assert simcardems.utils.compute_norm(coupling.Zetaw, coupling_.Zetaw) < 1e-12

    state_path.unlink()
