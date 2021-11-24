import shutil
from pathlib import Path
from unittest import mock

import dolfin
import h5py
import pytest
import simcardems


def test_DataCollector_reset_state_when_file_exists(mesh):
    outdir = Path("testdir")

    simcardems.DataCollector(outdir, mesh, mesh)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(outdir, mesh, mesh, reset_state=True)
    remove_file_mock.assert_called()
    assert Path(collector.results_file).is_file()
    shutil.rmtree(outdir)


def test_DataCollector_not_reset_state_when_file_exists(mesh):
    outdir = Path("testdir")

    simcardems.DataCollector(outdir, mesh, mesh)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(outdir, mesh, mesh, reset_state=False)
    remove_file_mock.assert_not_called()
    assert Path(collector.results_file).is_file()
    shutil.rmtree(outdir)


def test_DataCollector_create_file_with_mesh(mesh):
    outdir = Path("testdir")

    collector = simcardems.DataCollector(outdir, mesh, mesh)
    assert Path(collector.results_file).is_file()
    with h5py.File(collector.results_file, "r") as h5file:
        assert "ep/mesh" in h5file
        assert "mechanics/mesh" in h5file
    shutil.rmtree(outdir)


def test_DataCollector_register(mesh):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    f = dolfin.Function(V)
    outdir = Path("testdir")
    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(outdir, mech_mesh=mesh, ep_mesh=mesh)
    collector.register("ep", "func", f)
    assert "func" in collector.names["ep"]

    with pytest.raises(ValueError) as ex:
        simcardems.DataLoader(collector.results_file)
    assert str(ex.value) == "No functions found in results file"
    shutil.rmtree(outdir)


@pytest.mark.parametrize("group", ["ep", "mechanics"])
def test_DataCollector_store(group, mesh):
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    f = dolfin.Function(V)
    f.vector()[:] = 42
    outdir = Path("testdir")
    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(outdir, mech_mesh=mesh, ep_mesh=mesh)
    collector.register(group, "func", f)
    assert "func" in collector.names[group]

    collector.store(0)
    loader = simcardems.DataLoader(collector.results_file)
    assert loader.time_stamps == ["0.00"]
    g1 = loader.get(group, "func", "0.00")
    g2 = loader.get(group, "func", 0)
    assert all(g1.vector().get_local() == f.vector().get_local())
    assert all(g2.vector().get_local() == f.vector().get_local())
    shutil.rmtree(outdir)


def test_DataLoader_load_empty_files_raises_ValueError(mesh):
    outdir = Path("testdir")

    collector = simcardems.DataCollector(outdir, mesh, mesh)
    with pytest.raises(ValueError):
        simcardems.DataLoader(collector.results_file)
    shutil.rmtree(outdir)
