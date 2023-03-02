from pathlib import Path
from unittest import mock

import dolfin
import h5py
import pytest
import simcardems


def test_DataCollector_reset_state_when_file_exists(tmp_path, geo):
    simcardems.DataCollector(tmp_path, geo=geo)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(tmp_path, geo=geo, reset_state=True)
    remove_file_mock.assert_called()
    assert Path(collector.results_file).is_file()


def test_DataCollector_not_reset_state_when_file_exists(tmp_path, geo):
    simcardems.DataCollector(tmp_path, geo=geo)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(tmp_path, geo=geo, reset_state=False)
    remove_file_mock.assert_not_called()
    assert Path(collector.results_file).is_file()


def test_DataCollector_create_file_with_geo(tmp_path, geo):
    collector = simcardems.DataCollector(tmp_path, geo=geo)
    assert Path(collector.results_file).is_file()
    with h5py.File(collector.results_file, "r") as h5file:
        assert "geometry" in h5file


def test_DataCollector_register(tmp_path, geo):
    V = dolfin.FunctionSpace(geo.mesh, "CG", 1)
    f = dolfin.Function(V)

    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(tmp_path, geo=geo)
    collector.register("ep", "func", f)
    assert "func" in collector.names["ep"]

    with pytest.raises(ValueError) as ex:
        simcardems.DataLoader(collector.results_file)
    assert str(ex.value) == "No functions found in results file"


@pytest.mark.parametrize("group", ["ep", "mechanics"])
def test_DataCollector_store(group, geo, tmp_path):
    mesh = geo.mesh if group == "mechanics" else geo.ep_mesh
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    f = dolfin.Function(V)
    f.vector()[:] = 42

    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(tmp_path, geo=geo)
    collector.register(group, "func", f)
    assert "func" in collector.names[group]

    collector.store(0)
    loader = simcardems.DataLoader(collector.results_file)
    assert loader.time_stamps == ["0.00"]

    g1 = loader.get(group, "func", "0.00")
    g2 = loader.get(group, "func", 0)
    assert all(g1.vector().get_local() == f.vector().get_local())
    assert all(g2.vector().get_local() == f.vector().get_local())


def test_DataLoader_load_empty_files_raises_ValueError(tmp_path, geo):
    collector = simcardems.DataCollector(tmp_path, geo=geo)
    with pytest.raises(ValueError):
        simcardems.DataLoader(collector.results_file)


def test_DataCollector_store_version(tmp_path, geo):
    collector = simcardems.DataCollector(tmp_path, geo=geo)
    loader = simcardems.DataLoader(collector.results_file, empty_ok=True)
    from packaging.version import parse

    v = parse(simcardems.__version__)
    assert loader.version_major == v.major
    assert loader.version_minor == v.minor
    assert loader.version_micro == v.micro
