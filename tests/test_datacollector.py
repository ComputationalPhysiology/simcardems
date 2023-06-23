from pathlib import Path
from unittest import mock

import dolfin
import pytest
import simcardems


def test_DataCollector_reset_state_when_file_exists(mpi_tmp_path, geo):
    simcardems.DataCollector(mpi_tmp_path, geo=geo)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(mpi_tmp_path, geo=geo, reset_state=True)
    remove_file_mock.assert_called()
    assert Path(collector.results_file).is_file()


def test_DataCollector_not_reset_state_when_file_exists(mpi_tmp_path, geo):
    simcardems.DataCollector(mpi_tmp_path, geo=geo)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(mpi_tmp_path, geo=geo, reset_state=False)
    remove_file_mock.assert_not_called()
    assert Path(collector.results_file).is_file()


def test_DataCollector_create_file_with_geo(mpi_tmp_path, geo):
    collector = simcardems.DataCollector(mpi_tmp_path, geo=geo)
    assert Path(collector.results_file).is_file()
    with simcardems.save_load_functions.h5pyfile(collector.results_file, "r") as h5file:
        assert "geometry" in h5file


def test_DataCollector_register(mpi_tmp_path, geo):
    V = dolfin.FunctionSpace(geo.mesh, "CG", 1)
    f = dolfin.Function(V)

    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(mpi_tmp_path, geo=geo)
    collector.register("ep", "func", f)
    assert "func" in collector.names["ep"]

    with pytest.raises(ValueError) as ex:
        simcardems.DataLoader(collector.results_file)
    assert str(ex.value) == "No functions found in results file"


@pytest.mark.parametrize("group", ["ep", "mechanics"])
def test_DataCollector_store(group, geo, mpi_tmp_path):
    mesh = geo.mesh if group == "mechanics" else geo.ep_mesh
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    f = dolfin.Function(V)
    f.vector()[:] = 42

    simcardems.set_log_level(10)
    collector = simcardems.DataCollector(mpi_tmp_path, geo=geo)
    collector.register(group, "func", f)
    assert "func" in collector.names[group]

    collector.store(0)
    loader = simcardems.DataLoader(collector.results_file)
    assert loader.time_stamps == ["0.00"]

    g1 = loader.get(group, "func", "0.00")
    g2 = loader.get(group, "func", 0)
    assert all(g1.vector().get_local() == f.vector().get_local())
    assert all(g2.vector().get_local() == f.vector().get_local())


def test_DataLoader_load_empty_files_raises_ValueError(mpi_tmp_path, geo):
    collector = simcardems.DataCollector(mpi_tmp_path, geo=geo)
    with pytest.raises(ValueError):
        simcardems.DataLoader(collector.results_file)


def test_DataCollector_store_version(mpi_tmp_path, geo):
    collector = simcardems.DataCollector(mpi_tmp_path, geo=geo)
    loader = simcardems.DataLoader(collector.results_file, empty_ok=True)
    from packaging.version import parse

    v = parse(simcardems.__version__)
    assert loader.version_major == v.major
    assert loader.version_minor == v.minor
    assert loader.version_micro == v.micro
