import shutil
from pathlib import Path
from unittest import mock

import h5py
import pytest
import simcardems


def test_DataCollector_reset_state_when_file_exists(mesh):
    outdir = Path("testdir")

    simcardems.DataCollector(outdir, mesh)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(outdir, mesh, reset_state=True)
    remove_file_mock.assert_called()
    assert Path(collector.results_file).is_file()
    shutil.rmtree(outdir)


def test_DataCollector_not_reset_state_when_file_exists(mesh):
    outdir = Path("testdir")

    simcardems.DataCollector(outdir, mesh)

    with mock.patch("simcardems.utils.remove_file") as remove_file_mock:
        collector = simcardems.DataCollector(outdir, mesh, reset_state=False)
    remove_file_mock.assert_not_called()
    assert Path(collector.results_file).is_file()
    shutil.rmtree(outdir)


def test_DataCollector_create_file_with_mesh(mesh):
    outdir = Path("testdir")

    collector = simcardems.DataCollector(outdir, mesh)
    assert Path(collector.results_file).is_file()
    with h5py.File(collector.results_file, "r") as h5file:
        assert "mesh" in h5file
    shutil.rmtree(outdir)


def test_DataCollector_register():
    pass


def test_DataCollector_store():
    pass


def test_DataLoader_load_empty_files_raises_ValueError(mesh):
    outdir = Path("testdir")

    collector = simcardems.DataCollector(outdir, mesh)
    with pytest.raises(ValueError):
        simcardems.DataLoader(collector.results_file)
    shutil.rmtree(outdir)
