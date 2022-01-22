import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from simcardems.cli import run
from simcardems.cli import run_json

mesh_args = ["-lx", 1, "-ly", 1, "-lz", 1, "-dx", 1, "--num_refinements", 1]


@pytest.mark.slow
def test_run():
    runner = CliRunner()
    outdir = Path("test_outdir").absolute()
    if outdir.exists():
        shutil.rmtree(outdir)
    args = ["-T", 1, "--outdir", outdir.as_posix()] + mesh_args
    kwargs = dict(zip([arg.strip("-").strip("-") for arg in args[::2]], args[1::2]))
    result = runner.invoke(run, args)
    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()

    with open(outdir.joinpath("parameters.json"), "r") as f:
        dumped_arguments = json.load(f)

    for k, v in kwargs.items():
        assert dumped_arguments[k] == v, k


@pytest.mark.slow
def test_run_json():
    runner = CliRunner()
    outdir = Path("test_outdir").absolute()
    if outdir.exists():
        shutil.rmtree(outdir)

    args = ["-T", 1, "--outdir", outdir.as_posix()] + mesh_args
    kwargs = dict(zip([arg.strip("-").strip("-") for arg in args[::2]], args[1::2]))
    json_path = "args.json"

    with open(json_path, "w") as jsonfile:
        json.dump(kwargs, jsonfile)
    result = runner.invoke(run_json, [json_path])
    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()
    with open(outdir.joinpath("parameters.json"), "r") as f:
        dumped_arguments = json.load(f)
    for k, v in kwargs.items():
        assert dumped_arguments[k] == v
