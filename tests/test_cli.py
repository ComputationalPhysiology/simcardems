import json
import shutil
from pathlib import Path

from click.testing import CliRunner
from simcardems.cli import run
from simcardems.cli import run_json

mesh_args = ["-lx", 1, "-ly", 1, "-lz", 1, "-dx", 1]


def test_run():
    runner = CliRunner()
    outdir = Path("test_outdir")
    if outdir.exists():
        shutil.rmtree(outdir)
    result = runner.invoke(run, ["-T", 1, "-o", outdir.as_posix()] + mesh_args)
    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()


def test_run_json():
    runner = CliRunner()
    outdir = Path("test_outdir")
    if outdir.exists():
        shutil.rmtree(outdir)

    args = ["-T", 1, "--outdir", outdir.as_posix()] + mesh_args
    data = dict(zip([arg.strip("-").strip("-") for arg in args[::2]], args[1::2]))
    json_path = "args.json"

    with open(json_path, "w") as jsonfile:
        json.dump(data, jsonfile)
    result = runner.invoke(run_json, [json_path])
    # breakpoint()
    print(result.output)
    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()
