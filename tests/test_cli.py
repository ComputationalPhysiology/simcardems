import json
import math

import pytest
import simcardems
from click.testing import CliRunner
from simcardems.cli import postprocess
from simcardems.cli import run
from simcardems.cli import run_json

mesh_args = ["-lx", 1, "-ly", 1, "-lz", 1, "-dx", 1, "--num_refinements", 1]


@pytest.mark.slow
@pytest.mark.parametrize("coupling_type", simcardems.models.list_coupling_types())
def test_run(coupling_type, tmp_path, geo):
    geometry_path = tmp_path / "geo.h5"
    geometry_schema_path = geometry_path.with_suffix(".json")
    geo.dump(fname=geometry_path, schema_path=geometry_schema_path)
    geo_args = [geometry_path.as_posix(), "-s", geometry_schema_path.as_posix()]

    runner = CliRunner()
    outdir = tmp_path / "results"

    T = 1.0
    arguments = geo_args + [
        "-T",
        T,
        "--outdir",
        outdir.as_posix(),
        "--coupling-type",
        coupling_type,
    ]

    result = runner.invoke(run, arguments)

    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()

    with open(outdir.joinpath("parameters.json"), "r") as f:
        dumped_arguments = json.load(f)

    assert dumped_arguments["geometry_path"] == geometry_path.as_posix()
    assert dumped_arguments["geometry_schema_path"] == geometry_schema_path.as_posix()
    assert math.isclose(dumped_arguments["T"], T)
    assert dumped_arguments["outdir"] == outdir.as_posix()
    assert outdir.joinpath("state.h5").is_file()

    # Make sure we can restart the simulation from the current time point
    T = 2.0
    arguments = geo_args + [
        "-T",
        T,
        "--outdir",
        outdir.as_posix(),
        "--load_state",
    ]

    result = runner.invoke(run, arguments)
    assert result.exit_code == 0
    coupling = getattr(simcardems.models, coupling_type).EMCoupling.from_state(
        path=outdir.joinpath("state.h5"),
    )
    assert math.isclose(coupling.t, 2)
    assert coupling.coupling_type == coupling_type

    # Make sure we can plot the state traces
    result = runner.invoke(postprocess, outdir.as_posix(), "--plot-state-traces")
    assert result.exit_code == 0


@pytest.mark.slow
def test_run_json(tmp_path, geo):
    geometry_path = tmp_path / "geo.h5"
    geometry_schema_path = geometry_path.with_suffix(".json")
    geo.dump(fname=geometry_path, schema_path=geometry_schema_path)
    # geo_args = [geometry_path.as_posix(), "-s", geometry_schema_path.as_posix()]

    runner = CliRunner()
    outdir = tmp_path / "results"

    T = 0.2
    # arguments = geo_args + ["-T", 0.2, "--outdir", outdir.as_posix()]
    # Do not provide schema path to test that it picks up the default schema path
    kwargs = {
        "geometry_path": geometry_path.as_posix(),
        "outdir": outdir.as_posix(),
        "T": T,
    }
    json_path = "args.json"

    with open(json_path, "w") as jsonfile:
        json.dump(kwargs, jsonfile)
    result = runner.invoke(run_json, [json_path])
    assert result.exit_code == 0
    assert outdir.exists()
    assert outdir.joinpath("results.h5").is_file()
    with open(outdir.joinpath("parameters.json"), "r") as f:
        dumped_arguments = json.load(f)
    assert dumped_arguments["geometry_path"] == geometry_path.as_posix()
    assert dumped_arguments["geometry_schema_path"] == geometry_schema_path.as_posix()
    assert math.isclose(dumped_arguments["T"], T)
    assert dumped_arguments["outdir"] == outdir.as_posix()
