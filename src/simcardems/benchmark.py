import datetime
import json
import time
from pathlib import Path

import click
from simcardems.postprocess import extract_biomarkers
from simcardems.postprocess import extract_traces

t0 = time.perf_counter()
import simcardems  # noqa: E402

import_time = time.perf_counter() - t0


@click.command()
@click.argument("outdir", required=True, type=click.Path())
def main(outdir):
    data = {}

    data["import_time"] = import_time
    data["timestamp"] = datetime.datetime.now().isoformat()
    data["simcardems_version"] = simcardems.__version__

    config = simcardems.Config(
        outdir=outdir,
        T=1000,
        load_state=True,
        outfilename="benchmark_results.h5",
    )
    t0 = time.perf_counter()
    runner = simcardems.Runner(config)
    data["create_runner_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    if runner.t0 < config.T:
        runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
    data["solve_time"] = time.perf_counter() - t0

    loader = simcardems.DataLoader(Path(outdir) / "benchmark_results.h5")

    values = extract_traces(loader=loader)
    data.update(
        extract_biomarkers(
            V=values["ep"]["V"],
            Ta=values["mechanics"]["Ta"],
            time=values["time"],
            Ca=values["ep"]["Ca"],
            lmbda=values["mechanics"]["lambda"],
            inv_lmbda=1 - values["mechanics"]["lambda"],
            u=values["mechanics"]["u"],
        ),
    )

    path = Path(outdir) / "results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
