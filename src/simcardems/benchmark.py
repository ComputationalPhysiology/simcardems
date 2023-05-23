import datetime
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Dict
from typing import Literal
from typing import Set
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from simcardems.postprocess import extract_biomarkers
from simcardems.postprocess import extract_traces

t0 = time.perf_counter()
import simcardems  # noqa: E402

import_time = time.perf_counter() - t0


def underscore_to_space(x: str) -> str:
    return " ".join(x.split("_"))


def to_string(x):
    if isinstance(x, str):
        return x
    return f"{x:.7g}"


def trace_path(outdir: str, dx: float, dt: float) -> Path:
    return Path(outdir) / f"results_dx{int(100*dx)}_dt{int(1000*dt)}.npy"


def feature_path(outdir: str, dx: float, dt: float) -> Path:
    return Path(outdir) / f"results_dx{int(100*dx)}_dt{int(1000*dt)}.json"


def find_dx_dt(
    outdir: str,
) -> Set[Tuple[float, float]]:
    npy_pattern = re.compile(r"results_dx(?P<dx>\d+)_dt(?P<dt>\d+).npy")
    json_pattern = re.compile(r"results_dx(?P<dx>\d+)_dt(?P<dt>\d+).json")

    npy_dx_dt = set()
    json_dx_dt = set()

    for fname in Path(outdir).iterdir():
        if fname.suffix == ".json":
            m = json_pattern.match(fname.name)
            if m is None:
                continue

            json_dx_dt.add((int(m["dx"]) / 100, int(m["dt"]) / 1000))

        elif fname.suffix == ".npy":
            m = npy_pattern.match(fname.name)
            if m is None:
                continue

            npy_dx_dt.add((int(m["dx"]) / 100, int(m["dt"]) / 1000))

        else:
            continue

    return npy_dx_dt & json_dx_dt


@click.command()
@click.argument(
    "outdir",
    required=True,
    type=click.Path(),
)
@click.option("--dt", type=float, default=0.05, help="Time step used in the EP solver")
@click.option(
    "--dx",
    type=float,
    default=0.2,
    help="Spatial discretization for the mechanics mesh",
)
@click.option("--sha", type=str, default="", help="Current git SHA")
@click.option(
    "--pure-ep",
    is_flag=True,
    default=False,
    help="Whether to run the fully coupled EM model or a pure ep model",
)
def run(outdir: str, dt: float, dx: float, sha, pure_ep):
    coupling_type: Literal["fully_coupled_ORdmm_Land", "pureEP_ORdmm_Land"] = (
        "pureEP_ORdmm_Land" if pure_ep else "fully_coupled_ORdmm_Land"
    )

    data: Dict[str, Any] = {}
    data["import_time"] = import_time
    data["timestamp"] = datetime.datetime.now().isoformat()
    data["simcardems_version"] = simcardems.__version__
    data["dt"] = dt
    data["dx"] = dx
    data["sha"] = sha
    data["coupling_type"] = coupling_type

    config = simcardems.Config(
        outdir=outdir,
        T=1000,
        load_state=True,
        dt=dt,
        coupling_type=coupling_type,
        outfilename="benchmark_results.h5",
    )

    t0 = time.perf_counter()
    geo = simcardems.slabgeometry.SlabGeometry(parameters={"dx": dx})

    if not pure_ep:
        data["num_cells_mechanics"] = geo.mesh.num_cells()
        data["num_vertices_mechanics"] = geo.mesh.num_vertices()

    data["num_cells_ep"] = geo.ep_mesh.num_cells()
    data["num_vertices_ep"] = geo.ep_mesh.num_vertices()

    coupling = simcardems.models.em_model.setup_EM_model_from_config(
        config=config,
        geometry=geo,
    )
    runner = simcardems.Runner.from_models(config=config, coupling=coupling)
    data["create_runner_time"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    if runner.t0 < config.T:
        runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=False)
    data["solve_time"] = time.perf_counter() - t0

    loader = simcardems.DataLoader(Path(outdir) / "benchmark_results.h5")

    values = extract_traces(loader=loader)

    kwargs = {
        "V": values["ep"]["V"],
        "time": values["time"],
        "Ca": values["ep"]["Ca"],
    }
    if not pure_ep:
        kwargs.update(
            {
                "Ta": values["mechanics"]["Ta"],
                "lmbda": values["mechanics"]["lambda"],
                "inv_lmbda": 1 - values["mechanics"]["lambda"],
                "u": values["mechanics"]["u"],
            },
        )

    data.update(
        extract_biomarkers(**kwargs),
    )

    with open(feature_path(outdir, dx, dt), "w") as f:
        json.dump(data, f, indent=2)

    np.save(trace_path(outdir, dx, dt), kwargs)

    return 0


@click.command()
@click.argument(
    "outdir",
    required=True,
    type=click.Path(),
)
@click.pass_context
def pure_ep_report(ctx, outdir):
    for dx in [0.1, 0.2, 0.4]:
        ctx.invoke(run, outdir=outdir, dt=0.05, dx=dx, sha="", pure_ep=True)
    for dt in [0.025, 0.1]:
        ctx.invoke(run, outdir=outdir, dt=dt, dx=0.2, sha="", pure_ep=True)

    ctx.invoke(generate_report, outdir=outdir)


def create_table_dx(feature_keys, features, fixed_dt, dxs):
    body = ""
    for feature in feature_keys:
        body += (
            f" {underscore_to_space(feature)} & "
            + " & ".join(
                [
                    underscore_to_space(to_string(features[(dx, fixed_dt)][feature]))
                    for dx in dxs
                ],
            )
            + "\\\\ \n"
        )

    table = dedent(
        """
    \\begin{{center}}
    \\begin{{tabular}}{{ {c} }}
    {heading}
    \\hline
    {body}
    \\end{{tabular}}
    \\end{{center}}
    """,
    ).format(
        c="c" * (len(dxs) + 1),
        heading=" &".join(["feature"] + [f"dt={dx}" for dx in dxs]) + "\\\\ \n",
        body=body,
    )
    return table


def create_table_dt(feature_keys, features, fixed_dx, dts):
    body = ""
    for feature in feature_keys:
        body += (
            f" {underscore_to_space(feature)} & "
            + " & ".join(
                [
                    underscore_to_space(to_string(features[(fixed_dx, dt)][feature]))
                    for dt in dts
                ],
            )
            + "\\\\ \n"
        )

    table = dedent(
        """
    \\begin{{center}}
    \\begin{{tabular}}{{ {c} }}
    {heading}
    \\hline
    {body}
    \\end{{tabular}}
    \\end{{center}}
    """,
    ).format(
        c="c" * (len(dts) + 1),
        heading=" &".join(["feature"] + [f"dt={dt}" for dt in dts]) + "\\\\ \n",
        body=body,
    )
    return table


@click.command()
@click.argument(
    "outdir",
    required=True,
    type=click.Path(),
)
def generate_report(outdir):
    dxs_dts = find_dx_dt(outdir)

    traces = {}
    features = {}

    for dx, dt in dxs_dts:
        traces[(dx, dt)] = np.load(
            trace_path(outdir, dx=dx, dt=dt),
            allow_pickle=True,
        ).item()
        features[(dx, dt)] = json.loads(feature_path(outdir, dx=dx, dt=dt).read_text())

    # Find most frequent dx and dt
    fixed_dx = Counter(map(lambda x: x[0], dxs_dts)).most_common()[0][0]
    fixed_dt = Counter(map(lambda x: x[1], dxs_dts)).most_common()[0][0]

    dxs = sorted([dx for dx, dt in dxs_dts if np.isclose(dt, fixed_dt)])
    dts = sorted([dt for dx, dt in dxs_dts if np.isclose(dx, fixed_dx)])

    keys = [key for key in traces[tuple(traces.keys())[0]].keys() if key != "time"]
    feature_keys = [
        k
        for k in features[tuple(features.keys())[0]].keys()
        if k
        not in ["dt", "dx", "sha", "timestamp", "simcardems_version", "coupling_type"]
    ]

    timestamp = features[tuple(features.keys())[0]]["timestamp"]
    version = features[tuple(features.keys())[0]]["simcardems_version"]
    coupling_type = underscore_to_space(
        features[tuple(features.keys())[0]]["coupling_type"],
    )
    fig, ax = plt.subplots(len(keys), 1, sharex=True)

    for dx in dxs:
        d = traces[(dx, fixed_dt)]
        for i, name in enumerate(keys):
            ax[i].plot(d["time"], d[name], label=rf"$\Delta x = {dx}$")

    for i, name in enumerate(keys):
        ax[i].set_ylabel(name)
        ax[i].legend()
    ax[0].set_title("Spatial convergence")
    fig.savefig(Path(outdir) / "spatial_convergence.png", bbox_inches="tight")

    fig, ax = plt.subplots(len(keys), 1, sharex=True)

    for dt in dts:
        d = traces[(fixed_dx, dt)]
        for i, name in enumerate(keys):
            ax[i].plot(d["time"], d[name], label=rf"$\Delta t = {dt}$")

    for i, name in enumerate(keys):
        ax[i].set_ylabel(name)
        ax[i].legend()

    ax[0].set_title("Temporal convergence")
    fig.savefig(Path(outdir) / "temporal_convergence.png", bbox_inches="tight")

    text = dedent(
        f"""
    \\documentclass{{article}}
    \\usepackage{{graphicx}}
    \\begin{{document}}

    \\section{{Convergence}}
    \\begin{{itemize}}
     \\item Timestamp: {timestamp}
     \\item Simcardems version: {version}
     \\item Model: {coupling_type}
    \\end{{itemize}}

    \\subsection{{Spatial convergence}}
    {create_table_dx(feature_keys, features, fixed_dt, dxs)}
    \\includegraphics{{spatial_convergence.png}}



    \\subsection{{Temporal convergence}}
    {create_table_dt(feature_keys, features, fixed_dx, dts)}
    \\includegraphics{{temporal_convergence.png}}

    \\end{{document}}
    """,
    )

    (Path(outdir) / "report.tex").write_text(text)

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("Unable to generate pdf report since pdflatex is not install")

        msg = """
        export DEBIAN_FRONTEND=noninteractive
        apt-get update &&
        apt-get install texlive-latex-base
            texlive-fonts-recommended
            texlive-fonts-extra
            texlive-latex-extra
        """
        print(f"Try {msg}")
    else:
        subprocess.run([pdflatex, "report.tex"], cwd=outdir)


@click.group()
def main():
    pass


main.add_command(run)
main.add_command(pure_ep_report)
main.add_command(generate_report)

if __name__ == "__main__":
    raise SystemExit(main())
