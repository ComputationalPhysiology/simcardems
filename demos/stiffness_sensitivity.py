# # Simple demo
#
# In this demo we show the most simple usage of the `simcardems` library using the python API.
#
# Import the necessary libraries
#

from pathlib import Path
import simcardems
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).absolute().parent
geometry_path = here / "geometries/slab.h5"
geometry_schema_path = here / "geometries/slab.json"


def run(outdir: Path, a: float = 2.28):
    outdir.mkdir(exist_ok=True, parents=True)
    results_file = outdir.joinpath("results.h5")
    if results_file.exists():
        values = np.load(outdir.joinpath("values.npy"), allow_pickle=True).item()

        t = values["time"]
        lmbda = values["mechanics"]["lambda"]
        return t, lmbda
    config = simcardems.Config(
        outdir=outdir,
        geometry_path=geometry_path,
        geometry_schema_path=geometry_schema_path,
        coupling_type="fully_coupled_ORdmm_Land",
        T=1000,
        material_parameter_a=a,
    )
    runner = simcardems.Runner(config)
    runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)

    loader = simcardems.datacollector.DataLoader(results_file)
    values = simcardems.postprocess.extract_traces(loader, reduction="center")
    np.save(outdir.joinpath("values.npy"), values)

    t = values["time"]
    lmbda = values["mechanics"]["lambda"]
    fig, ax = plt.subplots()
    ax.plot(t, lmbda)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Lambda")
    ax.set_title(f"Material parameter a = {a}")
    fig.savefig(outdir.joinpath(f"lambda_{a}.png"))

    return t, lmbda
    # simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"), "center")


def main():
    outdir = here / "results_stiffness_sensitivity"
    fig, ax = plt.subplots()
    a_default = 2.28
    for a in [
        a_default / 4.0,
        a_default / 2.0,
        a_default,
        a_default * 2.0,
        a_default * 4.0,
    ]:
        t, lmbda = run(outdir=outdir / f"a_{a}", a=a)
        ax.plot(t, lmbda, label=f"a = {a}")
    ax.legend()
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Lambda")
    ax.set_title("Stiffness sensitivity")
    fig.savefig(outdir / "stiffness_sensitivity.png")


if __name__ == "__main__":
    main()
