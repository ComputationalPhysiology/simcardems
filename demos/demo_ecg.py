from pathlib import Path
import simcardems

# First we specifcy the path to the output directory where we want to store the results
here = Path(__file__).absolute().parent
outdir = here / "results_demo_ecg"

geometry_path = here / "geometries/slab_torso.h5"
geometry_schema_path = here / "geometries/slab_torso.json"

config = simcardems.Config(
    outdir=outdir,
    geometry_path=geometry_path,
    geometry_schema_path=geometry_schema_path,
    coupling_type="pureEP_ORdmm_Land",
    T=10,
    compute_ecg=False,
    # ecg_electrodes={"e1":[0,0,0], "e2":[0,1,0]}
)


runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)

# simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"), "center")

# simcardems.postprocess.make_xdmffiles(outdir.joinpath("results.h5"))
