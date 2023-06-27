"""Same as simple demo, but only 20 ms"""
from pathlib import Path

import simcardems


here = Path(__file__).absolute().parent
outdir = here / "smoke"


geometry_path = here / "../demos/geometries/slab.h5"
geometry_schema_path = here / "../demos/geometries/slab.json"

config = simcardems.Config(
    outdir=outdir,
    geometry_path=geometry_path,
    geometry_schema_path=geometry_schema_path,
    loglevel=10,
    coupling_type="fully_coupled_ORdmm_Land",
    T=20,
)

runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)

simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"), "center")


simcardems.postprocess.make_xdmffiles(outdir.joinpath("results.h5"))
