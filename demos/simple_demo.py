# -*- coding: utf-8 -*-
# # Simple demo
#
# In this demo we show the most simple usage of the `simcardems` library using the python API
#
# Import the necessary libraries
#

import pprint
from pathlib import Path

import simcardems

# Create configurations with custom output directory
here = Path(__file__).absolute().parent
outdir = here / "results_simple_demo"

# Specify paths to the geometry that we will use

geometry_path = here / "geometries/slab.h5"
geometry_schema_path = here / "geometries/slab.json"

# Please see https://computationalphysiology.github.io/cardiac_geometries/ for more info about the geometries


config = simcardems.Config(
    outdir=outdir,
    geometry_path=geometry_path,
    geometry_schema_path=geometry_schema_path,
    T=1000,
)


# This will set :
#
# ```
# {'PCL': 1000,
#  'T': 15,
#  'bnd_cond': <SlabBoundaryConditionTypes.dirichlet: 'dirichlet'>,
#  'cell_init_file': '',
#  'disease_state': 'healthy',
#  'drug_factors_file': '',
#  'dt': 0.05,
#  'ep_ode_scheme': 'GRL1',
#  'ep_preconditioner': 'sor',
#  'ep_theta': 0.5,
#  'fix_right_plane': False,
#  'geometry_path': 'demos/geometries/slab.h5',
#  'geometry_schema_path': 'demos/geometries/slab.json',
#  'linear_mechanics_solver': 'mumps',
#  'load_state': False,
#  'loglevel': 20,
#  'mechanics_ode_scheme': <Scheme.analytic: 'analytic'>,
#  'mechanics_use_continuation': False,
#  'mechanics_use_custom_newton_solver': False,
#  'num_refinements': 1,
#  'outdir': PosixPath('results_simple_demo'),
#  'popu_factors_file': '',
#  'pre_stretch': None,
#  'save_freq': 1,
#  'set_material': '',
#  'show_progress_bar': True,
#  'spring': None,
#  'traction': None}
# ```


# Print current configuration
pprint.pprint(config.as_dict())

runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)


# This will create the output directory `results_simple_demo` with the following output
#
# ```
# results_simple_demo
# ├── results.h5
# ├── state.h5
# ```
# The file `state.h5` contains the final state which can be used if you want use the final state as a starting point for the next simulation.
# The file `results.h5` contains the Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$) for each time step.
# We can also plot the traces using the postprocess module
#
#

simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"), "center")

#
# Here we also specify that we want the trace from the center of the slab
#

# This will create a figure in the output directory called `state_traces.png` which in this case is shown in {numref}`Figure {number} <simple_demo_state_traces>` we see the resulting state traces, and can also see the instant drop in the active tension ($T_a$) at the time of the triggered release.
#
# ```{figure} figures/simple_demo_state_traces.png
# ---
# name: simple_demo_state_traces
# ---
# Traces of the stretch ($\lambda$), the active tension ($T_a$), the membrane potential ($V$) and the intercellular calcium concentration ($Ca$) at the center of the geometry.
# ```

#
# We can also save the output to xdmf-files that can be viewed in Paraview
#

simcardems.postprocess.make_xdmffiles(outdir.joinpath("results.h5"))

#
# The `xdmf` files are can be opened in [Paraview](https://www.paraview.org/download/) to visualize the different variables such as in {numref}`Figure {number} <simple-demo-paraview>`.
#
# ```{figure} figures/simple_demo.png
# ---
# name: simple-demo-paraview
# ---
#
# Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$) visualized for a specific time point in Paraview.
# ```
