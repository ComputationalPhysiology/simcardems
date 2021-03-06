# -*- coding: utf-8 -*-
# # Simple demo
#
# In this demos we show the most simple usage of the `simcardems` library using the python API
#
# Import the necessary libraries
#

import pprint
from pathlib import Path

import simcardems


# Lets grab the default parameter and print them
#

parameters = simcardems.default_parameters()
pprint.pprint(parameters)


# This will output
#
# ```
# {'T': 1000,
#  'bnd_cond': <BoundaryConditions.dirichlet: 'dirichlet'>,
#  'cell_init_file': '',
#  'disease_state': 'healthy',
#  'drug_factors_file': '',
#  'dt': 0.05,
#  'dx': 0.2,
#  'fix_right_plane': True,
#  'hpc': False,
#  'load_state': False,
#  'loglevel': 20,
#  'lx': 2.0,
#  'ly': 0.7,
#  'lz': 0.3,
#  'num_refinements': 1,
#  'outdir': 'results',
#  'popu_factors_file': '',
#  'pre_stretch': None,
#  'save_freq': 1,
#  'set_material': '',
#  'spring': None,
#  'traction': None}
# ```
#
# These are the default parameters used for the simulation.
#

outdir = Path("results_simple_demo")
parameters["outdir"] = outdir
runner = simcardems.Runner(**parameters)
runner.solve(T=parameters["T"], save_freq=parameters["save_freq"], hpc=False)


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

simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"))

#
# And save the output to xdmf-files that can be viewed in Paraview
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


# This will create a figure in the output directory called `state_traces.png` which in this case is shown in {numref}`Figure {number} <simple_demo_state_traces>` we see the resulting state traces, and can also see the instant drop in the active tension ($T_a$) at the time of the triggered release.
#
# ```{figure} figures/simple_demo_state_traces.png
# ---
# name: simple_demo_state_traces
# ---
# Traces of the stretch ($\lambda$), the active tension ($T_a$), the membrane potential ($V$) and the intercellular calcium concentration ($Ca$) at the center of the geometry.
# ```
