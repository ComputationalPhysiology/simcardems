# Simple demo

In this demos we show the most simple usage of the `simcardems` library using the python API

Import the necessary libraries


```python
import pprint
from pathlib import Path
```

```python
import simcardems
```

```python
# Create configurations with custom output directory
outdir = Path("results_simple_demo")
config = simcardems.Config(outdir=outdir)
```


This will set :

```
{'PCL': 1000,
 'T': 1000,
 'bnd_cond': <BoundaryConditions.dirichlet: 'dirichlet'>,
 'cell_init_file': '',
 'disease_state': 'healthy',
 'drug_factors_file': '',
 'dt': 0.05,
 'dx': 0.2,
 'ep_ode_scheme': 'GRL1',
 'ep_preconditioner': 'sor',
 'ep_theta': 0.5,
 'fix_right_plane': False,
 'show_progress_bar': True,
 'linear_mechanics_solver': 'mumps',
 'load_state': False,
 'loglevel': 20,
 'lx': 2.0,
 'ly': 0.7,
 'lz': 0.3,
 'mechanics_ode_scheme': <Scheme.analytic: 'analytic'>,
 'mechanics_use_continuation': False,
 'mechanics_use_custom_newton_solver': False,
 'num_refinements': 1,
 'outdir': PosixPath('results_simple_demo'),
 'popu_factors_file': '',
 'pre_stretch': None,
 'save_freq': 1,
 'set_material': '',
 'spring': None,
 'traction': None}
```

```python
# Print current configuration
pprint.pprint(config.as_dict())
```

```python
runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
```


This will create the output directory `results_simple_demo` with the following output

```
results_simple_demo
├── results.h5
├── state.h5
```
The file `state.h5` contains the final state which can be used if you want use the final state as a starting point for the next simulation.
The file `results.h5` contains the Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$) for each time step.
We can also plot the traces using the postprocess module



```python
simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"))
```


And save the output to xdmf-files that can be viewed in Paraview


```python
simcardems.postprocess.make_xdmffiles(outdir.joinpath("results.h5"))
```


The `xdmf` files are can be opened in [Paraview](https://www.paraview.org/download/) to visualize the different variables such as in {numref}`Figure {number} <simple-demo-paraview>`.

```{figure} figures/simple_demo.png
---
name: simple-demo-paraview
---

Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$) visualized for a specific time point in Paraview.
```



This will create a figure in the output directory called `state_traces.png` which in this case is shown in {numref}`Figure {number} <simple_demo_state_traces>` we see the resulting state traces, and can also see the instant drop in the active tension ($T_a$) at the time of the triggered release.

```{figure} figures/simple_demo_state_traces.png
---
name: simple_demo_state_traces
---
Traces of the stretch ($\lambda$), the active tension ($T_a$), the membrane potential ($V$) and the intercellular calcium concentration ($Ca$) at the center of the geometry.
```
