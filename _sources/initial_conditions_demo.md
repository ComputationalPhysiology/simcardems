# initial conditions demo

In this demo we show how to use different initial conditions


```python
import pprint
from pathlib import Path
```

```python
import simcardems
```

```python
# Create configurations with custom output directory
here = Path(__file__).absolute().parent
outdir = here / "results_IC_demo"
```

```python
# Specify paths to the geometry that we will use
geometry_path = here / "geometries/slab.h5"
geometry_schema_path = here / "geometries/slab.json"
```

Please see https://computationalphysiology.github.io/cardiac_geometries/ for more info about the geometries

```python
# Specify path to the initial conditions for the cell model
initial_conditions_path = here / "initial_conditions/init_5000beats.json"
```

```python
config = simcardems.Config(
    outdir=outdir,
    geometry_path=geometry_path,
    geometry_schema_path=geometry_schema_path,
    T=1000,
    cell_init_file=initial_conditions_path,
)
```


```python
# Print current configuration
pprint.pprint(config.as_dict())
```

```python
runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
```


This will create the output directory `results_IC_demo` with the following output

```
results_IC_demo
├── results.h5
├── state.h5
```
The file `state.h5` contains the final state which can be used if you want use the final state as a starting point for the next simulation.
The file `results.h5` contains the Displacement ($u$), active tension ($T_a$), voltage ($V$) and calcium ($Ca$) for each time step.
We can also plot the traces using the postprocess module

The final state of this model can be used as the starting point of a new simulation, but is limited to changes in time, pacing rate, disease model and drug effects.


```python
config = simcardems.Config(
    outdir=outdir,
    T=2000,
    load_state=True,
)
runner = simcardems.Runner(config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
```

Analysis and visualization of the results is performed in the same way as described above

```python
simcardems.postprocess.plot_state_traces(outdir.joinpath("results.h5"), "center")
```
