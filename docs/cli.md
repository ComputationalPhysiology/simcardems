# Command line interface

Once installed, you can run a simulation using the command
```
python3 -m simcardems
```
Type
```
$ python3 -m simcardems --help
Usage: python -m simcardems [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  postprocess
  run
  run-json
```
to see all commands.
Run run a simulation using command line arguments you can use the `run` command. You can execute
```
$ python3 -m simcardems run --help
Usage: python -m simcardems run [OPTIONS]

Options:
  -o, --outdir PATH               Output directory
  --dt FLOAT                      Time step
  -T, --end-time FLOAT            End-time of simulation
  -n, --num_refinements INTEGER   Number of refinements of for the mesh using
                                  in the EP model
  --save_freq INTEGER             Set frequency of saving results to file
  --set_material TEXT             Choose material properties for mechanics
                                  model (default is HolzapfelOgden, option is
                                  Guccione
  -dx FLOAT                       Spatial discretization
  -lx FLOAT                       Size of mesh in x-direction
  -ly FLOAT                       Size of mesh in y-direction
  -lz FLOAT                       Size of mesh in z-direction
  --bnd_cond [dirichlet|rigid]    Boundary conditions for the mechanics
                                  problem
  --load_state                    If load existing state if exists, otherwise
                                  create a new state
  -IC, --cell_init_file TEXT      Path to file containing initial conditions
                                  (json or h5 file). If none is provided then
                                  the default initial conditions will be used
  --loglevel INTEGER              How much printing. DEBUG: 10, INFO:20
                                  (default), WARNING: 30
  --show_progress_bar / --hide_progress_bar
                                  Shows or hide the progress bar.
  --drug_factors_file TEXT        Set drugs scaling factors (json file)
  --popu_factors_file TEXT        Set population scaling factors (json file)
  --disease_state TEXT            Indicate disease state. Default is healthy.
  --mechanics-ode-scheme [fd|bd|analytic]
                                  Scheme used to solve the ODEs in the
                                  mechanics model
  --mechanics-use-continuation BOOLEAN
                                  Use continuation based mechanics solver
  --mechanics-use-custom-newton-solver BOOLEAN
                                  Use custom newton solver and solve ODEs at
                                  each Newton iteration
  --pcl FLOAT                     Pacing cycle length (ms)
  --spring FLOAT                  Set value of spring for Robin boundary
                                  condition
  --traction FLOAT                Set value of traction for Neumann boundary
                                  condition
  --help                          Show this message and exit.
```
to see all options.
For example if you want to run a simulations with `T 1000`, then use
```
python3 -m simcardems run -T 1000
```
You can also specify a json file containing all the settings, e.g a file called `args.json` with the following content

```json
{
    "T": 100,
    "outdir": "results",
    "bnd_cond": "rigid",
    "dt": 0.02,
    "dx": 0.2
}
```
and then run the simulation `run-json` command
```
python3 -m simcardems run-json args.json
```

`simcardems` relies on the seamlessly parallel MPI-based implementation offered by FEniCS, providing good scalability on high performance computing clusters.
You can run a simulation in parallel without changing anything in the code, using the `mpirun` command which takes the number of processors to be used (as option `-np`).
Although the progress bar indicating the progress of the simulation looks very nice on one processor, its display might become troublesome on HPC clusters.
We recommend using the option `--hide_progress_bar` when running simulations in parallel.
The command to run the previous simulation with `T 1000` on 2 processors then becomes :
```
mpirun -np 2 python3 -m simcardems run -T 1000 --hide_progress_bar
```
One shall also note that good scalability is only obtained on "big enough" problems (e.g. not on the small demo example)
