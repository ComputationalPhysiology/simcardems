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
  -o, --outdir PATH              Output directory
  --dt FLOAT                     Time step
  -T, --end-time FLOAT           End-time of simulation
  -n, --num_refinements INTEGER  Number of refinements of for the mesh using
                                 in the EP model
  --save_freq INTEGER            Set frequency of saving results to file
  --set_material TEXT            Choose material properties for mechanics
                                 model (default is HolzapfelOgden, option is
                                 Guccione
  -dx FLOAT                      Spatial discretization
  -lx FLOAT                      Size of mesh in x-direction
  -ly FLOAT                      Size of mesh in y-direction
  -lz FLOAT                      Size of mesh in z-direction
  --bnd_cond [dirichlet|rigid]   Boundary conditions for the mechanics problem
  --load_state                   If load existing state if exists, otherwise
                                 create a new state
  -IC, --cell_init_file TEXT     Path to file containing initial conditions
                                 (json or h5 file). If none is provided then
                                 the default initial conditions will be used
  --loglevel INTEGER             How much printing. DEBUG: 10, INFO:20
                                 (default), WARNING: 30
  --hpc                          Indicate if simulations runs on hpc. This
                                 turns off the progress bar.
  --drug_factors_file TEXT       Set drugs scaling factors (json file)
  --popu_factors_file TEXT       Set population scaling factors (json file)
  --help                         Show this message and exit.
```
to see all options.
For example if you want to run a simulations with `T=1000`, then use
```
python3 -m simcardems run -T=1000
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
