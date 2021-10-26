[![CI](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml/badge.svg)](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml)

# Simula Cardiac Electro-Mechanics Solver

`simcardems` is a FEniCS-based cardiac electro-mechanics solver and is developed as a part of the [SimCardio Test project](https://www.simcardiotest.eu/wordpress/). The solver depdens on [`pulse`](https://github.com/ComputationalPhysiology/pulse) and [`cbcbeat`](https://github.com/ComputationalPhysiology/cbcbeat).


## Installation

Before installing you need to install [FEniCS version 2019.1.0](https://fenicsproject.org/download/). Next you can install `simcardems` with pip

```
python -m pip install git+https://github.com/ComputationalPhysiology/simcardems.git@master
```
or clone the repository and install it from there

```
git clone git@github.com:ComputationalPhysiology/simcardems.git
cd simcardems
python -m pip install .
```

### Docker

We also provide a Dockerfile that contain all the instructions for installing the software using docker. To use this, you need to first create docker image. You can do this by executing the following command in the root folder of the project

```
docker build -t simcardems .
```
This will create a docker image with the name `simcardems`.


### Development installation

Developers should use editable install and install the development requirements using the following command
```
python -m pip install -e ".[dev]"
```
It is also recommended to install the `pre-commit` hook that comes with the package
```
pre-commit install
```
Note that linters and formatters will run in the CI system.


## Getting started
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
$ python -m simcardems run --help
Usage: python -m simcardems run [OPTIONS]

Options:
  -o, --outdir PATH             Output directory
  --dt FLOAT                    Time step
  -T, --end-time FLOAT          Endtime of simulation
  -dx FLOAT                     Spatial discretization
  -lx FLOAT                     Size of mesh in x-direction
  -ly FLOAT                     Size of mesh in y-direction
  -lz FLOAT                     Size of mesh in z-direction
  --bnd_cond [dirichlet|rigid]  Boundary conditions for the
                                mechanics problem
  --load_state                  If load existing state if exists,
                                otherwise create a new state
  -IC, --cell_init_file TEXT    Path to file containing initial
                                conditions (json or h5 file). If
                                none is provided then the default
                                initial conditions will be used
  --hpc                         Indicate if simulations runs on
                                hpc. This turns off the progress
                                bar.
  --help                        Show this message and exit.
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
python3 -m simcardems run-json rgs.json
```

### Run using docker

If you are using docker and first need to follow the instructions for creating a docker image described above.

#### Creating the container
You can create the container as follows

```
docker run --name simcardems -v "$(pwd)":/app -dit simcardems
```
This will create a a new container (aka a virtual machine) that you can use to execute the scripts.
Note that after executing the `docker run` command, the container will be created and it will run in the background (daemon-mode).

#### Execute command line scripts

You can now execute the command line script using the command
```
docker exec -it simcardems python3 -m simcardems
```
For example
```
 docker exec -it simcardems python3 -m simcardems run --help
```
or
```
 docker exec -it simcardems python3 -m simcardems run -T 1000
```

#### Stopping the container

When you are done using the script, you should stop the container so that it doesn't take up resources on your computer. You can do this using the command
```
docker stop simcardems
```

#### Starting the container again

To start the container again you can execute the command
```
docker start simcardems
```
You can now do ahead the [execute the command line scripts](#execute-command-line-scripts) again.

#### Deleting the container

If you don't want to use the container anymore, of your need to rebuild the image because there has been updates to the `simcardems` package, you can delete the container using the command
```
docker rm simcardems
```
Note that in order to use the container again, you need to first [create the container](#creating-the-container).



## Automated test

Tests are provided in the folder [tests](tests). You can run the tests with pytest

```
python3 -m pytest tests -vv
```


## Known issues

- Issue with h5py, see https://github.com/ComputationalPhysiology/pulse#known-issues


## Authors
- Henrik Finsberg (henriknf@simula.no)
- Ilsbeth van Herck (ilse@simula.no)
- Cécile Daversin-Catty (cecile@simula.no)
