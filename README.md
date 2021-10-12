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
python3 -m simcardems --help
```
to see all options.
For example if you want to run a simulations with `T=1000`, then use
```
python3 -m simcardems -T=1000
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
and then run the simulation using the `--from_json` flag
```
python3 -m simcardems --from_json=args.json
```

### Run using docker

If you are using docker and you used the instructions for creating a docker image described above, you can run the container as follows

```
docker run --name simcardems -v "$(pwd)":/app -it simcardems
```
And you can also provide arguments to this script in a similar fashion as described above, e.g
```
docker run --name simcardems -v "$(pwd)":/app -it simcardems --help
```
and
```
docker run --name simcardems -v "$(pwd)":/app -it simcardems -T=1000
```
Note that this will create a docker container called `simcardems`

To delete the container you can either pass the flag `--rm` with the docker run command or execute the command

```
docker rm simcardems
```
Note that `simcardems` uses FEniCS which is partly written in C++,  which will use quite a lot of time the first time it runs to compile all the forms, these forms will be saved in a cache which will make the runtime much faster the second time you run it. Therefore it is a good idea to not delete the container every time you run it, but rather reuse it, so that you get the benefit of the cache.

To execute an existing container you first need to make sure it is running
```
docker start simcardems
```
Next you can execute the container using the following command
```
docker exec simcardems python3 -m simcardems
```
And as before you can also provide command line arguments, e.g
```
docker exec simcardems python3 -m simcardems --help
```


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
