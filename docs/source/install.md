# Installation

Before installing you need to install [FEniCS version 2019.1.0](https://fenicsproject.org/download/). Next you can install `simcardems` with pip


```
python -m pip install simcardems
```

## Install with conda

TODO: We should make it possible to install `simcardems` with conda. Preferable, we should add it to conda-forge.

## Install from source

If you want the latest version or you want to develop `simcardems` you can install the code on the `master` branch

```
python -m pip install git+https://github.com/ComputationalPhysiology/simcardems.git@master
```
or clone the repository and install it from there

```
git clone git@github.com:ComputationalPhysiology/simcardems.git
cd simcardems
python -m pip install .
```

(section:docker-install)=
## Docker

`simcardems` is also available through [Docker](https://docs.docker.com/get-docker/). This is a good choice if you want to use `simcardems` in an isolated environment.

We provide both a pre-built docker image which you can get by pulling from docker hub
```
docker pull ghcr.io/computationalphysiology/simcardems:latest
```

### Building your own docker image

An alternative to pulling the image from docker hub, is to build it yourselves.
We provide a Dockerfile in the root of the repo that contain all the instructions for building the docker image. You can do this by executing the following command in the root folder of the project

```
docker build -t simcardems .
```
This will create a docker image with the name `simcardems`.


## Development installation

Developers should use editable install and install the development requirements using the following command
```
python -m pip install -e ".[dev]"
```
It is also recommended to install the `pre-commit` hook that comes with the package
```
pre-commit install
```
Note that linters and formatters will run in the CI system.
