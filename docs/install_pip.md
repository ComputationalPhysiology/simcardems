# Install with `pip`

Before installing `simcardems` with `pip` you need to [install the legacy version of FEniCS](https://fenicsproject.org/download/archive).

## Installing FEniCS

### Mac
If you are using Mac with Intel CPUs then you can install FEniCS with `conda` or use [Docker](install_docker.md). Alternatively you can try to build FEniCS from source.

If you are using Mac with Apple Silicon Chip then you can you should use [Docker](install_docker.md) or install from source.

### Linux
If you are running Linux then you can either install FEniCS using conda, Ubuntu package man


### Windows users
Windows users should use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or use [Docker](install_docker.md)

## Install `simcardems`

Once you have FEniCS installed, you can install `simcardems` with pip using the command
```
python3 -m pip install simcardems
```

If you want the latest version or you want to develop `simcardems` you can install the code directly from the GitHub repo

```
python3 -m pip install git+https://github.com/ComputationalPhysiology/simcardems.git
```
or clone the repository and install it from there

```
git clone git@github.com:ComputationalPhysiology/simcardems.git
cd simcardems
python3 -m pip install .
```
