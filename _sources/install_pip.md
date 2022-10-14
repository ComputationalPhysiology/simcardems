# Install with `pip`

Before installing `simcardems` with `pip` you need to install the [legacy version of FEniCS](https://fenicsproject.org/download/archive).

## Installing FEniCS

FEniCS provides prebuilt high-performance [Docker images](https://quay.io/repository/fenicsproject/stable), usable on most OS:
```
docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:latest
```

### Linux

Ubuntu users can also use the dedicated Ubuntu package:
```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics`
```

or the prebuilt Anaconda package:
```
conda create -n fenicsproject -c conda-forge fenics
source activate fenicsproject
```

### Mac
On Mac with Intel CPUs, both Docker and Anaconda installation are possible.
On Mac  with Apple Silicon Chip, the [Docker installation](https://quay.io/repository/fenicsproject/stable) is recommended.

### Windows
We recommend Windows users to use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
and follow the (Linux users)[#linux] instructions, or use [Docker](https://quay.io/repository/fenicsproject/stable)

Alternatively, FEniCS can also be built from source (only recommended for installation on HPC clusters),
following the installation instructions from the (possibly not up-to-date) [FEniCS Reference Manual](https://fenics.readthedocs.io/en/latest/installation.html).

## Install `simcardems`

Once you have FEniCS installed, you can install `simcardems` with pip using the command
```
python3 -m pip install simcardems
```

If you want the latest version or you want to contribute to `simcardems`,
you can install the code directly from the GitHub repo

```
python3 -m pip install git+https://github.com/ComputationalPhysiology/simcardems.git
```
or clone the repository and install it from there

```
git clone git@github.com:ComputationalPhysiology/simcardems.git
cd simcardems
python3 -m pip install .
```

## Development installation

Developers should use editable install, adding the appropriate option `-e`
to the previous `pip install` commands:
```
python -m pip install -e ".[dev]"
```
You should also install the `pre-commit` hook that comes with the package
```
pre-commit install
```
which will run a set of tests on the code that you commit to the repo.
Note that linters and formatters will run in the CI system.
