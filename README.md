[![CI](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml/badge.svg)](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/simcardems.svg)](https://badge.fury.io/py/simcardems)

# Simula Cardiac Electro-Mechanics Solver

`simcardems` is a FEniCS-based cardiac electro-mechanics solver and is developed as a part of the [SimCardio Test project](https://www.simcardiotest.eu/wordpress/). The solver depends on [`pulse`](https://github.com/ComputationalPhysiology/pulse) and [`cbcbeat`](https://github.com/ComputationalPhysiology/cbcbeat).


## Installation

See [Installation instructions](https://computationalphysiology.github.io/simcardems/install.html)

## Getting started

See [the demos](https://computationalphysiology.github.io/simcardems/demo.html)

## Documentation

Documentation is hosted at http://computationalphysiology.github.io/simcardems.

To build the documentation locally should should first install the documentation requirements
```
python -m pip install ".[docs]"
```
Then should should run the sphinx-apidoc command (from the root directory)
```
sphinx-apidoc -o docs/source src/simcardems
```
and finally build the html
```
cd docs
make html
```
Now open `docs/build/html/index.html` to see the documentation.


## Automated test

Tests are provided in the folder [tests](tests). You can run the tests with pytest

```
python3 -m pytest tests -vv
```

## Contributing
See [the contributing section](https://computationalphysiology.github.io/simcardems/CONTRIBUTING.html)


## Known issues

- Issue with h5py, see https://github.com/ComputationalPhysiology/pulse#known-issues


## Authors
- Henrik Finsberg (henriknf@simula.no)
- Ilsbeth van Herck (ilse@simula.no)
- Cécile Daversin-Catty (cecile@simula.no)
