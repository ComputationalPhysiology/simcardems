[![CI](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml/badge.svg)](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/main.yml)
[![PyPI version](https://badge.fury.io/py/simcardems.svg)](https://badge.fury.io/py/simcardems)
[![codecov](https://codecov.io/gh/ComputationalPhysiology/simcardems/branch/master/graph/badge.svg?token=V5DOQ1PUVF)](https://codecov.io/gh/ComputationalPhysiology/simcardems)
[![github pages](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/github-pages.yml/badge.svg)](https://github.com/ComputationalPhysiology/simcardems/actions/workflows/github-pages.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComputationalPhysiology/simcardems/master.svg)](https://results.pre-commit.ci/latest/github/ComputationalPhysiology/simcardems/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)

# Simula Cardiac Electro-Mechanics Solver

`simcardems` is a FEniCS-based cardiac electro-mechanics solver and is developed as a part of the [SimCardio Test project](https://www.simcardiotest.eu/wordpress/). The solver depends on [`pulse`](https://github.com/ComputationalPhysiology/pulse) and [`cbcbeat`](https://github.com/ComputationalPhysiology/cbcbeat).


## Installation

See [Installation instructions](https://computationalphysiology.github.io/simcardems/install.html)

## Getting started

See [the demos](https://computationalphysiology.github.io/simcardems/demo.html)

## Documentation

Documentation is hosted at http://computationalphysiology.github.io/simcardems.

## Automated test

Tests are provided in the folder [tests](https://github.com/ComputationalPhysiology/simcardems/tree/master/tests). You can run the tests with pytest

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
