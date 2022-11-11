# Installation

`simcardems` is a pure python package coupling the cardiac mechanics solver [pulse](https://github.com/finsberg/pulse)
and the cardiac electrophysiology solver [cbcbeat](https://github.com/ComputationalPhysiology/cbcbeat),
which are both python packages based on the legacy version of the open source finite element framework [FEniCS](https://fenicsproject.org/download/archive).

* We recommend using the dedicated [Docker](install_docker.md) container, which includes all the aforementioned dependencies.

* `simcardems` is also available on [PyPI](https://pypi.org/project/simcardems/) and can be installed with [pip](install_pip.md).
This option is recommended for developers who want to contribute to the software, but requires installation of the dependencies.
