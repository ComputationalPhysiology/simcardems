# Development installation

Developers should use editable install and install the development requirements using the following command
```
python -m pip install -e ".[dev]"
```
You should also install the `pre-commit` hook that comes with the package
```
pre-commit install
```
which will run a set of tests on the code that you commit to the repo.

Note that linters and formatters will run in the CI system.
