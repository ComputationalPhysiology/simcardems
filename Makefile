.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

export FENICS_PLOTLY_RENDERER=notebook

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-notebooks ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-docs:
	rm -rf docs/build

clean-notebooks: ## remove notebook checkpoints
	find . -name '*.ipynb_checkpoints' -exec rm -fr {} +

pre-commit-run: ## Run all pre-commit hooks
	pre-commit run --all

lint: ## check style with flake8
	python3 -m flake8 simcardems tests

type: ## Run mypy
	python3 -m mypy simcardems tests

test: ## run tests quickly with the default Python
	python3 -m pytest

docs: ## generate Sphinx HTML documentation, including API docs
	cp CONTRIBUTING.md docs/.
	jupytext demos/release_test.py -o docs/release_test.md
	jupytext demos/simple_demo.py -o docs/simple_demo.md
	mkdir -p docs/_build
	cp -r benchmarks docs/_build/
	jupyter book build -W docs

run-benchmark:
	python -m simcardems run-benchmark "benchmarks/$(shell git rev-parse --short HEAD)"

show-docs:
	open docs/build/html/index.html

release: dist ## package and upload a release
	python3 -m twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} dist/*

release-test: dist ## package and upload a release
	python3 -m twine upload --repository testpypi -u ${TEST_PYPI_USERNAME} -p ${TEST_PYPI_PASSWORD} dist/*

dist: clean ## builds source and wheel package
	python -m build

install: clean ## install the package to the active Python's site-packages
	python3 -m pip install --upgrade pip
	python3 -m pip install h5py --no-binary=h5py
	python3 -m pip install -r requirements.txt
	python3 -m pip install .

dev: clean ## Just need to make sure that libfiles remains
	python3 -m pip install -e ".[test,plot,docs,dev]"
	pre-commit install

bump-patch:  ## Bump patch version
	bump2version patch

bump-minor: ## Bump minor version
	bump2version minor

bump-major: ## Bump major version
	bump2version major
