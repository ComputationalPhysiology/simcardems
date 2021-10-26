#!/usr/bin/env python
"""The setup script."""
from pathlib import Path

from setuptools import setup

_here = Path(__file__).parent

with open(
    _here.joinpath("src").joinpath("simcardems").joinpath("version.py"),
    "r",
) as f:
    version = f.read().split('"')[1]


setup(version=version)
