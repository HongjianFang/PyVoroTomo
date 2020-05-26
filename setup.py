"""
setup.py adapted from https://github.com/kennethreitz/setup.py
"""
import io
import numpy as np
import os
from setuptools import setup


# Package meta-data.
name            = "pyvorotomo"
description     = "Parsimonious Voronoi-cell based tomograph (Fang et al., 2019)"
url             = "https://github.com/malcolmw/PyVoroTomo"
email           = "malcolm.white@.usc.edu"
author          = "Hongjian Fang and Malcolm C. A. White"
requires_python = ">=3.8"
packages        = ["pyvorotomo"]
required        = [
    "KDEpy>=1.0.3",
    "mpi4py",
    "numpy",
    "pandas",
    "pykonal>=0.2.3b",
    "tables",
    "scipy"
]
scripts         = ["bin/pyvorotomo"]
license         = "GNU GPLv3"

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = name.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setup(
    name=name,
    version=about["__version__"],
    description=description,
    author=author,
    author_email=email,
    python_requires=requires_python,
    url=url,
    packages=packages,
    scripts=scripts,
    install_requires=required,
    license=license,
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Physics"
    ]
)
