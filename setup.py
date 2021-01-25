# -*- coding: utf-8 -*-
try:
    from setuptools import setup, find_packages
except ImportError:
    raise ImportError(
        "Setuptools is needed to install all dependencies: https://pypi.python.org/pypi/setuptools"
    )


import platform
import os
import sys

name = "uncertainpy"

description = "A python toolbox for uncertainty quantification and sensitivity analysis tailored towards neuroscience models."
long_description = """Uncertainpy is a python toolbox for uncertainty quantification and sensitivity
analysis tailored towards computational neuroscience.

Uncertainpy is model independent and treats the model as a black box where the
model can be left unchanged. Uncertainpy implements both quasi-Monte Carlo
methods and polynomial chaos expansions using either point collocation or the
pseudo-spectral method. Both of the polynomial chaos expansion methods have
support for the rosenblatt transformation to handle dependent input parameters.

Uncertainpy is feature based, i.e., if applicable, it recognizes and calculates
the uncertainty in features of the model, as well as the model itself.
Examples of features in neuroscience can be spike timing and the action
potential shape.

Uncertainpy is tailored towards neuroscience models, and comes with several
common neuroscience models and features built in, but new models and features can
easily be implemented. It should be noted that while Uncertainpy is tailored
towards neuroscience, the implemented methods are general, and Uncertainpy can
be used for many other types of models and features within other fields.
"""


uncertainpy_require = [
    "chaospy>=4.0.0",
    "tqdm",
    "h5py",
    "multiprocess",
    "numpy>=1.16",
    "scipy>=1.4.1",
    "seaborn",
    "matplotlib>=3,<3.2",
    "xvfbwrapper",
    "six",
    "exdir",
    "ruamel.yaml",
    "salib",
]


efel_features = ["efel"]
network_features = ["elephant", "neo", "quantities"]

all_uncertainpy_requires = uncertainpy_require + efel_features + network_features

test_dependencies = ["click"]
tests_require = all_uncertainpy_requires + test_dependencies

docs_dependencies = ["sphinx", "sphinx_rtd_theme"]
docs_require = all_uncertainpy_requires + docs_dependencies

all_requires = docs_require + test_dependencies + docs_dependencies

extras_require = {
    "efel_features": efel_features,
    "network_features": network_features,
    "all": all_uncertainpy_requires,
    "docs": docs_require,
    "all_extras": all_requires,
    "tests": tests_require,
}

# To install on read the docs
if os.environ.get("READTHEDOCS") == "True":
    # uncertainpy_require = ["mock"]
    uncertainpy_require = []


help_text = """
Custom options:
  --all_extras        Install with all dependencies, along with extra dependencies.
  --all               Install with all dependencies required by Uncertainpy
  --tests             Install with dependencies required to run tests
  --docs              Install with dependencies required to build the docs
  --network_features  Install with dependencies required by NetworkFeatures
  --efel_features     Install with dependencies required by EfelFeatures
    """

if "--help" in sys.argv or "-h" in sys.argv:
    print(help_text)


if "--all_extras" in sys.argv:
    uncertainpy_require = all_requires
    sys.argv.remove("--all_extras")


if "--all" in sys.argv:
    uncertainpy_require = all_uncertainpy_requires
    sys.argv.remove("--all")

if "--tests" in sys.argv:
    uncertainpy_require = tests_require
    sys.argv.remove("--tests")


if "--network_features" in sys.argv:
    uncertainpy_require = uncertainpy_require + network_features
    sys.argv.remove("--network_features")


if "--efel_features" in sys.argv:
    uncertainpy_require = uncertainpy_require + efel_features
    sys.argv.remove("--efel_features")


# Get version
exec(open(os.path.join("src", "uncertainpy", "_version.py")).read())

setup(
    name=name,
    version=__version__,
    url="https://github.com/simetenn/uncertainpy",
    author="Simen Tennøe",
    description=description,
    license="GNU GPLv3",
    keywords="uncertainty quantification sensitivity analysis neuroscience",
    long_description=long_description,
    python_requires=">=3",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=uncertainpy_require,
    extras_require=extras_require,
)
