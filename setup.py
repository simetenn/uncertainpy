# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Setuptools is needed to install all dependencies")
    print("Setuptools: https://pypi.python.org/pypi/setuptools")

import platform
import os

if not platform.system() == "Linux":
    print("Warning: Uncertainpy not tested for current operating system")

name = "testme3"

uncertainpy_require = ["chaospy", "tqdm", "h5py", "multiprocess", "numpy",
                       "scipy", "seaborn", "matplotlib"]

all_requires = ["xvfbwrapper", "chaospy", "tqdm", "h5py",
                "multiprocess", "numpy", "scipy", "seaborn",
                "efel", "elephant",  "neo", "quantities", "matplotlib"]

tests_require = all_requires + ["click"]

extras_require = {"spike_features":  ["efel"],
                  "network_features": ["elephant", "neo", "quantities"],
                  "neuron": ["xvfbwrapper"],
                  "all": all_requires}


long_description = open("README.md").read()

# Remove badges from the description
long_description = "\n".join(long_description.split("\n")[4:])

execfile('src/uncertainpy/version.py')

#packages = ["uncertainpy", "uncertainpy.models", "uncertainpy.features", "uncertainpy.plotting", "uncertainpy.utils"]

setup(name=name,
      version=__version__,
    #   url="https://github.com/simetenn/uncertainpy",
    #   author="Simen Tennøe",
    #   description="Uncertainty quantification and sensitivity analysis",
    #   long_description=long_description,
      python_requires=">=2.7.*",
      packages=find_packages("src"),
      package_dir={"": "src"},
    #   data_files=("version", ["README.md", "VERSION"]),
      install_requires=all_requires,
    #   extras_require=extras_require,
      )
