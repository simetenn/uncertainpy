# -*- coding: utf-8 -*-

import platform

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Setuptools is needed to install all dependencies")
    print("Setuptools: https://pypi.python.org/pypi/setuptools")


if not platform.system() == "Linux":
    print("Warning: UncertainPy not tested for current operating system")

name = "uncertainpy"

uncertainpy_req = ["xvfbwrapper", "chaospy", "tqdm", "h5py",
                   "multiprocess", "numpy", "scipy", "seaborn"]


extras_require = {
    'spike_features':  ["efel"],
    'network_features': ["elephant", "neo", "quantities"],
}

anaconda_req = ["xvfbwrapper", "chaospy", "tqdm", "h5py",
                "multiprocess", "numpy", "scipy", "seaborn",
                "efel", "elephant",  "neo", "quantities"]

packages = ['uncertainpy', 'uncertainpy.models', 'uncertainpy.features', 'uncertainpy.plotting', 'uncertainpy.utils']
setup(name="uncertainpy",
      version="0.9",
      url="https://github.com/simetenn/uncertainpy",
      author="Simen Tennøe",
      description='Uncertainty quantification and sensitivity analysis',
      platforms='linux',
      packages=find_packages("src"),
      package_dir={"": "src", "uncertainpy.examples": "examples", "uncertainpy.tests": "tests",},
      install_requires=anaconda_req,
      extras_require=extras_require,
      )
