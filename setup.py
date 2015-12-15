# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    print "Setuptools is needed to install all dependencies"
    print "Setuptools: https://pypi.python.org/pypi/setuptools"

name = "uncertainpy"

setup(name=name,
      version="0.1",
      url="https://github.com/simetenn/parameter_estimation",
      author="Simen Tennøe",
      description='Parameter estimation and uncertainty quantification',
      platforms='linux',
      packages=find_packages(),  # ["uncertainpy", "uncertainpy/models"],
      setup_requires=["Cython"],
      install_requires=["chaospy", "xvfbwrapper", "h5py"],
      dependency_links=["https://github.com/cgoldberg/xvfbwrapper",
                        "https://github.com/hplgit/chaospy"]
      )
