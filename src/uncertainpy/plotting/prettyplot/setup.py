# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages

except ImportError:
    print("Setuptools is needed to install all dependencies")
    print("Setuptools: https://pypi.python.org/pypi/setuptools")


name = "prettyplot"
version = "0.9"

prettyplot_req = ["seaborn", "matplotlib", "numpy"]

setup(name=name,
      url="https://github.com/simetenn/prettyplot",
      author="Simen Tennøe",
      description='Pretty plotting',
      platforms='linux',
      packages=find_packages(),
      install_requires=prettyplot_req
      )
