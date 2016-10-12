# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import platform

try:
    from setuptools import setup, find_packages
    from setuptools.command.install import install as _install
    from setuptools.command.develop import develop as _develop

except ImportError:
    print("Setuptools is needed to install all dependencies")
    print("Setuptools: https://pypi.python.org/pypi/setuptools")



name = "uncertainpy"
virtual_enviroment = name


chaospy_req = ["networkx", "cython"]
uncertainpy_req = ["xvfbwrapper", "psutil", "odespy", "chaospy", "pandas"]
dependency_links = ["http://github.com/hplgit/odespy/tarball/master#egg=odespy",
                    "http://github.com/hplgit/chaospy/tarball/master#egg=chaospy"]



def activate_virtualev(virtual_enviroment=virtual_enviroment, system_site_packages=False):
    cmd = "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh && mkvirtualenv " + virtual_enviroment

    if system_site_packages:
        cmd = cmd + "--system-site-packages"

    subprocess.call(cmd, executable='bash', shell=True)
    virtual_path = os.environ["VIRTUALENVWRAPPER_HOOK_DIR"] + "/" + virtual_enviroment
    activate_this_file = virtual_path + "/bin/activate_this.py"

    execfile(activate_this_file, dict(__file__=activate_this_file))



def setupInstall():
    if not platform.system() == "Linux":
        print("Warning: OS not supported, installation may fail")

    subprocess.call("./install_scripts/install_dependencies.sh", shell=True)


class CustomDevelop(_develop):
    def run(self):
        setupInstall()
        _develop.run(self)


class CustomInstall(_install):
    def run(self):
        setupInstall()
        _install.run(self)


cmdclass = {'install': CustomInstall,
            'develop': CustomDevelop}



if "-h" in sys.argv:
    print("""
Custom commandline arguments:
    --virtual: Install in a virtual enviroment
    --neuron: Install neuron
    --no-dependencies: Only install uncertainpy
    install: Install uncertainpy with dependencies
    develop: Install uncertainpy with dependencies as a developer
    """)

if "--virtual" in sys.argv:
    # subprocess.call("sudo ./uncertainpy/install_scripts/install_virtual.sh", shell=True)
    activate_virtualev()
    sys.argv.remove("--virtual")

if "--neuron" in sys.argv:
    subprocess.call("sudo ./install_scripts/install_neuron.sh", shell=True)
    sys.argv.remove("--neuron")


if "--no-dependencies" in sys.argv:
    cmdclass = {}
    uncertainpy_req = []
    chaospy_req = []
    sys.argv.remove("--no-dependencies")


setup(name=name,
      version="0.1",
      url="https://github.com/simetenn/uncertainpy",
      author="Simen Tennøe",
      description='Parameter estimation and uncertainty quantification',
      platforms='linux',
      packages=find_packages(),
      cmdclass=cmdclass,
      install_requires=uncertainpy_req + chaospy_req,
      dependency_links=dependency_links
      )
