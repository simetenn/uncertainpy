# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import platform

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Setuptools is needed to install all dependencies")
    print("Setuptools: https://pypi.python.org/pypi/setuptools")


if not platform.system() == "Linux":
    print("Warning: OS not supported, installation may fail")

name = "uncertainpy"
virtual_enviroment = name

uncertainpy_req = []
dependency_links = []
uncertainpy_req = ["xvfbwrapper", "psutil", "odespy", "chaospy", "pandas", "tqdm"]
dependency_links = ["http://github.com/hplgit/odespy/tarball/master#egg=odespy"]



def activate_virtualev(virtual_enviroment=virtual_enviroment, system_site_packages=False):
    cmd = "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh && mkvirtualenv " + virtual_enviroment

    if system_site_packages:
        cmd = cmd + "--system-site-packages"

    subprocess.call(cmd, executable='bash', shell=True)
    virtual_path = os.environ["VIRTUALENVWRAPPER_HOOK_DIR"] + "/" + virtual_enviroment
    activate_this_file = virtual_path + "/bin/activate_this.py"

    execfile(activate_this_file, dict(__file__=activate_this_file))



# def setupInstall():
#     if not platform.system() == "Linux":
#         print("Warning: OS not supported, installation may fail")
#
#     subprocess.call("./install_scripts/install_dependencies.sh", shell=True)


# class CustomDevelop(_develop):
#     def run(self):
#         setupInstall()
#         _develop.run(self)
#
#
# class CustomInstall(_install):
#     def run(self):
#         setupInstall()
#         _install.run(self)
#
#
# cmdclass = {'install': CustomInstall,
#             'develop': CustomDevelop}

cmdclass = {}

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
    sys.argv.remove("--no-dependencies")
elif "develop" in sys.argv or "install" in sys.argv:
    subprocess.call("./install_scripts/install_dependencies.sh", shell=True)


packages = ['uncertainpy', 'uncertainpy.models', 'uncertainpy.features', 'uncertainpy.plotting', 'uncertainpy.utils']
setup(name="uncertainpy",
      version="0.9",
      url="https://github.com/simetenn/uncertainpy",
      author="Simen Tennøe",
      description='Uncertainty quantification',
      platforms='linux',
      packages=find_packages("src"),
      package_dir={"": "src", "uncertainpy.examples": "examples"},
      cmdclass=cmdclass,
      install_requires=uncertainpy_req,
      dependency_links=dependency_links
      )
