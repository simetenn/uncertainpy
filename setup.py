# -*- coding: utf-8 -*-

# TODO add option to install in virtual enviroment

import subprocess
import os
import sys
import platform
import argparse

try:
    from setuptools import setup, find_packages, Command
    from setuptools.command.install import install as _install
    from setuptools.command.develop import develop as _develop

except ImportError:
    print "Setuptools is needed to install all dependencies"
    print "Setuptools: https://pypi.python.org/pypi/setuptools"



name = "uncertainpy"
virtual_enviroment = name


def activate_virtualev(virtual_enviroment=virtual_enviroment):
    subprocess.call("source /usr/local/bin/virtualenvwrapper.sh && mkvirtualenv "
                    + virtual_enviroment + " --system-site-packages",
                    executable='bash', shell=True)
    virtual_path = os.environ["VIRTUALENVWRAPPER_HOOK_DIR"] + "/" + virtual_enviroment
    activate_this_file = virtual_path + "/bin/activate_this.py"
    execfile(activate_this_file, dict(__file__=activate_this_file))



def setupInstall():
    if not platform.system() == "Linux":
        print "Warning: OS not supported, the installation may fail"

    #
    # subprocess.call("./dependencies_apt-get.sh", shell=True)
    # subprocess.call("./dependencies_pip.sh", shell=True)
    subprocess.call("./install_dependencies.sh", shell=True)


class CustomDevelop(_develop):
    def run(self):
        setupInstall()
        _develop.run(self)


class CustomInstall(_install):
    def run(self):
        setupInstall()
        _install.run(self)



# parser = argparse.ArgumentParser(description="Install uncertainpy %s" % (name))
# parse.add_argument("-p", '--paper', action='store_true',
#                                  help='Compile for paper')
# parse.parse_known_args()


if "-h" in sys.argv:
    print """
Commandline arguments:
    --virtual: Install in a virtual enviroment
    install: Install uncertainpy
    develop: Install uncertainpy as a developer
    """

if "--virtual" in sys.argv:
    activate_virtualev()
    sys.argv.remove("--virtual")




cmdclass = {'install': CustomInstall,
            'develop': CustomDevelop}


setup(name=name,
      version="0.1",
      url="https://github.com/simetenn/parameter_estimation",
      author="Simen Tennøe",
      description='Parameter estimation and uncertainty quantification',
      platforms='linux',
      packages=find_packages(),
      cmdclass=cmdclass
      )
