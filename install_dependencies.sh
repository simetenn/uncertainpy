#!/bin/bash

# TODO Consider using anaconda instead of virtualenvwrapper + pip

set -x  # make sure each command is printed in the terminal


function apt_install {
  sudo apt-get -y install $@
  if [ $? -ne 0 ]; then
    echo "could not install $@ - abort"
    exit 1
  fi
}

function pip_install {
  pip install --upgrade "$@"
  if [ $? -ne 0 ]; then
    echo "could not install $p - abort"
    exit 1
  fi
}

sudo apt-get update --fix-missing


echo "Starting installation"

set -e  # make sure any failed command stops the script


echo "Installing system wide packages for chaospy, withouth sckit-learn"
apt_install gcc
apt_install python-scipy


echo "Installing system wide packages for odespy"
apt_install gfortran


echo "Installing system wide packages for uncertainpy"
apt_install python-dev
apt_install python-matplotlib
apt_install python-h5py


echo "Installing virtual envionment packages"
# pip_install --upgrade pip
# pip_install cython
# pip_install networkx
# pip_install pandas
pip_install numpy
echo "finished installing numpy"


# pip_install xvfbwrapper
# pip_install psutil
#pip_install -e git+https://github.com/hplgit/odespy.git#egg=odespy

# Testing tools
pip_install nose2
echo "finished installing nose2"
