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

echo "Installing system wide packages"
apt_install gcc
apt_install python-dev
apt_install python-matplotlib
apt_install python-scipy
apt_install python-h5py
apt_install build-essential
apt_install gfortran


# sudo apt-get install -y virtualenvwrapper
# source /etc/bash_completion.d/virtualenvwrapper
# set +e
# mkvirtualenv uncertainpy --system-site-packages
# workon uncertainpy
# set -e


echo "Installing virtual envionment packages"
pip install --upgrade pip
pip_install cython
pip_install networkx
pip_install pandas
pip_install numpy
pip_install -e git+https://github.com/hplgit/chaospy.git#egg=chaospy

pip_install xvfbwrapper
pip_install python-dev
pip_install psutils
pip_install -e git+https://github.com/hplgit/odespy.git#egg=odespy

# Testing tools
pip_install nose2
