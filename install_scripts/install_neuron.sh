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

# Neuron dependencies
apt_install libncurses5-dev

mkdir /usr/local/neuron/

cd /usr/local/neuron/
wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/iv-19.tar.gz

tar xzf iv-19.tar.gz
tar xzf nrn-7.4.tar.gz
rm iv-19.tar.gz
rm nrn-7.4.tar.gz
# renaming the new directories iv and nrn makes life simpler later on
mv iv-19 iv
mv nrn-7.4 nrn

cd iv
./configure --prefix='/usr/local/neuron/iv'
make
make install

cd ../nrn
./configure --prefix="/usr/local/neuron/nrn" --with-iv="/usr/local/neuron/iv" --with-nrnpython --with-numpy

make
make install

cd /usr/local/neuron/nrn/src/nrnpython
python setup.py install
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/site-packages/
