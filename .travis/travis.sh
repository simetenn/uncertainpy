#!/bin/bash
source .bashrc
# cd uncertainpy/tests/models/dLGN_modelDB
# nrnivmodl
cd uncertainpy
python setup.py install
python test.py --travis