#!/bin/bash
source .bashrc
# cd uncertainpy/tests/models/dLGN_modelDB
# nrnivmodl
cd uncertainpy
python setup.py test
python test.py --travis