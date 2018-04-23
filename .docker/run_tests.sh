#!/bin/bash
set -eo pipefail

#source .bashrc
cd uncertainpy
python setup.py develop --tests
coverage run test.py all
coverage xml