#!/bin/bash
set -eo pipefail

source .bashrc
cd uncertainpy
python setup.py install --tests
coverage run test.py all
