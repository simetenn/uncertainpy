#!/bin/bash
set -eo pipefail

source .bashrc
cd uncertainpy
python setup.py install --tests
coverage run test.py uncertainty_calculations
codecov --token=504e6f55-2e64-481b-88fe-8da7201c462e