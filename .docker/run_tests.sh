#!/bin/bash
set -eo pipefail

source .bashrc
cd uncertainpy
python setup.py develop --all
coverage run test.py spikes
coverage xml