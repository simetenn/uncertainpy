#!/bin/bash
set -eo pipefail

#source .bashrc
# cd uncertainpy
# python setup.py develop --tests
pwd

coverage run test.py all
coverage xml
# bash <(curl -s https://codecov.io/bash)
mv coverage.xml ../shared/coverage.xml