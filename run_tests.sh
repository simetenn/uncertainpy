#!/bin/bash
set -eo pipefail

#source .bashrc
cd uncertainpy
coverage run test.py all
coverage xml
ls
# bash <(curl -s https://codecov.io/bash)
mv coverage.xml ../shared/coverage.xml