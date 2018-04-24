#!/bin/bash
set -eo pipefail

cd uncertainpy
coverage run test.py all
ls
coverage xml
bash <(curl -s https://codecov.io/bash)