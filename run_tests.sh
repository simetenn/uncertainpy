#!/bin/bash
set -eo pipefail

cd uncertainpy
coverage run test.py all-no-simulators
bash <(curl -s https://codecov.io/bash)