#!/bin/bash
set -eo pipefail

source .bashrc
cd uncertainpy
python setup.py install --tests
coverage run test.py spike
echo DJDKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKkk?
bash <(curl -s https://codecov.io/bash)
# codecov --token=504e6f55-2e64-481b-88fe-8da7201c462e