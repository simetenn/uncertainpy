#!/bin/bash
source .bashrc
cd uncertainpy
python setup.py install
coverage run test.py all
codecov --token=504e6f55-2e64-481b-88fe-8da7201c462e