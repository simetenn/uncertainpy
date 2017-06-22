#!/bin/bash
source .bashrc
cd uncertainpy
python setup.py install
python test.py --travis