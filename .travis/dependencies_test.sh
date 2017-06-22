#!/bin/bash
cd uncertainpy
pip install -r test.txt
python setup.py install
python test.py -h