export DISPLAY=:99.0
sh -e /etc/init.d/xvfb start

python2.7 setup.py install
python2.7 test.py --fast
