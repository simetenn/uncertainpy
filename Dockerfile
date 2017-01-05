FROM neuralensemble/simulationx

RUN apt-get -qq update --fix-missing

#RUN apt-get install -qq sudo
# RUN sudo apt-get -qq update

RUN apt-get install -y python-pip
RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install h5py

RUN apt-get -y install gfortran
RUN apt-get -y install xvfb
RUN apt-get -y install python-dev
RUN apt-get -y install python-tk
RUN apt-get -y install texlive-latex-base
RUN apt-get -y install texlive-latex-extra
RUN apt-get -y install texlive-fonts-recommended
RUN apt-get -y install dvipng
RUN apt-get -y install libffi6
RUN apt-get -y install libffi-dev
RUN apt-get -y install libcairo2-dev
RUN apt-get -y install h5utils

RUN pip install xvfbwrapper
RUN pip install chaospy
RUN pip install tqdm
RUN pip install pandas
RUN pip install pyyaml
RUN pip install psutil

RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot
RUN pip install -e git+https://github.com/hplgit/odespy.git#egg=odespy

COPY . uncertainpy
WORKDIR uncertainpy

RUN cp -r tests/figures_docker/. tests/figures

RUN python2.7 setup.py install --no-dependencies
