FROM continuumio/anaconda

# Downgrade the pyqt package to solve a bug in anaconda
RUN conda install pyqt=4.11

RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot

RUN apt-get install -y gfortran
#RUN pip install -e git+https://github.com/hplgit/odespy.git#egg=odespy

# Uncertainpy dependencies
RUN conda install -c conda-forge xvfbwrapper



COPY . uncertainpy
WORKDIR uncertainpy
RUN cp -r tests/figures_docker/. tests/figures

RUN python setup.py install --no-dependencies

# Install neuron
RUN conda install -c mattions neuron=7.4

#Install nest
RUN conda install -c emka nest-simulator
