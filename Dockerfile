FROM continuumio/anaconda

# Install neuron
RUN conda install -c mattions neuron=7.4

#Install nest
RUN conda install -c emka nest-simulator

RUN conda install -c undy odespy



# Uncertainpy dependencies
RUN apt-get -y install texlive-latex-base
RUN apt-get -y install texlive-latex-extra
RUN apt-get -y install texlive-fonts-recommended
RUN apt-get -y install dvipng
RUN apt-get -y install Xvfb
RUN apt-get -y install h5utils
RUN apt-get -y install libx11-dev libxext-dev x11-apps

RUN conda install -c conda-forge xvfbwrapper

# Downgrade the pyqt package to solve a bug in anaconda
RUN conda install pyqt=4.11
RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot


COPY . uncertainpy
WORKDIR uncertainpy
RUN cp -r tests/figures_docker/. tests/figures

RUN python setup.py install --no-dependencies
