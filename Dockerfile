FROM continuumio/anaconda

# Install neuron
# RUN conda install -c mattions neuron=7.4

#Install nest
# RUN conda install -c emka nest-simulator

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
RUN conda install -c anaconda pandas
RUN conda install -c anaconda seaborn
RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot


# Neuron dependencies
RUN apt-get install -y build-essential
RUN apt-get install -y libcr-dev mpich2 mpich2-doc
RUN apt-get install -y libncurses-dev

ENV NRN_VER=7.4
ENV IV_VER=19
ENV NRN=nrn-$NRN_VER
ENV IV=iv-$IV_VER


RUN mkdir $HOME/packages
WORKDIR $HOME/packages
ADD http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$NRN.tar.gz .
# ADD http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$IV.tar.gz .
RUN tar xzf $NRN.tar.gz; rm $NRN.tar.gz
# RUN tar xzf $IV.tar.gz; rm $IV.tar.gz

RUN mkdir $HOME/build
# RUN mkdir $HOME/build/$IV
# WORKDIR $HOME/build/$IV
# RUN ../../packages/$IV/configure
# RUN make
# RUN make install


RUN mkdir $HOME/build/$NRN
WORKDIR $HOME/build/$NRN
RUN ../../packages/$NRN/configure --with-nrnpython --without-iv --disable-rx3d
RUN make
RUN make install
RUN cd src/nrnpython; python setup.py install

COPY . $HOME/uncertainpy
WORKDIR $HOME/uncertainpy
RUN cp -r tests/figures_docker/. tests/figures/.

RUN conda install conda-build
RUN conda build install_scripts/nest/.
RUN conda install --use-local nest-simulator

RUN python setup.py install --no-dependencies


# Run nivmodl
RUN chmod +x /build/$NRN/bin/nrnivmodl
RUN cd tests/models/dLGN_modelDB; ../../../../build/$NRN/bin/nrnivmodl

# RUN conda install -c conda-forge matplotlib=2.0.0
