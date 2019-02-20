FROM continuumio/anaconda3:latest

# Update the image since it sometimes are outdated
RUN conda update conda

# Install Nest and Neuron
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y automake libtool build-essential openmpi-bin libopenmpi-dev \
                                       libncurses5-dev libreadline-dev libgsl0-dev cmake > /dev/null


ENV HOME=/home/docker
ENV VENV=$HOME/simulators
RUN mkdir $HOME; mkdir $HOME/packages; mkdir $VENV
ENV PATH=$PATH:$VENV/bin

ENV NEST_VER=2.16.0 NRN_VER=7.6
ENV NEST=nest-$NEST_VER NRN=nrn-$NRN_VER

WORKDIR $HOME/packages
ADD http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$NRN.tar.gz .
ADD https://github.com/nest/nest-simulator/archive/v$NEST_VER.tar.gz .
RUN tar xzf v$NEST_VER.tar.gz; tar xzf $NRN.tar.gz; rm v$NEST_VER.tar.gz $NRN.tar.gz


RUN mkdir $VENV/build; mkdir $VENV/build/$NRN; mkdir $VENV/build/$NEST; mkdir $VENV/bin

WORKDIR $VENV/build/$NRN
RUN $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=python --disable-rx3d --without-iv --prefix=$VENV > /dev/null
RUN make > /dev/null
RUN make install > /dev/null
RUN cd src/nrnpython; python setup.py install > /dev/null
RUN cd $VENV/bin; ln -s ../x86_64/bin/nrnivmodl; ln -s ../x86_64/bin/nrngui; ln -s ../x86_64/bin/nrnoc; ln -s ../x86_64/bin/nrniv


WORKDIR $VENV/build/$NEST
RUN cmake -DCMAKE_INSTALL_PREFIX=$VENV -Dwith-mpi=ON -Dwith-python=3 $HOME/packages/nest-simulator-$NEST_VER > /dev/null
RUN make
RUN make install > /dev/null



WORKDIR /home/docker/
RUN echo "source $VENV/bin/nest_vars.sh" >> .bashrc
# Add nest to python path
ENV PYTHONPATH $PYTHONPATH:/home/docker/simulators/lib/python3.7/site-packages/
RUN conda install libgcc


# Get X working

RUN touch /home/docker/.Xauthority
RUN apt-get update; apt-get install -y libx11-dev libxext-dev x11-apps
EXPOSE 22



# Install test coverage dependencies
RUN pip install coverage

# Install uncertainpy dependencies
RUN apt-get update --fix-missing
RUN apt-get -y install xvfb

RUN pip install xvfbwrapper

# Make sure matplotlib uses agg
RUN mkdir .config/
RUN mkdir .config/matplotlib
RUN echo "backend : Agg" >> .config/matplotlib/matplotlibrc

# Temporary fix to chaospy
RUN git clone https://github.com/jonathf/chaospy.git
RUN cd chaospy; python setup.py install

# get exdir
# RUN conda install exdir -c cinpla -c conda-forge


# Ensure newest version of exdir, only temporary
RUN git clone https://github.com/CINPLA/exdir.git
RUN cd exdir; python setup.py install