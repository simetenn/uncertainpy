FROM continuumio/anaconda3

# https://bugs.debian.org/830696 (apt uses gpgv by default in newer releases, rather than gpg)
RUN set -x \
	&& apt-get update \
	&& { \
		which gpg \
# prefer gnupg2, to match APT's Recommends
		|| apt-get install -y --no-install-recommends gnupg2 \
		|| apt-get install -y --no-install-recommends gnupg \
	; } \
# Ubuntu includes "gnupg" (not "gnupg2", but still 2.x), but not dirmngr, and gnupg 2.x requires dirmngr
# so, if we're not running gnupg 1.x, explicitly install dirmngr too
	&& { \
		gpg --version | grep -q '^gpg (GnuPG) 1\.' \
		|| apt-get install -y --no-install-recommends dirmngr \
	; } \
	&& rm -rf /var/lib/apt/lists/*

# apt-key is a bit finicky during "docker build" with gnupg 2.x, so install the repo key the same way debian-archive-keyring does (/etc/apt/trusted.gpg.d)
# this makes "apt-key list" output prettier too!
RUN set -x \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& gpg --keyserver ha.pool.sks-keyservers.net --recv-keys DD95CC430502E37EF840ACEEA5D32F012649A5A9 \
	&& gpg --export DD95CC430502E37EF840ACEEA5D32F012649A5A9 > /etc/apt/trusted.gpg.d/neurodebian.gpg \
	&& rm -rf "$GNUPGHOME" \
	&& apt-key list | grep neurodebian

RUN { \
	echo 'deb http://neuro.debian.net/debian artful main'; \
	echo 'deb http://neuro.debian.net/debian data main'; \
	echo '#deb-src http://neuro.debian.net/debian-devel artful main'; \
} > /etc/apt/sources.list.d/neurodebian.sources.list






# ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y automake libtool build-essential openmpi-bin libopenmpi-dev git vim  \
                       wget python3 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3-pip python3-numpy python3-scipy python3-matplotlib python3-jinja2 python3-mock \
                       ipython3 python3-httplib2 python3-docutils python3-yaml \
                       subversion python3-venv python3-mpi4py python3-tables cmake

# RUN useradd -ms /bin/bash docker
# USER docker

ENV HOME=/home/docker
RUN mkdir $HOME; mkdir $HOME/env; mkdir $HOME/packages

ENV VENV=$HOME/env/neurosci
RUN mkdir $VENV

# we run venv twice because of this bug: https://bugs.python.org/issue24875
# using the workaround proposed by Georges Racinet
# RUN python3 -m venv $VENV && python3 -m venv --system-site-packages $VENV

# RUN $VENV/bin/pip3 install --upgrade pip
# RUN $VENV/bin/pip3 install parameters quantities neo django django-tagging future hgapi gitpython sumatra
# RUN $VENV/bin/pip3 install --upgrade nose ipython





ENV NEST_VER=2.12.0 NRN_VER=7.4
ENV NEST=nest-$NEST_VER NRN=nrn-$NRN_VER
# ENV PATH=$PATH:$VENV/bin
# RUN ln -s /usr/bin/2to3-3.4 $VENV/bin/2to3

WORKDIR $HOME/packages
ADD http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$NRN.tar.gz .
ADD https://github.com/nest/nest-simulator/releases/download/v$NEST_VER/nest-$NEST_VER.tar.gz .
# RUN wget https://github.com/nest/nest-simulator/releases/download/v$NEST_VER/nest-$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz;
# RUN wget http://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/$NRN.tar.gz
RUN tar xzf $NEST.tar.gz; tar xzf $NRN.tar.gz; rm $NEST.tar.gz $NRN.tar.gz
# RUN git clone --depth 1 https://github.com/INCF/libneurosim.git
# RUN cd libneurosim; ./autogen.sh

RUN mkdir $VENV/build
WORKDIR $VENV/build
# RUN mkdir libneurosim; \
#     cd libneurosim; \
#     PYTHON=$VENV/bin/python $HOME/packages/libneurosim/configure --prefix=$VENV; \
#     make; make install; ls $VENV/lib $VENV/include
RUN mkdir $NRN; \
    cd $NRN; \
    $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=python --disable-rx3d --without-iv --prefix=$VENV; \
    make; make install; \
    cd src/nrnpython; python setup.py install; \
    cd $VENV/bin; ln -s ../x86_64/bin/nrnivmodl

RUN mkdir $NEST; \
    cd $NEST; \
    # ln -s /usr/lib/python3.4/config-3.4m-x86_64-linux-gnu/libpython3.4.so $VENV/lib/; \
    cmake -DCMAKE_INSTALL_PREFIX=$VENV \
          -Dwith-mpi=ON  \
          -Dwith-python=3 \
        #   -DPYTHON_EXECUTABLE=python \
          -DPYTHON_EXECUTABLE=/opt/conda/bin/python3.6 \
          ###-Dwith-music=ON \
        #   -Dwith-libneurosim=ON \
          -DPYTHON_LIBRARY=/opt/conda/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6m.a \
          -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.6m \
        #   -DPYTHON_LIBRARY=$VENV/lib/libpython3.4.so \
        #   -DPYTHON_INCLUDE_DIR=/usr/include/python3.4 \
          $HOME/packages/$NEST; \
    make; make install;


# RUN $VENV/bin/pip3 install lazyarray nrnutils PyNN
# RUN $VENV/bin/pip3 install brian2

WORKDIR /home/docker/
RUN echo "source $VENV/bin/nest_vars.sh" >> .bashrc

RUN conda install libgcc
















# Install neuron
# RUN conda install -c mattions neuron=7.4

# RUN sudo apt-get -y install Anaconda

#Install nest
# RUN conda install -c emka nest-simulator

# RUN conda install -c undy odespy

# # RUN conda install -c conda-forge multiprocess

# # Uncertainpy dependencies
# RUN apt-get update --fix-missing

# RUN apt-get -y install texlive-latex-base
# RUN apt-get -y install texlive-latex-base
# RUN apt-get -y install texlive-latex-extra
# RUN apt-get -y install texlive-fonts-recommended
# RUN apt-get -y install dvipng
# RUN apt-get -y install Xvfb
# RUN apt-get -y install h5utils
# RUN apt-get -y install libx11-dev libxext-dev x11-apps

# RUN conda install -c conda-forge xvfbwrapper

# # Downgrade the pyqt package to solve a bug in anaconda
# RUN conda install pyqt=4.11
# RUN conda install -c anaconda pandas
# RUN conda install -c anaconda seaborn
# RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot


# COPY . $HOME/uncertainpy
# WORKDIR $HOME/uncertainpy
# RUN cp -r tests/figures_docker/. tests/figures/.

# # RUN conda install conda-build
# # RUN conda build install_scripts/nest/.
# # RUN conda install --use-local nest-simulator

# RUN pip install elephant

# RUN python setup.py install


# # RUN conda install -c conda-forge matplotlib=2.0.0
