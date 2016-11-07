FROM neuralensemble/simulationx


RUN apt-get -qq update --fix-missing

#RUN apt-get install -qq sudo
# RUN sudo apt-get -qq update

RUN apt-get install -y python-pip
RUN pip install -U pip
RUN pip install -U setuptools

RUN pip install h5py

RUN apt-get -y install gfortran\
                        xvfb\
                        python-dev
                        #texlive-full


RUN pip install xvfbwrapper\
                h5py\
                chaospy\
                tqdm\
                pandas\
                pyyaml

RUN pip install -e git+https://github.com/simetenn/prettyplot.git#egg=prettyplot
RUN pip install -e git+https://github.com/hplgit/odespy.git#egg=odespy
