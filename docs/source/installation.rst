.. _installation:

Installation
============

Uncertainpy works with with both Python 2 and 3.
Uncertainpy can easily be installed using pip. The minimum install is:

    pip install uncertainpy

To install all requirements you can write:

    pip install uncertainpy[all]

Specific optional requirements can also be installed,
see below for an explanation.
Uncertainpy can also be installed by cloning the `Github repository`_::

    $ git clone https://github.com/simetenn/uncertainpy
    $ cd /path/to/uncertainpy
    $ python setup.py install

``setup.py`` are able to install different set of dependencies.
For all options run::

    $ python setup.py --help



.. _Github repository: https://github.com/simetenn/uncertainpy


Dependencies
------------

Uncertainpy has the following dependencies:

* ``chaospy``
* ``tqdm``
* ``h5py``
* ``multiprocess``
* ``numpy``
* ``scipy``
* ``seaborn``
* ``matplotlib``
* ``xvfbwrapper``
* ``six``
* ``SALib``

These are installed with the minimum install.

``xvfbwrapper`` requires ``xvfb``, which can be installed with::

    sudo apt-get install xvfb

Two different fileformats (backends)  can be chosen to save the results in.
`HDF5` is chosen (and installed) by default, but the ``Exdir`` format can be
chosen as well. If you want to use ``Exdir`` it can be installed by:

    conda install -c cinpla exdir

Currently ``Exdir`` only supports Python 2.7 in an experimental branch.
To get the  Python 2.7 version of exdir,
run the above command to install all dependencies. Then clone the Exdir Github
repository, change to the python 2.7 branch and then install from the source:

    git clone https://github.com/CINPLA/exdir.git
    cd exdir
    git checkout python27
    python setup.py install

Additionally Uncertainpy has a few optional dependencies for specific classes
of models and for features of the models.

EfelFeatures
^^^^^^^^^^^^

``uncertainpy.EfelFeatures`` requires the Python package

* ``efel``

which can be installed with::

    pip install uncertainpy[efel_features]

or::

    pip install efel

NetworkFeatures
^^^^^^^^^^^^^^^

``uncertainpy.NetworkFeatures`` requires the Python packages

* ``elephant``
* ``neo``
* ``quantities``

which can be installed with::

    pip install uncertainpy[network_features]

or::

    pip install elephant, neo, quantities


NeuronModel
^^^^^^^^^^^

``uncertainpy.NeuronModel`` requires the external simulator `NEURON`_
(with Python), a simulator for neurons.
NEURON must be installed by the user.

.. _NEURON: https://www.neuron.yale.edu/neuron/download

NestModel
^^^^^^^^^

``uncertainpy.NestModel`` requires the external simulator
`NEST`_ (with Python),
a simulator for network of neurons.
NEST must be installed by the user.

.. _NEST: http://www.nest-simulator.org/installation



Test suite
----------

Uncertainpy comes with an extensive test suite that can be run with the ``test.py`` script.
For how to use ``test.py`` run::

    $ python test.py --help

``test.py`` has all the above dependencies in addition to:

* ``click``



