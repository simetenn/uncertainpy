.. _installation:

Installation
============

Uncertainpy works with with Python 3.
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
* ``exdir``

These are installed with the minimum install.

``xvfbwrapper`` requires ``xvfb``, which can be installed with::

    sudo apt-get install xvfb

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

or through::

    python setup.py install --efel_features



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

or through::

    python setup.py install --network_features


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

These dependencies can be installed with::

    pip install uncertainpy[tests]

or::

    pip install click

or through::

    python setup.py install --tests






Documentation
-------------

The documentation is generated through ``sphinx``, and has the following
dependencies:

* ``sphinx``
* ``sphinx_rtd_theme``


These dependencies can be installed with::

    pip install uncertainpy[docs]

or::

    pip install sphinx, sphinx_rtd_theme

or through::

    python setup.py install --docs


The documentation is build by::

    cd docs
    make html



