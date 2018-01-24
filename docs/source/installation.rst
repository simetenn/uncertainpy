.. _installation:

Installation
============

Uncertainpy currently only works with Python 2.
Uncertainpy can easily be installed using pip::

    pip install uncertainpy

or from source by cloning the Github repository::

    $ git clone https://github.com/simetenn/uncertainpy
    $ cd /path/to/uncertainpy
    $ sudo python setup.py install

Dependencies
------------

Uncertainpy has the following dependencies:

* ``xvfbwrapper``
* ``chaospy``
* ``tqdm``
* ``h5py``
* ``multiprocess``
* ``numpy``
* ``scipy``
* ``seaborn``

Additionally Uncertainpy has a few optional dependencies for specific classes of models and for features of the models.
The following external simulators are required for specific models:

* ``uncertainpy.NeuronModel``: Requires `Nest`_ (with Python), a simulator for neurons.
* ``uncertainpy.NestModel``: Requires `NEURON`_ (with Python), a simulator for network of neurons.

.. _Nest: http://www.nest-simulator.org/installation
.. _NEURON: https://www.neuron.yale.edu/neuron/download

And the following Python packages are required for specific features:

* ``uncertainpy.EfelFeatures``: ``efel``.
* ``uncertainpy.NetworkFeatures``: ``elephant``, ``neo``, and ``quantities``.

Test suite
----------

Uncertainpy comes with an extensive test suite that can be run with the ``test.py`` script.
For how to use test.py run::

    $ python test.py --help
