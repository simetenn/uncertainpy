[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/simetenn/uncertainpy.svg?branch=master)](https://travis-ci.org/simetenn/uncertainpy)
[![codecov](https://codecov.io/gh/simetenn/uncertainpy/branch/master/graph/badge.svg?token=BFXnBcPbMA)](https://codecov.io/gh/simetenn/uncertainpy)
[![Documentation Status](https://readthedocs.org/projects/uncertainpy/badge/?version=latest)](http://uncertainpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/34467409.svg)](https://zenodo.org/badge/latestdoi/34467409)


<!-- [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/simetenn/uncertainpy/master?filepath=examples%2Fcoffee_cup%2Fcoffee_cup.ipynb) -->


<img src="https://github.com/simetenn/uncertainpy/blob/master/logo/uncertainpy.png" width="700">


# A python toolbox for uncertainty quantification and sensitivity analysis tailored towards computational neuroscience.

Uncertainpy is a python toolbox for uncertainty quantification and sensitivity
analysis tailored towards computational neuroscience.

Uncertainpy is model independent and treats the model as a black box where the
model can be left unchanged. Uncertainpy implements both quasi-Monte Carlo
methods and polynomial chaos expansions using either point collocation or the
pseudo-spectral method. Both of the polynomial chaos expansion methods have
support for the rosenblatt transformation to handle dependent input parameters.

Uncertainpy is feature based, i.e., if applicable, it recognizes and calculates
the uncertainty in features of the model, as well as the model itself.
Examples of features in neuroscience can be spike timing and the action
potential shape.

Uncertainpy is tailored towards neuroscience models, and comes with several
common neuroscience models and features built in, but new models and features can
easily be implemented. It should be noted that while Uncertainpy is tailored
towards neuroscience, the implemented methods are general, and Uncertainpy can
be used for many other types of models and features within other fields.

### Table of contents

- [Example](#example)
- [Documentation](#documentation)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Citation](#citation)


## Example

Examples for how to use Uncertainpy can be found in the
[examples](https://github.com/simetenn/uncertainpy/tree/master/examples) folder
as well as in the [documentation](https://uncertainpy.readthedocs.io/en/latest/).
Here we show an example,
found in [examples/coffee_cup](https://github.com/simetenn/uncertainpy/tree/master/examples/coffee_cup),
where we examine the changes in temperature of a cooling coffee cup that
follows Newton’s law of cooling:

<!-- \frac{dT(t)}{dt} = -\kappa(T(t) - T_{env}) -->
![img](https://latex.codecogs.com/svg.latex?\frac{dT(t)}{dt}&space;=&space;-\kappa(T(t)&space;-&space;T_{env}))

This equation tells how the temperature ![img](http://latex.codecogs.com/svg.latex?T)
of the coffee cup changes with time ![img](http://latex.codecogs.com/svg.latex?t),
when it is in an environment with temperature
![img](http://latex.codecogs.com/svg.latex?T_{env}).
![img](http://latex.codecogs.com/svg.latex?\kappa) is a proportionality
constant that is characteristic of the system and regulates how fast the coffee
cup radiates heat to the environment.
For simplicity we set the initial temperature to a fixed value, ![img](http://latex.codecogs.com/svg.latex?95^\circ\text{C}&space;),
and let ![img](http://latex.codecogs.com/svg.latex?\kappa) and ![img](http://latex.codecogs.com/svg.latex?T_{env}) be uncertain input parameters.

We start by importing the packages we use:

```python
import uncertainpy as un
import numpy as np                   # For the time array
import chaospy as cp                 # To create distributions
from scipy.integrate import odeint   # To integrate our equation
```

To create the model we define a Python function `coffee_cup` that
takes the uncertain parameters `kappa` and `T_env` as input arguments.
Inside this function we solve our equation by integrating it using
`scipy.integrate.odeint`,
before we return the results.
The implementation of the model is:

```python
# Create the coffee cup model function
def coffee_cup(kappa, T_env):
    # Initial temperature and time array
    time = np.linspace(0, 200, 150)            # Minutes
    T_0 = 95                                   # Celsius

    # The equation describing the model
    def f(T, time, kappa, T_env):
        return -kappa*(T - T_env)

    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

    # Return time and model output
    return time, temperature
```

We could use this function directly in `UncertaintyQuantification`,
but we would like to have labels on the axes when plotting.
So we create a `Model` with the above run function and labels:

```python
# Create a model from the coffee_cup function and add labels
model = un.Model(run=coffee_cup, labels=["Time (min)", "Temperature (C)"])
```

The next step is to define the uncertain parameters.
We give the uncertain parameters in the cooling coffee cup model the following
distributions:

<!-- \begin{align}
    \kappa &= \mathrm{Uniform}(0.025, 0.075), \\
    T_{env} &= \mathrm{Uniform}(15, 25).
\end{align} -->

![img](http://latex.codecogs.com/svg.latex?\begin{align*}%0D%0A\kappa%26%3D\mathrm{Uniform}(0.025%2C0.075)%2C\\\\%0D%0AT_{env}%26%3D\mathrm{Uniform}(15%2C25).%0D%0A\end{align*})


We use Chaospy to create the distributions, and create a parameter dictionary:

```python
# Create the distributions
kappa_dist = cp.Uniform(0.025, 0.075)
T_env_dist = cp.Uniform(15, 25)

# Define the parameter dictionary
parameters = {"kappa": kappa_dist, "T_env": T_env_dist}
```

We can now calculate the uncertainty and sensitivity using polynomial chaos
expansions with point collocation,
which is the default option of `quantify`:

```python
# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify()
```

Here you see an example on how the results might look:

![Example of results](https://github.com/simetenn/uncertainpy/blob/dev/docs/images/coffee_cup.png "Results for an uncertainty quantification and sensitivity analysis of a cooling coffee cup model")


This plot shows the mean, variance, and 90% prediction interval
(A),
and the first-order Sobol indices (B),
which shows the sensitivity of the model to each parameter, for the cooling coffee cup model.
As the mean (blue line) in A shows,
the cooling gives rise to an exponential decay in the temperature,
towards the temperature of the environment ![img](http://latex.codecogs.com/svg.latex?T_{env}).
From the sensitivity analysis (B) we see that T is most
sensitive to ![img](http://latex.codecogs.com/svg.latex?\kappa)
early in the simulation,
and to ![img](http://latex.codecogs.com/svg.latex?T_{env}) towards the end of the simulation.
This is as expected, since ![img](http://latex.codecogs.com/svg.latex?\kappa)
determines the rate of the cooling,
while ![img](http://latex.codecogs.com/svg.latex?T_{env}) determines the final temperature.
After about 150 minutes,
the cooling is essentially completed,
and the uncertainty in T exclusively reflects the uncertainty of
![img](http://latex.codecogs.com/svg.latex?T_{env}).


## Documentation

The documentation for Uncertainpy can be found at [http://uncertainpy.readthedocs.io](http://uncertainpy.readthedocs.io),
and the Uncertainpy paper here:
[Tennøe S, Halnes G and Einevoll GT (2018) Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience. Front. Neuroinform. 12:49. doi: 10.3389/fninf.2018.00049](https://www.frontiersin.org/articles/10.3389/fninf.2018.00049/full).
## Installation

Uncertainpy works with Python 3.
Uncertainpy can easily be installed using pip. The minimum install is:

    pip install uncertainpy

To install all requirements you can write:

    pip install uncertainpy[all]

Specific optional requirements can also be installed,
see below for an explanation.
Uncertainpy can also be installed by cloning the Github repository:

    $ git clone https://github.com/simetenn/uncertainpy
    $ cd /path/to/uncertainpy
    $ python setup.py install

`setup.py` are able to install different set of dependencies.
For all options run::

    $ python setup.py --help

Alternatively, Uncertainpy can be easily installed (minimum install) with conda
using conda-forge channel::

    $ conda install -c conda-forge uncertainpy

The above installation within a conda environment is only compatible with Python 3.x.
By using conda, the installation will solves compatibility issues automatically.

### Dependencies

Uncertainpy has the following dependencies:

* `chaospy`
* `tqdm`
* `h5py`
* `multiprocess`
* `numpy`
* `scipy`
* `seaborn`
* `matplotlib`
* `xvfbwrapper`
* `six`
* `SALib`
* `exdir`

These are installed with the minimum install.

`xvfbwrapper` requires `xvfb`, which can be installed with:

    sudo apt-get install xvfb

Additionally Uncertainpy has a few optional dependencies for specific classes
of models and for features of the models.

#### EfelFeatures

`uncertainpy.EfelFeatures` requires the Python package

* `efel`

which can be installed with:

    pip install uncertainpy[efel_features]

or:

    pip install efel

#### NetworkFeatures

`uncertainpy.NetworkFeatures` requires the Python packages

* `elephant`
* `neo`
* `quantities`

which can be installed with:

    pip install uncertainpy[network_features]

or:

    pip install elephant, neo, quantities


#### NeuronModel

`uncertainpy.NeuronModel` requires the external simulator
[Neuron](https://www.neuron.yale.edu/neuron/download) (with Python),
a simulator for neurons. Neuron must be installed by the user.


#### NestModel

`uncertainpy.NestModel` requires the external simulator
[Nest](http://www.nest-simulator.org/installation) (with Python),
a simulator for network of neurons.
Nest must be installed by the user.


### Test suite

Uncertainpy comes with an extensive test suite that can be run with the `test.py` script.
For how to use test.py run:

    $ python test.py --help

`test.py` has all dependencies of Uncertainpy in addition to:

* `click`

These can be installed with pip:

    pip install uncertainpy[tests]

In addition, the following program must be installed:

* `hdf5-tools`


## Citation

If you use Uncertainpy in your work, please cite:
[Tennøe S, Halnes G and Einevoll GT (2018) Uncertainpy: A Python Toolbox for Uncertainty Quantification and Sensitivity Analysis in Computational Neuroscience. Front. Neuroinform. 12:49. doi: 10.3389/fninf.2018.00049](https://www.frontiersin.org/articles/10.3389/fninf.2018.00049/full).



## Commit messages

The style mostly used:

    API:         an (incompatible) API change
    Benchmark:   changes to the benchmark suite
    Build:       related to building
    Bug:         bug fix
    Deprecate:   deprecate something, or remove a deprecated object
    Doc:         documentation
    [ blank ]:   enhancement
    Refactor:    maintenance commit (refactoring, typos, etc.)
    Revert:      revert an earlier commit
    Style:       fix (whitespace, PEP8)
    Test:        addition or modification of tests
    Release:     related to releasing
    Tool:        development tool or utility
    WIP:         Work-in-progress
