"""
Uncertainpy is a python toolbox for uncertainty quantification and sensitivity
analysis of computational models and features of the models.

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
easily be implemented. It should be noted that while \uncertainpy is tailored
towards neuroscience, the implemented methods are general, and Uncertainpy can
be used for many other types of models and features within other fields.
"""

from .data import Data, DataFeature
from .distribution import uniform, normal
from .parameters import Parameter, Parameters
from .uncertainty import UncertaintyQuantification

from .plotting import PlotUncertainty
from .features import Features, NetworkFeatures, EfelFeatures, GeneralNetworkFeatures
from .features import GeneralSpikingFeatures, SpikingFeatures
from .models import Model, NeuronModel, NestModel
from ._version import __version__