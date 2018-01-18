"""
A module that contains different model classes. The main class is ``Model``,
for storing the model to perform uncertainty quantification and sensitivity
analysis on. This class does not implement any specific models itself.
Then there are two class for specific simulators: NEURON (``NeuronModel``) and
Nest (``NestModel``).

1. ``Model``
2. ``NestModel``
3. ``NeuronModel``
"""

__all__ = ["Model", "NeuronModel", "NestModel"]

from .model import Model
from .neuron_model import NeuronModel
from .nest_model import NestModel

