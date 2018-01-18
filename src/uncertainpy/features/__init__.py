"""
A module for calculating features of the model output.  The main class is
``Features``. This class does not implement any specific models itself, but
contain all common methods used by features. Then there are two general classes
for spiking (``GeneralSpikingFeatures``) and network features
(``GeneralNetworkFeatures``) that implements the common methods used the two
classes of features respectively. Lastly there are three classes that implements
actual features; ``SpikingFeatures`` and ``EfelFeatures`` for models with
spiking behavour (for example models of neurons that contain voltage traces),
and ``NetworkFeatures`` for network models that return spiketrains.
"""

__all__ = ["Features",
           "GeneralSpikingFeatures",
           "SpikingFeatures",
           "Spike",
           "Spikes",
           "NetworkFeatures",
           "GeneralNetworkFeatures",
           "EfelFeatures"]

from .features import Features
from .general_spiking_features import GeneralSpikingFeatures
from .spiking_features import SpikingFeatures
from .spikes import Spike, Spikes
from .network_features import NetworkFeatures
from .general_network_features import GeneralNetworkFeatures
from .efel_features import EfelFeatures