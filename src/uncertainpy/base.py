from .utils import create_logger
from .features import GeneralFeatures
from .models import Model
from .parameters import Parameters

class Base(object):
    def __init__(self,
                 model=None,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None):

        self._model = None
        self._features = None
        self._parameters = None

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        self.features = features
        self.model = model



    @property
    def features(self):
        return self._features


    @features.setter
    def features(self, new_features):
        if new_features is None:
            self._features = GeneralFeatures(features_to_run=None)
        elif isinstance(new_features, GeneralFeatures):
            self._features = new_features
        else:
            self._features = GeneralFeatures(features_to_run="all")
            self._features.add_features(new_features)


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        if isinstance(new_model, Model) or new_model is None:
            self._model = new_model
        elif callable(new_model):
            self._model = Model(new_model)
            # self._model.run = new_model
        else:
            raise TypeError("model must be a Model instance, callable or None")



class ParameterBase(Base):
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None):

        super(ParameterBase, self).__init__(model=model,
                                            features=features,
                                            verbose_level=verbose_level,
                                            verbose_filename=verbose_filename)

        self._parameters = None
        self.parameters = parameters


    @property
    def parameters(self):
        return self._parameters


    @parameters.setter
    def parameters(self, new_parameters):
        if isinstance(new_parameters, Parameters) or new_parameters is None:
            self._parameters = new_parameters
        else:
            self._parameters = Parameters(new_parameters)