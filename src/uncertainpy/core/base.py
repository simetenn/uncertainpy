from __future__ import absolute_import, division, print_function, unicode_literals

from ..utils import create_logger, load_config
from ..features import Features
from ..models import Model
from ..parameters import Parameters




class Base(object):
    """
    Set and update features and model.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all listed features will be calculated.
        Default is None.
    logger_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is ``"info"``.
    logger_config_filename : {None, "", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). If None,
        no configuration is loaded. Default is "".

    Attributes
    ----------
    model : uncertainpy.Model or subclass of uncertainpy.Model
        The model to perform uncertainty quantification on.
    features : uncertainpy.Features or subclass of uncertainpy.Features
        The features of the model to perform uncertainty quantification on.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.

    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.models.Model
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model=None,
                 features=None,
                 logger_level="info",
                 logger_config_filename=""):

        self._model = None
        self._features = None

        self.features = features
        self.model = model

        self.logger = create_logger(logger_level,
                                    __name__ + "." + self.__class__.__name__,
                                    logger_config_filename)




    @property
    def features(self):
        """
        Features to calculate from the model result.

        Parameters
        ----------
        new_features : {None, Features or Features subclass instance, list of feature functions}
            Features to calculate from the model result.
            If None, no features are calculated.
            If list of feature functions, all will be calculated.

        Returns
        -------
        features : {None, Features object}
             Features to calculate from the model result.
             If None, no features are calculated.

        See Also
        --------
        uncertainpy.features.Features
        uncertainpy.features.GeneralSpikingFeatures
        uncertainpy.features.SpikingFeatures
        uncertainpy.features.GeneralNetworkFeatures
        uncertainpy.features.NetworkFeatures
        """
        return self._features


    @features.setter
    def features(self, new_features):
        if isinstance(new_features, Features):
            self._features = new_features
        else:
            self._features = Features(new_features=new_features)


    @property
    def model(self):
        """
        Model to perform uncertainty quantification on. For requirements see
        Model.run.

        Parameters
        ----------
        new_model : {None, Model or Model subclass instance, model function}
            Model to perform uncertainty quantification on.

        Returns
        -------
        model : Model or Model subclass instance
            Model to perform uncertainty quantification on.

        See Also
        --------
        uncertainpy.models.Model
        uncertainpy.models.Model.run
        uncertainpy.models.NestModel
        uncertainpy.models.NeuronModel
        """
        return self._model

    @model.setter
    def model(self, new_model):
        if isinstance(new_model, Model) or new_model is None:
            self._model = new_model
        elif callable(new_model):
            self._model = Model(new_model)
        else:
            raise TypeError("model must be a Model or Model subclass instance, callable or None")



class ParameterBase(Base):
    """
    Set and update features, model and parameters.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}, optional
        Either None, a Parameters instance or a list of the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
        Default is None.
    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    logger_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    logger_config_filename : {None, "", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). If None,
        no configuration is loaded. Default is "".

    Attributes
    ----------
    model : Model or Model subclass
        The model to perform uncertainty quantification on.
    parameters : Parameters
        The uncertain parameters.
    features : Features or subclass of Features
        The features of the model to perform uncertainty quantification on.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.


    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.models.Model
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 logger_level="info",
                 logger_config_filename=""):

        super(ParameterBase, self).__init__(model=model,
                                            features=features,
                                            logger_level=logger_level,
                                            logger_config_filename=logger_config_filename)

        self._parameters = None
        self.parameters = parameters

        # self.logger = create_logger(logger_level,
        #                             __name__ + "." + self.__class__.__name__,
        #                             logger_config_filename)


    @property
    def parameters(self):
        """
        Model parameters.

        Parameters
        ----------
        new_parameters : {None, Parameters instance, list of Parameter instances, list [[name, value, distribution], ...]}
            Either None, a Parameters instance or a list of the parameters that should be created.
            The two lists are similar to the arguments sent to Parameters.
            Default is None.

        Returns
        -------
        parameters: {None, Parameters}
            Parameters of the model.
            If None, no parameters have been set.

        See Also
        --------
        uncertainpy.Parameter
        uncertainpy.Parameters
        """
        return self._parameters


    @parameters.setter
    def parameters(self, new_parameters):
        if isinstance(new_parameters, Parameters) or new_parameters is None:
            self._parameters = new_parameters
        else:
            self._parameters = Parameters(new_parameters)