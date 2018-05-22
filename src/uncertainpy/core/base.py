from __future__ import absolute_import, division, print_function, unicode_literals

from ..utils.logger import setup_module_logger
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
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging is performed.
        Default logger level is "info".

    Attributes
    ----------
    model : uncertainpy.Model or subclass of uncertainpy.Model
        The model to perform uncertainty quantification on.
    features : uncertainpy.Features or subclass of uncertainpy.Features
        The features of the model to perform uncertainty quantification on.

    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.models.Model
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model=None,
                 features=None,
                 logger_level="info"):

        setup_module_logger(class_instance=self, level=logger_level)

        self._model = None
        self._features = None

        self._logger_level = logger_level

        self.features = features
        self.model = model



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
            self._features = Features(new_features=new_features,
                                      logger_level=self._logger_level)


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
            self._model = Model(new_model,
                                logger_level=self._logger_level)
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
    parameters: {dict {name: parameter_object}, dict of {name: value or Chaospy distribution}, ...], list of Parameter instances, list [[name, value or Chaospy distribution], ...], list [[name, value, Chaospy distribution or callable that returns a Chaospy distribution],...],}
        List or dictionary of the parameters that should be created.
        On the form ``parameters =``

            * ``{name_1: parameter_object_1, name: parameter_object_2, ...}``
            * ``{name_1:  value_1 or Chaospy distribution, name_2:  value_2 or Chaospy distribution, ...}``
            * ``[parameter_object_1, parameter_object_2, ...]``,
            * ``[[name_1, value_1 or Chaospy distribution], ...]``.
            * ``[[name_1, value_1, Chaospy distribution or callable that returns a Chaospy distribution], ...]``

    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    logger_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.

    Attributes
    ----------
    model : Model or Model subclass
        The model to perform uncertainty quantification on.
    parameters : Parameters
        The uncertain parameters.
    features : Features or subclass of Features
        The features of the model to perform uncertainty quantification on.
    logger_level : {"info", "debug", "warning", "error", "critical", None}
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging is performed.

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
                 logger_level="info"):

        super(ParameterBase, self).__init__(model=model,
                                            features=features,
                                            logger_level=logger_level)

        self._parameters = None
        self.parameters = parameters



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