from .utils import create_logger
from .features import GeneralFeatures
from .models import Model
from .parameters import Parameters

class Base(object):
    """
    Set and update features and model.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on.
        Default is None.
    features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if a filename is given.
        Default is None.

    Attributes
    ----------
    model
    features
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.

    See Also
    --------
    uncertainpy.features.GeneralFeatures : General features class
    uncertainpy.models.Model : Model class
    """
    def __init__(self,
                 model=None,
                 features=None,
                 verbose_level="info",
                 verbose_filename=None):

        self._model = None
        self._features = None

        self.logger = create_logger(verbose_level,
                                    verbose_filename,
                                    self.__class__.__name__)

        self.features = features
        self.model = model



    @property
    def features(self):
        """
        Features to calculate from the model result.

        Parameters
        ----------
        new_features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}
            Features to calculate from the model result.
            If None, no features are calculated.
            If list of feature functions, all will be calculated.

        Returns
        -------
        features: {None, GeneralFeatures object}
             Features to calculate from the model result.
             If None, no features are calculated.

        See Also
        --------
        uncertainpy.features.GeneralFeatures : General features class
        uncertainpy.features.GeneralSpikingFeatures : General spiking features class
        uncertainpy.features.SpikingFeatures : Implemented spiking features class
        uncertainpy.features.GeneralNetworkFeatures : General network features class
        uncertainpy.features.NetworkFeatures : Implemented network features class
        """
        return self._features


    @features.setter
    def features(self, new_features):
        if isinstance(new_features, GeneralFeatures):
            self._features = new_features
        else:
            self._features = GeneralFeatures(new_features=new_features)


    @property
    def model(self):
        """
        Model to perform uncertainty quantification on.

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
        uncertainpy.models.Model : Model class
        uncertainpy.models.NestModel : Nest simulator model class
        uncertainpy.models.NeuronModel : Neuron simulator model class
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
        Model to perform uncertainty quantification on.
        Default is None.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}, optional
        Either None, a Parameters instance or a list the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
        Default is None.
    features : {None, GeneralFeatures or GeneralFeatures subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.

    Attributes
    ----------
    model
    parameters
    features
    logger : logging.Logger object
        Logger object responsible for logging to screen or file.


    See Also
    --------
    uncertainpy.features.GeneralFeatures : General features class
    uncertainpy.models.Model : Model class
    """
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
        """
        Model parameters.

        Parameters
        ----------
        new_parameters : {None, Parameters instance, list of Parameter instances, list [[name, value, distribution], ...]}
            Either None, a Parameters instance or a list the parameters that should be created.
            The two lists are similar to the arguments sent to Parameters.
            Default is None.

        Returns
        -------
        parameters: {None, Parameters object}
            Parameters of the model.
            If None, no parameters have been set.

        See Also
        --------
        uncertainpy.Parameter : Parameter class
        uncertainpy.Parameters : Parameters collection class
        """
        return self._parameters


    @parameters.setter
    def parameters(self, new_parameters):
        if isinstance(new_parameters, Parameters) or new_parameters is None:
            self._parameters = new_parameters
        else:
            self._parameters = Parameters(new_parameters)