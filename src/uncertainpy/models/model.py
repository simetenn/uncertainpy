import numpy as np

class Model(object):
    """
    Class for storing the model to perform uncertainty quantification and
    sensitivity analysis on.

    The ``run`` method must either be implemented or set to a
    function, and is responsible for running the model.
    If you want to calculate features directly from the original model results,
    but still need to postprocess the model results to perform the
    uncertainty quantification, you can implement the postprocessing in the
    ``postprocess`` method.

    Parameters
    ----------
    adaptive : bool, optional
        True if the model is adaptive, meaning it has a varying number of
        return values. False if not. Default is False.
    labels : list, optional
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output. Default is an empty list.
    run : {None, callable}, optional
        A function that implements the model. See the ``run`` method for
        requirements of the function. Default is None.
    suppress_graphics : bool, optional
        Suppress all graphics created by the model. Default is False.
    ignore : bool, optional
        Ignore the model results when calculating uncertainties, which means the
        uncertainty is not calculated for the model. Default is False.

    Attributes
    ----------
    labels : list
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
    adaptive : bool
        True if the model is adaptive, meaning it has a varying number of
        time values. False if not. Default is False.
    name : str
        Name of the model. Either the name of the class or the name of the
        function set as run.
    suppress_graphics : bool
        Suppress all graphics created by the model.
    ignore : bool
        Ignore the model results when calculating uncertainties, which means the
        uncertainty is not calculated for the model. Default is False.

    See Also
    --------
    uncertainpy.models.Model.run
    uncertainpy.models.Model.postprocess
    """
    def __init__(self,
                 run=None,
                 adaptive=False,
                 labels=[],
                 postprocess=None,
                 suppress_graphics=False,
                 ignore=False):

        self.adaptive = adaptive
        self.labels = labels
        self.ignore = ignore
        self.suppress_graphics = suppress_graphics

        if run is not None:
            self.run = run
        else:
            self.name = self.__class__.__name__

        if postprocess is not None:
            self.postprocess = postprocess


    @property
    def run(self):
        """
        Run the model and return time and model result.

        This method must either be implemented or set to a function and is
        responsible for running the model. See Notes for requirements.

        Parameters
        ----------
        **parameters : A number of named arguments (name=value).
            The parameters of the model. These parameters must be assigned to
            the model, either setting them with Python, or
            assigning them to the simulator.

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values of the model, if no time values returns None or
            numpy.nan.
        values : array_like
            Result of the model. Note that `values` myst either be regular
            (have the same number of points for different paramaters) or be able
            to be interpolated.
        info, optional
            Any number of info objects that is passed on to feature calculations.
            It is recommended to use a single dictionary with the information
            stored as key-value pairs.
            This is what the implemented features requires, as well as
            require that specific keys to be present.

        Raises
        ------
        NotImplementedError
            If no run method have been implemented or set to a function.

        Notes
        -----
        The ``run`` method must either be implemented or set to a
        function. Both options have the following requirements:

        1. **Input.**
           The model function takes a number of arguments which define the
           uncertain parameters of the model.

        2. **Run the model.**
           The model must then be run using the parameters given as arguments.

        3. **Output.**
           The model function must return at least two objects,
           the model time (or equivalent, if applicable) and model output.
           Additionally, any number of optional info objects can be returned.
           In Uncertainpy,
           we refer to the time object as ``time``,
           the model output object as ``values``,
           and the remaining objects as ``info``.
           Note that while we refer to these objects as ``time``,
           ``values`` and ``info`` in Uncertainpy,
           it does not matter what you call the objects returned by
           the run function.

            1. **Time** (``time``).
               The ``time`` can be interpreted as the x-axis of the model.
               It is used when interpolating (see below),
               and when certain features are calculated.
               We can return ``None`` if the model has no time
               associated with it.

            2. **Model output** (``values``).
               The model output must either be regular, or it must be possible to
               interpolate or postprocess the output to a regular form.

            3. **Additional info** (``info``).
               Some of the methods provided by Uncertainpy,
               such as the later defined model postprocessing,
               feature preprocessing,
               and feature calculations,
               require additional information from the model (e.g., the time a
               neuron receives an external stimulus).
               We recommend to use a
               single dictionary as info object,
               with key-value pairs for the information,
               to make debugging easier.
               Uncertainpy always uses a single dictionary as the
               ``info`` object.
               Certain features require that specific keys are present in this
               dictionary.

        The model does not need to be implemented in Python, you can use any
        model/simulator as long as you are able to set the model parameters of
        the model from the run method Python and return the results from the
        model into the run method.

        If you want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification, you can implement the postprocessing in the
        ``postprocess`` method.

        See also
        --------
        uncertainpy.features
        uncertainpy.features.Features.preprocess : Preprocessing of model results before feature calculation
        uncertainpy.model.Model.postprocess : Postprocessing of model result.
        """
        return self._run


    @run.setter
    def run(self, new_run):
        if not callable(new_run):
            raise TypeError("run function must be callable")

        self._run = new_run
        self.name = new_run.__name__


    def _run(self, **parameters):
        raise NotImplementedError("No run method implemented or set in {class_name}".format(class_name=self.__class__.__name__))


    @property
    def postprocess(self, *model_result):
        """
        Postprocessing of the time and results from the model.

        No postprocessing is performed, and the direct model results are
        currently returned.
        If postprocessing is needed it should follow the below format.

        Parameters
        ----------
        *model_result
            Variable length argument list. Is the values that ``run``
            returns. It contains `time` and `values`,
            and then any number of optional `info` values.
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values the model should return
            ``None`` or ``numpy.nan``.
        values : array_like
            Result of the model.
        info, optional
            Any number of info objects that is passed on to feature calculations.
            It is recommended to use a single dictionary with the information
            stored as key-value pairs.
            This is what the implemented features requires, as well as
            require that specific keys to be present.

        Returns
        -------
        time : {None, numpy.nan, array_like}
            Time values of the model, if no time values returns ``None`` or
            ``numpy.nan``.
        values : array_like
            The postprocessed model results, `values` must either be regular
            (have the same number of points for different paramaters) or be able
            to be interpolated.

        Notes
        -----
        Perform a postprocessing of the model results before they are sent to
        the uncertainty quantification.
        The model results must either be regular or be able to be interpolated.
        This is because the uncertainty quantification methods
        needs results with the same number of points for each set of parameters
        to be able to perform the uncertainty quantification.

        ``postprocess`` is implemented to make
        the model results regular, or on a form that can be interpolated.
        The results from the postprocessing is not
        used to calculate features, and is therefore used if you
        want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification.

        The requirements for a ``postprocess`` function are:

        1. **Input.**
           ``postprocess`` must take the objects returned by the
           model function as input arguments.

        2. **Postprocessing.**
           The model time (``time``) and output (``values``) must
           be postprocessed to a regular form, or to a form that can be
           interpolated to a regular form by Uncertainpy.
           If additional information is needed from the model, it can be passed
           along in the ``info`` object.

        3. **Output.**
           The ``postprocess`` function must return two objects:

           1. **Model time** (``time_postprocessed``).
              The first object is the postprocessed time (or equivalent)
              of the model.
              We can return ``None`` if the model has no time.
              Note that the automatic interpolation of the postprocessed
              time can only be performed if a postprocessed time is returned
              (if an interpolation is required).

           2. **Model output** (``values_postprocessed``).
              The second object is the postprocessed model output.
        """
        return self._postprocess


    def _postprocess(self, *model_result):
        return model_result[:2]


    @postprocess.setter
    def postprocess(self, new_postprocess_function):
        if not callable(new_postprocess_function):
            raise TypeError("postprocess function must be callable")

        self._postprocess = new_postprocess_function


    def validate_run_result(self, model_result):
        """
        Validate the results from ``run()``.

        This method ensures the results from returns `time`, `values`, and optional
        info objects.

        Parameters
        ----------
        model_results
            Any type of model results returned by ``run``.

        Raises
        ------
        ValueError
            If the model result does not fit the requirements.
        TypeError
            If the model result does not fit the requirements.

        Notes
        -----
        Tries to verify that at least, `time` and `values` are returned from ``run``.
        ``model_result`` should follow the format: ``return time, values, info_1, info_2, ...``.
        Where:

        * ``time`` : ``{None, numpy.nan, array_like}``.
          Time values of the model. If no time values it should return None or
          numpy.nan.
        * ``values`` : ``array_like``
          Result of the model.
        * ``info``, optional.
          Any number of info objects that is passed on to feature calculations.
          It is recommended to use a single dictionary with the information
          stored as key-value pairs.
          This is what the implemented features requires, as well as
          require that specific keys to be present.
        """
        if isinstance(model_result, np.ndarray):
            raise ValueError("model.run() returns an numpy array. "
                             "This indicates only t or U is returned. "
                             "model.run() or model function must return "
                             "time and values "
                             "(return time, values | return None, values)")

        if isinstance(model_result, str):
            raise ValueError("model.run() returns an string. "
                             "This indicates only t or U is returned. "
                             "model.run() or model function must return "
                             "time and values "
                             " (return time, values | return None, values)")

        try:
            time, values = model_result[:2]
        except (ValueError, TypeError) as error:
            msg = "model.run() or model function must return time and values (return time, values | return None, values)"
            if not error.args:
                error.args = ("",)
            error.args = error.args + (msg,)
            raise


    def set_parameters(self, **parameters):
        """
        Set all named arguments as attributes of the model class.

        Parameters
        ----------
        **parameters : A number of named arguments (name=value).
            All set as attributes of the class.
        """
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])
