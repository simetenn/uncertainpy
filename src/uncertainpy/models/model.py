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
    ``postprocess(time, values)`` method.

    Parameters
    ----------
    adaptive : bool, optional
        True if the model is adaptive, meaning it has a varying number of
        return values. False if not. Default is False.
    labels : list, optional
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
        Default is an empty list.
    run_function : {None, callable}, optional
        A function that implements the model. See the ``run`` method for
        requirements of the function. Default is None.

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

    See Also
    --------
    uncertainpy.models.Model.run : The run method.
    uncertainpy.models.Model.postprocess : The postprocessing method.
    """
    def __init__(self,
                 run_function=None,
                 adaptive=False,
                 labels=[],
                 ignore=False):
                 # TODO fix and document ignore option

        self.adaptive = adaptive
        self.labels = labels
        self.ignore = ignore

        if run_function is not None:
            self.run = run_function
        else:
            self.name = self.__class__.__name__


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

        1. **Receive parameters as input**.
            The run function takes a number of named arguments that are the
            parameters of the model.
        2. **Set the parameters of the model**.
            The arguments received as input (the model parameters) must be set
            in the model. Depending on the model this can be done by writing the
            parameters to an init file, assigning them to the correct variable,
            or simply using them directly.
        3. **Run the model**.
            The model must then be run. Examples of this is either to perform
            the calculations inside the run method, or calling an external
            simulator.
        4. **Return the model results**.
            Lastly we need to return the model results. If we use an external
            simulator the model results must be loaded into Python, so they can
            be returned. The run function must return at least two objects,
            but can return more.

            1. **Time**.
                The first object is the time values (or equivalent) of the model,
                denoted ``time`` in Uncertainpy. You can return ``None`` if the
                model has no time values.
            2. **Model results**.
                The second object is the model results, denoted ``values``
                in Uncertainpy. The model results ``time``, and ``values`` must
                either be regular (have the same number of points for each model
                evaluation), be able to be interpolated, or be able to be
                postprocessed to a regular form, or a form that can be
                interpolated. This is because the uncertainty quantification
                methods require results that have the same number of points for
                each set of model evaluations.
            3. **Additional info**.
                After ``time`` and ``values``, any number of additional info
                objects can be returned. These info objects are optional and are
                used in the ``Model.postprocess`` method, the
                ``Feature.preprocess`` method, and feature calculations if
                additional information about the model is needed in these steps.
                For ease of use we recommend to use a single dictionary as info
                object, with key-value pairs for all information needed. This
                makes debugging easier. All features implemented in
                Uncertainpy use a single dictionary as info object, denoted
                ``info``. Certain features require that certain keys are present
                in this dictionary.An example, models of neurons often contain
                an input stimulus that makes the neuron generate action potentials.
                The timing of this stimulus is used to calculate certain features,
                and ``info`` becomes::

                    info = {"stimulus_start": 150,  # ms
                            "stimulus_end":   300}  # ms

            Note that while we refer to these objects as ``time``,
            ``values`` and ``info`` in Uncertainpy,
            it does not matter what you call the objects returned by
            the run function.

        The model does not need to be implemented in Python, you can use any
        model/simulator as long as you are able to set the model parameters of
        the model from the run method Python and return the results from the
        model into the run method.

        If you want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification, you can implement the postprocessing in the
        ``postprocess(time, values, *info)`` method.

        See also
        --------
        uncertainpy.features : Features of models
        uncertainpy.features.Features.preprocess : Preprocessing of model results before feature calculation
        uncertainpy.model.Model.postprocess : Postprocessing of model result.
        """
        return self._run


    @run.setter
    def run(self, new_run_function):
        if not callable(new_run_function):
            raise TypeError("run() function must be callable")

        self._run = new_run_function
        self.name = new_run_function.__name__


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



    def _run(self, **parameters):
        raise NotImplementedError("No run() method implemented or set in {class_name}".format(class_name=self.__class__.__name__))



    def postprocess(self, *model_result):
        """
        Postprocessing of the time and results from the model.

        No postprocessing is performed, and the direct model results are
        currently returned.
        If postprocessing is needed it should follow the below format.

        Parameters
        ----------
        *model_result
            Variable length argument list. Is the values that ``run()``
            returns. It contains `time` and `values`,
            and then any number of optional `info` values.
        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it should return None or
            numpy.nan.
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
            Time values of the model, if no time values returns None or
            numpy.nan.
        values : array_like
            the postprocessed model results, `values` must either be regular
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

        ``postprocess(time, values)`` is implemented to make
        the model results regular, or on a form that can be interpolated.
        The results from the postprocessing is not
        used to calculate features, and is therefore used if you
        want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification.
        """
        return model_result[:2]


    def validate_run_result(self, model_result):
        """
        Validate the results from ``run()``.

        This method ensures the results from returns `time`, `values`, and optional
        info objects.

        Returns
        -------
        model_results
            Any type of model results returned by ``run()``.

        Raises
        ------
        ValueError
            If the model result does not fit the requirements.
        TypeError
            If the model result does not fit the requirements.

        Notes
        -----
        Tries to verify that at least, `time` and `values` are returned from ``run()``.
        ``run()`` should return on the following format
        ``return time, values, info_1, info_2, ...``.

        time : {None, numpy.nan, array_like}
            Time values of the model. If no time values it should return None or
            numpy.nan.
        values : array_like
            Result of the model.
        info, optional
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