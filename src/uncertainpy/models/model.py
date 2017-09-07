class Model(object):
    """
    Class for storing the model to perform uncertainty quantification on.

    The ``run(**parameters)`` method must either be implemented or set to a
    function, and is responsible for running the model.
    If you want to calculate features directly from the original model results,
    but still need to postprocess the model results to perform the
    uncertainty quantification, you can implement the postprocessing in the
    ``postprocess(t, U)`` method.

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
    run_function : {None, function}, optional
        A function that implements the model. See Note for requirements of the
        function. Default is None.

    Attributes
    ----------
    labels : list
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
    adaptive : bool
        True if the model is adaptive, meaning it has a varying number of
        time values. False if not. Default is False.

    See Also
    --------
    uncertainpy.models.Model.run : The run method.
    uncertainpy.models.Model.postprocess : The postprocessing method.
    """
    def __init__(self,
                 run_function=None,
                 adaptive=False,
                 labels=[]):

        self.adaptive = adaptive
        self.labels = labels

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
        t : {None, numpy.nan, array_like}
            Time values of the model, if no time values returns None or
            numpy.nan.
        U : array_like
            Result of the model. Note that `U` myst either be regular
            (have the same number of points for different paramaters) or be able
            to be interpolated.


        Raises
        ------
        NotImplementedError
            If no run method have been implemented or set to a function.

        Notes
        -----
        The ``run(**parameters)`` method must either be implemented or set to a
        function. Both options have the following requirements:

        1. ``run(**parameters)`` takes a number of named arguments which are
        the parameters to the model. These parameters must be assigned to
        the model, either setting them with Python, or
        assigning them to the simulator.

        2. ``run(**parameters)`` must return the time values or equivalent (``t``)
        and the model result (``U``). If the model have no time values,
        return None instead.

        The model does not need to be implemented in Python, you can use any
        model/simulator as long as you are able to set the model parameters of
        the model from the run method Python and return the results from the
        model into the run method.

        The model results ``t`` and ``U`` is used to calculate the features.

        The model results must either be regular or be able to be interpolated, or
        be able to be postprocessed to a regular form or a form that can
        be interpolated. This is because the uncertainty quantification methods
        needs results with the same number of points for each set of parameters
        to be able to perform the uncertainty quantification.

        If you want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification, you can implement the postprocessing in the
        ``postprocess(t, U)`` method.

        See also
        --------
        uncertainpy.model.Model.postprocess : Postprocessing of model result.
        """
        return self._run


    @run.setter
    def run(self, new_run_function):
        if not callable(new_run_function):
            raise TypeError("run function must be callable")

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
        raise NotImplementedError("No run() method implemented in {class_name}".format(class_name=self.__class__.__name__))


    def postprocess(self, t, U):
        """
        Postprocessing of the time and results from the model.

        No postprocessing is performed, and the direct model results are
        currently returned.
        If postprocessing is needed it should follow teh below format.

        Returns
        -------
        t : {None, numpy.nan, array_like}
            Time values of the model, if no time values returns None or
            numpy.nan.
        U : array_like
            Result of the model.

        Returns
        -------
        t : {None, numpy.nan, array_like}
            Time values of the model, if no time values returns None or
            numpy.nan.
        U : array_like
            the postprocessed model results, `U` must either be regular
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

        ``postprocess(t, U)`` is implemented to make
        the model results regular, or on a form that can be interpolated.
        The results from the postprocessing is not
        used to calculate features, and is therefore used if you
        want to calculate features directly from the original model results,
        but still need to postprocess the model results to perform the
        uncertainty quantification.
        """
        return t, U