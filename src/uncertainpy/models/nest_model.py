try:
    import nest

    prerequisites = True
except ImportError:
    prerequisites = False

import numpy as np

from .model import Model


class NestModel(Model):
    """
    Class for Nest simulator models.

    The ``run(**parameters)`` method must either be implemented or set to a
    function, and is responsible for running the Nest model.

    Parameters
    ----------
    adaptive : bool, optional
        True if the model is adaptive, meaning it has a varying number of
        time values. False if not. Default is False.
    labels : list, optional
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
        Default is ``["time [ms]", "Neuron nr", "Spiking probability"]``.
    run_function : {None, function}, optional
        A function that implements the model. See Note for requirements of the
        function. Default is None.

    Attributes
    ----------
    run : uncertainpy.models.Model.run
    labels : list, optional
        A list of label names for the axes when plotting the model.
    adaptive : bool
        True if the model is adaptive, meaning it has a varying number of
        return values. False if not. Default is False.

    See Also
    --------
    uncertainpy.models.NestModel.run : The run method.
    """
    def __init__(self,
                 run_function=None,
                 adaptive=False,
                 labels=["time [ms]", "Neuron nr", "Spiking probability"]):


        if not prerequisites:
            raise ImportError("NestModel requires: nest")

        super(NestModel, self).__init__(run_function=run_function,
                                        adaptive=adaptive,
                                        labels=labels)


    @Model.run.getter
    def run(self):
        """
        Run a Nest model and return the final simulation time and the
        spiketrains.

        This method must either be implemented or set to a function and is
        responsible for running the model. See Notes for requirements.

        Parameters
        ----------
        **parameters : A number of named arguments (name=value).
            The parameters of the model. These parameters must be assigned to
            the Nest model.

        Returns
        -------
        t_stop : {int, float}
            The final simulation time.
        spiketrains : list
            A list of spike trains for each neuron.

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
        the Nest model.

        2. ``run(**parameters)`` must return final simulation time  (``t_stop``)
        and a list of spike trains (``spiketrains``).

        The model results ``t_stop`` and ``spiketrains`` is used to calculate
        the features.
        The model result is postprocessed to create a regular result before
        the calculating the uncertainties of the model.

        See also
        --------
        uncertainpy.model.Model.postprocess : Postprocessing of model result.
        """
        return self._run


    def postprocess(self, t_stop, spiketrains):
        """
        Postprocessing of the spiketrains from a Nest model.

        For each neuron, convert a spiketrain to a list of the probability for
        a spike at each timestep, as well as creating a time array. For each
        timestep in the simulation the result is 0 if there is no spike
        and 1 if there is a spike.

        Parameters
        ----------
        t_stop : {int, float}
            The final simulation time.
        spiketrains : list
            A list of spike trains for each neuron.

        Returns
        -------
        t : array
            A time array of all time points in the Nest simulation.
        spiketrains : list
            A list of the probability for a spike at each timestep, for each
            neuron.

        Example
        -------
        In a simulation that gives the spiketrain ``[0, 2, 3]``, with a
        resolution of 0.5 and end time at 4, the resulting spike train become:
        ``[1, 0, 0, 0, 1, 0, 1, 0, 0]``.
        """

        dt = nest.GetKernelStatus()["resolution"]
        t = np.arange(0, t_stop, dt)

        expanded_spiketrains = []
        for spiketrain in spiketrains:
            binary_spike = np.zeros(len(t))
            binary_spike[np.in1d(t, spiketrain)] = 1

            expanded_spiketrains.append(binary_spike)

        U = np.array(expanded_spiketrains)

        return t, U