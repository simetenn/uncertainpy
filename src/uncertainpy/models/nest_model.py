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

    The ``run`` method must either be implemented or set to a
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
    uncertainpy.models.NestModel.run
    """
    def __init__(self,
                 run_function=None,
                 adaptive=False,
                 labels=["Time (ms)", "Neuron nr", "Spiking probability"]):


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
        simulation_end : {int, float}
            The final simulation time.
        spiketrains : list
            A list of spike trains for each neuron.

        Raises
        ------
        NotImplementedError
            If no ``run`` method have been implemented or set to a function.

        Notes
        -----
        The ``run`` method must either be implemented or set to a
        function. Both options have the following requirements:

        1. **Receive parameters as input**.
            The run function takes a number of named arguments that are the
            parameters of the model.
        2. **Set the parameters of the model**.
            The arguments received as input (the model parameters) must be
            assigned to the Nest model.
        3. **Run the model**.
            The Nest model must then be run.
        4. **Return the model results**.
            Lastly we need to return the model results.

            1. **Time**.
                The first object is the final simulation time (`simulation_end`).
            2. **Model results**.
                The second object is a list of spike trains (`spiketrains`).

        The model results `simulation_end` and `spiketrains` is used to calculate
        the features.
        The model result is postprocessed to create a regular result before
        the calculating the uncertainties of the model.

        See also
        --------
        uncertainpy.model.Model.postprocess
        """
        return self._run


    def postprocess(self, simulation_end, spiketrains):
        """
        Postprocessing of the spiketrains from a Nest model.

        For each neuron, convert a spiketrain to a list of the probability for
        a spike at each timestep, as well as creating a time array. For each
        timestep in the simulation the result is 0 if there is no spike
        and 1 if there is a spike.

        Parameters
        ----------
        simulation_end : {int, float}
            The final simulation time.
        spiketrains : list
            A list of spike trains for each neuron.

        Returns
        -------
        time : array
            A time array of all time points in the Nest simulation.
        spiketrains : list
            A list of the probability for a spike at each timestep, for each
            neuron.

        Example
        -------
        In a simulation that gives the spiketrain ``[0, 2, 3]``, with a
        time resolution of 0.5 ms and ends after 4 ms,
        the resulting spike train become:
        ``[1, 0, 0, 0, 1, 0, 1, 0, 0]``.
        """

        dt = nest.GetKernelStatus()["resolution"]
        time = np.arange(0, simulation_end, dt)

        expanded_spiketrains = []
        for spiketrain in spiketrains:
            binary_spike = np.zeros(len(time))
            binary_spike[np.in1d(time, spiketrain)] = 1

            expanded_spiketrains.append(binary_spike)

        values = np.array(expanded_spiketrains)

        return time, values