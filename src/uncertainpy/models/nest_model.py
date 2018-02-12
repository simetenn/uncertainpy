try:
    import nest

    prerequisites = True
except ImportError:
    prerequisites = False

import numpy as np

from .model import Model


class NestModel(Model):
    """
    Class for NEST simulator models.

    The ``run`` method must either be implemented or set to a
    function, and is responsible for running the NEST model.

    Parameters
    ----------
    adaptive : bool, optional
        True if the model is adaptive, meaning it has a varying number of
        time values. False if not. Default is False.
    labels : list, optional
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
        Default is ``["Time (ms)", "Neuron nr", "Spiking probability"]``.
    run : {None, function}, optional
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
                 run=None,
                 adaptive=False,
                 labels=["Time (ms)", "Neuron nr", "Spiking probability"]):


        if not prerequisites:
            raise ImportError("NestModel requires: nest")

        super(NestModel, self).__init__(run=run,
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
            the NEST model.

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

        1. **Input.**
           The model function takes a number of arguments which define the
           uncertain parameters of the model.

        2. **Run the model.**
           The NEST model must then be run using the parameters given as arguments.

        3. **Output.**
           The model function must return:

            1. **Time** (``simulation_end``).
               The final simulation time of the NEST model.

            2. **Model output** (``spiketrains``).
               A list if spike trains from each recorded neuron.


        The model results `simulation_end` and `spiketrains` are used to calculate
        the features, and is postprocessed to create a regular result before
        the calculating the uncertainty of the model.

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
        time resolution of 0.5 ms and that ends after 4 ms,
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