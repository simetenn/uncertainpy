from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from .model import Model
from ..utils.logger import setup_module_logger, get_logger


class NeuronModel(Model):
    """
    Class for Neuron simulator models.

    Loads a Neuron simulation, runs it, and measures the voltage in the soma.

    Parameters
    ----------
    file : str, optional
        Filename of the Neuron model. Default is ``"mosinit.hoc"``.
    path : {None, str}, optional
        Path to the Neuron model. If None, the file is considered to be in the
        current folder. Default is None.
    stimulus_start : {int, float, None}, optional
        The start time of any stimulus given to the neuron model. This
        is added to the info dictionary. If None, no stimulus_start is added to
        the info dictionary. Default is None.
    stimulus_end : {int, float, None}, optional
        The end time of any stimulus given to the neuron model. This
        is added to the info dictionary. If None, no stimulus_end is added to
        the info dictionary. Default is None.
    interpolate : bool, optional
        True if the model is irregular, meaning it has a varying number of
        return values between different model evaluations, and
        an interpolation of the results is performed. Default is False.
    name : {None, str}, optional
        Name of the model, if None the model gets the name of the current class.
        Default is None.
    ignore : bool, optional
        Ignore the model results when calculating uncertainties, which means the
        uncertainty is not calculated for the model. Default is False.
    run : {None, callable}, optional
        A function that implements the model. See the ``run`` method for
        requirements of the function. Default is None.
    labels : list, optional
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
        Default is ``["Time (ms)", "Membrane potential (mv)"]``.
    suppress_graphics : bool, optional
        Suppress all graphics created by the Neuron model. Default is True.
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging to file is performed
        Default logger level is "info".
    **kwargs :
        Additional key-value pairs added to info.

    Attributes
    ----------
    run : uncertainpy.models.Model.run
    labels : list
        A list of label names for the axes when plotting the model.
        On the form ``["x-axis", "y-axis", "z-axis"]``, with the number of axes
        that is correct for the model output.
    interpolate : bool
        True if the model is irregular, meaning it has a varying number of
        return values between different model evaluations, and
        an interpolation of the results is performed. Default is False.
    suppress_graphics : bool
        Suppress all graphics created by the model.
    ignore : bool
        Ignore the model results when calculating uncertainties, which means the
        uncertainty is not calculated for the model. The model results are still
        postprocessed if a postprocessing is implemented. Default is False.

    Raises
    ------
    RuntimeError
        If no section with name ``soma`` is found in the Neuron model.

    Notes
    -----
    Measures the voltage in the section with name ``soma``.
    """
    def __init__(self,
                 file="mosinit.hoc",
                 path=None,
                 interpolate=True,
                 stimulus_start=None,
                 stimulus_end=None,
                 name=None,
                 ignore=False,
                 run=None,
                 labels=["Time (ms)", "Membrane potential (mV)"],
                 suppress_graphics=True,
                 logger_level="info",
                 **kwargs):

        super(NeuronModel, self).__init__(interpolate=interpolate,
                                          ignore=ignore,
                                          labels=labels,
                                          suppress_graphics=suppress_graphics)

        self.file = file
        self.path = path
        self.info = {}

        if stimulus_end:
            self.info["stimulus_end"] = stimulus_end

        if stimulus_start:
            self.info["stimulus_start"] = stimulus_start

        for key in kwargs:
            self.info[key] = kwargs[key]

        if run is not None:
            self.run = run

        if name:
            self.name = name

        self.time = None
        self.V = None

        setup_module_logger(class_instance=self, level=logger_level)



    def load_neuron(self):
        """
        Import neuron and load neuron simulation file.
        """
        current_dir = os.getcwd()
        os.chdir(self.path)

        try:
            import neuron
        except ImportError:
            raise ImportError("NeuronModel requires: neuron")

        self.h = neuron.h

        # self.h("forall delete_section()")
        self.h.load_file(0, self.file.encode())
        # self.h.xopen(self.file.encode())
        # for section in self.h.allsec():
        #     print(section)
        # self.h.topology()

        os.chdir(current_dir)



    # Be really careful with these. Need to make sure that all references to
    # neuron are inside this class
    def _record(self, ref_data):
        """
        Record data from a neuron simulation.
        """
        data = self.h.Vector()
        data.record(getattr(self.h, ref_data))
        return data


    def _to_array(self, hocObject):
        """
        Convert a Neuron Vector object to an array.

        Parameters
        ----------
        hocObject : A Neuron Vector object.
            A Neuron Vector object to convert to an array.

        Returns
        -------
        array : array
            The converted array.
        """
        array = np.zeros(int(round(hocObject.size())))
        hocObject.to_python(array)
        return array


    def _record_v(self):
        """
        Record voltage in the soma.

        Raises
        ------
        RuntimeError
            If no section with name ``soma`` is found in the Neuron model.
        """
        # if not hasattr(self.h, "soma"):
        #     raise RuntimeError("No section with name soma found in: {}. Unable to record from soma".format(self.name))

        if not hasattr(self.h, "voltage_soma"):
            # self.h("objref voltage_soma")
            # self.h("voltage_soma = new Vector()")

            # self.h.voltage_soma.record(self.h.soma(0.5)._ref_v)

            for section in self.h.allsec():
                if section.name().lower() == "soma":
                    self.h("objref voltage_soma")
                    self.h("voltage_soma = new Vector()")

                    self.h.voltage_soma.record(section(0.5)._ref_v)
                    break

        if not hasattr(self.h, "voltage_soma"):
            raise RuntimeError("No section with name soma found in: {}. Unable to record from soma".format(self.name))


    def _record_t(self):
        """
        Record time values
        """
        if self.time is None:
            self.time = self._record("_ref_t")



    @Model.run.setter
    def run(self, new_run):
        """
        Load and run a Neuron simulation and return the model result.

        Parameters
        ----------
        **parameters : A number of named arguments (name=value).
            The parameters of the model which are set in Neuron.

        Returns
        -------
        time : array
            Time values of the model.
        values : array
            Voltage of the neuron. Note that `values` must either be regular
            (have the same number of points for different paramaters) or be able
            to be interpolated.
        info : dictionary
            A dictionary with information needed by features.
            Efel features require ``"stimulus_start"`` and ``"stimulus_end"``
            as keys, while spiking_features require ``stimulus_start"``.
        """
        Model.run.fset(self, new_run)



    def _run(self, **parameters):
        self.load_neuron()

        self.set_parameters(parameters)

        self._record_t()
        self._record_v()

        self.h.run()

        values = np.array(self.h.voltage_soma.to_python())
        # values = self._to_array(self.V)
        time = self._to_array(self.time)

        # self.V.resize(0)
        # self.time.resize(0)

        return time, values, self.info


    def set_parameters(self, parameters):
        """
        Set parameters in the neuron model.

        Parameters
        ----------
        parameters : dict
            A dictionary with parameter names as keys and the parameter value as
            value.
        """
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))


    def postprocess(self, time, values, info):
        """
        Postprocessing of the time and results from the Neuron model is
        generally not needed. The direct model result except the info
        is returned.

        Parameters
        ----------
        time : array_like
            Time values of the Neuron model.
        values : array_like
            Voltage of the neuron.
        info : dict
            Dictionary with information needed by features.

        Returns
        -------
        time : array_like
            Time values of the Neuron model.
        values : array_like
            Voltage of the neuron.
        """
        return time, values