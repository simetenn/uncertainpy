import numpy as np

try:
    import elephant
    import quantities as pq

    prerequisites = True
except ImportError:
    prerequisites = False

from .general_network_features import GeneralNetworkFeatures


class NetworkFeatures(GeneralNetworkFeatures):
    """
    Network features of a model result, works with Nest models.

    Parameters
    ----------
    new_features : {None, callable, list of callables}
        The new features to add. The feature functions have the requirements
        stated in ``reference_feature``. If None, no features are added.
        Default is None.
    features_to_run : {"all", None, str, list of feature names}, optional
        Which features to calculate uncertainties for.
        If ``"all"``, the uncertainties are calculated for all
        implemented and assigned features.
        If None, or an empty list ``[]``, no features are
        calculated.
        If str, only that feature is calculated.
        If list of feature names, all the listed features are
        calculated. Default is ``"all"``.
    adaptive : {None, "all", str, list of feature names}, optional
        Which features that are adaptive, meaning they have a varying number of
        time points between evaluations. An interpolation is performed on
        each adaptive feature to create regular results.
        If ``"all"``, all features are set to adaptive.
        If None, or an empty list, no features are adaptive.
        If str, only that feature is adaptive.
        If list of feature names, all listed are adaptive.
        Default is None.
    labels : dictionary, optional
        A dictionary with key as the feature name and the value as a list of
        labels for each axis. The number of elements in the list corresponds
        to the dimension of the feature. Example:

        .. code-block:: Python

            new_labels = {"0d_feature": ["x-axis"],
                          "1d_feature": ["x-axis", "y-axis"],
                          "2d_feature": ["x-axis", "y-axis", "z-axis"]
                         }
    units : {None, Quantities unit}, optional
        The Quantities unit of the time in the model. If None, ms is used.
        The default is None.
    instantaneous_rate_nr_samples : int
        The number of samples used to calculate the instantaneous rate.
        Default is 50.
    isi_bin_size : int
        The size of each bin in the ``binned_isi`` method.
        Default is 1.
    corrcoef_bin_size : int
        The size of each bin in the ``corrcoef`` method.
        Default is 1.
    covariance_bin_size : int
        The size of each bin in the ``covariance`` method.
        Default is 1.
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
    features_to_run : list
        Which features to calculate uncertainties for.
    adaptive : list
        A list of the adaptive features.
    utility_methods : list
        A list of all utility methods implemented. All methods in this class
        that is not in the list of utility methods is considered to be a feature.
    labels : dictionary
        Labels for the axes of each feature, used when plotting.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    instantaneous_rate_nr_samples : int
        The number of samples used to calculate the instantaneous rate.
        Default is 50.
    isi_bin_size : int
        The size of each bin in the ``binned_isi`` method.
        Default is 1.
    corrcoef_bin_size : int
        The size of each bin in the ``corrcoef`` method.
        Default is 1.
    covariance_bin_size : int
        The size of each bin in the ``covariance`` method.
        Default is 1.

    See also
    --------
    uncertainpy.features.Features.reference_feature : reference_feature showing the requirements of a feature function.
    """
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 labels={},
                 units=None,
                 instantaneous_rate_nr_samples=50,
                 isi_bin_size=1,
                 corrcoef_bin_size=1,
                 covariance_bin_size=1,
                 verbose_level="info",
                 verbose_filename=None):

        if not prerequisites:
            raise ImportError("Network features require: elephant and quantities")

        if units is None:
            units = pq.ms

        unit_string = str(units).split()[1]

        implemented_labels = {"cv": ["Neuron nr", "Coefficient of variation"],
                              "mean_cv": ["Mean coefficient of variation"],
                              "mean_isi": ["Mean interspike interval ({})".format(unit_string)],
                              "local_variation": ["Neuron nr", "Local variation"],
                              "mean_local_variation": ["Mean local variation"],
                              "mean_firing_rate": ["Neuron nr", "Rate (Hz)"],
                              "instantaneous_rate": ["Time (ms)", "Neuron nr", "Rate (Hz)"],
                              "fanofactor": ["Fanofactor"],
                              "van_rossum_dist": ["Neuron nr", "Neuron nr", ""],
                              "victor_purpura_dist": ["Neuron nr", "Neuron nr", ""],
                              "binned_isi": ["Interspike interval ({})".format(unit_string),
                                             "Neuron nr", "Count"],
                              "corrcoef": ["Neuron nr", "Neuron nr", "Correlation coefficient"],
                              "covariance": ["Neuron nr", "Neuron nr", "Covariance"]
                             }

        implemented_labels.update(labels)

        super(NetworkFeatures, self).__init__(new_features=new_features,
                                              features_to_run=features_to_run,
                                              adaptive=adaptive,
                                              labels=implemented_labels,
                                              units=units)

        self.instantaneous_rate_nr_samples = instantaneous_rate_nr_samples
        self.isi_bin_size = isi_bin_size
        self.corrcoef_bin_size = corrcoef_bin_size
        self.covariance_bin_size = covariance_bin_size


    def cv(self, simulation_end, spiketrains):
        """
        Calculate the coefficient of variation for each neuron.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        values : array
            The coefficient of variation for each spiketrain.
        """
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.array(cv)


    def mean_cv(self, simulation_end, spiketrains):
        """
        Calculate the mean coefficient of variation.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        values : float
            The mean coefficient of variation of each spiketrain.
        """
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.mean(cv)



    def binned_isi(self, simulation_end, spiketrains):
        """
        Calculate a histogram of the interspike interval.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : array
            The center of each bin.
        binned_isi : array
            The binned interspike intervals.
        """
        binned_isi = []
        bins = np.arange(0, spiketrains[0].t_stop.magnitude + self.isi_bin_size, self.isi_bin_size)

        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi = elephant.statistics.isi(spiketrain)
                binned_isi.append(np.histogram(isi, bins=bins)[0])

            else:
                binned_isi.append(np.zeros(len(bins) - 1))

        centers = bins[1:] - 0.5
        return centers, binned_isi


    def mean_isi(self, simulation_end, spiketrains):
        """
        Calculate the mean interspike interval (isi) variation for each neuron.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        mean_isi : float
           The mean interspike interval.
        """
        isi = []
        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi.append(np.mean(elephant.statistics.isi(spiketrain)))


        return None, np.mean(isi)


    def local_variation(self, simulation_end, spiketrains):
        """
        Calculate the measure of local variation.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        local_variation : list
            The local variation for each spiketrain.
        """
        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))
            else:
                local_variation.append(None)

        return None, local_variation



    def mean_local_variation(self, simulation_end, spiketrains):
        """
        Calculate the mean of the local variation.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        mean_local_variation : float
            The mean of the local variation for each spiketrain.
        """
        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))

        return None, np.mean(local_variation)


    def mean_firing_rate(self, simulation_end, spiketrains):
        """
        Calculate the mean firing rate.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        mean_firing_rate : float
            The mean firing rate of all neurons.
        """
        mean_firing_rates = []

        for spiketrain in spiketrains:
            mean_firing_rate = elephant.statistics.mean_firing_rate(spiketrain)
            mean_firing_rate.units = pq.Hz
            mean_firing_rates.append(mean_firing_rate.magnitude)

        return None, mean_firing_rates


    def instantaneous_rate(self, simulation_end, spiketrains):
        """
        Calculate the mean instantaneous firing rate.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : array
            Time of the instantaneous firing rate.
        instantaneous_rate : float
            The instantaneous firing rate.
        """
        instantaneous_rates = []
        t = None
        for spiketrain in spiketrains:
            if len(spiketrain) > 2:
                sampling_period = spiketrain.t_stop/self.instantaneous_rate_nr_samples
                # try/except to solve problem with elephant
                try:
                    instantaneous_rate = elephant.statistics.instantaneous_rate(spiketrain, sampling_period)
                    instantaneous_rates.append(np.array(instantaneous_rate).flatten())

                    if t is None:
                        t = instantaneous_rate.times.copy()
                        t.units = self.units
                except TypeError:
                    instantaneous_rates.append(None)

            else:
                instantaneous_rates.append(None)

        if t is None:
            return None, instantaneous_rates
        else:
            return t.magnitude, instantaneous_rates


    def fanofactor(self, simulation_end, spiketrains):
        """
        Calculate the fanofactor.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        fanofactor : float
            The fanofactor.
        """
        return None, elephant.statistics.fanofactor(spiketrains)


    def van_rossum_dist(self, simulation_end, spiketrains):
        """
        Calculate van Rossum distance.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        van_rossum_dist : 2D array
            The van Rossum distance.
        """
        van_rossum_dist = elephant.spike_train_dissimilarity.van_rossum_dist(spiketrains)

        # van_rossum_dist returns 0.j imaginary parts in some cases
        van_rossum_dist = np.real_if_close(van_rossum_dist)
        if np.any(np.iscomplex(van_rossum_dist)):
            return None, None

        return None, van_rossum_dist

    def victor_purpura_dist(self, simulation_end, spiketrains):
        """
        Calculate the Victor-Purpura's distance.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        values : 2D array
            The Victor-Purpura's distance.
        """
        victor_purpura_dist = elephant.spike_train_dissimilarity.victor_purpura_dist(spiketrains)

        return None, victor_purpura_dist


    def corrcoef(self, simulation_end, spiketrains):
        """
        Calculate the pairwise Pearson's correlation coefficients.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        values : 2D array
            The pairwise Pearson's correlation coefficients.
        """
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.corrcoef_bin_size*self.units)
        corrcoef = elephant.spike_train_correlation.corrcoef(binned_sts)

        return None, corrcoef

    def covariance(self, simulation_end, spiketrains):
        """
        Calculate the pairwise covariances.

        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.

        Returns
        -------
        time : None
        values : 2D array
            The pairwise covariances.
        """
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.covariance_bin_size*self.units)
        covariance = elephant.spike_train_correlation.covariance(binned_sts)

        return None, covariance
