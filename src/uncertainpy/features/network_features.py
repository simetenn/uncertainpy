import numpy as np

try:
    import elephant
    import quantities as pq

    prerequisites = True
except ImportError:
    prerequisites = False


from .general_network_features import GeneralNetworkFeatures


class NetworkFeatures(GeneralNetworkFeatures):
    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 adaptive=None,
                 labels={},
                 instantaneous_rate_nr_samples=50.,
                 isi_bin_size=1,
                 corrcoef_bin_size=1,
                 covariance_bin_size=1,
                 units=pq.ms):

        if not prerequisites:
            raise ImportError("Network features require: elephant and quantities")

        unit_string = str(units).split()[1]

        implemented_labels = {"cv": ["Neuron nr", "Coefficient of variation"],
                              "mean_cv": ["Mean coefficient of variation"],
                              "mean_isi": ["Mean interspike interval [{}]".format(unit_string)],
                              "local_variation": ["Neuron nr", "Local variation"],
                              "mean_local_variation": ["Mean local variation"],
                              "mean_firing_rate": ["Neuron nr", "Rate [Hz]"],
                              "instantaneous_rate": ["Time [ms]", "Neuron nr", "Rate [Hz]"],
                              "fanofactor": ["Fanofactor"],
                              "van_rossum_dist": ["Neuron nr", "Neuron nr", ""],
                              "victor_purpura_dist": ["Neuron nr", "Neuron nr", ""],
                              "binned_isi": ["Interspike interval [{}]".format(unit_string),
                                             "Neuron nr", "Count"],
                              "corrcoef": ["Neuron nr", "Neuron nr", ""],
                              "covariance": ["Neuron nr", "Neuron nr", ""]
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


    def cv(self, t_stop, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.array(cv)


    def mean_cv(self, t_stop, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.mean(cv)



    def binned_isi(self, t_stop, spiketrains):
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


    def mean_isi(self, t_stop, spiketrains):
        isi = []
        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi.append(np.mean(elephant.statistics.isi(spiketrain)))


        return None, np.mean(isi)


    def local_variation(self, t_stop, spiketrains):
        """
        Calculate the measure of local variation (LV) for a sequence of time intervals between events.
        """
        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))
            else:
                local_variation.append(None)

        return None, local_variation



    def mean_local_variation(self, t_stop, spiketrains):
        """
        Calculate the mean local variation (LV) for a sequence of time intervals between events.


        """
        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))

        return None, np.mean(lv)

    def mean_firing_rate(self, t_stop, spiketrains):
        mean_firing_rates = []

        for spiketrain in spiketrains:
            mean_firing_rate = elephant.statistics.mean_firing_rate(spiketrain)
            mean_firing_rate.units = pq.Hz
            mean_firing_rates.append(mean_firing_rate.magnitude)

        return None, mean_firing_rates


    def instantaneous_rate(self, t_stop, spiketrains):
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


    def fanofactor(self, t_stop, spiketrains):
        return None, elephant.statistics.fanofactor(spiketrains)


    def van_rossum_dist(self, t_stop, spiketrains):
        van_rossum_dist = elephant.spike_train_dissimilarity.van_rossum_dist(spiketrains)

        # van_rossum_dist returns 0.j imaginary parts in some cases
        van_rossum_dist = np.real_if_close(van_rossum_dist)
        if np.any(np.iscomplex(van_rossum_dist)):
            return None, None

        return None, van_rossum_dist

    def victor_purpura_dist(self, t_stop, spiketrains):
        victor_purpura_dist = elephant.spike_train_dissimilarity.victor_purpura_dist(spiketrains)

        return None, victor_purpura_dist


    def corrcoef(self, t_stop, spiketrains):
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.corrcoef_bin_size*self.units)
        corrcoef = elephant.spike_train_correlation.corrcoef(binned_sts)

        return None, corrcoef

    def covariance(self, t_stop, spiketrains):
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.covariance_bin_size*self.units)
        covariance = elephant.spike_train_correlation.covariance(binned_sts)

        return None, covariance
