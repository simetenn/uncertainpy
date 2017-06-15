from uncertainpy.features import GeneralFeatures
import numpy as np
import elephant

import neo.core
import quantities as pq

class NetworkFeatures(GeneralFeatures):
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

        unit_string = str(pq.m).split()[1]

        implemented_labels = {"cv": ["Neuron nr", "Coefficient of variation"],
                              "mean_cv": ["mean coefficient of variation"],
                              "mean_isi": ["Neuron nr",
                                           "Mean interspike interval [{}]".format(unit_string)],
                              "lv": ["Neuron nr", "Local variation"],
                              "mean_firing_rate": ["Neuron nr", "Hz"],
                              "instantaneous_rate": ["time ms", "Neuron nr", "Hz"],
                              "fanofactor": ["fanofactor"],
                              "van_rossum_dist": ["Neuron nr", "Neuron nr", ""],
                              "victor_purpura_dist": ["Neuron nr", "Neuron nr", ""],
                              "binned_isi": ["Interspike interval [{}]".format(unit_string),
                                             "Neuron nr", "count"],
                              "corrcoeff": ["Neuron nr", "Neuron nr", ""],
                              "covariance": ["Neuron nr", "Neuron nr", ""]
                             }

        super(NetworkFeatures, self).__init__(new_features=new_features,
                                              features_to_run=features_to_run,
                                              adaptive=adaptive,
                                              labels=implemented_labels)
        self.labels = labels

        self.instantaneous_rate_nr_samples = instantaneous_rate_nr_samples
        self.isi_bin_size = isi_bin_size
        self.corrcoef_bin_size = corrcoef_bin_size
        self.covariance_bin_size = covariance_bin_size
        self.units = units


    def preprocess(self, t, spiketrains):
        neo_spiketrains = []
        for spiketrain in spiketrains:
            neo_spiketrain = neo.core.SpikeTrain(spiketrain, t_stop=t, units=self.units)
            neo_spiketrains.append(neo_spiketrain)

        return None, neo_spiketrains


    def cv(self, t, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.array(cv)


    def mean_cv(self, t, spiketrains):
        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.mean(cv)



    def binned_isi(self, t, spiketrains):
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


    def mean_isi(self, t, spiketrains):
        isi = []
        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi.append(np.mean(elephant.statistics.isi(spiketrain)))


        return None, np.mean(isi)


    def lv(self, t, spiketrains):
        lv = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                lv.append(elephant.statistics.lv(isi))
            else:
                lv.append(None)

        return None, lv


    def mean_firing_rate(self, t, spiketrains):
        mean_firing_rates = []
        for spiketrain in spiketrains:
            mean_firing_rate = elephant.statistics.mean_firing_rate(spiketrain)
            mean_firing_rate.units = pq.Hz
            mean_firing_rates.append(mean_firing_rate)

        return None, mean_firing_rates


    def instantaneous_rate(self, t, spiketrains):
        instantaneous_rates = []
        t = None
        for spiketrain in spiketrains:
            if len(spiketrain) > 2:
                sampling_period = spiketrain.t_stop/self.instantaneous_rate_nr_samples
                instantaneous_rate = elephant.statistics.instantaneous_rate(spiketrain, sampling_period)
                instantaneous_rates.append(np.array(instantaneous_rate).flatten())

                if t is None:
                    t = instantaneous_rate.times.copy()
                    t.units = self.units

            else:
                instantaneous_rates.append(np.nan)

        if t is None:
            return None, instantaneous_rates
        else:
            return t.magnitude, instantaneous_rates


    def fanofactor(self, t, spiketrains):
        return None, elephant.statistics.fanofactor(spiketrains)


    def van_rossum_dist(self, t, spiketrains):
        van_rossum_dist = elephant.spike_train_dissimilarity.van_rossum_dist(spiketrains)

        return None, van_rossum_dist

    def victor_purpura_dist(self, t, spiketrains):
        victor_purpura_dist = elephant.spike_train_dissimilarity.victor_purpura_dist(spiketrains)

        return None, victor_purpura_dist


    def corrcoef(self, t, spiketrains):
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.corrcoef_bin_size*self.units)
        corrcoef = elephant.spike_train_correlation.corrcoef(binned_sts)

        return None, corrcoef

    def covariance(self, t, spiketrains):
        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.covariance_bin_size*self.units)
        covariance = elephant.spike_train_correlation.covariance(binned_sts)

        return None, covariance
