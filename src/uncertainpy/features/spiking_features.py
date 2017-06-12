import scipy.interpolate
import scipy.optimize

from uncertainpy.features import GeneralSpikingFeatures

class SpikingFeatures(GeneralSpikingFeatures):
    def __init__(self,
                 features_to_run="all",
                 adaptive=None,
                 thresh=-30,
                 extended_spikes=False,
                 labels={}):

        implemented_labels = {"nr_spikes": ["number of spikes"],
                              "spike_rate": ["spike rate [1/ms]"],
                              "time_before_first_spike": ["time [ms]"],
                              "accomondation_index": ["accomondation index"],
                              "average_AP_overshoot": ["voltage [mV]"],
                              "average_AHP_depth": ["voltage [mV]"],
                              "average_AP_width": ["voltage [mV]"]
                              }

        super(SpikingFeatures, self).__init__(features_to_run=features_to_run,
                                              adaptive=adaptive,
                                              thresh=thresh,
                                              extended_spikes=extended_spikes,
                                              labels=implemented_labels)
        self.labels = labels


    def nr_spikes(self, t, spikes):
        return None, spikes.nr_spikes


    # TODO get time of stimulus onset from model
    def time_before_first_spike(self, t, spikes):
        if spikes.nr_spikes <= 0:
            return None, None

        return None, spikes.spikes[0].t_spike


    def spike_rate(self, t, spikes):
        if spikes.nr_spikes < 0:
            return None, None

        return None, spikes.nr_spikes/float(t[-1] - t[0])


    def average_AP_overshoot(self, t, spikes):
        if spikes.nr_spikes <= 0:
            return None, None

        sum_AP_overshoot = 0
        for spike in spikes:
            sum_AP_overshoot += spike.U_spike
        return None, sum_AP_overshoot/float(spikes.nr_spikes)


    def average_AHP_depth(self, t, spikes):
        if spikes.nr_spikes <= 0:
            return None, None

        sum_AHP_depth = 0
        for i in range(spikes.nr_spikes - 1):
            sum_AHP_depth += min(self.U[spikes[i].global_index:spikes[i+1].global_index])

        return None, sum_AHP_depth/float(spikes.nr_spikes)


    def average_AP_width(self, t, spikes):
        if spikes.nr_spikes <= 0:
            return None, None

        sum_AP_width = 0
        for spike in spikes:
            U_width = (spike.U_spike + spike.U[0])/2.

            U_interpolation = scipy.interpolate.interp1d(spike.t, spike.U - U_width)

            # root1 = scipy.optimize.fsolve(U_interpolation, (spike.t_spike - spike.t[0])/2. + spike.t[0])
            # root2 = scipy.optimize.fsolve(U_interpolation, (spike.t[-1] - spike.t_spike)/2. + spike.t_spike)

            root1 = scipy.optimize.brentq(U_interpolation, spike.t[0], spike.t_spike)
            root2 = scipy.optimize.brentq(U_interpolation, spike.t_spike, spike.t[-1])

            sum_AP_width += abs(root2 - root1)

        return None, sum_AP_width/float(spikes.nr_spikes)


    def accomondation_index(self, t, spikes):
        N = spikes.nr_spikes
        if N <= 1:
            return None, None

        k = min(4, int(round(N-1)/5.))

        ISIs = []
        for i in range(N-1):
            ISIs.append(spikes[i+1].t_spike - spikes[i].t_spike)

        A = 0
        for i in range(k+1, N-1):
            A += (ISIs[i] - ISIs[i-1])/(ISIs[i] + ISIs[i-1])
        return None, A/(N - k - 1)
