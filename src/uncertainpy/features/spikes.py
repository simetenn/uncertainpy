from ..plotting.prettyplot import prettyPlot, create_figure

import matplotlib.pyplot as plt
import numpy as np


class Spike:
    def __init__(self, time, V, time_spike, V_spike, global_index,
                 xlabel="", ylabel=""):
        self.time = time
        self.V = V

        self.V_spike = V_spike
        self.time_spike = time_spike

        self.global_index = global_index

        self.xlabel = xlabel
        self.ylabel = ylabel


    def plot(self, save_name=None):
        prettyPlot(self.time, self.V,
                   title="Spike",
                   xlabel=self.xlabel,
                   ylabel=self.ylabel)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)




class Spikes:
    def __init__(self,
                 time=None,
                 V=None,
                 threshold=-30,
                 extended_spikes=False,
                 xlabel="",

                 ylabel=""):
        self.spikes = []
        self.nr_spikes = 0

        self.xlabel = xlabel
        self.ylabel = ylabel

        if time is not None and V is not None:
            self.find_spikes(time, V, threshold=threshold, extended_spikes=extended_spikes)


    def __iter__(self):
        for spike in self.spikes:
            yield spike


    def __len__(self):
        return self.nr_spikes


    def __getitem__(self, i):
        return self.spikes[i]


    def find_spikes(self, time, V, threshold=-30, extended_spikes=False):

        min_dist_from_peak = 1
        derivative_cutoff = 0.5

        self.spikes = []
        if threshold == "auto":
            threshold = np.sqrt(V.var())


        spike_start = 0
        start_flag = False

        if extended_spikes:
            dVdt = np.gradient(V)

            gt_derivative = np.where(dVdt >= derivative_cutoff)[0]
            lt_derivative = np.where(dVdt <= -derivative_cutoff)[0]

        prev_spike_end = 0

        for i in range(len(V)):
            if V[i] > threshold and start_flag is False:
                spike_start = i
                start_flag = True
                continue

            elif V[i] < threshold and start_flag is True:
                spike_end = i + 1
                start_flag = False

                time_spike = time[spike_start:spike_end]
                V_spike = V[spike_start:spike_end]

                spike_index = np.argmax(V_spike)
                global_index = spike_index + spike_start
                time_max = time[global_index]
                V_max = V[global_index]

                if extended_spikes:
                    spike_start = gt_derivative[(gt_derivative > prev_spike_end) & (gt_derivative < global_index)][0]
                    spike_end = self.consecutive(lt_derivative[lt_derivative > global_index])[-1] + 1

                else:
                    if global_index - min_dist_from_peak < spike_start:
                        spike_start = global_index - min_dist_from_peak

                    if global_index + min_dist_from_peak + 1 > spike_end:
                        spike_end = global_index + min_dist_from_peak + 1

                time_spike = time[spike_start:spike_end]
                V_spike = V[spike_start:spike_end]


                self.spikes.append(Spike(time_spike, V_spike, time_max, V_max, global_index))
                prev_spike_end = spike_end

        self.nr_spikes = len(self.spikes)



    def consecutive(self, data):
        result = [data[0]]
        d_prev = data[0]
        for d in data[1:]:
            if d_prev + 1 != d:
                return result
            d_prev = d

        return result


    def plot(self, save_name=None):
        V_max = []
        V_min = []
        time_max = []
        labels = []

        i = 1

        if self.nr_spikes == 0:
            raise RuntimeWarning("No spikes to plot")

        create_figure(nr_colors=self.nr_spikes)

        for spike in self.spikes:
            V_max.append(max(spike.V))
            V_min.append(min(spike.V))
            time_max.append(len(spike.time))

            prettyPlot(range(len(spike.time)), spike.V,
                       title="Spikes",
                       xlabel="index",
                       ylabel=self.ylabel,
                       new_figure=False,
                       nr_colors=self.nr_spikes)

            labels.append("spike %d" % (i))
            i += 1


        plt.ylim([min(V_min), max(V_max)])
        plt.xlim([0, max(time_max)*1.25])
        plt.legend(labels)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
