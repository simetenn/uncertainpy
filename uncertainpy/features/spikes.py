from prettyplot import prettyPlot

import pylab as plt
import numpy as np


class Spike:
    def __init__(self, t, U, t_spike, U_spike, global_index,
                 xlabel="", ylabel=""):
        self.t = t
        self.U = U

        self.U_spike = U_spike
        self.t_spike = t_spike

        self.global_index = global_index

        self.xlabel = xlabel
        self.xlabel = xlabel

    def plot(self, save_name=None):
        prettyPlot(self.t, self.U,
                   title="Spike",
                   xlabel=self.xlabel,
                   ylabel=self.xlabel)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)




class Spikes:
    def __init__(self, xlabel="", ylabel=""):
        self.spikes = []
        self.nr_spikes = 0

        self.xlabel = xlabel
        self.ylabel = ylabel


    def __iter__(self):
        for spike in self.spikes:
            yield spike


    def __len__(self):
        return self.nr_spikes


    def __getitem__(self, i):
        return self.spikes[i]


    def detectSpikes(self, t, U, thresh=-30, extended_spikes=False):

        min_dist_from_peak = 1
        derivative_cutoff = 0.5

        self.spikes = []
        if thresh == "auto":
            thresh = np.sqrt(U.var())


        spike_start = 0
        start_flag = False
        if extended_spikes:
            dUdt = np.gradient(U)
            gt_derivative = np.where(dUdt >= -derivative_cutoff)[0]
            lt_derivative = np.where(dUdt <= derivative_cutoff)[0]

        for i in range(len(U)):
            if U[i] > thresh and start_flag is False:
                spike_start = i
                start_flag = True
                continue

            elif U[i] < thresh and start_flag is True:
                spike_end = i + 1
                start_flag = False

                t_spike = t[spike_start:spike_end]
                U_spike = U[spike_start:spike_end]

                spike_index = np.argmax(U_spike)
                global_index = spike_index + spike_start
                t_max = t[global_index]
                U_max = U[global_index]

                if extended_spikes:
                    spike_start = lt_derivative[np.where(lt_derivative < global_index - min_dist_from_peak)][-1]
                    spike_end = gt_derivative[np.where(gt_derivative > global_index + min_dist_from_peak)][0]

                else:
                    if global_index - min_dist_from_peak < spike_start:
                        spike_start = global_index - min_dist_from_peak

                    if global_index + min_dist_from_peak + 1 > spike_end:
                        spike_end = global_index + min_dist_from_peak + 1

                t_spike = t[spike_start:spike_end]
                U_spike = U[spike_start:spike_end]

                self.spikes.append(Spike(t_spike, U_spike, t_max, U_max, global_index))


        self.nr_spikes = len(self.spikes)


    def plot(self, save_name=None):
        u_max = []
        u_min = []
        t_max = []
        labels = []

        i = 1
        for spike in self.spikes:
            u_max.append(max(spike.U))
            u_min.append(min(spike.U))
            t_max.append(len(spike.t))

            prettyPlot(range(len(spike.t)), spike.U,
                       title="Spikes",
                       xlabel="index",
                       ylabel=self.ylabel,
                       new_figure=False,
                       nr_hues=self.nr_spikes)

            labels.append("spike %d" % (i))
            i += 1


        plt.ylim([min(u_min), max(u_max)])
        plt.xlim([0, max(t_max)*1.25])
        plt.legend(labels)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
