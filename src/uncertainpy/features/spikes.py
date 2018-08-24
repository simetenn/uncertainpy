from __future__ import absolute_import, division, print_function, unicode_literals

from ..plotting.prettyplot import prettyPlot, create_figure, get_current_colormap

import matplotlib.pyplot as plt
import numpy as np


class Spike:
    """
    A single spike found in a voltage trace.

    Parameters
    ----------
    time : array_like
        The time array of the spike.
    V : array_like
        The voltage array of the spike.
    time_spike : {float, int}
        The timing of the peak of the spike.
    V_spike : {float, int}
        The voltage at the peak of the spike.
    global_index : int
        Index of the spike peak in the simulation.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Attributes
    ----------
    time : array_like
        The time array of the spike.
    V : array_like
        The voltage array of the spike.
    time_spike : {float, int}
        The timing of the peak of the spike.
    V_spike : {float, int}
        The voltage at the peak of the spike.
    global_index : int
        Index of the spike peak in the simulation.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    """
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
        """
        Plot the spike.

        Parameters
        ----------
        save_name : {str, None}
            Name of the plot file. If None, the plot is shown instead of saved
            to disk.
            Default is None.
        """
        prettyPlot(self.time, self.V,
                   title="Spike",
                   xlabel=self.xlabel,
                   ylabel=self.ylabel)

        plt.xlim([min(self.time), max(self.time)])

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
            plt.close()


    def trim(self, threshold):
        """
        Remove the first and last values of the spike that is below `threshold`.

        Parameters
        ----------
        threshold : {float, int}
            Remove all values from each side of the spike that is bellow this
            value.
        """
        indices = np.where(self.V > threshold)[0]

        if len(indices) < 2:
            raise RuntimeError("To few occurrences of threshold values")

        self.time = self.time[indices[0]:indices[-1] + 1]
        self.V = self.V[indices[0]:indices[-1] + 1]






    def __str__(self):
        """
        Convert the spike to a readable string.

        Returns
        -------
        str
           A human readable string of the spike information.
        """

        output_str = "time: {}\nV: {}\ntime_spike: {}\nV_spike: {}\nglobal_index: {}".format(self.time, self.V, self.time_spike, self.V_spike, self.global_index)
        return output_str


    def __add__(self, other):
        """
        Join two spikes. Assumes spikes are found in the same voltage trace.
        """


        if np.all(np.isin(self.time, other.time)):
            new_time = other.time
            new_V = other.V
            new_time_spike = other.time_spike
            new_V_spike =  other.V_spike
            new_global_index = other.global_index

        elif np.all(np.isin(other.time, self.time)):
            new_time = self.time
            new_V = self.V
            new_time_spike = self.time_spike
            new_V_spike =  self.V_spike
            new_global_index = self.global_index
        else:
            if self.V_spike < other.V_spike:
                new_V_spike = other.V_spike
                new_time_spike = other.time_spike
                new_global_index = other.global_index
            else:
                new_V_spike = self.V_spike
                new_time_spike = self.time_spike
                new_global_index = self.global_index


            if self.time[0] < other.time[0]:
                first_spike = self
                last_spike = other
            else:
                first_spike = other
                last_spike = self

            remaining_V = last_spike.V[last_spike.time > first_spike.time[-1]]
            new_V = np.concatenate([first_spike.V, remaining_V])

            remaining_time = last_spike.time[last_spike.time > first_spike.time[-1]]
            new_time = np.concatenate([first_spike.time, remaining_time])

        return Spike(new_time, new_V, new_time_spike, new_V_spike, new_global_index)



class Spikes:
    """
    Finds spikes in the given voltage trace and is a container for the resulting
    Spike objects.

    Parameters
    ----------
    time : array_like
        The time of the voltage trace.
    V : array_like
        The voltage trace.
    threshold : {int, "auto"}
        The threshold for what is considered a spike. If the voltage trace rise
        above and then fall below this threshold it is considered a spike. If
        "auto" the threshold is set to the standard deviation of the voltage trace.
        Default is -30.
    extended_spikes : bool
        If the spikes should be extended past the threshold, until the
        derivative of the voltage trace is below 0.5. Default is False.
    min_extent_from_peak : int, optional
        Minimum extent of the spike as the number of indices from the peak
        of the spike. A new spike is only found if it lays outside the
        extent of the previous spike. This is usefull to increase if the
        voltage trace is noisy. Default is 1.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Attributes
    ----------
    spikes : list
        A list of Spike objects.
    nr_spikes : int
        The number of spikes.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    time : array_like
        The time of the voltage trace.
    V : array_like
        The voltage trace.

    Notes
    -----
    The spikes are found by finding where the voltage trace goes above the
    threshold, and then later falls below this threshold. The spike is
    considered to be everything within this interval.

    The spike can be extended. If `extended_spikes` is True, the spike is
    extended around the above area until the derivative of the voltage trace
    falls below 0.5. This works badly with noisy voltage traces.

    See also
    --------
    Spike : The class for a single spike.
    find_spikes : Finding spikes in the voltage trace.
    """
    def __init__(self,
                 time=None,
                 V=None,
                 threshold=-30,
                 extended_spikes=False,
                 min_extent_from_peak=1,
                 allow_overlap=False,
                 xlabel="",
                 ylabel=""):
        self.spikes = []
        self.nr_spikes = 0

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.V = None
        self.time = None

        if time is not None and V is not None:
            self.find_spikes(time, V,
                             threshold=threshold,
                             extended_spikes=extended_spikes,
                             min_extent_from_peak=min_extent_from_peak,
                             allow_overlap=allow_overlap)


    def __iter__(self):
        """
        Iterate over all spikes.

        Yields
        ------
        Spike object
            A spike object.
        """
        for spike in self.spikes:
            yield spike


    def __len__(self):
        """
        Find the number of spikes.

        Returns
        -------
        int
            The number of spikes.
        """
        return self.nr_spikes


    def __getitem__(self, i):
        """
        Return spike number `i`.

        Parameters
        ----------
        i: int
         Spike number `i`.

        Returns
        -------
        Spike object
            The spike object number `i`.
        """
        return self.spikes[i]


    def find_spikes(self, time, V,
                    threshold=-30,
                    end_threshold=-40,
                    extended_spikes=False,
                    min_extent_from_peak=1,
                    allow_overlap=False):
        """
        Finds spikes in the given voltage trace.

        Parameters
        ----------
        time : array_like
            The time of the voltage trace.
        V : array_like
            The voltage trace.
        threshold : {int, "auto"}
            The threshold for what is considered a spike. If the voltage trace rise
            above and then fall below this threshold it is considered a spike. If
            "auto" the threshold is set to the standard deviation of the voltage trace.
            Default is -30.
        extended_spikes : bool, optional
            If the spikes should be extended past the threshold, until the
            derivative of the voltage trace is below 0.5. Default is False.
        min_extent_from_peak : int, optional
            Minimum extent of the spike as the number of indices from the peak
            of the spike. A new spike is only found if it lays outside the
            extent of the previous spike. This is usefull to increase if the
            voltage trace is noisy. Default is 1.
        allow_overlap : bool, optional
            If spikes are allowed to overlap.

        Notes
        -----
        The spikes are added to ``self.spikes`` and ``self.nr_spikes`` is
        updated.

        The spikes are found by finding where the voltage trace goes above the
        threshold, and then later falls below this threshold. The spike is
        considered to be everything within this interval.

        The spike can be extended. If `extended_spikes` is True, the spike is
        extended around the above area until the derivative of the voltage trace
        falls below 0.5. This works badly with noisy voltage traces.
        """
        self.time = time
        self.V = V

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

            elif V[i] < end_threshold and start_flag is True:
                start_flag = False
                spike_end = i + 1

                time_spike = time[spike_start:spike_end]
                V_spike = V[spike_start:spike_end]

                spike_index = np.argmax(V_spike)
                global_index = spike_index + spike_start
                time_max = time[global_index]
                V_max = V[global_index]

                # Discard the first spike if the spike max is at the first
                # point in the voltage trace
                if global_index == 0:
                    prev_spike_end = spike_end
                    continue


                if extended_spikes:
                    spike_start = gt_derivative[(gt_derivative > prev_spike_end) & (gt_derivative < global_index)][0]
                    spike_end = self.consecutive(lt_derivative[lt_derivative > global_index])[-1] + 1

                else:
                    if global_index - min_extent_from_peak < spike_start:
                        spike_start = global_index - min_extent_from_peak

                    if global_index + min_extent_from_peak + 1 > spike_end:
                        spike_end = global_index + min_extent_from_peak + 1


                time_spike = time[spike_start:spike_end]
                V_spike = V[spike_start:spike_end]


                # print("dat")
                # if len(self.spikes) >= 1:
                #     print(self.spikes[-1].time)
                #     print(self.spikes[0])


                # # Add two spikes if they are not allowed to overlapp and they
                # # overlap
                # spike = Spike(time_spike, V_spike, time_max, V_max, global_index)
                # if len(self.spikes) >= 1 and (not allow_overlap and time[spike_start] <= self.spikes[-1].time[-1]):
                #     self.spikes[-1] = self.spikes[-1] + spike
                # else:
                #     self.spikes.append(spike)


                # Only add a new spike if either does not overlap the
                # previous spike, or if the max V of this spike is greater than
                # the max V of the previous spike
                spike = Spike(time_spike, V_spike, time_max, V_max, global_index)
                if len(self.spikes) >= 1 and not allow_overlap and time[spike_start] <= self.spikes[-1].time[-1]:
                    # Replace old spike with new spike, else ignore the current spike
                    if np.max(V_spike) > np.max(self.spikes[-1].V_spike):
                        self.spikes[-1] = spike
                else:
                    self.spikes.append(spike)

                prev_spike_end = spike_end

        self.nr_spikes = len(self.spikes)




    def consecutive(self, data):
        """
        Returns the first consecutive array, from a discontinuous index array
        such as [2, 3, 4, 5, 12, 13, 14], which returns [2, 3, 4, 5]

        Parameters
        ----------
        data : array_like

        Returns
        -------
        array_like
            The first consecutive array
        """

        result = [data[0]]
        d_prev = data[0]
        for d in data[1:]:
            if d_prev + 1 != d:
                return result
            d_prev = d

        return result


    def plot_spikes(self, save_name=None):
        """
        Plot all spikes.

        Parameters
        ----------
        save_name : {str, None}
            Name of the plot file. If None, the plot is shown instead of saved
            to disk. Default is None.
        """
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
        plt.xlim([0, max(time_max)*1.2])
        plt.legend(labels)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
            plt.close()


    def plot_voltage(self, save_name):
        """
        Plot the voltage with the peak of each spike marked.

        Parameters
        ----------
        save_name : {str, None}
            Name of the plot file. If None, the plot is shown instead of saved
            to disk. Default is None.
        """
        ax = prettyPlot(self.time, self.V,
                        title=self.ylabel,
                        ylabel=self.ylabel,
                        xlabel="Time",
                        palette="deep")

        colors = get_current_colormap()

        for spike in self.spikes:
            ax.axvline(self.time[spike.global_index], color=colors[2])

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
            plt.close()