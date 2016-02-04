import matplotlib.pyplot as plt
import numpy as np

def prettyPlot(x, y, title="", xlabel="", ylabel="",
               color=0, new_figure=True):

    """
prettyPlot

Creates prettier plots, just a wrapper around matplotlib plot.
Customizing several matplotlib options

Parameters
----------
Required arguments

x : sequence to plot
   x values
y : sequence to plot
  y values

Optional arguments

title : str
    Title of the plot. Default is ""
xlabel : str
    Xlabel of the plot. Default is ""
ylabel : str
    Ylabel of the plot. Default is ""
color : int
    Color of the line, given as a int than then index Tablea20 colors.
    Defualt is 0.
new_figure : bool
    If a new figure should be made, or if teh plot should be made
    ontop of the last existing plot.
    Default is True.


Returns
----------
ax : matplotlib ax Object
tableau20 : list
    List of tableau20 colors
    """

    axis_grey = (0.5, 0.5, 0.5)
    titlesize = 18
    fontsize = 16
    labelsize = 14
    figsize = (10, 7.5)

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    color = color % len(tableau20)

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    plt.rcParams["figure.figsize"] = figsize
    if plt.gcf() == "None":
        plt.figure(figsize=figsize)
    else:
        if new_figure:
            plt.clf()

    ax = plt.subplot(111)

    ax.spines["top"].set_edgecolor("None")
    ax.spines["bottom"].set_edgecolor(axis_grey)
    ax.spines["right"].set_edgecolor("None")
    ax.spines["left"].set_edgecolor(axis_grey)


    ax.tick_params(axis="x", which="both", bottom="on", top="off",
                   labelbottom="on", color=axis_grey, labelcolor="black",
                   labelsize=labelsize)
    ax.tick_params(axis="y", which="both", right="off", left="on",
                   labelleft="on", color=axis_grey, labelcolor="black",
                   labelsize=labelsize)

    ax.plot(x, y, color=tableau20[color], linewidth=2, antialiased=True)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])


    return ax, tableau20


def prettyBar(x, error=None, title="", xlabels=[], ylabel="", new_figure=True):
        """
        Creates pretty bar plots
        """

        axis_grey = (0.5, 0.5, 0.5)
        titlesize = 18
        fontsize = 16
        labelsize = 14
        figsize = (10, 7.5)
        width = .2

        # These are the "Tableau 20" colors as RGB.
        tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        for i in range(len(tableau20)):
            r, g, b = tableau20[i]
            tableau20[i] = (r / 255., g / 255., b / 255.)

        plt.rcParams["figure.figsize"] = figsize
        if plt.gcf() == "None":
            plt.figure(figsize=figsize)
        else:
            if new_figure:
                plt.clf()

        ax = plt.subplot(111)

        ax.spines["top"].set_edgecolor("None")
        ax.spines["bottom"].set_edgecolor(axis_grey)
        ax.spines["right"].set_edgecolor("None")
        ax.spines["left"].set_edgecolor(axis_grey)


        ax.tick_params(axis="x", which="both", bottom="on", top="off",
                       labelbottom="on", color=axis_grey, labelcolor="black",
                       labelsize=labelsize)
        ax.tick_params(axis="y", which="both", right="off", left="on",
                       labelleft="on", color=axis_grey, labelcolor="black",
                       labelsize=labelsize)

        try:
            index = np.arange(len(x))
        except TypeError:
            index = [0]

        tmp_colors = []
        j = 0
        even = True
        for i in index:
            tmp_colors.append(tableau20[j])
            j += 2
            if j >= len(tableau20):
                if even:
                    j = 1
                    even = False
                else:
                    j = 0
                    even = True


        ax.bar(index, x, yerr=error, width=width, align='center', color=tmp_colors, linewidth=0,
               error_kw=dict(ecolor=axis_grey, lw=2, capsize=10, capthick=2))
        ax.set_xticks(index)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

        ax.set_title(title, fontsize=titlesize)
        #ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        #
        # ax.set_xlim([min(x), max(x)])
        # ax.set_ylim([min(y), max(y)])


        return ax, tableau20
