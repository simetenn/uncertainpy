import matplotlib.pyplot as plt
import numpy as np


axis_grey = (0.5, 0.5, 0.5)
titlesize = 20
fontsize = 18
labelsize = 16
figsize = (10, 7.5)

# TODO avoid this function
def title(title, ax=None):
    if ax is None:
        plt.title(title, fontsize=titlesize)
    else:
        ax.set_title(title, fontsize=titlesize)

def xlabel(xlabel, ax=None):
    if ax is None:
        plt.xlabel(xlabel, fontsize=labelsize)
    else:
        ax.set_xlabel(xlabel, fontsize=labelsize)


def ylabel(ylabel, ax=None):
    if ax is None:
        plt.ylabel(ylabel, fontsize=labelsize)
    else:
        ax.set_ylabel(ylabel, fontsize=labelsize)


def prettyPlot(x=[], y=None, title="", xlabel="", ylabel="",
               color=0, linestyle="solid", marker=None, new_figure=True, grid=True):

def prettyPlot(x=[], y=None, title="", xlabel="", ylabel="",
               color=0, linestyle="solid", marker=None, new_figure=True, grid=True):

    """
prettyPlot

Creates prettier plots, just a wrapper around matplotlib plot.
Customizing several matplotlib options

Parameters
----------
Required arguments

x : sequence to plot
   x values

Optional arguments

y : sequence to plot
  y values
title : str
    Title of the plot. Default is ""
xlabel : str
    Xlabel of the plot. Default is ""
ylabel : str
    Ylabel of the plot. Default is ""
linestyle: str
    ['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
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


    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    #Options
    params = {'text.usetex': True,
              'font.family': 'lmodern',
              'axes.grid': grid,
              'grid.color': 'white',
              'grid.linewidth': 1.3,
              'grid.linestyle': '-',
              'axes.facecolor': '0.95',
              'legend.fontsize': 16,
              "legend.fancybox": True}

    plt.rcParams.update(params)


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


    if y is None:
        y = x
        x = range(len(y))

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

    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if len(x) == 0:
        return ax, tableau20

    ax.plot(x, y, color=tableau20[color], linestyle=linestyle, marker=marker,
            markersize=8, markeredgewidth=2, linewidth=2, antialiased=True,
            zorder=3)


    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])


    return ax, tableau20


def prettyBar(x, error=None, index=None, colors=None, start_color=0, title="",
              linewidth=0, xlabels=[], ylabel="", width=0.2, new_figure=True,
              ax=None, grid=False, error_kw=None, **kwargs):
        """
        Creates pretty bar plots
        """


        plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
        #Options
        params = {'text.usetex': True,
                  'font.family': 'lmodern',
                  'axes.grid': grid,
                  'grid.color': 'white',
                  'grid.linewidth': 1.3,
                  'grid.linestyle': '-',
                  'axes.facecolor': '0.95',
                  'legend.fontsize': 16}

        plt.rcParams.update(params)


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

        if index is None:
            try:
                index = np.arange(len(x))
            except TypeError:
                index = [0]


        tmp_colors = []
        if colors is None:
            j = start_color
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
        else:
            for c in colors:
                c = c % len(tableau20)
                tmp_colors.append(tableau20[int(round(c, 0))])

        if error_kw is None:
            error_kw=dict(ecolor=axis_grey, lw=2, capsize=10, capthick=2)


        ax.bar(index, x, yerr=error, width=width, align='center', color=tmp_colors, linewidth=linewidth,
               error_kw=error_kw, edgecolor=axis_grey, **kwargs)
        ax.set_xticks(index)
        ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

        ax.set_title(title, fontsize=titlesize)
        #ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        #
        # ax.set_xlim([min(x), max(x)])
        # ax.set_ylim([min(y), max(y)])


        return ax, tableau20
