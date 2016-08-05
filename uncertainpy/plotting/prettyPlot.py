import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# axis_grey = (0.5, 0.5, 0.5)
axis_grey = (0.6, 0.6, 0.6)
titlesize = 20
fontsize = 14
labelsize = 16
ticksize = 5
figsize = (10, 7.5)


def set_figuresize():
    params = {"figure.figsize": figsize}

    plt.rcParams.update(params)


def set_legend(legend=None, ax=None):
    """
    legend options
    """
    params = {
        'legend.fontsize': 'medium',
        'legend.handlelength': 2.2,
        'legend.frameon': True,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        'legend.fontsize': fontsize,
        'legend.handlelength': 2.2,
        'legend.borderpad': 0.5,
        'legend.framealpha': 2,
        'legend.fancybox': True
    }
    plt.rcParams.update(params)

    if legend is not None:
        if ax is None:
            plt.legend(legend)
        else:
            ax.set_legend(legend)


def set_font():
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    #Options
    params = {'text.usetex': True,
              'font.family': 'lmodern',
            #   'axes.facecolor': "white",
              'axes.titlesize': titlesize,
              'axes.labelsize': labelsize,
              'axes.edgecolor': axis_grey,
              'axes.linewidth': 1,
              'lines.linewidth': 2,
              "xtick.major.size": ticksize,
              'xtick.color': axis_grey,
              "ytick.major.size": ticksize,
              'ytick.color': axis_grey
              }

    plt.rcParams.update(params)


def set_grid(ax, bgcolor="#EAEAF2", linecolor="w", linestyle="-", linewidth=1.3):
    """
    Set background color and grid line options
    Parameters
    ----------
    Required arguments
    ax : matplotlib.axis
        axis object
    bgcolor : str
        background color
    linecolor : str
        linecolor color
    linestyle : str
        linestyle
    linewidth : float
        linewidth
    """
    ax.set_axis_bgcolor(bgcolor)
    ax.set_axisbelow("True")
    ax.grid(True, color=linecolor, linestyle=linestyle, linewidth=linewidth,
            zorder=0)

# def set_spines_colors(ax):
#     ax.spines["top"].set_edgecolor("None")
#     ax.spines["bottom"].set_edgecolor(axis_grey)
#     ax.spines["right"].set_edgecolor("None")
#     ax.spines["left"].set_edgecolor(axis_grey)

def spines_edge_color(ax, edges={"top": "None", "bottom": axis_grey,
                                 "right": "None", "left": axis_grey}):
    """
    Set spines edge color
    Parameters
    ----------
    Required arguments
    ax : matplotlib.axis
        axis object
    edges : dictionary
        edges as keys with colors as key values
    """

    for edge, color in edges.iteritems():
        ax.spines[edge].set_edgecolor(color)


def remove_ticks(ax):
    ax.tick_params(axis="x", which="both", bottom="on", top="off",
                   labelbottom="on", color=axis_grey, labelcolor="black",
                   labelsize=labelsize)

    ax.tick_params(axis="y", which="both", right="off", left="on",
                   labelleft="on", color=axis_grey, labelcolor="black",
                   labelsize=labelsize)


def get_colormap_tableu20(color=None):
    tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    if color is None:
        return tableau20
    else:
        color = color % len(tableau20)

        return tableau20[color]



def get_colormap(nr_hues=6):
    return sns.color_palette("hls", nr_hues)


def get_current_colormap():
    return sns.color_palette()


def set_title(title, ax=None):
    if ax is None:
        plt.title(title, fontsize=titlesize)
    else:
        ax.set_title(title, fontsize=titlesize)


def set_xlabel(xlabel, ax=None, color="black"):
    if ax is None:
        plt.xlabel(xlabel, fontsize=labelsize, color=color)
    else:
        ax.set_xlabel(xlabel, fontsize=labelsize, color=color)


def set_ylabel(ylabel, ax=None, color="black"):
    if ax is None:
        plt.ylabel(ylabel, fontsize=labelsize, color=color)
    else:
        ax.set_ylabel(ylabel, fontsize=labelsize, color=color)


def set_style(sns_style="darkgrid", nr_hues=6, palette=None):
    sns.set_style(sns_style)

    if palette is None:
        sns.set_palette(get_colormap(nr_hues))
    else:
        sns.set_palette(palette)

    set_font()
    set_legend()
    set_figuresize()


def prettyPlot(x=[], y=None, title="", xlabel="", ylabel="",
               nr_hues=6, sns_style="darkgrid",
               linestyle="solid", marker=None,
               ax=None, palette=None,
               color=None, new_figure=True, **kwargs):

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
    Color of the line, given as a int.
    Uses #color from the current colormap
nr_hues : int
    the number of hues to be used with the seaborn color palette
    #hues different colors
sns_style : str
    ["darkgrid" | "whitegrid" | "dark" | "white" | "ticks"]
    which seaborn style to use base.
    Default is "darkgrid"
palette : hls | husl | matplotlib colormap | seaborn color palette
    Set the matplotlib color cycle using a seaborn palette.
    Availible seaborn palette names:
        deep, muted, bright, pastel, dark, colorblind
    Other options:
        hls, husl, any named matplotlib palette, list of colors
    Matplotlib paletes can be specified as reversed palettes by appending "_r"
    to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).
    Default is "hsl"
new_figure : bool
    If a new figure should be made, or if the plot should be made
    ontop of the last existing plot.
    Default is True.


Returns
----------
ax : matplotlib ax Object
    """

    set_style(sns_style, nr_hues=nr_hues, palette=palette)

    if ax is None:
        if new_figure:
            plt.figure()

            ax = plt.subplot(111)
        else:
            ax = plt.gca()


    if len(x) == 0:
        return ax

    spines_edge_color(ax)
    remove_ticks(ax)


    set_title(title, ax)
    set_xlabel(xlabel, ax)
    set_ylabel(ylabel, ax)


    if y is None:
        y = x
        x = range(len(y))

    # ax.plot(x, y, color=tableau20[color], linestyle=linestyle, marker=marker,
    #         markersize=8, markeredgewidth=2, linewidth=2, antialiased=True,
    #         zorder=3)

    if color is not None:
        colors = sns.color_palette()
        color = colors[color]


    ax.plot(x, y, linestyle=linestyle, marker=marker,
            markersize=8, markeredgewidth=2, linewidth=2, antialiased=True,
            zorder=3, color=color, **kwargs)

    ax.yaxis.offsetText.set_fontsize(labelsize)
    ax.yaxis.offsetText.set_color("black")
    # ax.ticklabel_format(useOffset=False)


    # ax.set_xlim([min(x), max(x)])
    # ax.set_ylim([min(y), max(y)])


    return ax




# TODO updated doc string
def prettyBar(x, error=None, index=None, colors=None, title="",
              linewidth=0, xlabels=[], ylabel="", width=0.2, new_figure=True, palette=None,
              ax=None, grid=False, sns_style="dark", nr_hues=6, error_kw=None, **kwargs):
    """
Creates pretty bar plots
    """

    set_style(sns_style, nr_hues=nr_hues, palette=palette)

    if new_figure:
        plt.figure()

        ax = plt.subplot(111)
    else:
        ax = plt.gca()

    spines_edge_color(ax)
    remove_ticks(ax)

    if index is None:
        try:
            index = np.arange(len(x))
        except TypeError:
            index = [0]


    if error_kw is None:
        error_kw = dict(ecolor=axis_grey, lw=2, capsize=10, capthick=2)


    ax.bar(index, x, yerr=error, color=sns.color_palette(), width=width,
           align='center', linewidth=linewidth, error_kw=error_kw,
           edgecolor=axis_grey, **kwargs)
    ax.set_xticks(index)
    ax.set_xticklabels(xlabels, fontsize=labelsize, rotation=0)

    # ax.set_xlim([min(x), max(x)])
    # ax.set_ylim([min(y), max(y)])

    set_title(title, ax)
    set_ylabel(ylabel, ax)


    return ax
