import h5py
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from uncertainpy.plotting.prettyplot.prettyplot import prettyPlot, prettyBar, set_latex_font, set_style, get_current_colormap

# plotting options
figure_width = 7.08
labelsize = 10
titlesize = 12
fontsize = 8
ticklabelsize = 8
linewidth = 1
figsize = (figure_width, figure_width*0.4)
figure_format = ".eps"
dpi = 300

# Load the analysed data
with h5py.File("pc_mc.h5", "r") as f:
    pc_evaluations_3 = f["pc_evaluations_3"][()]
    pc_mean_errors_3 = f["pc_mean_errors_3"][()]
    pc_variance_errors_3 = f["pc_variance_errors_3"][()]
    pc_sobol_evaluations_3 = f["pc_sobol_evaluations_3"][()]
    pc_sobol_errors_3 = f["pc_sobol_errors_3"][()]

    mc_evaluations_3 = f["mc_evaluations_3"][()]
    mc_mean_errors_3 = f["mc_mean_errors_3"][()]
    mc_variance_errors_3 = f["mc_variance_errors_3"][()]
    mc_sobol_evaluations_3 = f["mc_sobol_evaluations_3"][()]
    mc_sobol_errors_3 = f["mc_sobol_errors_3"][()]

    pc_evaluations_11 = f["pc_evaluations_11"][()]
    pc_mean_errors_11 = f["pc_mean_errors_11"][()]
    pc_variance_errors_11 = f["pc_variance_errors_11"][()]
    pc_sobol_evaluations_11 = f["pc_sobol_evaluations_11"][()]
    pc_sobol_errors_11 = f["pc_sobol_errors_11"][()]

    mc_evaluations_11 = f["mc_evaluations_11"][()]
    mc_mean_errors_11 = f["mc_mean_errors_11"][()]
    mc_variance_errors_11 = f["mc_variance_errors_11"][()]
    mc_sobol_evaluations_11 = f["mc_sobol_evaluations_11"][()]
    mc_sobol_errors_11 = f["mc_sobol_errors_11"][()]


# Plotting

set_latex_font()
set_style("seaborn-darkgrid")

fig, axes = plt.subplots(ncols=2, figsize=figsize, sharey=True)

nr_colors = 2

# 3 uncertain parameters
prettyPlot(pc_evaluations_3, pc_mean_errors_3, linewidth=linewidth, nr_colors=nr_colors, color=0, ax=axes[0])
prettyPlot(pc_evaluations_3, pc_variance_errors_3, linewidth=linewidth, nr_colors=nr_colors, color=0, linestyle="--", ax=axes[0])
prettyPlot(pc_sobol_evaluations_3, pc_sobol_errors_3, linewidth=linewidth, nr_colors=nr_colors, color=0, linestyle=":", ax=axes[0])


prettyPlot(mc_evaluations_3, mc_mean_errors_3,  linewidth=linewidth, nr_colors=nr_colors, color=1, ax=axes[0])
prettyPlot(mc_evaluations_3, mc_variance_errors_3, linewidth=linewidth, nr_colors=nr_colors, color=1, linestyle="--", ax=axes[0])
prettyPlot(mc_sobol_evaluations_3, mc_sobol_errors_3, linewidth=linewidth, nr_colors=nr_colors, color=1, linestyle=":", ax=axes[0])

axes[0].set_title("A) Three uncertain parameters", fontsize=titlesize)
axes[0].set_ylabel("Average absolute relative\nerror over time, $\\varepsilon$", fontsize=labelsize)
axes[0].set_xlabel("Number of model evaluations, $N_s$", fontsize=labelsize)
axes[0].set_yscale("log")
axes[0].set_xlim([0, 2000])
axes[0].legend(fontsize=fontsize, ncol=2)

grey = (0.5, 0.5, 0.5)
colors = get_current_colormap()
legend_elements = [Patch(facecolor=colors[0], label="PC"),
                   Patch(facecolor=colors[1], label="QMC"),
                   Line2D([0], [0], linewidth=0, label=""),
                   Line2D([0], [0], color=grey, linewidth=linewidth, label="Mean"),
                   Line2D([0], [0], color=grey, linewidth=linewidth, linestyle="--", label="Variance"),
                   Line2D([0], [0], color=grey, linewidth=linewidth, linestyle=":", label="First-order Sobol")]

axes[0].legend(handles=legend_elements, fontsize=fontsize, ncol=2)


# 11 uncertain parameters
prettyPlot(pc_evaluations_11, pc_mean_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=0, ax=axes[1])
prettyPlot(pc_evaluations_11, pc_variance_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=0, linestyle="--", ax=axes[1])
prettyPlot(pc_sobol_evaluations_11, pc_sobol_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=0, linestyle=":", ax=axes[1])

prettyPlot(mc_evaluations_11, mc_mean_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=1, ax=axes[1])
prettyPlot(mc_evaluations_11, mc_variance_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=1, linestyle="--", ax=axes[1])
prettyPlot(mc_sobol_evaluations_11, mc_sobol_errors_11, linewidth=linewidth, nr_colors=nr_colors, color=1, linestyle=":", ax=axes[1])



axes[1].set_title("B) Eleven uncertain parameters", fontsize=titlesize)
axes[1].set_ylabel("Average absolute relative\nerror over time, $\\varepsilon$", fontsize=labelsize)
axes[1].set_xlabel("Number of model evaluations, $N_s$", fontsize=labelsize)
axes[1].set_yscale("log")
axes[1].set_xlim([0, 10000])
axes[1].set_ylim([5*10**-7, 10**8])
axes[1].set_yticks([10**-6, 10**-3, 10**0, 10**3, 10**6])
axes[1].legend(handles=legend_elements, fontsize=fontsize, ncol=2)



plt.tight_layout()
plt.savefig("mc_vs_pc" + figure_format, dpi=dpi)