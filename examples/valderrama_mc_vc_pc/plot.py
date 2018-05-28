import h5py
import matplotlib.pyplot as plt

from uncertainpy.plotting.prettyplot.prettyplot import prettyPlot, set_latex_font, set_style

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
with h5py.File("analysed_data.h5", "r") as f:
    pc_evaluations_3 = f["pc_evaluations_3"][()]
    pc_mean_errors_3 = f["pc_mean_errors_3"][()]
    pc_variance_errors_3 = f["pc_variance_errors_3"][()]

    mc_evaluations_3 = f["mc_evaluations_3"][()]
    mc_mean_errors_3 = f["mc_mean_errors_3"][()]
    mc_variance_errors_3 = f["mc_variance_errors_3"][()]

    pc_evaluations_11 = f["pc_evaluations_11"][()]
    pc_mean_errors_11 = f["pc_mean_errors_11"][()]
    pc_variance_errors_11 = f["pc_variance_errors_11"][()]

    mc_evaluations_11 = f["mc_evaluations_11"][()]
    mc_mean_errors_11 = f["mc_mean_errors_11"][()]
    mc_variance_errors_11 = f["mc_variance_errors_11"][()]



# Plotting

set_latex_font()
set_style("seaborn-darkgrid")

fig, axes = plt.subplots(ncols=2, figsize=figsize)

nr_colors = 2

# 3 uncertain parameters
prettyPlot(pc_evaluations_3, pc_mean_errors_3, nr_colors=nr_colors, color=0, ax=axes[0], label="Polynomial chaos, mean")
prettyPlot(pc_evaluations_3, pc_variance_errors_3, nr_colors=nr_colors, color=0, linestyle="--", ax=axes[0], label="Polynomial chaos, variance")
prettyPlot(mc_evaluations_3, mc_mean_errors_3, nr_colors=nr_colors, color=1, ax=axes[0], label="quasi-Monte Carlo, mean")
prettyPlot(mc_evaluations_3, mc_variance_errors_3, nr_colors=nr_colors, color=1, linestyle="--", ax=axes[0], label="quasi-Monte Carlo, variance")

axes[0].set_title("A) 3 uncertain parameters", fontsize=titlesize)
axes[0].set_ylabel("Relative error", fontsize=labelsize)
axes[0].set_xlabel("Number of model evaluations", fontsize=labelsize)
axes[0].set_yscale("log")
# axes[0].set_xlim([-20, max([max(pc_evaluations_3), max(mc_evaluations_3)])])
axes[0].set_xlim([-20, 2000])
axes[0].legend(fontsize=fontsize)



# 11 uncertain parameters
prettyPlot(pc_evaluations_11, pc_mean_errors_11, nr_colors=nr_colors, color=0, ax=axes[1], label="Polynomial chaos, mean")
prettyPlot(pc_evaluations_11, pc_variance_errors_11, nr_colors=nr_colors, color=0, linestyle="--", ax=axes[1], label="Polynomial chaos, variance")
prettyPlot(mc_evaluations_11, mc_mean_errors_11, nr_colors=nr_colors, color=1, ax=axes[1], label="quasi-Monte Carlo, mean")
prettyPlot(mc_evaluations_11, mc_variance_errors_11, nr_colors=nr_colors, color=1, linestyle="--", ax=axes[1], label="quasi-Monte Carlo, variance")


axes[1].set_title("B) 11 uncertain parameters", fontsize=titlesize)
axes[1].set_ylabel("Relative error", fontsize=labelsize)
axes[1].set_xlabel("Number of model evaluations", fontsize=labelsize)
axes[1].set_yscale("log")
#axes[1].set_xlim([-20, max([max(pc_evaluations_11), max(mc_evaluations_11)])])
axes[1].set_xlim([-20, 5000])
axes[1].legend(fontsize=fontsize)

plt.tight_layout()
plt.savefig("mc_vs_pc" + figure_format, dpi=dpi)