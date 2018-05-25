import glob
from tqdm import tqdm

import uncertainpy as un
import numpy as np
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



def calculate_error(glob_pattern, correct_data, base="data/"):
    files = glob.glob(base + glob_pattern)

    mean_errors = {}
    variance_errors = {}
    for file in tqdm(files):
        data = un.Data(file)

        mean = data["valderrama"].mean
        correct_mean = correct_data["valderrama"].mean
        variance = data["valderrama"].variance
        correct_variance = correct_data["valderrama"].variance

        dt = data["valderrama"].time[1] - data["valderrama"].time[0]
        nr_evaluations = len(data["valderrama"].evaluations)

        mean_error = dt*np.sum(np.abs((correct_mean - mean)/correct_mean))
        variance_error = dt*np.sum(np.abs((correct_variance - variance)/correct_variance))

        if nr_evaluations not in mean_errors:
            mean_errors[nr_evaluations] = [mean_error]
        else:
            mean_errors[nr_evaluations].append(mean_error)

        if nr_evaluations not in variance_errors:
            variance_errors[nr_evaluations] = [variance_error]
        else:
            variance_errors[nr_evaluations].append(variance_error)

        del data

    sorted_nr_evaluations = []
    average_mean_errors = []
    average_variance_errors = []

    for evaluation in sorted(mean_errors.keys()):
        sorted_nr_evaluations.append(evaluation)
        average_mean_errors.append(np.mean(mean_errors[evaluation]))
        average_variance_errors.append(np.mean(variance_errors[evaluation]))

    return sorted_nr_evaluations, average_mean_errors, average_variance_errors




# 3 uncertain parameters
correct_data_3 =  un.Data("data/parameters_3/correct.h5")

pc_evaluations_3, pc_mean_errors_3, pc_variance_errors_3 = calculate_error("parameters_3/pc_*",  correct_data_3)
mc_evaluations_3, mc_mean_errors_3, mc_variance_errors_3 = calculate_error("parameters_3/mc_*",  correct_data_3)


# 11 uncertain parameters
correct_data_11 =  un.Data("data/parameters_11/correct.h5")

pc_evaluations_11, pc_mean_errors_11, pc_variance_errors_11 = calculate_error("parameters_11/pc_*", correct_data_11)
mc_evaluations_11, mc_mean_errors_11, mc_variance_errors_11 = calculate_error("parameters_11/mc_*", correct_data_11)


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
axes[0].set_xlim([-20, max([max(pc_evaluations_3), max(mc_evaluations_3)])])
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
axes[1].set_xlim([-20, max([max(pc_evaluations_11), max(mc_evaluations_11)])])
axes[1].legend(fontsize=fontsize)

plt.tight_layout()
plt.savefig("mc_vs_pc" + figure_format, dpi=dpi)