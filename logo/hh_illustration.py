import uncertainpy as un

import matplotlib.pyplot as plt
from HodgkinHuxley import HodgkinHuxley
from prettyplot import prettyPlot, set_xlabel, set_ylabel

# plt.xkcd()

scale1 = 1.6
scale2 = 0.4
linewidth = 10

model = HodgkinHuxley()
# parameters_1 = {"gbar_Na": 120,
#                 "gbar_K": 36,
#                 "gbar_l": 0.5}


# time, V, info = model.run(**parameters_1)
# prettyPlot(time, V, nr_colors=3, style="seaborn-white", linewidth=linewidth)

# parameters_2 = {"gbar_Na": scale1*120,
#                 "gbar_K": scale1*36,
#                 "gbar_l": 0.5}

# time, V, info = model.run(**parameters_2)
# prettyPlot(time, V, new_figure=False, nr_colors=3, style="seaborn-white", linewidth=linewidth)



# parameters_3 = {"gbar_Na": scale2*120,
#                 "gbar_K": scale2*36,
#                 "gbar_l": scale2*0.3}

# time, V, info = model.run(**parameters_3)
# prettyPlot(time, V, new_figure=False, nr_colors=3, style="seaborn-white", linewidth=linewidth)

# set_xlabel("Time (ms)")
# set_ylabel("Voltage (mv)")
# plt.xlim([10, 55])
# plt.savefig("hh.pdf")
# plt.show()


# Define a parameter list
parameter_list = [["gbar_Na", 120],
                  ["gbar_K", 36],
                  ["gbar_l", 0.3]]

# Create the parameters
parameters = un.Parameters(parameter_list)

# Set all parameters to have a uniform distribution
# within a 50% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.25))

# Initialize the model
model = HodgkinHuxley()

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters)
UQ.quantify(plot=None, nr_pc_mc_samples=10**2)


time = UQ.data["HodgkinHuxley"].time
mean = UQ.data["HodgkinHuxley"].mean
percentile_95 = UQ.data["HodgkinHuxley"].percentile_95
percentile_5 = UQ.data["HodgkinHuxley"].percentile_5

ax = prettyPlot(time, mean, color=0, palette="deep", linewidth=2)

ax.fill_between(time,
                percentile_5,
                percentile_95,
                color=(0.45, 0.65, 0.9))

plt.savefig("hh_prediction.pdf")
plt.show()