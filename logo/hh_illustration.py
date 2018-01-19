import uncertainpy
import matplotlib.pyplot as plt
from HodgkinHuxleyModel import HodgkinHuxleyModel
from prettyplot import prettyPlot, set_xlabel, set_ylabel

# plt.xkcd()

scale1 = 1.6
scale2 = 0.4
linewidth = 10

parameter_list = [["gbar_Na", 120, None],
                  ["gbar_K", 36, None],
                  ["gbar_l", 0.3, None]]

parameters = uncertainpy.Parameters(parameter_list)
model = HodgkinHuxleyModel(parameters=parameters)

xlabel = "time [ms]"
ylabel = "voltage [mv]"

model.run()
prettyPlot(model.t, model.values, nr_hues=3, sns_style="white", linewidth=linewidth)

parameter_list1 = {"gbar_Na": scale1*120,
                   "gbar_K": scale1*36,
                   "gbar_l": 0.5}

model.set_parameters_file(parameter_list1)
model.run()
prettyPlot(model.t, model.values, new_figure=False, nr_hues=3, sns_style="white", linewidth=linewidth)



parameter_list2 = {"gbar_Na": scale2*120,
                   "gbar_K": scale2*36,
                   "gbar_l": scale2*0.3}

model.set_parameters_file(parameter_list2)
model.run()
prettyPlot(model.t, model.values, new_figure=False, nr_hues=3, sns_style="white", linewidth=linewidth)

set_xlabel("Time [ms]")
set_ylabel("Voltage [mv]")
plt.xlim([10, 55])
plt.savefig("hh.pdf")
# plt.show()
