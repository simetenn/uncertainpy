import uncertainpy as un
import chaospy as cp
import numpy as np

import time

from valderrama import valderrama


reruns = 100
mc_evaluations = [10, 100, 200, 300, 400, 500, 1000, 1500, 2000, 10000, 20000, 50000]
polynomial_orders_3 = np.arange(1, 8)
polynomial_orders_11 = np.arange(1, 5)
correct_mc = 1000000


# Few parameters
model = un.Model(run=valderrama,
                 labels=["Time (ms)", "Membrane potential (mV)"])

# Define a parameter list
parameters = {"gbar_Na": 120,
              "gbar_K": 36,
              "gbar_L": 0.3}


# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Setup the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters)


t1 = time.time()
correct_data = UQ.quantify(method="mc", nr_mc_samples=correct_mc, plot=None, filename="correct", data_folder="data/parameters_3")
t2 = time.time()

print("time:", t2-t1)

mc_data = {}
for rerun in range(reruns):
    for nr_evaluations in mc_evaluations:
        name = "mc_" + str(nr_evaluations) + "_rerun_" + str(rerun)
        mc_data[name] = UQ.quantify(method="mc",
                                    nr_mc_samples=nr_evaluations,
                                    plot=None,
                                    filename=name,
                                    data_folder="data/parameters_3")


pc_data = {}
for polynomial_order in polynomial_orders_3:
    name = "pc_" + str(polynomial_order)
    pc_data[name] = UQ.quantify(polynomial_order=polynomial_order,
                                plot=None,
                                filename=name,
                                data_folder="data/parameters_3")



model = un.Model(run=valderrama,
                 labels=["Time (ms)", "Membrane potential (mV)"])

# Define a parameter list
parameters = {"V_0": -10,
              "C_m": 1,
              "gbar_Na": 120,
              "gbar_K": 36,
              "gbar_L": 0.3,
              "m_0": 0.0011,
              "n_0": 0.0003,
              "h_0": 0.9998,
              "E_Na": 112,
              "E_K": -12,
              "E_l": 10.613}

# Create the parameters
parameters = un.Parameters(parameters)

# Set all parameters to have a uniform distribution
# within a 20% interval around their fixed value
parameters.set_all_distributions(un.uniform(0.2))

# Setup the uncertainty quantification
UQ = un.UncertaintyQuantification(model,
                                  parameters=parameters)


correct_data = UQ.quantify(method="mc", nr_mc_samples=correct_mc, plot=None, filename="correct", data_folder="data/parameters_11")


mc_data = {}
for rerun in range(reruns):
    for nr_evaluations in mc_evaluations:
        name = "mc_" + str(nr_evaluations) + "_rerun_" + str(rerun)
        mc_data[name] = UQ.quantify(method="mc",
                                    nr_mc_samples=nr_evaluations,
                                    plot=None,
                                    filename=name,
                                    data_folder="data/parameters_11")



pc_data = {}
for polynomial_order in polynomial_orders_11:
    name = "pc_" + str(polynomial_order)
    pc_data[name] = UQ.quantify(polynomial_order=polynomial_order,
                                plot=None,
                                filename=name,
                                data_folder="data/parameters_11")


