import uncertainpy as un
import chaospy as cp
import numpy as np

from valderrama import valderrama

reruns = 50

exact_mc = 100000
mc_evaluations = [10, 100, 200, 300, 400, 500, 1000, 1500, 2000, 10000]
polynomial_orders_3 = np.arange(1, 8)
polynomial_orders_11 = np.arange(1, 5)


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

folder = "data/parameters_3/"

# exact_data = UQ.quantify(method="mc", nr_mc_samples=exact_mc, plot=None, save=False)

# # This is to not save all model evaluations,
# # otherwise the filesize would be on the order of Gb
# sobol_evaluations = len(exact_data["valderrama"].evaluations)
# exact_data["valderrama"].evaluations = [exact_mc, sobol_evaluations]
# exact_data.save(folder + "exact.h5")


# for rerun in range(reruns):
#     print("Rerun: ", rerun)
#     for nr_evaluations in mc_evaluations:
#         data = UQ.quantify(method="mc",
#                            nr_mc_samples=nr_evaluations,
#                            plot=None,
#                            save=False)

#         name = "mc_" + str(nr_evaluations) + "_rerun_" + str(rerun)
#         sobol_evaluations = len(data["valderrama"].evaluations)
#         data["valderrama"].evaluations = [nr_evaluations, sobol_evaluations]
#         data.save(folder + name + ".h5")


# for polynomial_order in polynomial_orders_3:
#     data = UQ.quantify(polynomial_order=polynomial_order,
#                        plot=None,
#                        save=False)

#     name = "pc_" + str(polynomial_order)
#     sobol_evaluations = len(data["valderrama"].evaluations)
#     data["valderrama"].evaluations = [sobol_evaluations, sobol_evaluations]
#     data.save(folder + name + ".h5")


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
                                  parameters=parameters, CPUs=8)


folder = "data/parameters_11/"

# exact_data = UQ.quantify(method="mc", nr_mc_samples=exact_mc, plot=None, save=False)

# # This is to not save all model evaluations,
# # otherwise the filesize would be on the order of Gb
# sobol_evaluations = len(exact_data["valderrama"].evaluations)
# exact_data["valderrama"].evaluations = [exact_mc, sobol_evaluations]
# exact_data.save(folder + "exact.h5")

for rerun in range(reruns):
    print("Rerun: ", rerun)
    for nr_evaluations in mc_evaluations:
        data = UQ.quantify(method="mc",
                           nr_mc_samples=nr_evaluations,
                           plot=None,
                           save=False)

        name = "mc_" + str(nr_evaluations) + "_rerun_" + str(rerun)
        sobol_evaluations = len(data["valderrama"].evaluations)
        data["valderrama"].evaluations = [nr_evaluations, sobol_evaluations]
        data.save(folder + name + ".h5")


# for polynomial_order in polynomial_orders_11:
#     data = UQ.quantify(polynomial_order=polynomial_order,
#                        plot=None,
#                        save=False)

#     name = "pc_" + str(polynomial_order)
#     nr_evaluations = len(data["valderrama"].evaluations)
#     sobol_evaluations = len(data["valderrama"].evaluations)
#     data["valderrama"].evaluations = [sobol_evaluations, sobol_evaluations]
#     data.save(folder + name + ".h5")
