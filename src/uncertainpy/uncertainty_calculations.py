import numpy as np
import multiprocess as mp
from tqdm import tqdm
import chaospy as cp

from .run_model import RunModel
from .base import ParameterBase

# Model is now potentially set two places, is that a problem?
class UncertaintyCalculations(ParameterBase):
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 CPUs=mp.cpu_count(),
                 suppress_model_graphics=True,
                 M=3,
                 nr_pc_samples=None,
                 nr_mc_samples=10*3,
                 nr_pc_mc_samples=10*5,
                 seed=None,
                 verbose_level="info",
                 verbose_filename=None):

        self.runmodel = RunModel(model=model,
                                 parameters=parameters,
                                 features=features,
                                 verbose_level=verbose_level,
                                 verbose_filename=verbose_filename,
                                 CPUs=CPUs,
                                 suppress_model_graphics=suppress_model_graphics)


        super(UncertaintyCalculations, self).__init__(parameters=parameters,
                                                      model=model,
                                                      features=features,
                                                      verbose_level=verbose_level,
                                                      verbose_filename=verbose_filename)

        self.nr_pc_samples = nr_pc_samples
        self.nr_mc_samples = nr_mc_samples
        self.nr_pc_mc_samples = nr_pc_mc_samples
        self.M = M

        self.P = None
        self.distribution = None
        self.data = None
        self.U_hat = {}
        self.U_mc = {}


        if seed is not None:
            np.random.seed(seed)


    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self.runmodel.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self.runmodel.model = self.model


    @ParameterBase.parameters.setter
    def parameters(self, new_parameters):
        ParameterBase.parameters.fset(self, new_parameters)

        self.runmodel.parameters = self.parameters


    def create_distribution(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        parameter_distributions = self.parameters.get("distribution", uncertain_parameters)

        self.distribution = cp.J(*parameter_distributions)


    def create_mask(self, nodes, feature, weights=None):
        if feature not in self.data:
            raise AttributeError("Error: {} is not a feature".format(feature))

        i = 0
        masked_U = []
        mask = np.ones(len(self.data[feature]["U"]), dtype=bool)

        # TODO use numpy masked array
        for result in self.data[feature]["U"]:
            if not isinstance(result, np.ndarray) and np.all(np.isnan(result)):
            # if result is None:
                mask[i] = False
            else:
                masked_U.append(result)

            i += 1


        if len(nodes.shape) > 1:
            masked_nodes = nodes[:, mask]
        else:
            masked_nodes = nodes[mask]

        if weights is not None:
            # TODO is this needed?
            if len(weights.shape) > 1:
                masked_weights = weights[:, mask]
            else:
                masked_weights = weights[mask]


        if not np.all(mask):
            self.logger.warning("Feature: {} does not yield".format(feature)
                                + " results for all parameter combinations")


        if weights is None:
            return np.array(masked_nodes), np.array(masked_U)
        else:
            return np.array(masked_nodes), np.array(masked_U), np.array(masked_weights)


    def convert_uncertain_parameters(self, uncertain_parameters):
        if self.model is None:
            raise RuntimeError("No model is set")

        if uncertain_parameters is None:
            uncertain_parameters = self.parameters.get_from_uncertain("name")

        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        return uncertain_parameters


    # TODO not tested
    def create_PCE_quadrature(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.create_distribution(uncertain_parameters=uncertain_parameters)

        self.P = cp.orth_ttr(self.M, self.distribution)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 2


        nodes, weights = cp.generate_quadrature(3, self.distribution, rule="J", sparse=True)

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)

        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_U, masked_weights = self.create_mask(nodes, feature, weights)

            self.U_hat[feature] = cp.fit_quadrature(self.P, masked_nodes,
                                                    masked_weights, masked_U)

        # TODO perform for directComparison outside, since masking is not needed.?
        # self.U_hat[self.model.name] = cp.fit_quadrature(self.P, nodes,
        #                                                    masked_weights,
        #                                                    self.data.U[feature])




    def create_PCE_regression(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.create_distribution(uncertain_parameters=uncertain_parameters)

        self.P = cp.orth_ttr(self.M, self.distribution)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 2

        nodes = self.distribution.sample(self.nr_pc_samples, "M")

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)


        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_U = self.create_mask(nodes, feature)

            # print masked_nodes, masked_U
            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")



        # TODO perform for directComparison outside, since masking is not needed.?
        # self.U_hat[self.model.name] = cp.fit_regression(self.P, nodes,
        #                                                    self.data.U[self.model.name],
        #                                                    rule="T")



    # TODO not tested
    def create_create_PCE_quadrature_rosenblatt(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariat normal distribution
        dist_MvNormal = []
        for parameter in self.data.uncertain_parameters:
            dist_MvNormal.append(cp.Normal())

        dist_MvNormal = cp.J(*dist_MvNormal)

        self.P = cp.orth_ttr(self.M, dist_MvNormal)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 1

        nodes_MvNormal, weights_MvNormal = cp.generate_quadrature(3, dist_MvNormal,
                                                                  rule="J", sparse=True)
        # TODO Is this correct, copy pasted from below.
        nodes = self.distribution.inv(dist_MvNormal.fwd(nodes_MvNormal))

        weights = weights_MvNormal*self.distribution.pdf(nodes)/dist_MvNormal.pdf(nodes_MvNormal)

        self.distribution = dist_MvNormal

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)

        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_U, masked_weights = self.create_mask(nodes_MvNormal,
                                                                      feature,
                                                                      weights)

            self.U_hat[feature] = cp.fit_quadrature(self.P, masked_nodes, masked_weights, masked_U)


        # # perform for directComparison outside, since masking is not needed.
        # # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
        # self.U_hat[self.model.name] = cp.fit_regression(self.P, nodes,
        #                                                    self.data.U[self.model.name],
        #                                                    rule="T")



    def create_PCE_regression_rosenblatt(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariat normal distribution
        dist_MvNormal = []
        for parameter in uncertain_parameters:
            dist_MvNormal.append(cp.Normal())

        dist_MvNormal = cp.J(*dist_MvNormal)


        self.P = cp.orth_ttr(self.M, dist_MvNormal)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 1

        nodes_MvNormal = dist_MvNormal.sample(self.nr_pc_samples, "M")
        nodes = self.distribution.inv(dist_MvNormal.fwd(nodes_MvNormal))

        self.distribution = dist_MvNormal

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)

        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_U = self.create_mask(nodes_MvNormal, feature)

            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")

        # TODO perform for directComparison outside, since masking is not needed.
        # self.U_hat[self.model.name] = cp.fit_regression(self.P, nodes_MvNormal,
        #                                                    self.data.U[self.model.name],
        #                                                    rule="T")



    def analyse_PCE(self):
        if len(self.data.uncertain_parameters) == 1:
            self.logger.info("Only 1 uncertain parameter. Sensitivity is not calculated")

        for feature in self.data:
            self.data[feature]["E"] = cp.E(self.U_hat[feature], self.distribution)
            self.data[feature]["Var"] = cp.Var(self.U_hat[feature], self.distribution)

            samples = self.distribution.sample(self.nr_pc_mc_samples, "R")

            if len(self.data.uncertain_parameters) > 1:
                self.U_mc[feature] = self.U_hat[feature](*samples)

                self.data[feature]["sensitivity_1"] = cp.Sens_m(self.U_hat[feature], self.distribution)
                self.data[feature]["sensitivity_t"] = cp.Sens_t(self.U_hat[feature], self.distribution)
                self.calculate_total_sensitivity(sensitivity="sensitivity_1")
                self.calculate_total_sensitivity(sensitivity="sensitivity_t")

            else:
                self.U_mc[feature] = self.U_hat[feature](samples)

            self.data[feature]["p_05"] = np.percentile(self.U_mc[feature], 5, -1)
            self.data[feature]["p_95"] = np.percentile(self.U_mc[feature], 95, -1)





    def create_PCE_custom(self, uncertain_parameters=None):
        raise NotImplementedError("Custom Polynomial Chaos Expansion method not implemented")


    def custom_uncertainty_quantification(self, **kwargs):
        raise NotImplementedError("Custom uncertainty calculation method not implemented")


    def polynomial_chaos(self, uncertain_parameters=None, method="regression", rosenblatt=False):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        if method == "regression":
            if rosenblatt:
                self.create_PCE_regression_rosenblatt(uncertain_parameters)
            else:
                self.create_PCE_regression(uncertain_parameters)
        elif method == "custom":
            self.create_PCE_custom(uncertain_parameters)

        # TODO add support for more methods here by using
        # try:
        #     getattr(self, method)
        # excpect AttributeError:
        #     raise NotImplementedError("{} not implemented".format{method})



        else:
            raise ValueError("No method with name {}".format(method))


        self.analyse_PCE()

        return self.data


    def monte_carlo(self, uncertain_parameters=None):
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.create_distribution(uncertain_parameters=uncertain_parameters)

        nodes = self.distribution.sample(self.nr_mc_samples, "M")

        self.data = self.runmodel.run(nodes, uncertain_parameters)


        for feature in self.data:
            self.data[feature]["E"] = np.mean(self.data[feature]["U"], 0)
            self.data[feature]["Var"] = np.var(self.data[feature]["U"], 0)

            self.data[feature]["p_05"] = np.percentile(self.data[feature]["U"], 5, 0)
            self.data[feature]["p_95"] = np.percentile(self.data[feature]["U"], 95, 0)

        return self.data


    def calculate_total_sensitivity(self, sensitivity="sensitivity_1"):
        for feature in self.data:
            if sensitivity in self.data[feature]:
                total_sensitivity = 0
                total_sense = []
                for i in range(0, len(self.data.uncertain_parameters)):
                    tmp_sum_sensitivity = np.sum(self.data[feature][sensitivity][i])

                    total_sensitivity += tmp_sum_sensitivity
                    total_sense.append(tmp_sum_sensitivity)

                for i in range(0, len(self.data.uncertain_parameters)):
                    if total_sensitivity != 0:
                        total_sense[i] /= float(total_sensitivity)

                self.data[feature]["total_" + sensitivity] = np.array(total_sense)
