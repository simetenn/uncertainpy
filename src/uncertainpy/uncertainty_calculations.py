import chaospy as cp
import numpy as np
import multiprocessing as mp

from uncertainpy import RunModel

class UncertaintyCalculations:
    def __init__(self,
                 model,
                 features,
                 rosenblatt=False,
                 CPUs=mp.cpu_count(),
                 M=3,
                 nr_pc_samples=None,
                 seed=None):

        self.model = model
        self.features = features

        self.P = None
        self.M = M
        self.distribution = None

        self.runmodel = RunModel(self.model, self.features)

        self.rosenblatt = rosenblatt

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 2

        if self.rosenblatt:
            self.nr_pc_samples -= 1

        if seed is not None:
            cp.seed(seed)
            np.random.seed(seed)


    def createDistribution(self, parameter_names=None):
        if parameter_names is None:
            parameter_space = self.model.parameters.getUncertain("distribution")
            self.data.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.data.uncertain_parameters = [parameter_name]



        self.distribution = cp.J(*parameter_space)


    def createMask(self, nodes, feature):

        if feature not in self.data.feature_list:
            raise AttributeError("Error: {} is not a feature".format(feature))

        i = 0
        masked_U = []
        masked_nodes = []
        mask = np.ones(len(self.data.U[feature]), dtype=bool)

        for result in self.data.U[feature]:
            if not isinstance(result, np.ndarray) and np.isnan(result):
                mask[i] = False
            else:
                if feature in self.data.feature_list:
                    masked_U.append(result)

                else:
                    raise AttributeError("{} is not a feature".format(feature))

            i += 1


        if len(nodes.shape) > 1:
            masked_nodes = nodes[:, mask]
        else:
            masked_nodes = nodes[mask]


        if not np.all(mask):
            # raise RuntimeWarning("Feature: {} does not yield results for all parameter combinations".format(feature))
            self.logger.warning("Feature: {} does not yield results for all parameter combinations".format(feature))

        return np.array(masked_nodes), np.array(masked_U)




    def createPCE(self):
        self.nr_pc_samples = nr_pc_samples

        # TODO find a good way to solve the parameter_name poblem


        self.P = cp.orth_ttr(self.M, self.distribution)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 2


        nodes = self.distribution.sample(self.nr_pc_samples, "M")
        # nodes, weights = cp.generate_quadrature(3, self.distribution, rule="J", sparse=True)

        # Running the model
        self.data = self.runmodel.run()

        # Calculate PC for each feature
        for feature in self.data.feature_list:
            masked_nodes, masked_U = self.createMask(nodes, feature)

            # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")

        # TODO perform for directComparison outside, since masking is not needed.?
        # # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
        # self.U_hat["directComparison"] = cp.fit_regression(self.P, nodes,
        #                                                    self.data.U["directComparison"], rule="T")




    def createPCERosenblatt(self):
        self.nr_pc_samples = nr_pc_samples

        # TODO find a good way to solve the parameter_name poblem
        if parameter_name is None:
            parameter_space = self.model.parameters.getUncertain("parameter_space")
            self.data.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.data.uncertain_parameters = [parameter_name]


        # Create the Multivariat normal distribution
        dist_MvNormal = []
        for parameter in self.model.parameters.getUncertain("value"):
            dist_MvNormal.append(cp.Normal())

        dist_MvNormal = cp.J(*dist_MvNormal)


        self.distribution = cp.J(*parameter_space)
        self.P = cp.orth_ttr(self.M, dist_MvNormal)

        if self.nr_pc_samples is None:
            self.nr_pc_samples = 2*len(self.P) + 1

        nodes_MvNormal = dist_MvNormal.sample(self.nr_pc_samples, "M")
        # nodes_MvNormal, weights_MvNormal = cp.generate_quadrature(3, dist_MvNormal,
        #                                                           rule="J", sparse=True)

        nodes = self.distribution.inv(dist_MvNormal.fwd(nodes_MvNormal))
        # weights = weights_MvNormal*\
        #    self.distribution.pdf(nodes)/dist_MvNormal.pdf(nodes_MvNormal)

        self.distribution = dist_MvNormal


        # Running the model
        solves = self.evaluateNodes(nodes)

        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        self.storeResults(solves)





        # Calculate PC for each feature
        for feature in self.data.feature_list:
            masked_nodes, masked_U = self.createMask(nodes_MvNormal, feature)

            # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
            self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                    masked_U, rule="T")

        # # perform for directComparison outside, since masking is not needed.
        # # self.U_hat = cp.fit_quadrature(self.P, nodes, weights, interpolated_solves)
        # self.U_hat["directComparison"] = cp.fit_regression(self.P, nodes,
        #                                                    self.data.U["directComparison"], rule="T")



    def calculatePC(self, nodes, U):
        U_hat = cp.fit_regression(self.P, nodes, U, rule="T")
        return U_hat


    def calculateNodes(self):
        return nodes


    def PCAnalysis(self):
        for feature in self.data.feature_list:
            self.data.E[feature] = cp.E(self.U_hat[feature], self.distribution)
            self.data.Var[feature] = cp.Var(self.U_hat[feature], self.distribution)

            samples = self.distribution.sample(self.nr_pc_mc_samples, "R")

            if len(self.data.uncertain_parameters) > 1:

                self.U_mc[feature] = self.U_hat[feature](*samples)
                self.data.sensitivity_1[feature] = cp.Sens_m(self.U_hat[feature], self.distribution)
                self.data.sensitivity_t[feature] = cp.Sens_t(self.U_hat[feature], self.distribution)

            else:
                self.U_mc[feature] = self.U_hat[feature](samples)
                self.data.sensitivity_1[feature] = None
                self.data.sensitivity_t[feature] = None


            self.data.p_05[feature] = np.percentile(self.U_mc[feature], 5, -1)
            self.data.p_95[feature] = np.percentile(self.U_mc[feature], 95, -1)

        self.totalSensitivity(sensitivity="sensitivity_1")
        self.totalSensitivity(sensitivity="sensitivity_t")



    def MCanalysis(self, parameter_name=None):
        if parameter_name is None:
            parameter_space = self.model.parameters.getUncertain("parameter_space")
            self.data.uncertain_parameters = self.model.parameters.getUncertain()
        else:
            parameter_space = [self.model.parameters.get(parameter_name).parameter_space]
            self.data.uncertain_parameters = [parameter_name]

        self.distribution = cp.J(*parameter_space)
        nodes = self.distribution.sample(self.nr_mc_samples, "M")

        solves = self.evaluateNodes(nodes)

        # Find 1d and 2d features
        # Store the results from the runs in self.U and self.t, and interpolate U if there is a t
        self.storeResults(solves)

        for feature in self.data.feature_list:
            self.data.E[feature] = np.mean(self.data.U[feature], 0)
            self.data.Var[feature] = np.var(self.data.U[feature], 0)

            self.data.p_05[feature] = np.percentile(self.data.U[feature], 5, 0)
            self.data.p_95[feature] = np.percentile(self.data.U[feature], 95, 0)
            self.data.sensitivity_1[feature] = None
            self.data.total_sensitivity_1[feature] = None
            self.data.sensitivity_t[feature] = None
            self.data.total_sensitivity_t[feature] = None


    def totalSensitivity(self, sensitivity="sensitivity_1"):

        sense = getattr(self.data, sensitivity)
        total_sense = {}

        for feature in sense:
            if sense[feature] is None:
                continue

            total_sensitivity = 0
            total_sense[feature] = []
            for i in xrange(0, len(self.data.uncertain_parameters)):
                tmp_sum_sensitivity = np.sum(sense[feature][i])

                total_sensitivity += tmp_sum_sensitivity
                total_sense[feature].append(tmp_sum_sensitivity)

            for i in xrange(0, len(self.data.uncertain_parameters)):
                if not total_sensitivity == 0:
                    total_sense[feature][i] /= float(total_sensitivity)


        setattr(self.data, "total_" + sensitivity, total_sense)



    def getNrSamples(self):
        return self.M
