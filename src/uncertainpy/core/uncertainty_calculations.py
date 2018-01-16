import numpy as np
import multiprocess as mp
from tqdm import tqdm
import chaospy as cp

from .run_model import RunModel
from .base import ParameterBase


class UncertaintyCalculations(ParameterBase):
    """
    Class for performing the uncertainty calculations.

    This is the class that performs the uncertainty quantifications and
    sensitivity analysis of the model and features. It implements both
    quasi-Monte Carlo methods as well as polynomial chaos expansions using
    point collocation and the pseudo-spectral method. Both of the polynomial
    chaos expansion methods have support for the rosenblatt transformation to
    handle dependent variables.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on.
        Default is None.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}, optional
        Either None, a Parameters instance or a list the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
        Default is None.
    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    CPUs : int
        The number of CPUs used when calculating the model and features.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.
    suppress_model_graphics : bool, optional
        Suppress all model graphics created by the model.
        Default is True.


    Attributes
    ----------
    model : uncertainpy.Model or subclass of uncertainpy.Model
        The model to perform uncertainty quantification on.
    parameters : uncertainpy.Parameters
        The uncertain parameters.
    features : uncertainpy.Features or subclass of uncertainpy.Features
        The features of the model to perform uncertainty quantification on.
    runmodel : uncertainpy.core.RunModel
        Runmodel object responsible for evaluating the model and calculating features.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    P : chaospy.Poly object
        Polynomial to use when approximating the model U.
    distribution : chaospy.Dist
        Distribution of the uncertain parameters.
    data : uncertainpy.Data
        Stored data calculated in the uncertainty quantification.
    U_hat : chaospy.Poly
        Polynomial chaos approximation of the model.
    U_mc : array
        Monte Carlo evaluations of the polynomial chaos approximation of the model.

    See Also
    --------
    uncertainpy.features.Features : General features class
    uncertainpy.Parameter : Parameter class
    uncertainpy.Parameters : Parameters collection class
    uncertainpy.models.Model : Model class
    uncertainpy.core.RunModel : RunModel class
    """
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 CPUs=mp.cpu_count(),
                 suppress_model_graphics=True,
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
        self.P = None
        self.distribution = None
        self.data = None
        self.U_hat = {}
        self.U_mc = {}


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
        """
        Create a joint multivariate distribution for the selected parameters from
        univariate distributions.

        Parameters
        ----------
        uncertain_parameters : {None, list}, optional
            If a multivariate distribution is defined in the Parameters, return
            that multivariate distribution. Otherwise create the joint
            multivariate distribution for the selected parameters from their
            univariate distributions. If uncertain_parameters is None, the
            joint multivariate distribution for all parameters is returned.
            Default is None.

        Returns
        -------
        """
        if self.parameters.distribution is None:
            uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

            parameter_distributions = self.parameters.get("distribution", uncertain_parameters)

            distribution = cp.J(*parameter_distributions)
        else:
            self.logger.warning("A multivariate distribution is defined. Returns that distribution.")
            distribution = self.parameters.distribution

        return distribution



    def create_mask(self, nodes, feature, weights=None):
        if feature not in self.data:
            raise AttributeError("Error: {} is not a feature".format(feature))

        masked_values = []
        mask = np.ones(len(self.data[feature]["values"]), dtype=bool)

        # TODO use numpy masked array
        for i, result in enumerate(self.data[feature]["values"]):
            if np.any(np.isnan(result)):
                mask[i] = False
            else:
                masked_values.append(result)


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
            self.logger.warning("Feature: {} only yields ".format(feature) +
                                "results for {}/{} ".format(sum(mask), len(mask)) +
                                "parameter combinations.")


        if weights is None:
            return np.array(masked_nodes), np.array(masked_values), mask
        else:
            return np.array(masked_nodes), np.array(masked_values), np.array(masked_weights), mask



    def convert_uncertain_parameters(self, uncertain_parameters):
        if isinstance(uncertain_parameters, str):
            uncertain_parameters = [uncertain_parameters]

        if self.parameters.distribution is not None:
            if uncertain_parameters is None:
                uncertain_parameters = self.parameters.get("name")
            elif sorted(uncertain_parameters) != sorted(self.parameters.get("name")):
                 raise ValueError("A common multivariate distribution is given, " +
                                  "and all uncertain parameters must be used. " +
                                  "Set uncertain_parameters to None or a list of all " +
                                  "uncertain parameters.")
        else:
            if uncertain_parameters is None:
                uncertain_parameters = self.parameters.get_from_uncertain("name")

        return uncertain_parameters



    def create_PCE_spectral(self,
                            uncertain_parameters=None,
                            polynomial_order=3,
                            quadrature_order=None,
                            allow_incomplete=False):

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        self.P = cp.orth_ttr(polynomial_order, self.distribution)

        if quadrature_order is None:
            quadrature_order = polynomial_order + 2

        nodes, weights = cp.generate_quadrature(quadrature_order,
                                                self.distribution,
                                                rule="J",
                                                sparse=True)

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)

        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_values, masked_weights, mask = self.create_mask(nodes, feature, weights)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                self.U_hat[feature] = cp.fit_quadrature(self.P, masked_nodes,
                                                        masked_weights, masked_values)
            else:
                self.logger.warning("Uncertainty quantification is not performed " +\
                                    "for feature: {} ".format(feature) +\
                                    "due too not all parameter combinations " +\
                                    "giving a result. Set allow_incomplete=True to " +\
                                    "calculate the uncertainties anyway.")

            if not np.all(mask):
                self.data.incomplete.append(feature)



    def create_PCE_collocation(self,
                               uncertain_parameters=None,
                               polynomial_order=3,
                               nr_collocation_nodes=None,
                               allow_incomplete=False):

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        self.P = cp.orth_ttr(polynomial_order, self.distribution)

        if nr_collocation_nodes is None:
            nr_collocation_nodes = 2*len(self.P) + 2

        nodes = self.distribution.sample(nr_collocation_nodes, "M")

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)


        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            masked_nodes, masked_values, mask = self.create_mask(nodes, feature)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                self.U_hat[feature] = cp.fit_regression(self.P, masked_nodes,
                                                        masked_values, rule="T")
            else:
                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result. Set allow_incomplete=True to " +
                                    "calculate the uncertainties anyway.")


            if not np.all(mask):
                self.data.incomplete.append(feature)



    def create_PCE_spectral_rosenblatt(self,
                                       uncertain_parameters=None,
                                       polynomial_order=3,
                                       quadrature_order=None,
                                       allow_incomplete=False):

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)

        self.P = cp.orth_ttr(polynomial_order, dist_R)

        if quadrature_order is None:
            quadrature_order = polynomial_order + 2

        nodes_R, weights_R = cp.generate_quadrature(quadrature_order,
                                                    dist_R,
                                                    rule="J",
                                                    sparse=True)

        # TODO Is this correct, copy pasted from below.
        nodes = self.distribution.inv(dist_R.fwd(nodes_R))
        weights = weights_R*self.distribution.pdf(nodes)/dist_R.pdf(nodes_R)

        self.distribution = dist_R

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)


        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):

            # The tutorial version
            # masked_nodes, masked_values, masked_weights, mask = self.create_mask(nodes_R,
            #                                                           feature,
            #                                                           weights)

            # The version thats seems to be working
            masked_nodes, masked_values, masked_weights, mask = self.create_mask(nodes_R,
                                                                      feature,
                                                                      weights_R)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                self.U_hat[feature] = cp.fit_quadrature(self.P,
                                                        masked_nodes,
                                                        masked_weights,
                                                        masked_values)

                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result. Set allow_incomplete=True to " +
                                    "calculate the uncertainties anyway.")

            if not np.all(mask):
                self.data.incomplete.append(feature)



    def create_PCE_collocation_rosenblatt(self,
                                          uncertain_parameters=None,
                                          polynomial_order=3,
                                          nr_collocation_nodes=None,
                                          allow_incomplete=False):

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        # dist_R = cp.Iid(cp.Normal(), len(uncertain_parameters))
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)


        self.P = cp.orth_ttr(polynomial_order, dist_R)

        if nr_collocation_nodes is None:
            nr_collocation_nodes = 2*len(self.P) + 2

        nodes_R = dist_R.sample(nr_collocation_nodes, "M")
        nodes = self.distribution.inv(dist_R.fwd(nodes_R))

        self.distribution = dist_R

        # Running the model
        self.data = self.runmodel.run(nodes, uncertain_parameters)

        # Calculate PC for each feature
        for feature in tqdm(self.data,
                            desc="Calculating PC for each feature",
                            total=len(self.data)):
            # masked_nodes, masked_values, mask = self.create_mask(nodes_R, feature)
            masked_nodes, masked_values, mask = self.create_mask(nodes_R, feature)


            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                self.U_hat[feature] = cp.fit_regression(self.P,
                                                        masked_nodes,
                                                        masked_values,
                                                        rule="T")
            else:
                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result. Set allow_incomplete=True to " +
                                    "calculate the uncertainties anyway.")

            if not np.all(mask):
                self.data.incomplete.append(feature)


    def analyse_PCE(self,
                    nr_samples=10**4):

        if len(self.data.uncertain_parameters) == 1:
            self.logger.info("Only 1 uncertain parameter. Sensitivity is not calculated")

        for feature in self.data:
            if feature in self.U_hat:
                self.data[feature]["mean"] = cp.E(self.U_hat[feature], self.distribution)
                self.data[feature]["variance"] = cp.Var(self.U_hat[feature], self.distribution)

                samples = self.distribution.sample(nr_samples, "H")

                if len(self.data.uncertain_parameters) > 1:
                    self.U_mc[feature] = self.U_hat[feature](*samples)

                    self.data[feature]["sensitivity_1"] = cp.Sens_m(self.U_hat[feature], self.distribution)
                    self.data[feature]["sensitivity_t"] = cp.Sens_t(self.U_hat[feature], self.distribution)
                    self.calculate_sensitivity_sum(sensitivity="sensitivity_1")
                    self.calculate_sensitivity_sum(sensitivity="sensitivity_t")

                else:
                    self.U_mc[feature] = self.U_hat[feature](samples)

                self.data[feature]["percentile_5"] = np.percentile(self.U_mc[feature], 5, -1)
                self.data[feature]["percentile_95"] = np.percentile(self.U_mc[feature], 95, -1)





    def create_PCE_custom(self, uncertain_parameters=None):
        raise NotImplementedError("Custom Polynomial Chaos Expansion method not implemented")


    def custom_uncertainty_quantification(self, **kwargs):
        raise NotImplementedError("Custom uncertainty calculation method not implemented")


    def polynomial_chaos(self,
                         uncertain_parameters=None,
                         method="collocation",
                         rosenblatt=False,
                         polynomial_order=3,
                         nr_collocation_nodes=None,
                         quadrature_order=4,
                         nr_pc_mc_samples=10**4,
                         allow_incomplete=False,
                         seed=None,
                         **custom_kwargs):

        if seed is not None:
            np.random.seed(seed)

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        if method == "collocation":
            if rosenblatt:
                self.create_PCE_collocation_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                       polynomial_order=polynomial_order,
                                                       nr_collocation_nodes=nr_collocation_nodes,
                                                       allow_incomplete=allow_incomplete)
            else:
                self.create_PCE_collocation(uncertain_parameters=uncertain_parameters,
                                            polynomial_order=polynomial_order,
                                            nr_collocation_nodes=nr_collocation_nodes,
                                            allow_incomplete=allow_incomplete)

        elif method == "spectral":
            if rosenblatt:
                self.create_PCE_spectral_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                    polynomial_order=polynomial_order,
                                                    quadrature_order=quadrature_order,
                                                    allow_incomplete=allow_incomplete)
            else:
                self.create_PCE_spectral(uncertain_parameters=uncertain_parameters,
                                         polynomial_order=polynomial_order,
                                         quadrature_order=quadrature_order,
                                         allow_incomplete=allow_incomplete)

        elif method == "custom":
            self.create_PCE_custom(uncertain_parameters, **custom_kwargs)

        # TODO add support for more methods here by using
        # try:
        #     getattr(self, method)
        # except AttributeError:
        #     raise NotImplementedError("{} not implemented".format{method})

        else:
            raise ValueError("No polynomial chaos method with name {}".format(method))

        self.analyse_PCE(nr_samples=nr_pc_mc_samples)

        return self.data


    def monte_carlo(self,
                    uncertain_parameters=None,
                    nr_samples=10**3,
                    seed=None):

        if seed is not None:
            np.random.seed(seed)

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        self.distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        nodes = self.distribution.sample(nr_samples, "M")

        self.data = self.runmodel.run(nodes, uncertain_parameters)


        # TODO mask data
        for feature in self.data:
            self.data[feature]["mean"] = np.mean(self.data[feature]["values"], 0)
            self.data[feature]["variance"] = np.var(self.data[feature]["values"], 0)

            self.data[feature]["percentile_5"] = np.percentile(self.data[feature]["values"], 5, 0)
            self.data[feature]["percentile_95"] = np.percentile(self.data[feature]["values"], 95, 0)

        return self.data


    def calculate_sensitivity_sum(self, sensitivity="sensitivity_1"):
        if sensitivity not in ["sensitivity_1", "sensitivity_t", "1", "t"]:
            raise ValueError("Sensitivity must be either: sensitivity_1, sensitivity_t, 1, or t, not {}".format(sensitivity))

        if sensitivity == "1":
            sensitivity = "sensitivity_1"
        elif sensitivity == "t":
            sensitivity = "sensitivity_t"

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

                self.data[feature][sensitivity + "_sum"] = np.array(total_sense)
