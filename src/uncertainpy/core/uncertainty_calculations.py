import numpy as np
import multiprocess as mp
from tqdm import tqdm
import chaospy as cp
import types

from .run_model import RunModel
from .base import ParameterBase


class UncertaintyCalculations(ParameterBase):
    """
    Perform the calculations for the uncertainty quantification and
    sensitivity analysis.

    This class performs the calculations for the uncertainty quantification and
    sensitivity analysis of the model and features. It implements both
    quasi-Monte Carlo methods and polynomial chaos expansions using either
    point collocation or pseudo-spectral method. Both of the polynomial chaos
    expansion methods have support for the rosenblatt transformation to handle
    dependent variables.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}, optional
        Either None, a Parameters instance or a list of the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
        Default is None.
    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    create_PCE_custom : callable, optional
        A custom method for calculating the polynomial chaos approximation.
        For the requirements of the function see
        ``UncertaintyCalculations.create_PCE_custom``. Overwrites existing
        ``create_PCE_custom`` method.
        Default is None.
    custom_uncertainty_quantification : callable, optional
        A custom method for calculating uncertainties.
        For the requirements of the function see
        ``UncertaintyCalculations.custom_uncertainty_quantification``.
        Overwrites existing ``custom_uncertainty_quantification`` method.
        Default is None.
    CPUs : int, optional
        The number of CPUs used when calculating the model and features.
        By default all CPUs are used.
    verbose_level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level.
        Logging messages less severe than this level is ignored.
        Default is `"info"`.
    verbose_filename : {None, str}, optional
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.

    Attributes
    ----------
    model : Model or Model subclass
        The model to perform uncertainty quantification on.
    parameters : Parameters
        The uncertain parameters.
    features : Features or Features subclass
        The features of the model to perform uncertainty quantification on.
    runmodel : RunModel
        Runmodel object responsible for evaluating the model and calculating features.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.

    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.Parameter
    uncertainpy.Parameters
    uncertainpy.models.Model
    uncertainpy.core.RunModel
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 create_PCE_custom=None,
                 custom_uncertainty_quantification=None,
                 CPUs=mp.cpu_count(),
                 verbose_level="info",
                 verbose_filename=None):

        self.runmodel = RunModel(model=model,
                                 parameters=parameters,
                                 features=features,
                                 verbose_level=verbose_level,
                                 verbose_filename=verbose_filename,
                                 CPUs=CPUs)

        if create_PCE_custom is not None:
            self.create_PCE_custom = create_PCE_custom

        if custom_uncertainty_quantification is not None:
            self.custom_uncertainty_quantification = custom_uncertainty_quantification

        super(UncertaintyCalculations, self).__init__(parameters=parameters,
                                                      model=model,
                                                      features=features,
                                                      verbose_level=verbose_level,
                                                      verbose_filename=verbose_filename)

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




    def convert_uncertain_parameters(self, uncertain_parameters=None):
        """
        Converts uncertain_parameter(s) to a list of uncertain parameter(s), and
        checks if it is a legal set of uncertain parameter(s).

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The name(s) of the uncertain parameters to use. If None, a list of
            all uncertain parameters are returned.
            Default is None.

        Returns
        -------
        uncertain_parameters : list
            A list with the name of all uncertain parameters.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        See Also
        --------
        uncertainpy.Parameters
        """
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


    def create_distribution(self, uncertain_parameters=None):
        """
        Create a joint multivariate distribution for the selected parameters from
        univariate distributions.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the joint multivariate
            distribution. If None, the joint multivariate distribution for all
            uncertain parameters is created.
            Default is None.

        Returns
        -------
        distribution : chaospy.Dist
            The joint multivariate distribution for the given parameters.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        If a multivariate distribution is defined in the Parameters.distribution,
        that multivariate distribution is returned. Otherwise the joint
        multivariate distribution for the selected parameters is created from
        the univariate distributions.

        See also
        --------
        uncertainpy.Parameters
        """
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        if self.parameters.distribution is None:
            parameter_distributions = self.parameters.get("distribution", uncertain_parameters)

            distribution = cp.J(*parameter_distributions)
        else:
            distribution = self.parameters.distribution

        return distribution



    def create_mask(self, data, nodes, feature, weights=None):
        """
        Mask all model and feature evaluations that do not give results
        (anything but np.nan) and the corresponding nodes.

        Parameters
        ----------
        data : Data
            A Data object with evaluations for the model and each feature.
            Must contain `data[feature].evaluations`.
        nodes : array_like
            The nodes used to evaluate the model.
        feature : str
            Name of the feature or model to mask.
        weights : array_like, optional
            Weights corresponding to each node.
            Default is None.

        Returns
        -------
        masked_nodes : array_like
            The nodes that correspond to the evaluations with results.
        masked_evaluations : array_like
            The evaluations which have results.
        mask : boolean array
            The mask itself, used to create the masked arrays.
        masked_weights : array_like, optional
            Masked weights that correspond to evaluations with results,
            only returned when weights are given.
        """
        if feature not in data:
            raise AttributeError("Error: {} is not a feature".format(feature))

        masked_evaluations = []
        mask = np.ones(len(data[feature].evaluations ), dtype=bool)

        # TODO use numpy masked array
        for i, result in enumerate(data[feature].evaluations ):
            if np.any(np.isnan(result)):
                mask[i] = False
            else:
                masked_evaluations.append(result)


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
            return np.array(masked_nodes), np.array(masked_evaluations), mask
        else:
            return np.array(masked_nodes), np.array(masked_evaluations), mask, np.array(masked_weights)



    def create_PCE_spectral(self,
                            uncertain_parameters=None,
                            polynomial_order=3,
                            quadrature_order=None,
                            allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 3.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The model and feature do not necessarily give results for each
        node. The pseudo-spectral methods is sensitive to missing values, so
        `allow_incomplete` should be used with care.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation. Then we use the
        pseudo-spectral projection to find the expansion coefficients for the
        model and each feature of the model.

        Pseudo-spectral projection is based on least squares minimization and
        finds the expansion coefficients through numerical integration. The
        integration uses a quadrature scheme with weights and nodes. We use Leja
        quadrature with Smolyak sparse grids to reduce the number of nodes
        required. For each of the nodes we evaluate the model and calculate the
        features, and the polynomial approximation is created from these results.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        P = cp.orth_ttr(polynomial_order, distribution)

        if quadrature_order is None:
            quadrature_order = polynomial_order + 2

        nodes, weights = cp.generate_quadrature(quadrature_order,
                                                distribution,
                                                rule="J",
                                                sparse=True)

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with the pseudo-spectral method. polynomial_order={}, quadrature_order={}".format(polynomial_order, quadrature_order)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                    continue

            masked_nodes, masked_evaluations, mask, masked_weights = self.create_mask(data, nodes, feature, weights)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_quadrature(P, masked_nodes,
                                                    masked_weights, masked_evaluations)
            else:
                self.logger.warning("Uncertainty quantification is not performed " +\
                                    "for feature: {} ".format(feature) +\
                                    "due too not all parameter combinations " +\
                                    "giving a result.")

            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, distribution, data


    def create_PCE_collocation(self,
                               uncertain_parameters=None,
                               polynomial_order=3,
                               nr_collocation_nodes=None,
                               allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 3.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The model and feature do not necessarily give results for each
        node. The collocation method is robust towards missing values as long as
        the number of results that remain is high enough.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation. Then we use point collocation
        to find the expansion coefficients for the model and each feature of the
        model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the `distribution`. We evaluate the model and each feature in parallel,
        and solve the resulting set of linear equations with Tikhonov
        regularization.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        P = cp.orth_ttr(polynomial_order, distribution)

        if nr_collocation_nodes is None:
            nr_collocation_nodes = 2*len(P) + 2

        nodes = distribution.sample(nr_collocation_nodes, "M")

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with point collocation. polynomial_order={}, nr_collocation_nodes={}".format(polynomial_order, nr_collocation_nodes)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                continue

            masked_nodes, masked_evaluations, mask = self.create_mask(data, nodes, feature)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_regression(P, masked_nodes,
                                                        masked_evaluations, rule="T")
            else:
                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result. Set allow_incomplete=True to " +
                                    "calculate the uncertainties anyway.")


            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, distribution, data


    def create_PCE_spectral_rosenblatt(self,
                                       uncertain_parameters=None,
                                       polynomial_order=3,
                                       quadrature_order=None,
                                       allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection and the Rosenblatt transformation. Works for dependend
        uncertain parameters.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 3.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        `data` should contain (but not necessarily) the following, if
           applicable:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The model and feature do not necessarily give results for each
        node. The pseudo-spectral methods is sensitive to missing values, so
        `allow_incomplete` should be used with care.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        We use the Rosenblatt transformation to transform from dependent to
        independent variables before we create the polynomial chaos expansion.
        We first find the polynomials using the three-therm recurrence relation
        from the independent distributions. Then we use the
        pseudo-spectral projection with the Rosenblatt transformation to find
        the expansion coefficients for the model and each feature of the model.

        Pseudo-spectral projection is based on least squares
        minimization and finds the expansion coefficients through numerical
        integration. The integration uses a quadrature scheme with weights
        and nodes. We use Leja quadrature with Smolyak sparse grids to reduce the
        number of nodes required.
        We use the Rosenblatt transformation to transform the quadrature nodes
        before they are sent to the model evaluation.
        For each of the nodes we evaluate the model and calculate the features,
        and the polynomial approximation is created from these results.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)

        P = cp.orth_ttr(polynomial_order, dist_R)

        if quadrature_order is None:
            quadrature_order = polynomial_order + 2

        nodes_R, weights_R = cp.generate_quadrature(quadrature_order,
                                                    dist_R,
                                                    rule="G",
                                                    sparse=True)


        nodes = distribution.inv(dist_R.fwd(nodes_R))
        # weights = weights_R*distribution.pdf(nodes)/dist_R.pdf(nodes_R)

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with the pseudo-spectral method and the Rosenblatt transformation. polynomial_order={}, quadrature_order={}".format(polynomial_order, quadrature_order)


        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):

            if feature == self.model.name and self.model.ignore:
                continue

            # The tutorial version
            # masked_nodes, masked_values, mask, masked_weights = self.create_mask(data,
            #                                                           nodes_R,
            #                                                           feature,
            #                                                           weights)

            # The version thats seems to be working
            masked_nodes, masked_evaluations, mask, masked_weights = self.create_mask(data,
                                                                    nodes_R,
                                                                    feature,
                                                                    weights_R)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_quadrature(P,
                                                masked_nodes,
                                                masked_weights,
                                                masked_evaluations)
            else:
                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result. Set allow_incomplete=True to " +
                                    "calculate the uncertainties anyway.")

            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, dist_R, data


    def create_PCE_collocation_rosenblatt(self,
                                          uncertain_parameters=None,
                                          polynomial_order=3,
                                          nr_collocation_nodes=None,
                                          allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection and the Rosenblatt transformation. Works for dependend
        uncertain parameters.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 3.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The model and feature do not necessarily give results for each node. The
        collocation method is robust towards missing values as long as the number
        of results that remain is high enough.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        We use the Rosenblatt transformation to transform from dependent to
        independent variables before we create the polynomial chaos expansion.
        We first find the polynomials using the three-therm recurrence relation
        from the independent distributions. Then we use the
        point collocation with the Rosenblatt transformation to find
        the expansion coefficients for the model and each feature of the model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the independent distribution. We then transform the nodes using the
        Rosenblatte transformation and evaluate the model and each
        feature in parallel. We solve the resulting set of linear equations
        with Tikhonov regularization.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        # dist_R = cp.Iid(cp.Normal(), len(uncertain_parameters))
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)

        P = cp.orth_ttr(polynomial_order, dist_R)

        if nr_collocation_nodes is None:
            nr_collocation_nodes = 2*len(P) + 2

        nodes_R = dist_R.sample(nr_collocation_nodes, "M")
        nodes = distribution.inv(dist_R.fwd(nodes_R))

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with point collocation and the Rosenblatt transformation. polynomial_order={}, nr_collocation_nodes={}".format(polynomial_order, nr_collocation_nodes)


        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                continue

            masked_nodes, masked_evaluations, mask = self.create_mask(data, nodes_R, feature)


            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_regression(P,
                                                masked_nodes,
                                                masked_evaluations,
                                                rule="T")
            else:
                self.logger.warning("Uncertainty quantification is not performed " +
                                    "for feature: {} ".format(feature) +
                                    "due too not all parameter combinations " +
                                    "giving a result.")

            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, dist_R, data


    def analyse_PCE(self, U_hat, distribution, data, nr_samples=10**4):
        """
        Calculate the statistical metrics from the polynomial chaos
        approximation.

        Parameters
        ----------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.
        nr_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation.
            Default is 10**4.

        Returns
        -------
        data : Data
            The `data` parameter given as input with the statistical metrics added.

        Notes
        -----
        The `data` parameter should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        When returned `data` additionally contains:

            7. ``data["model/features"].mean``
            8. ``data["model/features"].variance``
            9. ``data["model/features"].percentile_5``
            10. ``data["model/features"].percentile_95``
            11. ``data["model/features"].sobol_first``, if more than 1 parameter
            12. ``data["model/features"].sobol_total``, if more than 1 parameter
            13. ``data["model/features"].sobol_first_sum``, if more than 1 parameter
            14. ``data["model/features"].sobol_total_sum``, if more than 1 parameter

        See also
        --------
        uncertainpy.Data
        """

        if len(data.uncertain_parameters) == 1:
            self.logger.info("Only 1 uncertain parameter. Sensitivity is not calculated")

        U_mc = {}
        for feature in tqdm(data,
                            desc="Calculating statistics from PCE",
                            total=len(data)):
            if feature in U_hat:
                data[feature].mean = cp.E(U_hat[feature], distribution)
                data[feature].variance = cp.Var(U_hat[feature], distribution)

                samples = distribution.sample(nr_samples, "H")

                if len(data.uncertain_parameters) > 1:
                    U_mc[feature] = U_hat[feature](*samples)

                    data[feature].sobol_first = cp.Sens_m(U_hat[feature], distribution)
                    data[feature].sobol_total = cp.Sens_t(U_hat[feature], distribution)
                    data = self.sensitivity_sum(data, sensitivity="sobol_first")
                    data = self.sensitivity_sum(data, sensitivity="sobol_total")

                else:
                    U_mc[feature] = U_hat[feature](samples)

                data[feature].percentile_5 = np.percentile(U_mc[feature], 5, -1)
                data[feature].percentile_95 = np.percentile(U_mc[feature], 95, -1)

        return data



    @property
    def create_PCE_custom(self, uncertain_parameters=None, **kwargs):
        """
        A custom method for calculating the polynomial chaos approximation.
        Must follow the below requirements.

        Parameters
        ----------
        self : UncertaintyCalculation
            An explicit self is required as the first argument.
            self can be used inside the custom function.
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        **kwargs
            Any number of optional arguments.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        This method can be implemented to create a custom method to calculate
        the polynomial chaos expansion. The method must calculate and return
        the return arguments described above.

        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The method `analyse_PCE` is called after the polynomial approximation
        has been created.

        Usefull methods in Uncertainpy are:

        1. uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters
        2. uncertainpy.core.Uncertaintycalculations.create_distribution
        3. uncertainpy.core.RunModel.run

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters : Converts uncertain parameters to allowed list
        uncertainpy.core.Uncertaintycalculations.create_distribution : Creates the uncertain parameter distribution
        uncertainpy.core.RunModel.run : Runs the model
        """
        return self._create_PCE_custom

    @create_PCE_custom.setter
    def create_PCE_custom(self, new_create_PCE_custom):
        if not callable(new_create_PCE_custom):
            raise TypeError("create_PCE_custom function must be callable")

        self._create_PCE_custom = types.MethodType(new_create_PCE_custom, self)
        # self._create_PCE_custom = new_create_PCE_custom


    def _create_PCE_custom(self, uncertain_parameters=None, **kwargs):
        raise NotImplementedError("No custom Polynomial Chaos expansion method implemented")


    @property
    def custom_uncertainty_quantification(self, **kwargs):
        """
        A custom uncertainty quantification method. Must follow the below
        requirements.

        Parameters
        ----------
        self : UncertaintyCalculation
            An explicit self is required as the first argument.
            self can be used inside the custom function.
        **kwargs
            Any number of optional arguments.

        Returns
        -------
        data : Data
            A Data object with calculated uncertainties.

        Notes
        -----
        Usefull methods in Uncertainpy are:

        1. uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters
           - Converts uncertain parameters to an allowed list.
        2. uncertainpy.core.Uncertaintycalculations.create_distribution
           - Creates the uncertain parameter distribution
        3. uncertainpy.core.RunModel.run - Runs the model and all features.

        See also
        --------
        uncertainpy.Data
        uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters : Converts uncertain parameters to list
        uncertainpy.core.Uncertaintycalculations.create_distribution : Create uncertain parameter distribution
        uncertainpy.core.RunModel.run : Runs the model
        """

        return self._custom_uncertainty_quantification

    @custom_uncertainty_quantification.setter
    def custom_uncertainty_quantification(self, new_custom_uncertainty_quantification):
        if not callable(new_custom_uncertainty_quantification):
            raise TypeError("custom_uncertainty_quantification function must be callable")

        self._custom_uncertainty_quantification = types.MethodType(new_custom_uncertainty_quantification, self)
        # self._custom_uncertainty_quantification = new_custom_uncertainty_quantification

    def _custom_uncertainty_quantification(self, **kwargs):
        raise NotImplementedError("No custom uncertainty calculation method implemented")


    def polynomial_chaos(self,
                         method="collocation",
                         rosenblatt=False,
                         uncertain_parameters=None,
                         polynomial_order=3,
                         nr_collocation_nodes=None,
                         quadrature_order=None,
                         nr_pc_mc_samples=10**4,
                         allow_incomplete=True,
                         seed=None,
                         **custom_kwargs):
        """
        Perform an uncertainty quantification and sensitivity analysis
        using polynomial chaos expansions.

        Parameters
        ----------
        method : {"collocation", "spectral", "custom"}, optional
            The method to use when creating the polynomial chaos approximation.
            "collocation" is the point collocation method "spectral" is
            pseudo-spectral projection, and "custom" is the custom polynomial
            method.
            Default is "collocation".
        rosenblatt : bool, optional
            If the Rosenblatt transformation should be used. The Rosenblatt
            transformation must be used if the uncertain parameters have
            dependent variables.
            Default is False.
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 3.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose, if point collocation is
            used. If None, `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method, if pseudo-spectral
            projection is used. If None, ``quadrature_order = polynomial_order + 2``.
            Default is None.
        nr_pc_mc_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.

        Returns
        -------
        data : Data
            A data object with all model and feature values, as well as all
            calculated statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.
        ValueError
            If `method` not one of "collocation", "spectral" or "custom".
        NotImplementedError
            If "custom" is chosen and have not been implemented.

        Notes
        -----
        The `data` parameter contains the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data["model/features"].mean``
            8. ``data["model/features"].variance``
            9. ``data["model/features"].percentile_5``
            10. ``data["model/features"].percentile_95``
            11. ``data["model/features"].sobol_first``, if more than 1 parameter
            12. ``data["model/features"].sobol_total``, if more than 1 parameter
            13. ``data["model/features"].sobol_first_sum``, if more than 1 parameter
            14. ``data["model/features"].sobol_total_sum``, if more than 1 parameter

        The model and feature do not necessarily give results for each
        node. The collocation method is robust towards missing values as long as
        the number of results that remain is high enough. The pseudo-spectral
        method on the other hand, is sensitive to missing values, so
        `allow_incomplete` should be used with care in that case.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation. Then we use point collocation
        or pseudo-spectral projection to find the expansion coefficients for the
        model and each feature of the model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the `distribution`. We evaluate the model and each feature in parallel,
        and solve the resulting set of linear equations with Tikhonov
        regularization.

        Pseudo-spectral projection is based on least squares minimization and
        finds the expansion coefficients through numerical integration. The
        integration uses a quadrature scheme with weights and nodes. We use Leja
        quadrature with Smolyak sparse grids to reduce the number of nodes
        required. For each of the nodes we evaluate the model and calculate the
        features, and the polynomial approximation is created from these results.

        If we have dependent uncertain parameters we must use the Rosenblatt
        transformation. We use the Rosenblatt transformation to transform from
        dependent to independent variables before we create the polynomial chaos
        expansion. We first find the polynomials using the three-term
        recurrence relation from the independent distributions.

        Both pseudo-spectral projection and point collocation is performed using
        the independent distribution, the only difference is that we use the
        Rosenblatt transformation to transform the nodes from the independent
        distribution to the dependent distribution.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """
        if seed is not None:
            np.random.seed(seed)

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        if method == "collocation":
            if rosenblatt:
                U_hat, distribution, data = \
                    self.create_PCE_collocation_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                           polynomial_order=polynomial_order,
                                                           nr_collocation_nodes=nr_collocation_nodes,
                                                           allow_incomplete=allow_incomplete)
            else:
                U_hat, distribution, data = \
                    self.create_PCE_collocation(uncertain_parameters=uncertain_parameters,
                                                polynomial_order=polynomial_order,
                                                nr_collocation_nodes=nr_collocation_nodes,
                                                allow_incomplete=allow_incomplete)

        elif method == "spectral":
            if rosenblatt:
                U_hat, distribution, data = \
                    self.create_PCE_spectral_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                        polynomial_order=polynomial_order,
                                                        quadrature_order=quadrature_order,
                                                        allow_incomplete=allow_incomplete)
            else:
                U_hat, distribution, data = \
                    self.create_PCE_spectral(uncertain_parameters=uncertain_parameters,
                                             polynomial_order=polynomial_order,
                                             quadrature_order=quadrature_order,
                                             allow_incomplete=allow_incomplete)

        elif method == "custom":
            U_hat, distribution, data = \
                self.create_PCE_custom(uncertain_parameters, **custom_kwargs)

        # TODO add support for more methods here by using
        # try:
        #     getattr(self, method)
        # except AttributeError:
        #     raise NotImplementedError("{} not implemented".format{method})

        else:
            raise ValueError("No polynomial chaos method with name {}".format(method))

        data = self.analyse_PCE(U_hat, distribution, data, nr_samples=nr_pc_mc_samples)

        data.seed = seed

        return data


    def monte_carlo(self,
                    uncertain_parameters=None,
                    nr_samples=10**3,
                    seed=None):
        """
        Perform an uncertainty quantification using the quasi-Monte Carlo method.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        nr_samples : int, optional
            Number of samples for the Monte Carlo sampling.
            Default is 10**3.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.

        Returns
        -------
        data : Data
            A data object with all model and feature evaluations, as well as all
            calculated statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The `data` parameter contains the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.method``
            6. ``data["model/features"].mean``
            7. ``data["model/features"].variance``
            8. ``data["model/features"].percentile_5``
            9. ``data["model/features"].percentile_95``

        In the quasi-Monte Carlo method we quasi-randomly draw 10**3 (by default)
        parameter samples using the Hammersley sequence. We evaluate the model
        for each of these parameter samples and calculate the features from each
        of the model results. This step is performed in parallel to speed up the
        calculations. Lastly we use the model and feature results to calculate
        the mean, variance, and 5th and 95th percentile for the model and each
        feature.

        Sensitivity analysis is currently not yet available for the quasi-Monte
        Carlo method.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        if seed is not None:
            np.random.seed(seed)

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        nodes = distribution.sample(nr_samples, "M")

        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "monte carlo method. nr_samples={}".format(nr_samples)
        data.seed = seed

        # TODO mask data
        for feature in data:
            if feature == self.model.name and self.model.ignore:
                continue

            data[feature].mean = np.mean(data[feature].evaluations , 0)
            data[feature].variance = np.var(data[feature].evaluations , 0)

            data[feature].percentile_5 = np.percentile(data[feature].evaluations , 5, 0)
            data[feature].percentile_95 = np.percentile(data[feature].evaluations , 95, 0)

        return data


    def sensitivity_sum(self, data, sensitivity="sobol_first"):
        """
        Calculate the normalized sum of the sensitivities for the model and all
        features and add them to `data`.

        Parameters
        ----------
        data : Data
            A data object with all model and feature evaluations, as well as all
            calculated statistical metrics.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            The sensitivity to normalize and sum. "sobol_first" and "1" are
            for the first order Sobol indice while "sobol_total" and "t" is
            for the total order Sobol indices.
            Default is "sobol_first".

        Returns
        ----------
        data : Data
            The `data` object with the normalized sum of the sensitivities for
            the model and all features added.

        See also
        --------
        uncertainpy.Data
        """
        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"

        for feature in data:
            if sensitivity in data[feature]:
                total_sensitivity = 0
                total_sense = []
                for i in range(0, len(data.uncertain_parameters)):
                    tmp_sum_sensitivity = np.sum(data[feature][sensitivity][i])

                    total_sensitivity += tmp_sum_sensitivity
                    total_sense.append(tmp_sum_sensitivity)

                for i in range(0, len(data.uncertain_parameters)):
                    if total_sensitivity != 0:
                        total_sense[i] /= float(total_sensitivity)

                data[feature][sensitivity + "_sum"] = np.array(total_sense)

        return data