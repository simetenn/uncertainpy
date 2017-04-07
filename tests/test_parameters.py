import unittest
import os
import shutil
import subprocess

import chaospy as cp


from uncertainpy import Parameter, Parameters


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.parameter = Parameter("gbar_Na", 120)

        self.parameter_filename = "example_hoc.hoc"

        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "example_hoc.hoc"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_initNone(self):
        parameter = Parameter("gbar_Na", 120)

        self.assertEqual(parameter.name, "gbar_Na")
        self.assertEqual(parameter.value, 120)


    def test_initFunction(self):
        def distribution(x):
            return cp.Uniform(x - 10, x + 10)

        parameter = Parameter("gbar_Na", 120, distribution)

        self.assertEqual(parameter.name, "gbar_Na")
        self.assertEqual(parameter.value, 120)
        self.assertIsInstance(parameter.distribution, cp.Dist)


    def test_initChaospy(self):
        parameter = Parameter("gbar_Na", 120, cp.Uniform(110, 130))

        self.assertTrue(parameter.name, "gbar_Na")
        self.assertTrue(parameter.value, 120)
        self.assertIsInstance(parameter.distribution, cp.Dist)


    def test_set_distributionNone(self):
        distribution = None
        self.parameter.distribution = distribution

        self.assertIsNone(self.parameter.distribution)


    def test_set_distributionFunction(self):
        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.parameter.distribution = distribution_function

        # self.assertEqual(self.parameter.distribution, cp.Uniform(110, 130))
        self.assertIsInstance(self.parameter.distribution, cp.Dist)

        def test_set_distributionFunctionNotDistReturn(self):
            def distribution_function(x):
                return x

            with self.assertRaises(TypeError):
                self.parameter.distribution = distribution_function


    def test_set_distributionInt(self):
        distribution = 1
        with self.assertRaises(TypeError):
            self.parameter.distribution = distribution


    def test_set_distributionChaospy(self):
        distribution = cp.Uniform(110, 130)
        self.parameter.distribution = distribution

        # self.assertEqual(self.parameter.distribution, cp.Uniform(110, 130))
        self.assertIsInstance(self.parameter.distribution, cp.Dist)


    def test_set_parameters_file(self):
        parameter_file = os.path.join(self.output_test_dir, self.parameter_filename)

        shutil.copy(os.path.join(self.test_data_dir, self.parameter_filename),
                    parameter_file)

        self.parameter = Parameter("test", 120)

        self.parameter.set_parameter_file(parameter_file, 12)


        compare_file = os.path.join(self.test_data_dir, "example_hoc_control.hoc")

        result = subprocess.call(["diff", parameter_file, compare_file])
        self.assertEqual(result, 0)



    def test_reset_parameter_file(self):
        parameter_file = os.path.join(self.output_test_dir, self.parameter_filename)

        shutil.copy(os.path.join(self.test_data_dir, self.parameter_filename),
                    parameter_file)

        self.parameter = Parameter("test", 12)

        self.parameter.reset_parameter_file(parameter_file)

        compare_file = os.path.join(self.test_data_dir, "example_hoc_control.hoc")

        result = subprocess.call(["diff", parameter_file, compare_file])
        self.assertEqual(result, 0)


    def test_str(self):
        result = str(self.parameter)

        self.assertEqual(result, "gbar_Na: 120")


    def test_strUncertain(self):

        self.parameter = Parameter("gbar_Na", 120, cp.Uniform(110, 130))

        result = str(self.parameter)

        self.assertEqual(result, "gbar_Na: 120 - Uncertain")






class TestParameters(unittest.TestCase):
    def setUp(self):
        self.parameter_filename = "example_hoc.hoc"

        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.test_data_dir = os.path.join(self.folder, "data")
        self.data_file = "example_hoc.hoc"
        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_initListNone(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        parameters = Parameters(parameterlist)

        self.assertIsInstance(parameters, Parameters)
        self.assertIsInstance(parameters["gbar_Na"], Parameter)
        self.assertIsInstance(parameters["gbar_K"], Parameter)
        self.assertIsInstance(parameters["gbar_l"], Parameter)

        self.assertEqual(parameters["gbar_Na"].distribution, None)
        self.assertEqual(parameters["gbar_K"].distribution, None)
        self.assertEqual(parameters["gbar_l"].distribution, None)


    def test_initListChaospy(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        parameters = Parameters(parameterlist)


        self.assertIsInstance(parameters, Parameters)
        self.assertIsInstance(parameters["gbar_Na"], Parameter)
        self.assertIsInstance(parameters["gbar_K"], Parameter)
        self.assertIsInstance(parameters["gbar_l"], Parameter)

        self.assertIsInstance(parameters["gbar_Na"].distribution, cp.Dist)
        self.assertIsInstance(parameters["gbar_K"].distribution, cp.Dist)
        self.assertIsInstance(parameters["gbar_l"].distribution, cp.Dist)


    def test_initObject(self):
        parameterlist = [Parameter("gbar_Na", 120, cp.Uniform(110, 130)),
                         Parameter("gbar_K", 36),
                         Parameter("gbar_l", 10.3)]

        parameters = Parameters(parameterlist)

        self.assertIsInstance(parameters, Parameters)
        self.assertIsInstance(parameters["gbar_Na"], Parameter)
        self.assertIsInstance(parameters["gbar_K"], Parameter)
        self.assertIsInstance(parameters["gbar_l"], Parameter)



    def test_init_list_to_long(self):
        parameterlist = [["gbar_Na", 120, None, 1]]

        with self.assertRaises(TypeError):
            Parameters(parameterlist)


    def test_init_no_list(self):
        parameterlist = 1

        with self.assertRaises(TypeError):
            Parameters(parameterlist)

    def test_getitem(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)

        self.assertIsInstance(self.parameters["gbar_Na"], Parameter)


    def test_iter(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        parameters = Parameters(parameterlist)

        result = [parameter for parameter in parameters]

        self.assertEqual(len(result), 3)

        self.assertIsInstance(result[0], Parameter)
        self.assertIsInstance(result[1], Parameter)
        self.assertIsInstance(result[2], Parameter)


    def test_set_distribution(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)

        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.parameters.set_distribution("gbar_Na", distribution_function)

        self.assertIsInstance(self.parameters["gbar_Na"].distribution, cp.Dist)


    def set_all_distributions(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)


        def distribution_function(x):
            return cp.Uniform(x - 10, x + 10)

        self.parameters.set_all_distributions(distribution_function)

        self.assertIsInstance(self.parameters["gbar_Na"].distribution, cp.Dist)
        self.assertIsInstance(self.parameters["gbar_K"].distribution, cp.Dist)
        self.assertIsInstance(self.parameters["gbar_l"].distribution, cp.Dist)


    def test_get_from_uncertainName(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)
        result = self.parameters.get_from_uncertain()

        self.assertIn("gbar_Na", result)
        self.assertIn("gbar_K", result)
        self.assertNotIn("gbar_l", result)


    def test_get_from_uncertainValue(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)
        result = self.parameters.get_from_uncertain("value")


        self.assertIn(120, result)
        self.assertIn(36, result)
        self.assertNotIn(0.3, result)


    def test_get_from_uncertainDistribution(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, None]]

        self.parameters = Parameters(parameterlist)
        result = self.parameters.get_from_uncertain("distribution")

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], cp.Dist)
        self.assertIsInstance(result[1], cp.Dist)


    def test_getName(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        self.parameters = Parameters(parameterlist)

        result = self.parameters.get()

        self.assertIn("gbar_Na", result)
        self.assertIn("gbar_K", result)
        self.assertIn("gbar_l", result)


    def test_getValue(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]
        self.parameters = Parameters(parameterlist)

        result = self.parameters.get("value")


        self.assertIn(120, result)
        self.assertIn(36, result)
        self.assertIn(0.3, result)


    def test_getValueList(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]
        self.parameters = Parameters(parameterlist)

        result = self.parameters.get("value", ["gbar_Na", "gbar_K"])


        self.assertIn(120, result)
        self.assertIn(36, result)


    def test_getError(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        self.parameters = Parameters(parameterlist)

        with self.assertRaises(AttributeError):
            self.parameters.get("not_a_parameter")


    def test_getDistribution(self):
        parameterlist = [["gbar_Na", 120, cp.Uniform(110, 130)],
                         ["gbar_K", 36, cp.Normal(36, 1)],
                         ["gbar_l", 0.3, cp.Chi(1, 1, 0.3)]]

        self.parameters = Parameters(parameterlist)
        result = self.parameters.get("distribution")

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], cp.Dist)
        self.assertIsInstance(result[1], cp.Dist)
        self.assertIsInstance(result[2], cp.Dist)



    def test_set_parameters_file(self):
        parameter_file = os.path.join(self.output_test_dir, self.parameter_filename)

        shutil.copy(os.path.join(self.test_data_dir, self.parameter_filename),
                    parameter_file)

        parameter_change = {"soma gbar_nat": 10, "basal gbar_ih": 10, "iseg vshift2_nat": 10}

        parameterlist = [["soma gbar_nat", 284.546493, None],
                         ["basal gbar_ih", 15.709707, None],
                         ["iseg vshift2_nat", -9.802976, None]]

        self.parameters = Parameters(parameterlist)

        self.parameters.set_parameters_file(parameter_file, parameter_change)


        compare_file = os.path.join(self.test_data_dir, "example_hoc_control_parameters.hoc")

        result = subprocess.call(["diff", parameter_file, compare_file])
        self.assertEqual(result, 0)

    def test_reset_parameter_file(self):
        parameter_file = os.path.join(self.output_test_dir, self.parameter_filename)

        shutil.copy(os.path.join(self.test_data_dir, self.parameter_filename),
                    parameter_file)

        parameterlist = [["soma gbar_nat", 10, None],
                         ["basal gbar_ih", 10, None],
                         ["iseg vshift2_nat", 10, None]]

        self.parameters = Parameters(parameterlist)

        self.parameters.reset_parameter_file(parameter_file)

        compare_file = os.path.join(self.test_data_dir, "example_hoc_control_parameters.hoc")

        result = subprocess.call(["diff", parameter_file, compare_file])
        self.assertEqual(result, 0)


    def test_str(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, cp.Uniform(0.1, 0.5)]]

        parameters = Parameters(parameterlist)

        result = str(parameters)

        self.assertEqual(result, "gbar_K: 36\ngbar_Na: 120\ngbar_l: 0.3 - Uncertain")


if __name__ == "__main__":
    unittest.main()
