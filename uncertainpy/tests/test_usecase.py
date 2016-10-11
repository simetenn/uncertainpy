import unittest
import os
import shutil

import uncertainpy


class TestUseCases(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.seed = 10

        self.folder = os.path.dirname(os.path.realpath(__file__))


        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def list_files(self):
        result = []
        for dir_content in os.walk(self.output_test_dir):
            for filename in dir_content[2]:
                result.append(os.path.join(os.path.sep.join(dir_content[0].split(os.path.sep)[1:]),
                                           filename))

        result.sort()

        return result


    def test_CoffeeCupPointModelExploreParameters(self):
        parameterlist = [["kappa", -0.05, None],
                         ["u_env", 20, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.CoffeeCupPointModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list=None,
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)

        result = self.list_files()

        files = ['uniform_0.1/CoffeeCupPointModel',
                 'uniform_0.1/CoffeeCupPointModel_single-parameter-kappa',
                 'uniform_0.1/CoffeeCupPointModel_single-parameter-u_env',
                 'uniform_0.2/CoffeeCupPointModel',
                 'uniform_0.2/CoffeeCupPointModel_single-parameter-kappa',
                 'uniform_0.2/CoffeeCupPointModel_single-parameter-u_env',
                 'uniform_0.3/CoffeeCupPointModel',
                 'uniform_0.3/CoffeeCupPointModel_single-parameter-kappa',
                 'uniform_0.3/CoffeeCupPointModel_single-parameter-u_env']

        self.assertEqual(files, result)




    def test_CoffeeCupPointModelCompareMC(self):
        parameterlist = [["kappa", -0.05, None],
                         ["u_env", 20, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.CoffeeCupPointModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)

        result = self.list_files()

        files = ['mc_10/CoffeeCupPointModel',
                 'mc_100/CoffeeCupPointModel',
                 'pc/CoffeeCupPointModel']

        self.assertEqual(files, result)



    def test_HodkinHuxleyModelExploreParameters(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.HodkinHuxleyModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)

        result = self.list_files()

        files = ['uniform_0.1/HodkinHuxleyModel',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_K',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_Na',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_l',
                 'uniform_0.2/HodkinHuxleyModel',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_K',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_Na',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_l',
                 'uniform_0.3/HodkinHuxleyModel',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_K',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_Na',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_l']

        self.assertEqual(files, result)


    def test_HodkinHuxleyModelCompareMC(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.HodkinHuxleyModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)


        result = self.list_files()

        files = ['mc_10/HodkinHuxleyModel',
                 'mc_100/HodkinHuxleyModel',
                 'pc/HodkinHuxleyModel']

        self.assertEqual(files, result)


    def test_IzhikevichModelExploreParameters(self):
        parameterlist = [["a", 0.02, None],
                         ["b", 0.2, None],
                         ["c", -65, None],
                         ["d", 8, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.IzhikevichModel(parameters)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)

        result = self.list_files()

        files = ['uniform_0.1/IzhikevichModel',
                 'uniform_0.1/IzhikevichModel_single-parameter-a',
                 'uniform_0.1/IzhikevichModel_single-parameter-b',
                 'uniform_0.1/IzhikevichModel_single-parameter-c',
                 'uniform_0.1/IzhikevichModel_single-parameter-d',
                 'uniform_0.2/IzhikevichModel',
                 'uniform_0.2/IzhikevichModel_single-parameter-a',
                 'uniform_0.2/IzhikevichModel_single-parameter-b',
                 'uniform_0.2/IzhikevichModel_single-parameter-c',
                 'uniform_0.2/IzhikevichModel_single-parameter-d',
                 'uniform_0.3/IzhikevichModel',
                 'uniform_0.3/IzhikevichModel_single-parameter-a',
                 'uniform_0.3/IzhikevichModel_single-parameter-b',
                 'uniform_0.3/IzhikevichModel_single-parameter-c',
                 'uniform_0.3/IzhikevichModel_single-parameter-d']

        self.assertEqual(files, result)


    def test_IzhikevichModelCompareMC(self):
        parameterlist = [["a", 0.02, None],
                         ["b", 0.2, None],
                         ["c", -65, None],
                         ["d", 8, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.IzhikevichModel(parameters)
        model.setAllDistributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)


        result = self.list_files()

        files = ['mc_10/IzhikevichModel',
                 'mc_100/IzhikevichModel',
                 'pc/IzhikevichModel']

        self.assertEqual(files, result)


    def test_LgnExploreParameters(self):
        model_file = "INmodel.hoc"
        model_path = "../models/neuron_models/dLGN_modelDB/"

        full_model_path = os.path.join(self.folder, model_path)

        # parameterlist = [["cap", 1.1, None],
        #                  ["Rm", 22000, None],
        #                  ["Vrest", -63, None],
        #                  ["Epas", -67, None],
        #                  ["gna", 0.09, None],
        #                  ["nash", -52.6, None],
        #                  ["gkdr", 0.37, None],
        #                  ["kdrsh", -51.2, None],
        #                  ["gahp", 6.4e-5, None],
        #                  ["gcat", 1.17e-5, None]]

        parameterlist = [["cap", 1.1, None],
                         ["Rm", 22000, None]]

        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.NeuronModel(parameters=parameters,
                                        model_file=model_file,
                                        model_path=full_model_path,
                                        adaptive_model=True)
        model.setAllDistributions(uncertainpy.Distribution(0.05).uniform)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        percentages = [0.1, 0.2, 0.3]
        test_distributions = {"uniform": percentages}
        exploration.exploreParameters(test_distributions)



    def test_LgnModelCompareMC(self):
        model_file = "INmodel.hoc"
        model_path = "../models/neuron_models/dLGN_modelDB/"

        full_model_path = os.path.join(self.folder, model_path)

        # parameterlist = [["cap", 1.1, None],
        #                  ["Rm", 22000, None],
        #                  ["Vrest", -63, None],
        #                  ["Epas", -67, None],
        #                  ["gna", 0.09, None],
        #                  ["nash", -52.6, None],
        #                  ["gkdr", 0.37, None],
        #                  ["kdrsh", -51.2, None],
        #                  ["gahp", 6.4e-5, None],
        #                  ["gcat", 1.17e-5, None]]

        parameterlist = [["cap", 1.1, None],
                         ["Rm", 22000, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.NeuronModel(parameters=parameters,
                                        model_file=model_file,
                                        model_path=full_model_path,
                                        adaptive_model=True)

        model.setAllDistributions(uncertainpy.Distribution(0.05).uniform)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.compareMC(mc_samples)



if __name__ == "__main__":
    unittest.main()
