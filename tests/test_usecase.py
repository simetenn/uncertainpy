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

        files = ['uniform_0.1/CoffeeCupPointModel.h5',
                 'uniform_0.1/CoffeeCupPointModel_single-parameter-kappa.h5',
                 'uniform_0.1/CoffeeCupPointModel_single-parameter-u_env.h5',
                 'uniform_0.2/CoffeeCupPointModel.h5',
                 'uniform_0.2/CoffeeCupPointModel_single-parameter-kappa.h5',
                 'uniform_0.2/CoffeeCupPointModel_single-parameter-u_env.h5',
                 'uniform_0.3/CoffeeCupPointModel.h5',
                 'uniform_0.3/CoffeeCupPointModel_single-parameter-kappa.h5',
                 'uniform_0.3/CoffeeCupPointModel_single-parameter-u_env.h5']

        self.assertEqual(files, result)




    def test_CoffeeCupPointModelComparemonte_carlo(self):
        parameterlist = [["kappa", -0.05, None],
                         ["u_env", 20, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.CoffeeCupPointModel(parameters)
        model.set_all_distributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.comparemonte_carlo(mc_samples)

        result = self.list_files()

        files = ['mc_10/CoffeeCupPointModel.h5',
                 'mc_100/CoffeeCupPointModel.h5',
                 'pc/CoffeeCupPointModel.h5']

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

        files = ['uniform_0.1/HodkinHuxleyModel.h5',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_K.h5',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_Na.h5',
                 'uniform_0.1/HodkinHuxleyModel_single-parameter-gbar_l.h5',
                 'uniform_0.2/HodkinHuxleyModel.h5',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_K.h5',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_Na.h5',
                 'uniform_0.2/HodkinHuxleyModel_single-parameter-gbar_l.h5',
                 'uniform_0.3/HodkinHuxleyModel',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_K.h5',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_Na.h5',
                 'uniform_0.3/HodkinHuxleyModel_single-parameter-gbar_l.h5']

        self.assertEqual(files, result)


    def test_HodkinHuxleyModelComparemonte_carlo(self):
        parameterlist = [["gbar_Na", 120, None],
                         ["gbar_K", 36, None],
                         ["gbar_l", 0.3, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.HodkinHuxleyModel(parameters)
        model.set_all_distributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.comparemonte_carlo(mc_samples)


        result = self.list_files()

        files = ['mc_10/HodkinHuxleyModel.h5',
                 'mc_100/HodkinHuxleyModel.h5',
                 'pc/HodkinHuxleyModel.h5']

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

        files = ['uniform_0.1/IzhikevichModel.h5',
                 'uniform_0.1/IzhikevichModel_single-parameter-a.h5',
                 'uniform_0.1/IzhikevichModel_single-parameter-b.h5',
                 'uniform_0.1/IzhikevichModel_single-parameter-c.h5',
                 'uniform_0.1/IzhikevichModel_single-parameter-d.h5',
                 'uniform_0.2/IzhikevichModel.h5',
                 'uniform_0.2/IzhikevichModel_single-parameter-a.h5',
                 'uniform_0.2/IzhikevichModel_single-parameter-b.h5',
                 'uniform_0.2/IzhikevichModel_single-parameter-c.h5',
                 'uniform_0.2/IzhikevichModel_single-parameter-d.h5',
                 'uniform_0.3/IzhikevichModel.h5',
                 'uniform_0.3/IzhikevichModel_single-parameter-a.h5',
                 'uniform_0.3/IzhikevichModel_single-parameter-b.h5',
                 'uniform_0.3/IzhikevichModel_single-parameter-c.h5',
                 'uniform_0.3/IzhikevichModel_single-parameter-d.h5']

        self.assertEqual(files, result)


    def test_IzhikevichModelComparemonte_carlo(self):
        parameterlist = [["a", 0.02, None],
                         ["b", 0.2, None],
                         ["c", -65, None],
                         ["d", 8, None]]


        parameters = uncertainpy.Parameters(parameterlist)
        model = uncertainpy.IzhikevichModel(parameters)
        model.set_all_distributions(uncertainpy.Distribution(0.5).uniform)


        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.comparemonte_carlo(mc_samples)


        result = self.list_files()

        files = ['mc_10/IzhikevichModel.h5',
                 'mc_100/IzhikevichModel.h5',
                 'pc/IzhikevichModel.h5']

        self.assertEqual(files, result)


    def test_LgnExploreParameters(self):
        file = "INmodel.hoc"
        path = "../models/neuron_models/dLGN_modelDB/"

        full_path = os.path.join(self.folder, path)

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
                                        file=file,
                                        path=full_path,
                                        adaptive=True)
        model.set_all_distributions(uncertainpy.Distribution(0.05).uniform)

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



    def test_LgnModelComparemonte_carlo(self):
        file = "INmodel.hoc"
        path = "../models/neuron_models/dLGN_modelDB/"

        full_path = os.path.join(self.folder, path)

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
                                        file=file,
                                        path=full_path,
                                        adaptive=True)

        model.set_all_distributions(uncertainpy.Distribution(0.05).uniform)

        exploration = uncertainpy.UncertaintyEstimations(model,
                                                         feature_list="all",
                                                         output_dir_data=self.output_test_dir,
                                                         output_dir_figures=self.output_test_dir,
                                                         nr_mc_samples=10**1,
                                                         nr_pc_mc_samples=10**2,
                                                         verbose_level="Error")


        mc_samples = [10, 100]
        exploration.comparemonte_carlo(mc_samples)



if __name__ == "__main__":
    unittest.main()
