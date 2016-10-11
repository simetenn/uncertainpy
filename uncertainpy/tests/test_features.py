import numpy as np
import unittest
import os

from uncertainpy.features import GeneralFeatures, GeneralNeuronFeatures
from uncertainpy.features import TestingFeatures, NeuronFeatures


class TestGeneralFeatures(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        self.features = GeneralFeatures(t=t, U=U)


    def test_initNone(self):
        features = GeneralFeatures()

        self.assertIsInstance(features, GeneralFeatures)

    def test_initUt(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        features = GeneralFeatures(t=t, U=U)

        self.assertIsInstance(features, GeneralFeatures)

        self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))



    def test_initUtility(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        new_utility_methods = ["new"]

        features = GeneralFeatures(t=t, U=U, new_utility_methods=new_utility_methods)

        self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))
        self.assertIn("new", features.utility_methods)


    def test_kwargs(self):
        self.assertEqual(self.features.kwargs(), {"features_to_run": []})

        self.features = GeneralFeatures(test1=1, test2=2)

        self.assertEqual(self.features.kwargs(), {"features_to_run": [], "test1": 1, "test2": 2})

    def test_cmd(self):
        result = self.features.cmd()

        self.assertEqual('general_features', result[1].split(".")[0])
        self.assertEqual('GeneralFeatures', result[2])


    def test_calculateFeatureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculateFeature("not_in_class")


    def test_calculateFeatureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculateFeature("cmd")


    def test_implementedFeatures(self):
        self.assertEqual(self.features.implementedFeatures(), [])


    def test_calculateAllFeatures(self):
        self.assertEqual(self.features.calculateAllFeatures(), {})

    def test_intitFeatureList(self):
        features = GeneralFeatures(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = GeneralFeatures(features_to_run=["feature1", "feature2"])
        self.assertEqual(features.features_to_run,
                         ["feature1", "feature2"])


    def test_set_properties(self):
        cmds = {"a": 1, "b": 2}

        self.features.set_properties(cmds)

        self.assertEqual(self.features.a, 1)
        self.assertEqual(self.features.b, 2)

        self.assertIn("a", self.features.additional_kwargs)
        self.assertIn("b", self.features.additional_kwargs)



class TestGeneralNeuronFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))


    def test_initNone(self):
        self.features = GeneralNeuronFeatures()

        self.assertIsInstance(self.features, GeneralNeuronFeatures)
        self.assertIsNone(self.features.spikes)
        self.assertIn("calculateSpikes", self.features.utility_methods)


    def test_calculateSpikes(self):
        self.features = GeneralNeuronFeatures()

        self.features.t = self.t
        self.features.U = self.U

        self.features.calculateSpikes()

        self.assertEqual(self.features.spikes.nr_spikes, 12)


    def test_calculateSpikesTNone(self):
        self.features = GeneralNeuronFeatures()

        self.features.U = self.U
        with self.assertRaises(AttributeError):
            self.features.calculateSpikes()


    def test_calculateSpikesUNone(self):
        self.features = GeneralNeuronFeatures()

        self.features.t = self.t
        with self.assertRaises(AttributeError):
            self.features.calculateSpikes()


    def test_init(self):
        self.features = GeneralNeuronFeatures(t=self.t, U=self.U)

        self.assertIsInstance(self.features, GeneralNeuronFeatures)
        self.assertIsNotNone(self.features.spikes)
        self.assertEqual(self.features.spikes.nr_spikes, 12)





class TestNeuronFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = ["nrSpikes", "timeBeforeFirstSpike",
                                     "spikeRate", "averageAPOvershoot",
                                     "averageAHPDepth", "averageAPWidth",
                                     "accomondationIndex"]

        self.features = NeuronFeatures(t=self.t, U=self.U)


    def test_initNone(self):
        self.features = NeuronFeatures()

        self.assertIsInstance(self.features, NeuronFeatures)
        self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, NeuronFeatures)
        self.assertIsNotNone(self.features.spikes)
        self.assertEqual(self.features.spikes.nr_spikes, 12)


    def test_implementedFeatures(self):
        self.assertEqual(set(self.features.implementedFeatures()), set(self.implemented_features))


    def test_nrSpikes(self):
        self.assertEqual(self.features.nrSpikes(), 12)


    def test_timeBeforeFirstSpike(self):
        self.assertGreater(self.features.timeBeforeFirstSpike(), 10)


    def test_timeBeforeFirstSpikeNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertIsNone(self.features.timeBeforeFirstSpike())


    def test_spikeRate(self):
        self.assertEqual(self.features.spikeRate(), 0.12)


    def test_spikeRateNone(self):
        self.features.spikes.nr_spikes = -1
        self.assertIsNone(self.features.spikeRate())


    def test_averageAPOvershoot(self):
        self.assertEqual(self.features.averageAPOvershoot(), 30)


    def test_averageAPOvershootNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertIsNone(self.features.averageAPOvershoot())


    # TODO Find correct test, this is a rough bound only
    def test_averageAHPDepth(self):
        self.assertLess(self.features.averageAHPDepth(), 0)


    def test_averageAHPDepthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertIsNone(self.features.averageAHPDepth())

    # TODO Find correct test, this is a rough bound only
    def test_averageAPWidth(self):
        self.assertLess(self.features.averageAPWidth(), 5)


    def test_averageAPWidthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertIsNone(self.features.averageAPWidth())


    # TODO Find correct test, this is a rough bound only
    def test_accomondationIndex(self):
        self.assertIsNotNone(self.features.accomondationIndex())


    def test_accomondationIndexNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertIsNone(self.features.accomondationIndex())


    def test_calculateAllFeatures(self):
        self.assertEqual(set(self.features.calculateAllFeatures().keys()),
                         set(self.implemented_features))



class TestTestingFeatures(unittest.TestCase):
    def setUp(self):
        self.features = TestingFeatures()

        self.implemented_features = ["feature0d", "feature1d",
                                     "feature2d", "featureInvalid"]

    def test_init(self):
        self.features = TestingFeatures()


    def test_feature0d(self):
        self.assertEqual(self.features.feature0d(), 1)


    def test_feature1d(self):
        self.assertTrue((self.features.feature1d(), np.arange(0, 10)))


    def test_feature2d(self):
        self.assertTrue((self.features.feature2d(),
                         np.array([np.arange(0, 10), np.arange(0, 10)])))


    def test_featureInvalid(self):
        self.assertIsNone(self.features.featureInvalid())


    def test_calculateAllFeatures(self):
        self.assertEqual(set(self.features.calculateAllFeatures().keys()),
                         set(self.implemented_features))

    def test_intitFeatureList(self):
        features = TestingFeatures(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2d"])

        features = TestingFeatures(features_to_run="all")
        self.assertEqual(features.features_to_run, self.implemented_features)

    def test_kwargs(self):
        self.assertEqual(self.features.kwargs(), {"features_to_run": self.implemented_features})


if __name__ == "__main__":
    unittest.main()
