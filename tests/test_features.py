import numpy as np
import unittest
import os

from uncertainpy.features import GeneralFeatures, GeneralNeuronFeatures, NeuronFeatures
from testing_classes import TestingFeatures


class TestGeneralFeatures(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 1

        self.features = GeneralFeatures()
        self.features.t = t
        self.features.U = U

    def test_initNone(self):
        features = GeneralFeatures()

        self.assertIsInstance(features, GeneralFeatures)

    def test_t(self):
        t = np.arange(0, 10)

        features = GeneralFeatures()
        features.t = t

        self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))


    def test_U(self):
        U = np.arange(0, 10) + 1

        features = GeneralFeatures()
        features.U = U

        self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))



    def test_initUtility(self):
        new_utility_methods = ["new"]

        features = GeneralFeatures(new_utility_methods=new_utility_methods)

        self.assertIn("new", features.utility_methods)



    def test_setup(self):
        features = GeneralFeatures()
        features.setup()



    def test_calculateFeatureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculateFeature("not_in_class")


    def test_calculateFeatureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculateFeature("setup")


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

        features = GeneralFeatures(features_to_run="all")
        self.assertEqual(features.features_to_run, [])


    def test_intitAdaptiveList(self):
        features = GeneralFeatures(adaptive_features=None)
        self.assertEqual(features.adaptive_features, [])

        features = GeneralFeatures(adaptive_features=["feature1", "feature2"])
        self.assertEqual(features.adaptive_features,
                         ["feature1", "feature2"])


        features = GeneralFeatures(adaptive_features="all")
        self.assertEqual(features.adaptive_features, [])


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


    def test_setup(self):
        self.features = GeneralNeuronFeatures()

        self.features.t = self.t
        self.features.U = self.U

        self.features.setup()

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




class TestNeuronFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = ["nrSpikes", "timeBeforeFirstSpike",
                                     "spikeRate", "averageAPOvershoot",
                                     "averageAHPDepth", "averageAPWidth",
                                     "accomondationIndex"]

        self.features = NeuronFeatures()
        self.features.t = self.t
        self.features.U = self.U
        self.features.setup()



    def test_initNone(self):
        self.features = NeuronFeatures()

        self.assertIsInstance(self.features, NeuronFeatures)
        self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, NeuronFeatures)
        self.assertIsNotNone(self.features.spikes)
        self.assertEqual(self.features.spikes.nr_spikes, 12)


    def test_features_to_run_all(self):
        features = NeuronFeatures(features_to_run="all")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_adaptive_features_all(self):
        features = NeuronFeatures(adaptive_features="all")
        self.assertEqual(set(features.adaptive_features), set(self.implemented_features))


    def test_implementedFeatures(self):
        self.assertEqual(set(self.features.implementedFeatures()), set(self.implemented_features))


    def test_nrSpikes(self):
        self.assertEqual(self.features.nrSpikes(), (None, 12))


    def test_timeBeforeFirstSpike(self):
        self.assertGreater(self.features.timeBeforeFirstSpike()[1], 10)


    def test_timeBeforeFirstSpikeNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.timeBeforeFirstSpike(), (None, None), (None, None))


    def test_spikeRate(self):
        self.assertEqual(self.features.spikeRate(), (None, 0.12))


    def test_spikeRateNone(self):
        self.features.spikes.nr_spikes = -1
        self.assertEqual(self.features.spikeRate(), (None, None))


    def test_averageAPOvershoot(self):
        self.assertEqual(self.features.averageAPOvershoot(), (None, 30))


    def test_averageAPOvershootNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.averageAPOvershoot(), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_averageAHPDepth(self):
        self.assertLess(self.features.averageAHPDepth()[1], 0)


    def test_averageAHPDepthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.averageAHPDepth(), (None, None))

    # TODO Find correct test, this is a rough bound only
    def test_averageAPWidth(self):
        self.assertLess(self.features.averageAPWidth()[1], 5)


    def test_averageAPWidthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.averageAPWidth(), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_accomondationIndex(self):
        self.assertIsNotNone(self.features.accomondationIndex()[1])


    def test_accomondationIndexNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.accomondationIndex(), (None, None))


    def test_calculateAllFeatures(self):
        self.assertEqual(set(self.features.calculateAllFeatures().keys()),
                         set(self.implemented_features))



class TestTestingFeatures(unittest.TestCase):
    def setUp(self):

        self.implemented_features = ["feature0d", "feature1d",
                                     "feature2d", "featureInvalid",
                                     "feature_adaptive"]

        self.features = TestingFeatures(features_to_run=self.implemented_features)

    def test_init(self):
        self.features = TestingFeatures()


    def test_feature0d(self):
        self.assertEqual(self.features.feature0d(), (None, 1))


    def test_feature1d(self):
        self.assertTrue((self.features.feature1d(), (np.arange(0, 10), np.arange(0, 10))))


    def test_feature2d(self):
        self.assertTrue((self.features.feature2d(),
                         (np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)]))))


    def test_featureInvalid(self):
        self.assertEqual(self.features.featureInvalid(), (None, None))


    def test_calculate_features(self):
        self.assertEqual(set(self.features.calculateFeatures().keys()),
                         set(self.implemented_features))


    def test_feature_no_time(self):
        features = TestingFeatures(features_to_run="feature_no_time")

        with self.assertRaises(ValueError):
            features.calculateFeatures()


    def test_intitFeatureList(self):
        features = TestingFeatures(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2d"])

        features = TestingFeatures(features_to_run="all")
        self.assertEqual(features.features_to_run, self.implemented_features + ["feature_no_time"])

    # def test_kwargs(self):
    #     self.assertEqual(self.features.kwargs(), {"features_to_run": self.implemented_features})


if __name__ == "__main__":
    unittest.main()
