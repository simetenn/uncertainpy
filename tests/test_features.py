import numpy as np
import unittest
import os

from uncertainpy.features import GeneralFeatures, GeneralSpikingFeatures, SpikingFeatures
from uncertainpy import Spikes
from testing_classes import TestingFeatures


class TestGeneralFeatures(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 10)
        self.U = np.arange(0, 10) + 1

        self.features = GeneralFeatures()

    def test_initNone(self):
        features = GeneralFeatures()

        self.assertIsInstance(features, GeneralFeatures)

    # def test_t(self):
    #     t = np.arange(0, 10)

    #     features = GeneralFeatures()
    #     features.t = t

    #     self.assertTrue(np.array_equal(features.t, np.arange(0, 10)))


    # def test_U(self):
    #     U = np.arange(0, 10) + 1

    #     features = GeneralFeatures()
    #     features.U = U

    #     self.assertTrue(np.array_equal(features.U, np.arange(0, 10) + 1))



    def test_initUtility(self):
        new_utility_methods = ["new"]

        features = GeneralFeatures(new_utility_methods=new_utility_methods)

        self.assertIn("new", features.utility_methods)



    def test_preprocess(self):
        features = GeneralFeatures()
        t, U = features.preprocess(self.t, self.U)

        self.assertTrue(np.array_equal(t, self.t))
        self.assertTrue(np.array_equal(U, self.U))

    def test_calculate_featureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculate_feature(self.t, self.U, "not_in_class")


    def test_calculate_featureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculate_feature(self.t, self.U, "preprocess")


    def test_implemented_features(self):
        self.assertEqual(self.features.implemented_features(), [])


    def test_calculate_all_features(self):
        self.assertEqual(self.features.calculate_all_features(self.t, self.U), {})


    def test_calculate(self):
        self.assertEqual(self.features.calculate(self.t, self.U), {})


    def test_intitFeatureList(self):
        features = GeneralFeatures(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = GeneralFeatures(features_to_run=["feature1d", "feature2"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2"])

        features = GeneralFeatures(features_to_run="all")
        self.assertEqual(features.features_to_run, [])


    def test_intitAdaptiveList(self):
        features = GeneralFeatures(adaptive_features=None)
        self.assertEqual(features.adaptive_features, [])

        features = GeneralFeatures(adaptive_features=["feature1d", "feature2"])
        self.assertEqual(features.adaptive_features,
                         ["feature1d", "feature2"])


        features = GeneralFeatures(adaptive_features="all")
        self.assertEqual(features.adaptive_features, [])


    def test_add_feature(self):
        def feature_function(t, U):
                return "t", "U"

        features = GeneralFeatures()

        features.add_features(feature_function)

        t, U = features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        self.assertEqual(features.features_to_run,
                         ["feature_function"])


    def test_add_features(self):
        def feature_function(t, U):
            return "t", "U"


        def feature_function2(t, U):
            return "t2", "U2"

        features = GeneralFeatures()

        features.add_features([feature_function, feature_function2])

        t, U = features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")

        self.assertEqual(features.implemented_features(),
                         ["feature_function", "feature_function2"])

        self.assertEqual(features.features_to_run,
                         ["feature_function", "feature_function2"])



class TestGeneralSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))


    def test_initNone(self):
        self.features = GeneralSpikingFeatures()

        self.assertIsInstance(self.features, GeneralSpikingFeatures)
        self.assertIsNone(self.features.spikes)
        self.assertIn("calculate_spikes", self.features.utility_methods)


    def test_calculate_spikes(self):
        self.features = GeneralSpikingFeatures()

        self.features.calculate_spikes(self.t, self.U)

        self.assertEqual(self.features.spikes.nr_spikes, 12)


    def test_preprocess(self):
        self.features = GeneralSpikingFeatures()

        t, spikes = self.features.preprocess(self.t, self.U)

        self.assertEqual(spikes.nr_spikes, 12)

        self.assertIsInstance(spikes, Spikes)
        self.assertTrue(np.array_equal(self.t, t))


    # def test_calculate_spikesTNone(self):
    #     self.features = GeneralSpikingFeatures()

    #     self.features.U = self.U
    #     with self.assertRaises(AttributeError):
    #         self.features.calculate_spikes()


    # def test_calculate_spikesUNone(self):
    #     self.features = GeneralSpikingFeatures()

    #     self.features.t = self.t
    #     with self.assertRaises(AttributeError):
    #         self.features.calculate_spikes()



class TestSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        t = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = ["nrSpikes", "time_before_first_spike",
                                     "spike_rate", "average_AP_overshoot",
                                     "average_AHP_depth", "average_AP_width",
                                     "accomondation_index"]

        self.features = SpikingFeatures()

        self.t, self.spikes = self.features.preprocess(t, U)


    def test_initNone(self):
        self.features = SpikingFeatures()

        self.assertIsInstance(self.features, SpikingFeatures)
        self.assertIsNone(self.features.spikes)


    # def test_init(self):
    #     self.assertIsInstance(self.features, SpikingFeatures)
    #     self.assertIsNotNone(self.features.spikes)
    #     self.assertEqual(self.features.spikes.nr_spikes, 12)


    def test_features_to_run_all(self):
        features = SpikingFeatures(features_to_run="all")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_adaptive_features_all(self):
        features = SpikingFeatures(adaptive_features="all")
        self.assertEqual(set(features.adaptive_features), set(self.implemented_features))


    def test_implemented_features(self):
        self.assertEqual(set(self.features.implemented_features()), set(self.implemented_features))


    def test_nrSpikes(self):
        self.assertEqual(self.features.nrSpikes(self.t, self.spikes), (None, 12))


    def test_time_before_first_spike(self):
        self.assertGreater(self.features.time_before_first_spike(self.t, self.spikes)[1], 10)


    def test_time_before_first_spikeNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.time_before_first_spike(self.t, self.spikes), (None, None))


    def test_spike_rate(self):
        self.assertEqual(self.features.spike_rate(self.t, self.spikes), (None, 0.12))


    def test_spike_rateNone(self):
        self.features.spikes.nr_spikes = -1
        self.assertEqual(self.features.spike_rate(self.t, self.spikes), (None, None))


    def test_average_AP_overshoot(self):
        self.assertEqual(self.features.average_AP_overshoot(self.t, self.spikes), (None, 30))


    def test_average_AP_overshootNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_overshoot(self.t, self.spikes), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_average_AHP_depth(self):
        self.features.average_AHP_depth(self.t, self.spikes)
        self.assertLess(self.features.average_AHP_depth(self.t, self.spikes)[1], 0)


    def test_average_AHP_depthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AHP_depth(self.t, self.spikes), (None, None))

    # TODO Find correct test, this is a rough bound only
    def test_average_AP_width(self):
        self.assertLess(self.features.average_AP_width(self.t, self.spikes)[1], 5)


    def test_average_AP_widthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_width(self.t, self.spikes), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_accomondation_index(self):
        self.assertIsNotNone(self.features.accomondation_index(self.t, self.spikes)[1])


    def test_accomondation_indexNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.accomondation_index(self.t, self.spikes), (None, None))


    def test_calculate_all_features(self):
        result = self.features.calculate_all_features(self.t, self.spikes)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))



class TestTestingFeatures(unittest.TestCase):
    def setUp(self):

        self.implemented_features = ["feature0d", "feature1d",
                                     "feature2d", "feature_invalid",
                                     "feature_adaptive"]

        self.features = TestingFeatures(features_to_run=self.implemented_features)

    def test_init(self):
        self.features = TestingFeatures()


    def test_feature0d(self):
        self.assertEqual(self.features.feature0d(None, None), (None, 1))


    def test_feature1d(self):
        self.assertTrue((self.features.feature1d(None, None), (np.arange(0, 10), np.arange(0, 10))))


    def test_feature2d(self):
        self.assertTrue((self.features.feature2d(None, None),
                         (np.arange(0, 10), np.array([np.arange(0, 10), np.arange(0, 10)]))))


    def test_feature_invalid(self):
        self.assertEqual(self.features.feature_invalid(None, None), (None, None))


    def test_calculate_features(self):
        self.assertEqual(set(self.features.calculate_features(None, None).keys()),
                         set(self.implemented_features))


    def test_calculate_none(self):
        self.assertEqual(set(self.features.calculate(None, None).keys()),
                         set(self.implemented_features))

    def test_calculate_all(self):
        with self.assertRaises(ValueError):
            self.features.calculate(None, None, "all")


    def test_calculate_one(self):
        self.assertEqual(self.features.calculate(None, None, "feature2d").keys(),
                         ["feature2d"])

    def test_feature_no_time(self):
        features = TestingFeatures(features_to_run="feature_no_time")

        with self.assertRaises(ValueError):
            features.calculate_features(None, None)


    def test_intitFeatureList(self):
        features = TestingFeatures(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = TestingFeatures(features_to_run=["feature1d", "feature2d"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2d"])

        features = TestingFeatures(features_to_run="all")
        self.assertEqual(features.features_to_run.sort(), (self.implemented_features + ["feature_no_time"]).sort())



if __name__ == "__main__":
    unittest.main()
