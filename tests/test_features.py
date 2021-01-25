import unittest
import os
import neo
import elephant
import efel

import numpy as np
import quantities as pq

from uncertainpy.features import Features, GeneralSpikingFeatures
from uncertainpy.features import SpikingFeatures, NetworkFeatures, GeneralNetworkFeatures
from uncertainpy.features import EfelFeatures
from uncertainpy.features import Spikes
from .testing_classes import TestingFeatures

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0, 10)
        self.values = np.arange(0, 10) + 1
        self.info = {"info": 1}

        self.features = Features(logger_level="error")

    def test_initNone(self):
        features = Features(logger_level="error")

        self.assertIsInstance(features, Features)


    def test_initUtility(self):
        new_utility_methods = ["new"]

        features = Features(new_utility_methods=new_utility_methods)

        self.assertIn("new", features.utility_methods)



    def test_preprocess(self):
        features = Features(logger_level="error")
        time, values = features.preprocess(self.time, self.values)

        self.assertTrue(np.array_equal(time, self.time))
        self.assertTrue(np.array_equal(values, self.values))


    def test_preprocess_args(self):
        features = Features(logger_level="error")
        time, values, a, b = features.preprocess(self.time, self.values, 1, 2)

        self.assertTrue(np.array_equal(time, self.time))
        self.assertTrue(np.array_equal(values, self.values))
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)


    def test_validate(self):
        self.features.validate("name", "t", "U")

        with self.assertRaises(ValueError):
            self.features.validate("123456")
            self.features.validate(np.linspace(0, 1, 100))
            self.features.validate(1)
            self.features.validate((1, 2, 3))


    def test_preprocess_assign(self):
        def preprocess(time, values):
            return "time", "values"

        features = Features(preprocess=preprocess, logger_level="error")

        time, values = features.preprocess(self.time, self.values)

        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        features = Features(preprocess=None)
        features.preprocess = preprocess

        time, values = features.preprocess(self.time, self.values)

        self.assertEqual(time, "time")
        self.assertEqual(values, "values")

        features = Features(preprocess=None)
        with self.assertRaises(TypeError):
            features.preprocess = 12



    def test_calculate_featureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculate_feature("not_in_class", self.time, self.values)


    def test_calculate_featureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculate_feature("preprocess", self.time, self.values)


    def test_implemented_features(self):
        self.assertEqual(self.features.implemented_features(), [])


    def test_calculate_all_features(self):
        self.assertEqual(self.features.calculate_all_features(self.time, self.values), {})


    # def test_calculate(self):
    #     self.assertEqual(self.features.calculate(self.time, self.values), {})


    def test_intitFeatureList(self):
        features = Features(features_to_run=None, logger_level="error")
        self.assertEqual(features.features_to_run, [])

        features = Features(features_to_run=["feature1d", "feature2"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2"])

        features = Features(features_to_run="all")
        self.assertEqual(features.features_to_run, [])


    def test_intitNewFeatures(self):
        def feature_function(time, values):
            return "t", "U"

        def feature_function2(time, values):
            return "t2", "U2"

        features = Features(new_features=[feature_function, feature_function2],
                            labels={"feature_function": ["x", "y"]},
                            logger_level="error")



        time, values = features.feature_function(None, None)
        self.assertEqual(time, "t")
        self.assertEqual(values, "U")

        time, values = features.feature_function2(None, None)
        self.assertEqual(time, "t2")
        self.assertEqual(values, "U2")

        self.assertEqual(features.implemented_features(),
                         ["feature_function", "feature_function2"])

        self.assertEqual(features.features_to_run,
                         ["feature_function", "feature_function2"])

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})



    def test_intitinterpolateList(self):
        features = Features(interpolate=None, logger_level="error")
        self.assertEqual(features.interpolate, [])

        features = Features(interpolate=["feature1d", "feature2"])
        self.assertEqual(features.interpolate,
                         ["feature1d", "feature2"])


        features = Features(interpolate="all")
        self.assertEqual(features.interpolate, [])


    def test_add_feature(self):
        def feature_function(time, values):
                return "t", "U"

        features = Features()

        features.add_features(feature_function,
                              labels={"feature_function": ["x", "y"]})

        time, values = features.feature_function(None, None)
        self.assertEqual(time, "t")
        self.assertEqual(values, "U")

        features.features_to_run = "all"
        self.assertEqual(features.features_to_run,
                         ["feature_function"])

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})


    def test_add_features(self):
        def feature_function(time, values):
            return "t", "U"


        def feature_function2(time, values):
            return "t2", "U2"

        features = Features(logger_level="error")

        features.add_features([feature_function, feature_function2],
                               labels={"feature_function": ["x", "y"]})

        time, values = features.feature_function(None, None)
        self.assertEqual(time, "t")
        self.assertEqual(values, "U")

        time, values = features.feature_function2(None, None)
        self.assertEqual(time, "t2")
        self.assertEqual(values, "U2")


        self.assertEqual(features.implemented_features(),
                         ["feature_function", "feature_function2"])

        self.assertEqual(features.features_to_run,
                         [])

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})


    def test_add_features_error(self):
        with self.assertRaises(TypeError):
            self.features.add_features({"test": 1})
            self.features.add_features([{"test": 1}])



    def test_reference_feature(self):
        time, values = self.features.reference_feature(self.time, self.values, self.info)

        self.assertIsNone(time)
        self.assertIsNone(values)




class TestGeneralSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.time = np.load(os.path.join(folder, "data/t_test.npy"))
        self.values = np.load(os.path.join(folder, "data/U_test.npy"))
        self.info = {"info": 1}


    def test_initNone(self):
        self.features = GeneralSpikingFeatures(logger_level="error")

        self.assertIsInstance(self.features, GeneralSpikingFeatures)
        self.assertIsNone(self.features.spikes)
        self.assertIn("calculate_spikes", self.features.utility_methods)


    def test_calculate_spikes(self):
        self.features = GeneralSpikingFeatures(logger_level="error")

        spikes = self.features.calculate_spikes(self.time, self.values)

        self.assertEqual(spikes.nr_spikes, 12)


    def test_preprocess(self):
        self.features = GeneralSpikingFeatures(logger_level="error")

        time, spikes, info = self.features.preprocess(self.time, self.values, self.info)

        self.assertEqual(spikes.nr_spikes, 12)

        self.assertIsInstance(spikes, Spikes)
        self.assertTrue(np.array_equal(self.time, time))


    def test_preprocess_normalize(self):
        self.features = GeneralSpikingFeatures(logger_level="error", normalize=True, threshold=0.4, end_threshold=-0.1)

        time, spikes, info = self.features.preprocess(self.time, self.values, self.info)

        self.assertEqual(spikes.nr_spikes, 12)

        self.assertIsInstance(spikes, Spikes)
        self.assertTrue(np.array_equal(self.time, time))
        self.assertTrue(spikes[0].V_spike > 1)


    def test_reference_feature(self):
        self.features = GeneralSpikingFeatures(logger_level="error")
        time, values = self.features.reference_feature(1, 1, 1)

        self.assertIsNone(time)
        self.assertIsNone(values)



class TestSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.V = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = ["nr_spikes", "time_before_first_spike",
                                     "spike_rate", "average_AP_overshoot",
                                     "average_AHP_depth", "average_AP_width",
                                     "accommodation_index", "average_duration"]

        self.implemented_labels = {"nr_spikes": ["Number of spikes"],
                                   "spike_rate": ["Spike rate (1/ms)"],
                                   "time_before_first_spike": ["Time (ms)"],
                                   "accommodation_index": ["Accommodation index"],
                                   "average_AP_overshoot": ["Voltage (mV)"],
                                   "average_AHP_depth": ["Voltage (mV)"],
                                   "average_AP_width": ["Time (ms)"],
                                   "average_duration": ["Time (ms)"]
                                   }

        self.features = SpikingFeatures(logger_level="error")

        self.info = {"stimulus_start": self.t[0], "stimulus_end": self.t[-1]}

        self.time, self.spikes, info = self.features.preprocess(self.t, self.V, self.info)


    def test_initNone(self):
        self.features = SpikingFeatures(logger_level="error")

        self.assertIsInstance(self.features, SpikingFeatures)
        self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, SpikingFeatures)
        self.assertIsNotNone(self.features.spikes)
        self.assertEqual(self.features.spikes.nr_spikes, 12)
        self.assertEqual(self.features.labels, self.implemented_labels)

    def test_initLabels(self):
        features = SpikingFeatures(labels={"nr_spikes": ["changed"],
                                           "new": ["new"]},
                                   logger_level="error")

        labels = {"nr_spikes": ["changed"],
                  "new": ["new"],
                  "spike_rate": ["Spike rate (1/ms)"],
                  "time_before_first_spike": ["Time (ms)"],
                  "accommodation_index": ["Accommodation index"],
                  "average_AP_overshoot": ["Voltage (mV)"],
                  "average_AHP_depth": ["Voltage (mV)"],
                  "average_AP_width": ["Time (ms)"],
                  "average_duration": ["Time (ms)"]
                  }

        self.assertEqual(features.labels, labels)

    def test_features_to_run_all(self):
        features = SpikingFeatures(features_to_run="all", logger_level="error")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_interpolate_all(self):
        features = SpikingFeatures(interpolate="all", logger_level="error")
        self.assertEqual(set(features.interpolate), set(self.implemented_features))


    def test_implemented_features(self):
        self.assertEqual(set(self.features.implemented_features()), set(self.implemented_features))


    def test_nr_spikes(self):
        self.assertEqual(self.features.nr_spikes(self.time, self.spikes, self.info), (None, 12))


    def test_nr_spikes_no_stimulus(self):
        info = {"stimulus_start": -2, "stimulus_end": -1}
        self.assertEqual(self.features.nr_spikes(self.time, self.spikes, info), (None, 0))


    def test_nr_spikes_error(self):
        self.features.strict = True
        with self.assertRaises(ValueError):
            self.features.nr_spikes(self.time, self.spikes, {})
            self.features.nr_spikes(self.time, self.spikes, {"stimulus_start": 1})
            self.features.nr_spikes(self.time, self.spikes, {"stimulus_end": 1})
            self.features.nr_spikes(self.time, self.spikes, {"stimulus_end": 1, "stimulus_start": -1})


    def test_time_before_first_spike(self):
        self.assertGreater(self.features.time_before_first_spike(self.time, self.spikes, self.info)[1], 10)


    def test_time_before_first_spikeNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.time_before_first_spike(self.time, self.spikes, self.info), (None, None))


    def test_time_before_first_spike_no_strict(self):
        self.features.strict = False
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.time_before_first_spike(self.time, self.spikes, {}), (None, None))


    def test_time_before_first_spike_error(self):
        with self.assertRaises(ValueError):
            self.features.time_before_first_spike(self.time, self.spikes, {})
            self.features.time_before_first_spike(self.time, self.spikes, {"stimulus_end": 1})


    def test_spike_rate(self):
        self.assertEqual(self.features.spike_rate(self.time, self.spikes, self.info), (None, 0.12))


    def test_spike_rate_no_strict(self):
        self.features.strict = False
        self.assertEqual(self.features.spike_rate(self.time, self.spikes, {}), (None, 0.12))


    def test_spike_rateNone(self):
        self.features.spikes.nr_spikes = -1
        self.assertEqual(self.features.spike_rate(self.time, self.spikes, self.info), (None, None))


    def test_spike_rate_error(self):
        with self.assertRaises(ValueError):
            self.features.spike_rate(self.time, self.spikes, {})
            self.features.spike_rate(self.time, self.spikes, {"stimulus_start": 1})
            self.features.spike_rate(self.time, self.spikes, {"stimulus_end": 1})
            self.features.spike_rate(self.time, self.spikes, {"stimulus_end": 1, "stimulus_start": -1})


    def test_average_AP_overshoot(self):
        self.assertEqual(self.features.average_AP_overshoot(self.time, self.spikes, self.info), (None, 30))


    def test_average_AP_overshootNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_overshoot(self.time, self.spikes, self.info), (None, None))


    # TODO Find correct testime, this is a rough bound only
    def test_average_AHP_depth(self):
        self.features.average_AHP_depth(self.time, self.spikes, self.info)
        self.assertLess(self.features.average_AHP_depth(self.time, self.spikes, self.info)[1], 0)


    def test_average_AHP_depthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AHP_depth(self.time, self.spikes, self.info), (None, None))

    # TODO Find correct testime, this is a rough bound only
    def test_average_AP_width(self):
        self.assertLess(self.features.average_AP_width(self.time, self.spikes, self.info)[1], 5)


    # TODO Find correct testime, this is a rough bound only
    def test_average_duration(self):
        self.assertLess(self.features.average_duration(self.time, self.spikes, self.info)[1], 5)


    def test_average_AP_widthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_width(self.time, self.spikes, self.info), (None, None))


    # TODO Find correct testime, this is a rough bound only
    def test_accommodation_index(self):
        self.assertIsNotNone(self.features.accommodation_index(self.time, self.spikes, self.info)[1])


    def test_accommodation_indexNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.accommodation_index(self.time, self.spikes, self.info), (None, None))


    def test_calculate_all_features(self):

        result = self.features.calculate_all_features(self.t, self.V, self.info)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))


    def test_calculate_all_features_normalize(self):
        self.features = SpikingFeatures(logger_level="error", threshold=0.4, end_threshold=-0.1)

        self.info = {"stimulus_start": self.t[0], "stimulus_end": self.t[-1]}

        self.time, self.spikes, info = self.features.preprocess(self.t, self.V, self.info)

        result = self.features.calculate_all_features(self.t, self.V, self.info)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))


    def test_reference_feature(self):
        time, values = self.features.reference_feature(1, 1, 1)

        self.assertIsNone(time)
        self.assertIsNone(values)


class TestEfelFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.time = np.load(os.path.join(folder, "data/t_test.npy"))
        self.values = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = efel.getFeatureNames()

        self.features = EfelFeatures(logger_level="error")

        self.info = {}
        self.info["stimulus_start"] = self.time[0]
        self.info["stimulus_end"] = self.time[-10]

        # self.time, self.spikes = self.features.preprocess(time, values)


    # def test_initNone(self):
    #     self.features = SpikingFeatures()

    #     self.assertIsInstance(self.features, SpikingFeatures)
    #     self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, EfelFeatures)


    def test_features_to_run_all(self):
        features = EfelFeatures(features_to_run="all", logger_level="error")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_implemented_features(self):
        self.assertEqual(set(self.features.implemented_features()), set(self.implemented_features))


    def test_spikecount(self):
        time, values = self.features.Spikecount(self.time, self.values, self.info)

        self.assertIsNone(time)
        self.assertEqual(values, 12)


    def test_calculate_all_features(self):
        result = self.features.calculate_all_features(self.time, self.values, self.info)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))


    def test_reference_feature(self):
        time, values = self.features.reference_feature(self.time, self.values, self.info)

        self.assertIsNone(time)
        self.assertIsNone(values)


    def test_spikecount_error(self):
        with self.assertRaises(ValueError):
            self.features.Spikecount(self.time, self.values, {})
            self.features.Spikecount(self.time, self.values, {"stimulus_start": self.time[0]})
            self.features.Spikecount(self.time, self.values, {"stimulus_end": self.time[-1]})
            self.features.Spikecount(self.time, self.values, {"stimulus_end": 0, "stimulus_start": 1})


    def test_spikecount_no_strict(self):
        self.features = EfelFeatures(strict=False, logger_level="error")

        time, values = self.features.Spikecount(self.time, self.values, {})
        self.assertIsNone(time)
        self.assertEqual(values, 12)

        time, values = self.features.Spikecount(self.time, self.values, {"stimulus_start": self.time[0]})
        self.assertIsNone(time)
        self.assertEqual(values, 12)

        time, values = self.features.Spikecount(self.time, self.values, {"stimulus_end": self.time[-1]})
        self.assertIsNone(time)
        self.assertEqual(values, 12)



class TestGeneralNetworkFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.time_original = 8
        spiketrain = np.array([1, 3, 5, 6])
        self.values = [spiketrain, spiketrain, spiketrain, np.array([1])]


        self.features = NetworkFeatures(logger_level="error")


    def test_initNone(self):
        self.features = GeneralNetworkFeatures(logger_level="error")

        self.assertIsInstance(self.features, GeneralNetworkFeatures)


    def test_init(self):
        def feature(time, values):
            return "t", "U"

        features = GeneralNetworkFeatures(new_features=feature,
                                          features_to_run=None,
                                          interpolate=["cv"],
                                          labels={"cv": ["test"]},
                                          units="",
                                          logger_level="error")

        self.assertIsInstance(features, GeneralNetworkFeatures)
        self.assertEqual(features.features_to_run, [])
        self.assertEqual(features.interpolate, ["cv"])
        self.assertEqual(features.units, "")
        self.assertEqual(set(features.implemented_features()),
                         set(["feature"]))


    def test_preprocess(self):
        self.features = GeneralNetworkFeatures(logger_level="error")

        time, spiketrains = self.features.preprocess(self.time_original, self.values)

        self.assertEqual(time, self.time_original)
        self.assertIsInstance(spiketrains[0], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[1], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[2], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[3], neo.core.SpikeTrain)

        self.assertTrue(np.array_equal(spiketrains[0], self.values[0]))
        self.assertTrue(np.array_equal(spiketrains[1], self.values[1]))
        self.assertTrue(np.array_equal(spiketrains[2], self.values[2]))
        self.assertTrue(np.array_equal(spiketrains[3], self.values[3]))

        self.assertEqual(spiketrains[0].t_stop, self.time_original)


class TestNetworkFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.time_original = 8
        spiketrain = np.array([1, 3, 5, 6])
        self.values = [spiketrain, spiketrain, spiketrain, np.array([1])]

        self.implemented_features = ["cv", "average_cv", "binned_isi",
                                     "average_isi", "local_variation", "average_firing_rate",
                                     "fanofactor", "instantaneous_rate",
                                     "van_rossum_dist", "victor_purpura_dist",
                                     "corrcoef", "covariance", "average_local_variation"]


        self.features = NetworkFeatures(instantaneous_rate_nr_samples=2,
                                        logger_level="error")

        self.time, self.spiketrains = self.features.preprocess(self.time_original, self.values)


    def test_initNone(self):
        self.features = NetworkFeatures(logger_level="error")

        self.assertIsInstance(self.features, NetworkFeatures)


    def test_init(self):
        def feature(time, values):
            return "t", "U"

        features = NetworkFeatures(new_features=feature,
                                   features_to_run=None,
                                   interpolate=["cv"],
                                   labels={"cv": ["test"]},
                                   instantaneous_rate_nr_samples=-1.,
                                   isi_bin_size=-1,
                                   corrcoef_bin_size=-1,
                                   covariance_bin_size=-1,
                                   units=pq.ms)

        self.assertIsInstance(features, NetworkFeatures)
        self.assertEqual(features.features_to_run, [])
        self.assertEqual(features.interpolate, ["cv"])
        self.assertEqual(features.instantaneous_rate_nr_samples, -1)
        self.assertEqual(features.isi_bin_size, -1)
        self.assertEqual(features.corrcoef_bin_size, -1)
        self.assertEqual(features.covariance_bin_size, -1)
        self.assertEqual(features.units, pq.ms)
        self.assertEqual(set(features.implemented_features()),
                         set(["feature"] + self.implemented_features))


    def test_preprocess(self):
        self.features = NetworkFeatures()

        time, spiketrains = self.features.preprocess(self.time_original, self.values)

        self.assertEqual(time, self.time_original)
        self.assertIsInstance(spiketrains[0], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[1], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[2], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[3], neo.core.SpikeTrain)

        self.assertTrue(np.array_equal(spiketrains[0], self.values[0]))
        self.assertTrue(np.array_equal(spiketrains[1], self.values[1]))
        self.assertTrue(np.array_equal(spiketrains[2], self.values[2]))
        self.assertTrue(np.array_equal(spiketrains[3], self.values[3]))

        self.assertEqual(spiketrains[0].t_stop, self.time_original)


    def test_cv(self):
        time, values = self.features.cv(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertTrue(np.array_equal([0.51207638319124049,
                                        0.51207638319124049,
                                        0.51207638319124049,
                                        0],
                                       values))

    def test_empty_spiketrain(self):
        for feature in self.implemented_features:
            time, values = getattr(self.features, feature)(self.time, [])

            self.assertIsNone(time)
            self.assertIsNone(values)


    def test_average_cv(self):
        time, values = self.features.average_cv(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(0.38405728739343037, values)


    def test_binned_isi(self):
        time, values = self.features.binned_isi(self.time, self.spiketrains)

        centers = np.arange(0, self.time_original + 1)[1:] - 0.5

        self.assertTrue(np.array_equal(centers, time))
        self.assertTrue(np.array_equal(values, [[0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0]]))


    def test_average_isi(self):
        time, values = self.features.average_isi(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(1.66666666666666667, values)


    def test_lv(self):
        time, values = self.features.local_variation(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertTrue(np.array_equal([0.16666666666666666,
                                        0.16666666666666666,
                                        0.16666666666666666,
                                        None],
                                       values))

    def test_mean_lv(self):
        time, values = self.features.average_local_variation(self.time, self.spiketrains)

        self.assertIsNone(time)
        mean = np.mean([0.16666666666666666, 0.16666666666666666, 0.16666666666666666])
        self.assertEqual(mean, values)


    def test_average_firing_rate(self):
        time, values = self.features.average_firing_rate(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertTrue(np.array_equal([500, 500, 500, 125], values))


    def test_instantaneous_rate(self):
        time, values = self.features.instantaneous_rate(self.time, self.spiketrains)

        rates = [345.49198943, 303.02788156]

        correct_t = np.linspace(0, 8, 3)[:-1]

        self.assertTrue(np.all(np.isclose(values[0], rates)))
        self.assertTrue(np.all(np.isclose(values[1], rates)))
        self.assertTrue(np.all(np.isclose(values[2], rates)))
        self.assertIsNone(values[3])
        self.assertTrue(np.array_equal(time, correct_t))



    def test_fanofactor(self):
        time, values = self.features.fanofactor(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(values, 0.51923076923076927)


    def test_van_rossum_dist(self):
        time, values = self.features.van_rossum_dist(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(values.shape, (4, 4))

        # correct_values = [[0.0, 5.9604644775390625e-08, 5.9604644775390625e-08, 2.9980016657780828],
        #              [5.9604644775390625e-08, 0.0, 5.9604644775390625e-08, 2.9980016657780828],
        #              [5.9604644775390625e-08, 5.9604644775390625e-08, 0.0, 2.9980016657780828],
        #              [2.9980016657780828, 2.9980016657780828, 2.9980016657780828, 0.0]]

        # self.assertTrue(np.array_equal(values, correct_values))

        diag = np.diag_indices(4)
        self.assertTrue(np.all(values[diag] == 0))

    def test_victor_purpura_dist(self):
        time, values = self.features.victor_purpura_dist(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(values.shape, (4, 4))
        diag = np.diag_indices(4)
        self.assertTrue(np.all(values[diag] == 0))


    def test_corrcoef(self):
        time, values = self.features.corrcoef(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(values.shape, (4, 4))
        diag = np.diag_indices(4)
        self.assertTrue(np.allclose(values[diag], [1]*4, rtol=1e-14, atol=1e-14,))

    def test_covariance(self):
        time, values = self.features.covariance(self.time, self.spiketrains)

        self.assertIsNone(time)
        self.assertEqual(values.shape, (4, 4))


    def test_reference_feature(self):
        time, values = self.features.reference_feature(1, 1)

        self.assertIsNone(time)
        self.assertIsNone(values)



class TestTestingFeatures(unittest.TestCase):
    def setUp(self):

        self.implemented_features = ["feature0d", "feature1d",
                                     "feature2d", "feature_invalid",
                                     "feature_interpolate"]

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


    # def test_calculate_none(self):
    #     self.assertEqual(set(self.features.calculate(None, None).keys()),
    #                      set(self.implemented_features))

    # def test_calculate_all(self):
    #     with self.assertRaises(ValueError):
    #         self.features.calculate(None, None, "all")


    # def test_calculate_one(self):
    #     self.assertEqual(self.features.calculate(None, None, "feature2d").keys(),
    #                      ["feature2d"])

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




    def test_calculate_feature_info(self):
        time, values,= self.features.feature_info(1, 2, "info")

        self.assertEqual(self.features.info, "info")


    def test_calculate_feature_error_one(self):
        with self.assertRaises(TypeError):
            self.features.calculate_feature("feature_error_one")


    def test_calculate_feature_error_one(self):
        with self.assertRaises(ValueError):
            self.features.calculate_feature("feature_error_value")


if __name__ == "__main__":
    unittest.main()
