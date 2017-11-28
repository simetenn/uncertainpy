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
from uncertainpy import Spikes
from .testing_classes import TestingFeatures

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 10)
        self.U = np.arange(0, 10) + 1
        self.info = {"info": 1}

        self.features = Features()

    def test_initNone(self):
        features = Features()

        self.assertIsInstance(features, Features)


    def test_initUtility(self):
        new_utility_methods = ["new"]

        features = Features(new_utility_methods=new_utility_methods)

        self.assertIn("new", features.utility_methods)



    def test_preprocess(self):
        features = Features()
        t, U = features.preprocess(self.t, self.U)

        self.assertTrue(np.array_equal(t, self.t))
        self.assertTrue(np.array_equal(U, self.U))


    def test_preprocess_args(self):
        features = Features()
        t, U, a, b = features.preprocess(self.t, self.U, 1, 2)

        self.assertTrue(np.array_equal(t, self.t))
        self.assertTrue(np.array_equal(U, self.U))
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)


    def test_calculate_featureNotImplemented(self):
        with self.assertRaises(AttributeError):
            self.features.calculate_feature("not_in_class", self.t, self.U)


    def test_calculate_featureUtilityMethod(self):
        with self.assertRaises(TypeError):
            self.features.calculate_feature("preprocess", self.t, self.U)


    def test_implemented_features(self):
        self.assertEqual(self.features.implemented_features(), [])


    def test_calculate_all_features(self):
        self.assertEqual(self.features.calculate_all_features(self.t, self.U), {})


    # def test_calculate(self):
    #     self.assertEqual(self.features.calculate(self.t, self.U), {})


    def test_intitFeatureList(self):
        features = Features(features_to_run=None)
        self.assertEqual(features.features_to_run, [])

        features = Features(features_to_run=["feature1d", "feature2"])
        self.assertEqual(features.features_to_run,
                         ["feature1d", "feature2"])

        features = Features(features_to_run="all")
        self.assertEqual(features.features_to_run, [])


    def test_intitNewFeatures(self):
        def feature_function(t, U):
            return "t", "U"

        def feature_function2(t, U):
            return "t2", "U2"

        features = Features(new_features=[feature_function, feature_function2],
                                   labels={"feature_function": ["x", "y"]})



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

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})



    def test_intitAdaptiveList(self):
        features = Features(adaptive=None)
        self.assertEqual(features.adaptive, [])

        features = Features(adaptive=["feature1d", "feature2"])
        self.assertEqual(features.adaptive,
                         ["feature1d", "feature2"])


        features = Features(adaptive="all")
        self.assertEqual(features.adaptive, [])


    def test_add_feature(self):
        def feature_function(t, U):
                return "t", "U"

        features = Features()

        features.add_features(feature_function,
                              labels={"feature_function": ["x", "y"]})

        t, U = features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        features.features_to_run = "all"
        self.assertEqual(features.features_to_run,
                         ["feature_function"])

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})

    def test_add_features(self):
        def feature_function(t, U):
            return "t", "U"


        def feature_function2(t, U):
            return "t2", "U2"

        features = Features()

        features.add_features([feature_function, feature_function2],
                              labels={"feature_function": ["x", "y"]})

        t, U = features.feature_function(None, None)
        self.assertEqual(t, "t")
        self.assertEqual(U, "U")

        t, U = features.feature_function2(None, None)
        self.assertEqual(t, "t2")
        self.assertEqual(U, "U2")


        self.assertEqual(features.implemented_features(),
                         ["feature_function", "feature_function2"])

        self.assertEqual(features.features_to_run,
                         [])

        self.assertEqual(features.labels, {"feature_function": ["x", "y"]})



class TestGeneralSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))
        self.info = {"info": 1}


    def test_initNone(self):
        self.features = GeneralSpikingFeatures()

        self.assertIsInstance(self.features, GeneralSpikingFeatures)
        self.assertIsNone(self.features.spikes)
        self.assertIn("calculate_spikes", self.features.utility_methods)


    def test_calculate_spikes(self):
        self.features = GeneralSpikingFeatures()

        spikes = self.features.calculate_spikes(self.t, self.U)

        self.assertEqual(spikes.nr_spikes, 12)


    def test_preprocess(self):
        self.features = GeneralSpikingFeatures()

        t, spikes, info = self.features.preprocess(self.t, self.U, self.info)

        self.assertEqual(spikes.nr_spikes, 12)

        self.assertIsInstance(spikes, Spikes)
        self.assertTrue(np.array_equal(self.t, t))




class TestSpikingFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        t = np.load(os.path.join(folder, "data/t_test.npy"))
        U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = ["nr_spikes", "time_before_first_spike",
                                     "spike_rate", "average_AP_overshoot",
                                     "average_AHP_depth", "average_AP_width",
                                     "accommodation_index"]

        self.implemented_labels = {"nr_spikes": ["number of spikes"],
                                   "spike_rate": ["spike rate [Hz]"],
                                   "time_before_first_spike": ["time [ms]"],
                                   "accommodation_index": ["accommodation index"],
                                   "average_AP_overshoot": ["voltage [mV]"],
                                   "average_AHP_depth": ["voltage [mV]"],
                                   "average_AP_width": ["time [ms]"],
                                  }

        self.features = SpikingFeatures(verbose_level="error")

        self.info = {"stimulus_start": t[0], "stimulus_end": t[-1]}

        self.t, self.spikes, info = self.features.preprocess(t, U, self.info)


    def test_initNone(self):
        self.features = SpikingFeatures()

        self.assertIsInstance(self.features, SpikingFeatures)
        self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, SpikingFeatures)
        self.assertIsNotNone(self.features.spikes)
        self.assertEqual(self.features.spikes.nr_spikes, 12)
        self.assertEqual(self.features.labels, self.implemented_labels)

    def test_initLabels(self):
        features = SpikingFeatures(labels={"nr_spikes": ["changed"],
                                           "new": ["new"]})

        labels = {"nr_spikes": ["changed"],
                  "spike_rate": ["spike rate [Hz]"],
                  "time_before_first_spike": ["time [ms]"],
                  'average_AP_width': ['time [ms]'],
                  "accommodation_index": ["accommodation index"],
                  "average_AP_overshoot": ["voltage [mV]"],
                  "average_AHP_depth": ["voltage [mV]"],
                  "new": ["new"]
                 }

        self.assertEqual(features.labels, labels)

    def test_features_to_run_all(self):
        features = SpikingFeatures(features_to_run="all")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_adaptive_all(self):
        features = SpikingFeatures(adaptive="all")
        self.assertEqual(set(features.adaptive), set(self.implemented_features))


    def test_implemented_features(self):
        self.assertEqual(set(self.features.implemented_features()), set(self.implemented_features))


    def test_nr_spikes(self):
        self.assertEqual(self.features.nr_spikes(self.t, self.spikes, self.info), (None, 12))


    def test_time_before_first_spike(self):
        self.assertGreater(self.features.time_before_first_spike(self.t, self.spikes, self.info)[1], 10)


    def test_time_before_first_spikeNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.time_before_first_spike(self.t, self.spikes, self.info), (None, None))


    def test_time_before_first_spike_no_strict(self):
        self.features.strict = False
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.time_before_first_spike(self.t, self.spikes, {}), (None, None))


    def test_time_before_first_spike_error(self):
        with self.assertRaises(RuntimeError):
            self.features.time_before_first_spike(self.t, self.spikes, {})
            self.features.time_before_first_spike(self.t, self.spikes, {"stimulus_end": 1})


    def test_spike_rate(self):
        self.assertEqual(self.features.spike_rate(self.t, self.spikes, self.info), (None, 0.12))


    def test_spike_rate_no_strict(self):
        self.features.strict = False
        self.assertEqual(self.features.spike_rate(self.t, self.spikes, {}), (None, 0.12))


    def test_spike_rateNone(self):
        self.features.spikes.nr_spikes = -1
        self.assertEqual(self.features.spike_rate(self.t, self.spikes, self.info), (None, None))


    def test_spike_rate_error(self):
        with self.assertRaises(RuntimeError):
            self.features.spike_rate(self.t, self.spikes, {})
            self.features.spike_rate(self.t, self.spikes, {"stimulus_start": 1})
            self.features.spike_rate(self.t, self.spikes, {"stimulus_end": 1})


    def test_average_AP_overshoot(self):
        self.assertEqual(self.features.average_AP_overshoot(self.t, self.spikes, self.info), (None, 30))


    def test_average_AP_overshootNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_overshoot(self.t, self.spikes, self.info), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_average_AHP_depth(self):
        self.features.average_AHP_depth(self.t, self.spikes, self.info)
        self.assertLess(self.features.average_AHP_depth(self.t, self.spikes, self.info)[1], 0)


    def test_average_AHP_depthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AHP_depth(self.t, self.spikes, self.info), (None, None))

    # TODO Find correct test, this is a rough bound only
    def test_average_AP_width(self):
        self.assertLess(self.features.average_AP_width(self.t, self.spikes, self.info)[1], 5)


    def test_average_AP_widthNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.average_AP_width(self.t, self.spikes, self.info), (None, None))


    # TODO Find correct test, this is a rough bound only
    def test_accommodation_index(self):
        self.assertIsNotNone(self.features.accommodation_index(self.t, self.spikes, self.info)[1])


    def test_accommodation_indexNone(self):
        self.features.spikes.nr_spikes = 0
        self.assertEqual(self.features.accommodation_index(self.t, self.spikes, self.info), (None, None))


    def test_calculate_all_features(self):
        result = self.features.calculate_all_features(self.t, self.spikes, self.info)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))




class TestEfelFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.implemented_features = efel.getFeatureNames()

        # self.implemented_labels = {"nr_spikes": ["number of spikes"],
        #                            "spike_rate": ["spike rate [Hz]"],
        #                            "time_before_first_spike": ["time [ms]"],
        #                            "accommodation_index": ["accommodation index"],
        #                            "average_AP_overshoot": ["voltage [mV]"],
        #                            "average_AHP_depth": ["voltage [mV]"],
        #                            "average_AP_width": ["time [ms]"]
        #                           }

        self.features = EfelFeatures()

        self.info = {}
        self.info["stimulus_start"] = self.t[0]
        self.info["stimulus_end"] = self.t[-10]

        # self.t, self.spikes = self.features.preprocess(t, U)


    # def test_initNone(self):
    #     self.features = SpikingFeatures()

    #     self.assertIsInstance(self.features, SpikingFeatures)
    #     self.assertIsNone(self.features.spikes)


    def test_init(self):
        self.assertIsInstance(self.features, EfelFeatures)

    # def test_initLabels(self):
    #     features = SpikingFeatures(labels={"nr_spikes": ["changed"],
    #                                        "new": ["new"]})

    #     labels = {"nr_spikes": ["changed"],
    #               "spike_rate": ["spike rate [Hz]"],
    #               "time_before_first_spike": ["time [ms]"],
    #               'average_AP_width': ['time [ms]'],
    #               "accommodation_index": ["accommodation index"],
    #               "average_AP_overshoot": ["voltage [mV]"],
    #               "average_AHP_depth": ["voltage [mV]"],
    #               "new": ["new"]
    #              }

    #     self.assertEqual(features.labels, labels)

    def test_features_to_run_all(self):
        features = EfelFeatures(features_to_run="all")
        self.assertEqual(set(features.features_to_run), set(self.implemented_features))


    def test_implemented_features(self):
        self.assertEqual(set(self.features.implemented_features()), set(self.implemented_features))


    def test_spikecount(self):
        t, U = self.features.Spikecount(self.t, self.U, self.info)

        self.assertIsNone(t)
        self.assertEqual(U, 12)


    def test_calculate_all_features(self):
        result = self.features.calculate_all_features(self.t, self.U, self.info)
        self.assertEqual(set(result.keys()),
                         set(self.implemented_features))



    def test_spikecount_error(self):
        with self.assertRaises(RuntimeError):
            self.features.Spikecount(self.t, self.U, {})
            self.features.Spikecount(self.t, self.U, {"stimulus_start": self.t[0]})
            self.features.Spikecount(self.t, self.U, {"stimulus_end": self.t[-1]})



class TestGeneralNetworkFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t_original = 8
        spiketrain = np.array([1, 3, 5, 6])
        self.U = [spiketrain, spiketrain, spiketrain, np.array([1])]


        self.features = NetworkFeatures()

    def test_initNone(self):
        self.features = GeneralNetworkFeatures()

        self.assertIsInstance(self.features, GeneralNetworkFeatures)


    def test_init(self):
        def feature(t, U):
            return "t", "U"

        features = GeneralNetworkFeatures(new_features=feature,
                                          features_to_run=None,
                                          adaptive=["cv"],
                                          labels={"cv": ["test"]},
                                          units="")

        self.assertIsInstance(features, GeneralNetworkFeatures)
        self.assertEqual(features.features_to_run, [])
        self.assertEqual(features.adaptive, ["cv"])
        self.assertEqual(features.units, "")
        self.assertEqual(set(features.implemented_features()),
                         set(["feature"]))


    def test_preprocess(self):
        self.features = GeneralNetworkFeatures()

        t, spiketrains = self.features.preprocess(self.t_original, self.U)

        self.assertEqual(t, self.t_original)
        self.assertIsInstance(spiketrains[0], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[1], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[2], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[3], neo.core.SpikeTrain)

        self.assertTrue(np.array_equal(spiketrains[0], self.U[0]))
        self.assertTrue(np.array_equal(spiketrains[1], self.U[1]))
        self.assertTrue(np.array_equal(spiketrains[2], self.U[2]))
        self.assertTrue(np.array_equal(spiketrains[3], self.U[3]))

        self.assertEqual(spiketrains[0].t_stop, self.t_original)


class TestNetworkFeatures(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t_original = 8
        spiketrain = np.array([1, 3, 5, 6])
        self.U = [spiketrain, spiketrain, spiketrain, np.array([1])]

        self.implemented_features = ["cv", "mean_cv", "binned_isi",
                                     "mean_isi", "local_variation", "mean_firing_rate",
                                     "fanofactor", "instantaneous_rate",
                                     "van_rossum_dist", "victor_purpura_dist",
                                     "corrcoef", "covariance", "mean_local_variation"]



        self.features = NetworkFeatures()

        self.t, self.spiketrains = self.features.preprocess(self.t_original, self.U)


    def test_initNone(self):
        self.features = NetworkFeatures()

        self.assertIsInstance(self.features, NetworkFeatures)


    def test_init(self):
        def feature(t, U):
            return "t", "U"

        features = NetworkFeatures(new_features=feature,
                                   features_to_run=None,
                                   adaptive=["cv"],
                                   labels={"cv": ["test"]},
                                   instantaneous_rate_nr_samples=-1.,
                                   isi_bin_size=-1,
                                   corrcoef_bin_size=-1,
                                   covariance_bin_size=-1,
                                   units=pq.ms)

        self.assertIsInstance(features, NetworkFeatures)
        self.assertEqual(features.features_to_run, [])
        self.assertEqual(features.adaptive, ["cv"])
        self.assertEqual(features.instantaneous_rate_nr_samples, -1)
        self.assertEqual(features.isi_bin_size, -1)
        self.assertEqual(features.corrcoef_bin_size, -1)
        self.assertEqual(features.covariance_bin_size, -1)
        self.assertEqual(features.units, pq.ms)
        self.assertEqual(set(features.implemented_features()),
                         set(["feature"] + self.implemented_features))


    def test_preprocess(self):
        self.features = NetworkFeatures()

        t, spiketrains = self.features.preprocess(self.t_original, self.U)

        self.assertEqual(t, self.t_original)
        self.assertIsInstance(spiketrains[0], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[1], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[2], neo.core.SpikeTrain)
        self.assertIsInstance(spiketrains[3], neo.core.SpikeTrain)

        self.assertTrue(np.array_equal(spiketrains[0], self.U[0]))
        self.assertTrue(np.array_equal(spiketrains[1], self.U[1]))
        self.assertTrue(np.array_equal(spiketrains[2], self.U[2]))
        self.assertTrue(np.array_equal(spiketrains[3], self.U[3]))

        self.assertEqual(spiketrains[0].t_stop, self.t_original)


    def test_cv(self):
        t, U = self.features.cv(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertTrue(np.array_equal([0.51207638319124049,
                                        0.51207638319124049,
                                        0.51207638319124049,
                                        0],
                                       U))

    def test_mean_cv(self):
        t, U = self.features.mean_cv(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(0.38405728739343037, U)


    def test_binned_isi(self):
        t, U = self.features.binned_isi(self.t, self.spiketrains)

        centers = np.arange(0, self.t_original + 1)[1:] - 0.5

        self.assertTrue(np.array_equal(centers, t))
        self.assertTrue(np.array_equal(U, [[0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 1, 2, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0]]))


    def test_mean_isi(self):
        t, U = self.features.mean_isi(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(1.66666666666666667, U)


    def test_lv(self):
        t, U = self.features.local_variation(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertTrue(np.array_equal([0.16666666666666666,
                                        0.16666666666666666,
                                        0.16666666666666666,
                                        None],
                                       U))

    def test_mean_lv(self):
        t, U = self.features.mean_local_variation(self.t, self.spiketrains)

        self.assertIsNone(t)
        mean = np.mean([0.16666666666666666, 0.16666666666666666, 0.16666666666666666])
        self.assertEqual(mean, U)


    def test_mean_firing_rate(self):
        t, U = self.features.mean_firing_rate(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertTrue(np.array_equal([500, 500, 500, 125], U))


    def test_instantaneous_rate(self):
        t, U = self.features.instantaneous_rate(self.t, self.spiketrains)

        rates = [201.7976071573258992, 270.7685498293570845,
                 345.7639606392203859, 420.3430010464927022,
                 486.7768188367210769, 537.5579992556051820,
                 567.1860438550103254, 573.7223146658689075,
                 559.5994608897037779, 531.3529919395189154,
                 498.2464223998736657, 470.1236785019651734,
                 455.0277266711299262, 457.2299460378653748,
                 476.1919997742103305, 506.7447214089696104,
                 540.4430043620118340, 567.7746503588338101,
                 580.6406099880132388, 574.4777562099577608,
                 549.4763516278245561, 510.6019227450684639,
                 466.4604005235715363, 427.3459903488383134,
                 402.9914602412301292, 400.5619369321500471,
                 423.3169253559639174, 470.1674223081157038,
                 536.1340587481936382, 613.5114750284128604,
                 693.3963029036556236, 767.1795921375620537,
                 827.6464085719730974, 869.5173515546268845,
                 889.4655805998627329, 885.8505116405049193,
                 858.4580587318871494, 808.4356634670250514,
                 738.4088696230276128, 652.5870483357980447,
                 556.6158845552261027, 457.0407336107435299,
                 360.4412580216865081, 272.4742619759052218,
                 197.1057281755237511, 136.2592191048793211,
                 89.9202782005130530, 56.5991363488668640,
                 33.9579695054921231, 19.4106207950416447]

        correct_U = [np.array(rates), np.array(rates),
                     np.array(rates), None]
        correct_t = np.linspace(0, 8, 51)[:-1]

        self.assertTrue(np.array_equal(U[0], rates))
        self.assertTrue(np.array_equal(U[1], rates))
        self.assertTrue(np.array_equal(U[2], rates))
        self.assertIsNone(U[3])
        self.assertTrue(np.array_equal(t, correct_t))



    def test_fanofactor(self):
        t, U = self.features.fanofactor(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(U, 0.51923076923076927)


    def test_van_rossum_dist(self):
        t, U = self.features.van_rossum_dist(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(U.shape, (4, 4))

        # correct_U = [[0.0, 5.9604644775390625e-08, 5.9604644775390625e-08, 2.9980016657780828],
        #              [5.9604644775390625e-08, 0.0, 5.9604644775390625e-08, 2.9980016657780828],
        #              [5.9604644775390625e-08, 5.9604644775390625e-08, 0.0, 2.9980016657780828],
        #              [2.9980016657780828, 2.9980016657780828, 2.9980016657780828, 0.0]]

        # self.assertTrue(np.array_equal(U, correct_U))

        diag = np.diag_indices(4)
        self.assertTrue(np.all(U[diag] == 0))

    def test_victor_purpura_dist(self):
        t, U = self.features.victor_purpura_dist(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(U.shape, (4, 4))
        diag = np.diag_indices(4)
        self.assertTrue(np.all(U[diag] == 0))


    def test_corrcoef(self):
        t, U = self.features.corrcoef(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(U.shape, (4, 4))
        diag = np.diag_indices(4)
        self.assertTrue(np.all(U[diag] == 1))

    def test_covariance(self):
        t, U = self.features.covariance(self.t, self.spiketrains)

        self.assertIsNone(t)
        self.assertEqual(U.shape, (4, 4))



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
        t, U,= self.features.feature_info(1, 2, "info")

        self.assertEqual(self.features.info, "info")


    def test_calculate_feature_error_one(self):
        with self.assertRaises(TypeError):
            self.features.calculate_feature("feature_error_one")


    def test_calculate_feature_error_one(self):
        with self.assertRaises(ValueError):
            self.features.calculate_feature("feature_error_value")


if __name__ == "__main__":
    unittest.main()
