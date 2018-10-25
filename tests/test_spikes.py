import os
import unittest
import shutil
import subprocess
import numpy as np


# import matplotlib
# matplotlib.use("Agg")


from uncertainpy.features.spikes import Spike, Spikes

from .testing_classes import TestCasePlot

class TestSpikes(TestCasePlot):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.time = np.load(os.path.join(folder, "data/t_test.npy"))
        self.values = np.load(os.path.join(folder, "data/U_test.npy"))

        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.figureformat = ".png"


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init_no_input(self):
        self.spikes = Spikes()
        self.assertIsInstance(self.spikes, Spikes)


    def test_init_input(self):
        self.spikes = Spikes(self.time, self.values)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_default(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_auto(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, threshold="auto")
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_min_amplitude(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, min_amplitude=1000)
        self.assertEqual(self.spikes.nr_spikes, 0)


    def test_find_spikes_min_duration(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, min_duration=1000)
        self.assertEqual(self.spikes.nr_spikes, 0)


    def test_str(self):
        self.spikes = Spikes(self.time, self.values)
        self.assertIsInstance(str(self.spikes), str)


    def test_find_spikes_extended(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_default_auto_extended(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, threshold="auto", extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_normalize(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.time, self.values, threshold=0.4, end_threshold=-0.1, normalize=True)
        self.assertEqual(self.spikes.nr_spikes, 12)
        self.assertTrue(self.spikes[0].V_spike > 1)


    def test_find_spikes_normalize_error(self):
        self.spikes = Spikes()

        with self.assertRaises(ValueError):
            self.spikes.find_spikes(self.time, self.values, threshold=12, end_threshold=-0.1, normalize=True)

        with self.assertRaises(ValueError):
            self.spikes.find_spikes(self.time, self.values, threshold=0.1, end_threshold=12, normalize=True)


    def test_iter(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.time, self.values)

        for spike in self.spikes:
            self.assertIsInstance(spike, Spike)


    def test_len(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.time, self.values)

        self.assertEqual(len(self.spikes), 12)


    def test_getitem(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.time, self.values)

        result = self.spikes[0]
        self.assertIsInstance(result, Spike)


    def test_plot_spikes(self):
        self.spikes = Spikes(self.time, self.values, xlabel="xlabel", ylabel="ylabel")

        self.spikes.plot_spikes(os.path.join(self.output_test_dir, "spikes.png"))
        self.plot_exists("spikes")


    def test_plot_voltage(self):
        self.spikes = Spikes(self.time, self.values, xlabel="xlabel", ylabel="ylabel")

        self.spikes.plot_spikes(os.path.join(self.output_test_dir, "voltage.png"))
        self.plot_exists("voltage")

    def test_plot_extended(self):
        self.spikes = Spikes(self.time, self.values, xlabel="xlabel", ylabel="ylabel", extended_spikes=True)

        self.spikes.plot_spikes(os.path.join(self.output_test_dir, "spikes_extended.png"))
        self.plot_exists("spikes_extended")


    def plot_exists(self, name):
        plot_file = os.path.join(self.output_test_dir, name + ".png")
        self.assertTrue(os.path.isfile(plot_file))


    def test_bad_start(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        time = np.load(os.path.join(folder, "data/t_spike.npy"))
        values = np.load(os.path.join(folder, "data/V_spike.npy"))

        spikes = Spikes()
        spikes.find_spikes(time, values)

        self.assertEqual(len(spikes), 14)


    def test_noisy(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        time = np.load(os.path.join(folder, "data/t_noise.npy"))
        values = np.load(os.path.join(folder, "data/V_noise.npy"))

        spikes = Spikes()
        spikes.find_spikes(time, values)

        spikes.plot_voltage("voltage.png")

        self.assertEqual(len(spikes), 16)





if __name__ == "__main__":
    unittest.main()
