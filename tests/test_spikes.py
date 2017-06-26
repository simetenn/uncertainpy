import os
import unittest
import shutil
import subprocess
import numpy as np


import matplotlib
matplotlib.use("Agg")

from .testing_classes import TestCaseExact

from uncertainpy.features.spikes import Spike, Spikes



class TestSpikes(TestCaseExact):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)

        self.figureformat = ".png"


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def compare_plot(self, name):
        if self.exact_plots:
            folder = os.path.dirname(os.path.realpath(__file__))
            compare_file = os.path.join(folder, "figures",
                                        name + self.figureformat)

            plot_file = os.path.join(self.output_test_dir, name + self.figureformat)

            result = subprocess.call(["diff", plot_file, compare_file])
            self.assertEqual(result, 0)
        else:
            plot_file = os.path.join(self.output_test_dir, name + self.figureformat)
            self.assertTrue(os.path.isfile(plot_file))


    def test_init_no_input(self):
        self.spikes = Spikes()
        self.assertIsInstance(self.spikes, Spikes)


    def test_init_input(self):
        self.spikes = Spikes(self.t, self.U)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_default(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.t, self.U)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_auto(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.t, self.U, thresh="auto")
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_extended(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.t, self.U, extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_find_spikes_default_auto_extended(self):
        self.spikes = Spikes()

        self.spikes.find_spikes(self.t, self.U, thresh="auto", extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_iter(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.t, self.U)

        for spike in self.spikes:
            self.assertIsInstance(spike, Spike)

    def test_len(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.t, self.U)

        self.assertEqual(len(self.spikes), 12)


    def test_getitem(self):
        self.spikes = Spikes()
        self.spikes.find_spikes(self.t, self.U)

        result = self.spikes[0]
        self.assertIsInstance(result, Spike)


    def test_plot(self):
        self.spikes = Spikes(self.t, self.U, xlabel="xlabel", ylabel="ylabel")

        self.spikes.plot(os.path.join(self.output_test_dir, "spikes.png"))
        self.plot_exists("spikes")


    def test_plot_extended(self):
        self.spikes = Spikes(self.t, self.U, xlabel="xlabel", ylabel="ylabel", extended_spikes=True)

        self.spikes.plot(os.path.join(self.output_test_dir, "spikes_extended.png"))
        self.plot_exists("spikes_extended")


    def plot_exists(self, name):
        plot_file = os.path.join(self.output_test_dir, name + ".png")
        self.assertTrue(os.path.isfile(plot_file))


if __name__ == "__main__":
    unittest.main()
