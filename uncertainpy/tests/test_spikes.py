import numpy as np
import os
import unittest
import shutil
import subprocess

from uncertainpy.features.spikes import Spike, Spikes


class TestSpikes(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))

        self.output_test_dir = ".tests/"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def compare_plot(self, name):
        folder = os.path.dirname(os.path.realpath(__file__))
        compare_file = os.path.join(folder, "data/",
                                    name + ".png")

        plot_file = os.path.join(self.output_test_dir, name + ".png")

        result = subprocess.call(["diff", plot_file, compare_file])
        self.assertEqual(result, 0)


    def test_init_no_input(self):
        self.spikes = Spikes()
        self.assertIsInstance(self.spikes, Spikes)


    def test_init_input(self):
        self.spikes = Spikes(self.t, self.U)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_detectSpikesDefault(self):
        self.spikes = Spikes()

        self.spikes.detectSpikes(self.t, self.U)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_detectSpikesAuto(self):
        self.spikes = Spikes()

        self.spikes.detectSpikes(self.t, self.U, thresh="auto")
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_detectSpikesExtended(self):
        self.spikes = Spikes()

        self.spikes.detectSpikes(self.t, self.U, extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_detectSpikesDefaultAutoExtended(self):
        self.spikes = Spikes()

        self.spikes.detectSpikes(self.t, self.U, thresh="auto", extended_spikes=True)
        self.assertEqual(self.spikes.nr_spikes, 12)


    def test_iter(self):
        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U)

        for spike in self.spikes:
            self.assertIsInstance(spike, Spike)

    def test_len(self):
        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U)

        self.assertEqual(len(self.spikes), 12)


    def test_getitem(self):
        self.spikes = Spikes()
        self.spikes.detectSpikes(self.t, self.U)

        result = self.spikes[0]
        self.assertIsInstance(result, Spike)


    def test_plot(self):
        self.spikes = Spikes(self.t, self.U, xlabel="xlabel", ylabel="ylabel")

        self.spikes.plot(os.path.join(self.output_test_dir, "spikes.png"))
        self.compare_plot("spikes")


    def test_plotExtended(self):
        self.spikes = Spikes(self.t, self.U, xlabel="xlabel", ylabel="ylabel", extended_spikes=True)

        self.spikes.plot(os.path.join(self.output_test_dir, "spikes_extended.png"))
        self.compare_plot("spikes_extended")


if __name__ == "__main__":
    unittest.main()
