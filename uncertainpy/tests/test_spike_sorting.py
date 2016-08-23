import numpy as np
import os
import unittest

from uncertainpy.spikes import Spike, Spikes

class TestSpike(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 10
        t_spike = 5
        U_spike = 10
        global_index = 50

        self.spike = Spike(t, U, t_spike, U_spike, global_index)

    def test_init(self):
        self.assertIsInstance(self.spike, Spike)
        self.assertTrue(np.array_equal(self.spike.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.spike.U, np.arange(0, 10) + 10))
        self.assertEqual(self.spike.t_spike, 5)
        self.assertEqual(self.spike.U_spike, 10)
        self.assertEqual(self.spike.global_index, 50)


class TestSpikes(unittest.TestCase):
    def setUp(self):
        folder = os.path.dirname(os.path.realpath(__file__))

        self.t = np.load(os.path.join(folder, "data/t_test.npy"))
        self.U = np.load(os.path.join(folder, "data/U_test.npy"))


    def test_init(self):
        self.spikes = Spikes()
        self.assertIsInstance(self.spikes, Spikes)


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


if __name__ == "__main__":
    unittest.main()
