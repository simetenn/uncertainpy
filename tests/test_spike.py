import os
import unittest
import shutil
import subprocess

import numpy as np

from uncertainpy.features.spikes import Spike

from .testing_classes import TestCasePlot

class TestSpike(TestCasePlot):
    def setUp(self):
        time = np.arange(0, 10)
        V = np.arange(0, 10) + 10
        time_spike = 5
        V_spike = 10
        global_index = 50

        self.spike = Spike(time, V, time_spike, V_spike, global_index,
                           xlabel="time", ylabel="voltage")

        self.output_test_dir = ".tests/"
        self.figureformat = ".png"

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_init(self):
        self.assertIsInstance(self.spike, Spike)
        self.assertTrue(np.array_equal(self.spike.time, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.spike.V, np.arange(0, 10) + 10))
        self.assertEqual(self.spike.time_spike, 5)
        self.assertEqual(self.spike.V_spike, 10)
        self.assertEqual(self.spike.global_index, 50)


    def test_plot(self):
        self.spike.plot(os.path.join(self.output_test_dir, "spike.png"))

        self.compare_plot("spike")


    def test_str(self):
        self.assertIsInstance(str(self.spike), str)


    def test_trim(self):
        V = np.concatenate([np.arange(0, 11), np.arange(0, 10)[::-1]])
        time = np.arange(0, len(V))
        time_spike = 5
        V_spike = 10
        global_index = 50

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")

        spike.trim(5)

        self.assertTrue(np.array_equal(spike.time, np.arange(5, 16)))
        self.assertTrue(np.array_equal(spike.V, [5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5]))
        self.assertEqual(spike.V_spike, 10)
        self.assertEqual(spike.time_spike, 5)
        self.assertEqual(spike.global_index, 50)


        V = np.concatenate([np.arange(0, 11), np.arange(0, 10)[::-1]])
        time = np.arange(0, len(V))
        time_spike = 5
        V_spike = 10
        global_index = 50

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")

        spike.trim(-5)

        self.assertTrue(np.array_equal(spike.time, time))
        self.assertTrue(np.array_equal(spike.V, V))
        self.assertEqual(spike.V_spike, V_spike)
        self.assertEqual(spike.time_spike, time_spike)
        self.assertEqual(spike.global_index, global_index)


        V = np.concatenate([np.arange(0, 11), np.arange(0, 10)[::-1]])
        time = np.arange(0, len(V))
        time_spike = 5
        V_spike = 10
        global_index = 50

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")

        # with self.assertRaises(RuntimeError):
        spike.trim(15)



    def test_trim_min(self):

        V = np.concatenate([np.arange(0, 11), np.arange(0, 10)[::-1]])
        time = np.arange(0, len(V))
        time_spike = 5
        V_spike = 10
        global_index = 50

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")

        spike.trim(7, min_extent_from_peak=5)

        self.assertTrue(np.array_equal(spike.time, np.arange(5, 16)))
        self.assertTrue(np.array_equal(spike.V, [5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5]))
        self.assertEqual(spike.V_spike, 10)
        self.assertEqual(spike.time_spike, 5)
        self.assertEqual(spike.global_index, 50)



    def test_trim_asymetric_min(self):

        V = np.array([0.401, 0.399, 0.381, 0.373, 0.368, 0.371])
        time = np.arange(0, len(V))
        time_spike = 5
        V_spike = 0.401
        global_index = 50

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")

        spike.trim(0.4)

        self.assertTrue(np.array_equal(spike.time, np.arange(0, 2)))
        self.assertTrue(np.array_equal(spike.V, [0.401, 0.399]))
        self.assertEqual(spike.V_spike, 0.401)
        self.assertEqual(spike.time_spike, 5)
        self.assertEqual(spike.global_index, 50)



    def test_add(self):

        time = np.arange(5, 15)
        V = np.arange(0, 10) - 10
        time_spike = 10
        V_spike = 5
        global_index = 150

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")


        new_spike = self.spike + spike

        self.assertTrue(np.array_equal(new_spike.time, np.arange(0, 15)))
        self.assertTrue(np.array_equal(new_spike.V, np.concatenate([np.arange(0, 10) + 10, np.arange(5, 10) - 10])))
        self.assertEqual(new_spike.V_spike, 10)
        self.assertEqual(new_spike.time_spike, 5)
        self.assertEqual(new_spike.global_index, 50)

        new_spike = spike + self.spike

        self.assertTrue(np.array_equal(new_spike.time, np.arange(0, 15)))
        self.assertTrue(np.array_equal(new_spike.V, np.concatenate([np.arange(0, 10) + 10, np.arange(5, 10) - 10])))
        self.assertEqual(new_spike.V_spike, 10)
        self.assertEqual(new_spike.time_spike, 5)
        self.assertEqual(new_spike.global_index, 50)


        time = np.arange(0, 15)
        V = np.arange(0, 15) - 10
        time_spike = 10
        V_spike = 5
        global_index = 150

        spike = Spike(time, V, time_spike, V_spike, global_index,
                      xlabel="time", ylabel="voltage")


        new_spike = spike + self.spike

        self.assertTrue(np.array_equal(new_spike.time, time))
        self.assertTrue(np.array_equal(new_spike.V, V))
        self.assertEqual(new_spike.V_spike, V_spike)
        self.assertEqual(new_spike.time_spike, time_spike)
        self.assertEqual(new_spike.global_index, global_index)

        new_spike = self.spike + spike

        self.assertTrue(np.array_equal(new_spike.time, time))
        self.assertTrue(np.array_equal(new_spike.V, V))
        self.assertEqual(new_spike.V_spike, V_spike)
        self.assertEqual(new_spike.time_spike, time_spike)
        self.assertEqual(new_spike.global_index, global_index)