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

