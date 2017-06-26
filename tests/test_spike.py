import os
import unittest
import shutil
import subprocess

import numpy as np

from uncertainpy.features.spikes import Spike

from .testing_classes import TestCasePlot

class TestSpike(TestCasePlot):
    def setUp(self):
        t = np.arange(0, 10)
        U = np.arange(0, 10) + 10
        t_spike = 5
        U_spike = 10
        global_index = 50

        self.spike = Spike(t, U, t_spike, U_spike, global_index,
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
        self.assertTrue(np.array_equal(self.spike.t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(self.spike.U, np.arange(0, 10) + 10))
        self.assertEqual(self.spike.t_spike, 5)
        self.assertEqual(self.spike.U_spike, 10)
        self.assertEqual(self.spike.global_index, 50)


    def test_plot(self):
        self.spike.plot(os.path.join(self.output_test_dir, "spike.png"))

        self.compare_plot("spike")
