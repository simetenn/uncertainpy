import numpy as np
import os
import unittest
import subprocess

from uncertainpy.models import TestingModel0d, TestingModel1d, TestingModel2d

class TestRunModel(unittest.TestCase):
    def test_runModel0d(self):
        model = TestingModel0d()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 1, "b": 2}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ut, err = simulation.communicate()

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        self.assertEqual(t, 1)
        self.assertEqual(U, 2)



    def test_runModel1d(self):
        model = TestingModel1d()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 2}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ut, err = simulation.communicate()

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) + 4))


    def test_runModel2d(self):
        model = TestingModel2d()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))


        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ut, err = simulation.communicate()

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.array([np.arange(0, 10) + 2,
                                                    np.arange(0, 10) + 3])))


if __name__ == "__main__":
    unittest.main()
