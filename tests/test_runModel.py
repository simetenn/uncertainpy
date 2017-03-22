import numpy as np
import os
import unittest
import subprocess

from testing_classes import TestingModel0d, TestingModel1d, TestingModel2d
from testing_classes import TestingModelNoTime, TestingModelNoTimeU
from testing_classes import TestingModelAdaptive, TestingModelConstant
from testing_classes import TestingModelNewProcess


class TestRunModel(unittest.TestCase):
    def run_subprocess(self, model, tmp_parameters):
        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=os.environ.copy())
        ut, err = simulation.communicate()

        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        return t, U


    def test_runModel0d(self):
        model = TestingModel0d()
        tmp_parameters = {"a": 1, "b": 2}

        t, U = self.run_subprocess(model, tmp_parameters)

        self.assertEqual(t, 1)
        self.assertEqual(U, 2)



    def test_runModel1d(self):
        model = TestingModel1d()
        tmp_parameters = {"a": 2, "b": 2}

        t, U = self.run_subprocess(model, tmp_parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) + 4))




    def test_runModel2d(self):
        model = TestingModel2d()
        tmp_parameters = {"a": 2, "b": 3}

        t, U = self.run_subprocess(model, tmp_parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.array([np.arange(0, 10) + 2,
                                                    np.arange(0, 10) + 3])))


    def test_runModelNewProcess(self):
        model = TestingModelNewProcess()
        tmp_parameters = {"a": 2, "b": 2}

        t, U = self.run_subprocess(model, tmp_parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) + 4))


    def test_runModelAdaptive(self):
        model = TestingModelAdaptive()

        tmp_parameters = {"a": 2, "b": 2}

        t, U = self.run_subprocess(model, tmp_parameters)

        self.assertTrue(np.array_equal(t, np.arange(0, 10 + 4)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10 + 4) + 4))



    def test_runModelConstant(self):
        model = TestingModelConstant()

        tmp_parameters = {"a": 2, "b": 2}

        t, U = self.run_subprocess(model, tmp_parameters)


        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.arange(0, 10)))



    def test_runModelNoTime(self):
        model = TestingModelNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))


        simulation = subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=os.environ.copy())
        ut, err = simulation.communicate()

        exc = err.split("\n")[-2].split(":")[0]
        msg = err.split("\n")[-2].split(":")[-1].strip()

        self.assertEqual(exc, "RuntimeError")
        self.assertEqual(msg, "model.run() must return t and U (return t, U | return None, U)")
        self.assertNotEqual(simulation.returncode, 0)


    def test_runModelNoTimeU(self):
        model = TestingModelNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=os.environ.copy())
        ut, err = simulation.communicate()

        exc = err.split("\n")[-2].split(":")[0]
        msg = err.split("\n")[-2].split(":")[-1].strip()

        self.assertEqual(exc, "RuntimeError")
        self.assertEqual(msg, "model.run() must return t and U (return t, U | return None, U)")
        self.assertNotEqual(simulation.returncode, 0)


    def test_runModelWrongParameters(self):
        model = TestingModelNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))


        simulation = subprocess.Popen(cmd[:-1],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=os.environ.copy())
        ut, err = simulation.communicate()

        exc = err.split("\n")[-2].split(":")[0]
        msg = err.split("\n")[-2].split(":")[-1].strip()

        self.assertEqual(exc, "ValueError")
        self.assertEqual(msg, "Number of parameters does not match number of parametervalues sent to simulation")
        self.assertNotEqual(simulation.returncode, 0)


if __name__ == "__main__":
    unittest.main()
