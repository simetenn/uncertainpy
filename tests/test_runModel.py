import numpy as np
import os
import unittest
import subprocess

from TestingModel import TestingModel0d, TestingModel1d, TestingModel2d
from TestingModel import TestingModel0dNoTime, TestingModel1dNoTime
from TestingModel import TestingModel2dNoTime, TestingModelNoU



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

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()

        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)

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

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()

        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


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


        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()


        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        self.assertTrue(np.array_equal(t, np.arange(0, 10)))
        self.assertTrue(np.array_equal(U, np.array([np.arange(0, 10) + 2,
                                                    np.arange(0, 10) + 3])))


    def test_runModel0dNoTime(self):
        model = TestingModel0dNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 1, "b": 2}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()


        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


        self.assertEqual(simulation.returncode, 0)

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        self.assertTrue(np.isnan(t))
        self.assertEqual(U, 2)



    def test_runModel1dNoTime(self):
        model = TestingModel1dNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 2}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))

        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()

        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


        self.assertEqual(simulation.returncode, 0)

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))


        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(U, np.arange(0, 10) + 4))


    def test_runModel2dNoTime(self):
        model = TestingModel2dNoTime()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))


        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()


        if simulation.returncode != 0:
            print ut
            raise RuntimeError(err)


        self.assertEqual(simulation.returncode, 0)

        U = np.load(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        t = np.load(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        os.remove(os.path.join(filedir, ".tmp_U_%s.npy" % current_process))
        os.remove(os.path.join(filedir, ".tmp_t_%s.npy" % current_process))

        self.assertTrue(np.isnan(t))
        self.assertTrue(np.array_equal(U, np.array([np.arange(0, 10) + 2,
                                                    np.arange(0, 10) + 3])))


    def test_runModelNoU(self):
        model = TestingModelNoU()

        current_process = "0"
        filepath = os.path.abspath(__file__)
        filedir = os.path.dirname(filepath)

        cmd = model.cmd() + ["--CPU", current_process, "--save_path", filedir, "--parameters"]

        tmp_parameters = {"a": 2, "b": 3}

        for parameter in tmp_parameters:
            cmd.append(parameter)
            cmd.append("{:.16f}".format(tmp_parameters[parameter]))


        simulation = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
        ut, err = simulation.communicate()

        exc = err.split("\n")[-2].split(":")[0]
        msg = err.split("\n")[-2].split(":")[-1].strip()

        self.assertEqual(exc, "ValueError")
        self.assertEqual(msg, "U has not been calculated")
        self.assertNotEqual(simulation.returncode, 0)



if __name__ == "__main__":
    unittest.main()
