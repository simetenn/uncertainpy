import os
import unittest
import subprocess


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.dirname(os.path.realpath(__file__))

        self.example_folder = os.path.join(os.path.dirname(self.folder),
                                           "examples")
        self.current_dir = os.getcwd()


    def test_bahl(self):
        folder = os.path.join(self.example_folder, "bahl")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_bahl.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_brunel(self):
        folder = os.path.join(self.example_folder, "brunel")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_brunel.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_coffee(self):
        folder = os.path.join(self.example_folder, "coffee")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffe.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_coffee_dependent(self):
        folder = os.path.join(self.example_folder, "coffee_dependent")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffe_dependent.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_hodgkin_huxley(self):
        folder = os.path.join(self.example_folder, "hodgkin_huxley")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_hodgkin_huxley.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_izhikevich(self):
        folder = os.path.join(self.example_folder, "izhikevich")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_izhikevich.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_interneuron(self):
        folder = os.path.join(self.example_folder, "interneuron")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_interneuron.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_valderrama(self):
        folder = os.path.join(self.example_folder, "valderrama")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_valderrama.py"], env=os.environ.copy())
        self.assertEqual(result, 0)
