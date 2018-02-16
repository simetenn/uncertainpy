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


    def test_coffee_function(self):
        folder = os.path.join(self.example_folder, "coffee_cup")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffee_function.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_coffee_class(self):
        folder = os.path.join(self.example_folder, "coffee_cup")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffee_class.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_coffee_dependent_function(self):
        folder = os.path.join(self.example_folder, "coffee_cup_dependent")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffee_dependent_function.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_coffee_dependent_class(self):
        folder = os.path.join(self.example_folder, "coffee_cup_dependent")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_coffee_dependent_class.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_hodgkin_huxley(self):
        folder = os.path.join(self.example_folder, "hodgkin_huxley")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_hodgkin_huxley.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_izhikevich_function(self):
        folder = os.path.join(self.example_folder, "izhikevich")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_izhikevich_function.py"], env=os.environ.copy())
        self.assertEqual(result, 0)


    def test_izhikevich_class(self):
        folder = os.path.join(self.example_folder, "izhikevich")
        os.chdir(folder)

        result = subprocess.call(["python", "uq_izhikevich_class.py"], env=os.environ.copy())
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

