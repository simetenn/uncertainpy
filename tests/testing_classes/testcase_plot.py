import unittest
import os
import subprocess

class TestCasePlot(unittest.TestCase):
    def __init__(self, test_name, parameter=False):
        super(TestCasePlot, self).__init__(test_name)

        self.exact_plots = parameter

        self.figureformat = ".png"
        self.output_test_dir = ".tests/"


    def compare_plot(self, name):
        if self.exact_plots:
            folder = os.path.dirname(os.path.realpath(__file__))
            compare_file = os.path.join(folder, "figures",
                                        name + self.figureformat)

            plot_file = os.path.join(self.output_test_dir, name + self.figureformat)

            result = subprocess.call(["diff", plot_file, compare_file])
            self.assertEqual(result, 0)
        else:
            plot_file = os.path.join(self.output_test_dir, name + self.figureformat)
            self.assertTrue(os.path.isfile(plot_file))