import unittest
import os
import sys

folder = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(folder, "uncertainpy")
tests = unittest.TestLoader().discover(path)

test_runner = unittest.TextTestRunner()
results = test_runner.run(tests)

sys.exit(results.wasSuccessful())
