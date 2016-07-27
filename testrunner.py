import unittest
import os

folder = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(folder, "uncertainpy")
tests = unittest.TestLoader().discover(path)

test_runner = unittest.TextTestRunner()
test_runner.run(tests)
