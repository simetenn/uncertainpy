import unittest
import os
import shutil

from uncertainpy.utils import create_logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.logfile = "test.log"
        self.full_path = os.path.join(self.output_test_dir, self.logfile)

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)


    def tearDown(self):
        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)


    def test_debug(self):
        self.logger = create_logger("debug",
                                    self.full_path,
                                    "test_logger")

        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")
        self.logger.error("error message")
        print self.logger.handlers

        message = """DEBUG - test_logger - test_logger.py - 29 - debug message
info message
WARNING - test_logger - warning message
ERROR - test_logger - test_logger.py - 32 - error message"""

        self.assertTrue(message in open(self.full_path).read())


    def test_info(self):
        self.logger = create_logger("info",
                                    self.full_path,
                                    "test_logger")

        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")
        self.logger.error("error message")


        message = """info message
WARNING - test_logger - warning message
ERROR - test_logger - test_logger.py - 51 - error message"""

        self.assertTrue(message in open(self.full_path).read())


    def test_warning(self):
        self.logger = create_logger("warning",
                                    self.full_path,
                                    "test_logger")

        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")
        self.logger.error("error message")

        message = """WARNING - test_logger - warning message
ERROR - test_logger - test_logger.py - 69 - error message"""

        self.assertTrue(message in open(self.full_path).read())


    def test_error(self):
        self.logger = create_logger("error",
                                    self.full_path,
                                    "test_logger")

        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")
        self.logger.error("error message")

        message = "ERROR - test_logger - test_logger.py - 85 - error message"

        self.assertTrue(message in open(self.full_path).read())



if __name__ == "__main__":
    unittest.main()
