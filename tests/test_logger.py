import unittest
import os
import shutil
import logging
import os.path
import time

from uncertainpy.utils.logger import setup_module_logging, has_handlers, setup_logging
from uncertainpy import Model

from .testing_classes import TestingModel1d

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.output_test_dir = ".tests/"
        self.logfile = "uncertainpy.log"
        self.full_path = os.path.join(self.output_test_dir, self.logfile)

        logger = logging.getLogger("uncertainpy")
        logger.handlers = []

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)
        os.makedirs(self.output_test_dir)



    def tearDown(self):
        logger = logging.getLogger("uncertainpy")
        logger.handlers = []

        if os.path.isdir(self.output_test_dir):
            shutil.rmtree(self.output_test_dir)



    def test_has_handlers(self):
        logger = logging.getLogger("has_handlers")
        stream = logging.StreamHandler()
        logger.addHandler(stream)

        logging.getLogger("has_handlers.test")
        logger = logging.getLogger("has_handlers.test.test")

        result = has_handlers(logger)
        self.assertTrue(result)

    def test_has_handlers_no_propagate(self):
        logger = logging.getLogger("has_handlers")
        stream = logging.StreamHandler()
        logger.addHandler(stream)
        logger_1 = logging.getLogger("has_handlers.test")
        logger_1.propagate = False
        logger = logging.getLogger("has_handlers.test.test")

        result = has_handlers(logger)
        self.assertFalse(result)


    def test_has_handlers_none(self):
        logging.getLogger("has_handlers")
        logging.getLogger("has_handlers.test")
        logger = logging.getLogger("has_handlers.test.test")

        result = has_handlers(logger)
        self.assertFalse(result)



    def test_create_logger(self):
        setup_logging("uncertainpy.test1", level="info", filename=self.full_path)
        logger = logging.getLogger("uncertainpy.test1")

        level = logger.getEffectiveLevel()
        self.assertEqual(level, 20)
        logger = logging.getLogger("uncertainpy")
        self.assertEqual(len(logger.handlers), 2)


    def test_create_logger_config(self):

        setup_logging("test_logger", level="info", filename=self.full_path)
        logger = logging.getLogger("test_logger")
        level = logger.getEffectiveLevel()
        self.assertEqual(level, 20)
        logger = logging.getLogger("uncertainpy")
        self.assertEqual(len(logger.handlers), 2)


    def test_setup_logging_debug(self):


        setup_logging("uncertainpy", level="debug", filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 5)

    def test_setup_logging_info(self):
        setup_logging("uncertainpy", level="info", filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)


    def test_setup_logging_warning(self):
        setup_logging("uncertainpy", level="warning", filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        # print(open(self.full_path).readlines())
        self.assertEqual(len(open(self.full_path).readlines()), 3)


    def test_setup_logging_error(self):
        setup_logging("uncertainpy", level="error", filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 2)


    def test_setup_logging_critical(self):
        setup_logging("uncertainpy", level="critical", filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 1)


    def test_setup_logging_None(self):
        setup_logging("uncertainpy", level=None, filename=self.full_path)

        logger = logging.getLogger("uncertainpy")

        self.assertEqual(len(logger.handlers), 0)

        logger.debug("debug message")

        time.sleep(0.1)

        self.assertFalse(os.path.exists(self.full_path))

    def test_setup_setup_module_logging(self):

        model = Model(logger_level=None)
        setup_module_logging(class_instance=model, level="info", filename=self.full_path)

        logger = logging.getLogger("uncertainpy.models.model.Model")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)

    def test_setup_setup_module_logging_external(self):

        model = TestingModel1d()
        setup_module_logging(class_instance=model, level="info", filename=self.full_path)
        logger = logging.getLogger("uncertainpy.tests.testing_classes.testing_models.TestingModel1d")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)

    def test_setup_logging_no_file(self):
        setup_logging("uncertainpy", level="info", filename=None)

        logger = logging.getLogger("uncertainpy")
        self.assertEqual(len(logger.handlers), 1)





if __name__ == "__main__":
    unittest.main()
