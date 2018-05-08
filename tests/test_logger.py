import unittest
import os
import shutil
import logging
import os.path
import time

from uncertainpy.utils.logger import setup_module_logger, has_handlers, setup_logger, add_screen_handler, add_file_handler
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

        # for handler in handlers:
        #     handler.close()
        #     main_logger.removeHandler(handler)

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


    def test_add_screen_handler(self):
        logger = logging.getLogger("screen")

        add_screen_handler("screen")

        self.assertEqual(len(logger.handlers), 1)

        add_screen_handler("screen")
        self.assertEqual(len(logger.handlers), 1)


    def test_add_file_handler(self):
        logger = logging.getLogger("file")

        add_file_handler("file")

        self.assertEqual(len(logger.handlers), 1)

        add_file_handler("file")
        self.assertEqual(len(logger.handlers), 1)

    def test_add_screen_handler(self):
        logger = logging.getLogger("screen")

        add_screen_handler("screen")

        self.assertEqual(len(logger.handlers), 1)

        add_screen_handler("screen")
        self.assertEqual(len(logger.handlers), 1)


    def test_add_file_and_screen_handlers(self):
        logger = logging.getLogger("both")

        add_screen_handler("both")

        self.assertEqual(len(logger.handlers), 1)

        add_file_handler("both")
        self.assertEqual(len(logger.handlers), 2)



    def test_setup_logger(self):
        setup_logger("uncertainpy.test1", level="warning")
        logger = logging.getLogger("uncertainpy.test1")

        level = logger.getEffectiveLevel()
        self.assertEqual(level, 30)
        self.assertEqual(len(logger.handlers), 0)



    def test_setup_logger_debug(self):
        setup_logger("uncertainpy", level="debug")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 5)


    def test_setup_logger_info(self):
        setup_logger("uncertainpy", level="info")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)


    def test_setup_logger_warning(self):
        setup_logger("uncertainpy", level="warning")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 3)



    def test_setup_logger_error(self):
        setup_logger("uncertainpy", level="error")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 2)


    def test_setup_logger_critical(self):
        setup_logger("uncertainpy", level="critical")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 1)



    def test_setup_logger_debug(self):
        setup_logger("uncertainpy", level="debug")

        logger = logging.getLogger("uncertainpy")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 5)


    def test_setup_setup_module_logger(self):

        model = Model(logger_level=None)
        setup_module_logger(class_instance=model, level="info")

        logger = logging.getLogger("uncertainpy.models.model.Model")

        add_file_handler("uncertainpy", filename=self.full_path)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)



    def test_setup_setup_module_logger_external(self):

        model = TestingModel1d()
        setup_module_logger(class_instance=model, level="info")
        logger = logging.getLogger("uncertainpy.tests.testing_classes.testing_models.TestingModel1d")


        add_file_handler("uncertainpy", filename=self.full_path)


        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        time.sleep(0.1)

        self.assertEqual(len(open(self.full_path).readlines()), 4)



if __name__ == "__main__":
    unittest.main()
