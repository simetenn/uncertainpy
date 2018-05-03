from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import os
import yaml
import tqdm
import sys


class MyFormatter(logging.Formatter):
    """
    The logging formater.
    """
    # debug_format = "%(levelname)s - %(name)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    # info_format = "%(message)s"
    # warning_format = "%(levelname)s - %(message)s"
    # error_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    debug_format = "%(levelname)s - %(name)s - %(funcName)s - %(filename)s - %(lineno)d - %(message)s"
    info_format = "%(levelname)s - %(name)s - %(funcName)s - %(filename)s - %(lineno)d - %(message)s"
    warning_format = "%(levelname)s - %(name)s - %(funcName)s - %(filename)s - %(lineno)d - %(message)s"
    error_format = "%(levelname)s - %(name)s - %(funcName)s - %(filename)s - %(lineno)d - %(message)s"
    critical_format = "%(levelname)s - %(name)s - %(funcName)s - %(filename)s - %(lineno)d - %(message)s"

    debug_fmt = logging.Formatter(debug_format)
    info_fmt = logging.Formatter(info_format)
    warning_fmt = logging.Formatter(warning_format)
    error_fmt = logging.Formatter(error_format)
    critical_fmt = logging.Formatter(critical_format)


    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        super(MyFormatter, self).__init__(fmt)


    def format(self, record):
        if record.levelno == logging.DEBUG:
            return self.debug_fmt.format(record)
        elif record.levelno == logging.INFO:
            return self.info_fmt.format(record)
        elif record.levelno == logging.WARNING:
            return self.warning_fmt.format(record)
        elif record.levelno == logging.ERROR:
            return self.error_fmt.format(record)
        elif record.levelno == logging.CRITICAL:
            return self.critical_fmt.format(record)

class TqdmLoggingHandler(logging.StreamHandler):
    """
    Set logging so logging to  stream works with Tqdm,
    logging now uses tqdm.write.
    """
    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)


# Adapted from Logger.hasHandlers()
def has_handlers(logger):
    """
    See if this logger has any handlers configured.
    Loop through all handlers for this logger and its parents in the
    logger hierarchy. Return True if a handler was found, else False.
    Stop searching up the hierarchy whenever a logger with the "propagate"
    attribute set to zero is found - that will be the last logger which
    is checked for the existence of handlers.
    """
    current_logger = logger
    has_handler = False
    while current_logger:
        if current_logger.handlers:
            has_handler = True
            break
        if not current_logger.propagate:
            break
        else:
            current_logger = current_logger.parent
    return has_handler




def create_logger(level="info", name="logger", config_filename=""):
    """
    Create a logger object.

    Parameters
    ----------
    level : {"info", "debug", "warning", "error", "critical"}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. Default is info.
    name : str, optional
        Name of the logger. Default is logger.
    config_filename : {None, "", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). If None,
        no configuration is loaded. Default is "".

    Returns
    -------
    logger : Logger object
        The logger object.
    """
    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)

    logger.setLevel(numeric_level)

    if config_filename is not None:
        load_config(filename=config_filename)

    return logger


def load_config(filename=""):
    """
    Load logger configuration yaml file with `filename`.

    Parameters
    ----------
    filename : {"", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). Default is "".
    """
    if filename == "":
        folder = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(folder, "logging.yaml")

    with open(config_file) as f:
        logger_config = yaml.safe_load(f)

    logging.config.dictConfig(logger_config)

