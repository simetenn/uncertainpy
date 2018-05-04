from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import os
import yaml
import tqdm
import sys
import threading
import multiprocess
import traceback


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


class MultiprocessLoggingHandler(logging.Handler):
    """
    Adapted from:
    https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
    """
    def __init__(self, filename, mode):
        logging.Handler.__init__(self)

        self._handler = logging.FileHandler(filename, mode)
        manager = multiprocess.Manager()
        self.queue = manager.Queue(-1)

        self.t = threading.Thread(target=self.receive)
        self.t.daemon = True
        self.t.start()


    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)


    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)


    def _format_record(self, record):
        # ensure that exc_info and args
        # have been stringified.  Removes any chance of
        # unpickleable things inside and possibly reduces
        # message size sent over the pipe
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self.t.join(5.0)
        self._handler.close()
        logging.Handler.close(self)


# Adapted from Logger.hasHandlers()
def has_handlers(logger):
    """
    See if this logger has any handlers configured.

    Loop through all handlers for this logger and its parents in the
    logger hierarchy. Return True if a handler was found, else False.
    Stop searching up the hierarchy whenever a logger with the "propagate"
    attribute set to zero is found - that will be the last logger which
    is checked for the existence of handlers.

    Returns
    -------
    bool
        True if the logger or any parent logger has handlers attached.
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


def get_logger(class_instance):
    """
    Get a logger with name given from `class_instance`:
    ``class_instance.__module__ + "." +  class_instance.__class__.__name__.``

    Parameters
    ----------
    class_instance : instance
        Class instance used to get the logger name.

    Returns
    -------
    logger : Logger object
        The logger object.
    """
    return logging.getLogger(class_instance.__module__ + "." +  class_instance.__class__.__name__)


def _create_module_logger(class_instance, level="info", config_filename=""):
    """
    Create a logger with `name`

    Parameters
    ----------
    class_instance : instance
        Class instance used to get the logger name.
        ``class_instance.__module__ + "." +  class_instance.__class__.__name__.``
    level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logger level is set. Setting
        logger level overwrites the logger level set from configuration file.
        Default logger level is info.
    config_filename : {None, "", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). If None,
        no configuration is loaded. Default is "". The configuration file is
        not loaded if any existing handlers are found.
    """
    create_logger(class_instance.__module__ + "." +  class_instance.__class__.__name__,
                  level=level,
                  config_filename=config_filename)




def create_logger(name, level="info", config_filename=""):
    """
    Create a logger with `name`

    Parameters
    ----------
    name : str
        Name of the logger
    level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logger level is set. Setting
        logger level overwrites the logger level set from configuration file.
        Default logger level is info.
    config_filename : {None, "", str}, optional
        Name of the logger configuration yaml file. If "", the default logger
        configuration is loaded (/uncertainpy/utils/logging.yaml). If None,
        no configuration is loaded. Default is "". The configuration file is
        not loaded if any existing handlers are found.
    """
    logger = logging.getLogger(name)

    if config_filename is not None and not has_handlers(logger):
        load_config(filename=config_filename)

        # To make sure that the logger is not overwritten by loading the config.
        logger = logging.getLogger(name)


    if level is not None:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % level)

        logger.setLevel(numeric_level)


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
        filename = os.path.join(folder, "logging.yaml")

    with open(filename) as f:
        logger_config = yaml.safe_load(f)

    logging.config.dictConfig(logger_config)
