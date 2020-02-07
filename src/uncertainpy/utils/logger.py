from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import logging.config
import os
import tqdm
import sys
import threading
import multiprocess
import traceback
import queue


class MyFormatter(logging.Formatter):
    """
    The logging formater.
    """
    # debug_format = "%(levelname)s - %(name)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    # info_format = "%(message)s"
    # warning_format = "%(levelname)s - %(message)s"
    # error_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"

    debug_format = "%(levelname)s - %(name)s - %(funcName)s  - %(lineno)d - %(message)s"
    info_format = "%(message)s"
    warning_format = "%(levelname)s - %(message)s"
    error_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    critical_format = "%(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s"


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

        self.handler = logging.FileHandler(filename, mode)
        manager = multiprocess.Manager()
        self.queue = manager.Queue(-1)
        # self.queue = multiprocess.Queue(-1)

        self.is_closed = False

        self.t = threading.Thread(target=self.receive)
        self.t.daemon = True
        self.t.start()


    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self.handler.setFormatter(fmt)


    def receive(self):
        # while True:
        while not (self.is_closed and self.queue.empty()):
            try:
                record = self.queue.get()
                self.handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except queue.Empty:
                pass # This periodically checks if the logger is closed.
            except:
                traceback.print_exc(file=sys.stderr)

        # self.queue.close()
        # self.queue.join_thread()

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
        if not self.is_closed:
            self.is_closed = True

            self.t.join(5.0)
            self.handler.close()
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


def setup_module_logger(class_instance, level="info"):
    """
    Create a logger with a name from the current class. "uncertainpy." is added
    to the beginning of the name if the module name does not start with
    "uncertainpy.". If no handlers, adds handlers to the logger named uncertainpy.

    Parameters
    ----------
    class_instance : instance
        Class instance used to set the logger name.
        ``class_instance.__module__ + "." +  class_instance.__class__.__name__.``
    level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logger level is set. Setting
        logger level overwrites the logger level set from configuration file.
        Default logger level is "info".
    """
    if level is None:
        return

    name = class_instance.__module__ + "." +  class_instance.__class__.__name__

    if not name.startswith("uncertainpy."):
        name = "uncertainpy." + name

    setup_logger(name, level=level)

    add_screen_handler()



def setup_logger(name, level="info"):
    """
    Create a logger with `name`.

    Parameters
    ----------
    name : str
        Name of the logger
    level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logger is set up. Default
        logger level is info.
    """
    if level is None:
        return

    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)

    logger.setLevel(numeric_level)



def add_screen_handler(name="uncertainpy"):
    """
    Adds a logging to console (a console handler) to logger with `name`, if no screen handler already
    exists for the given logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default name is "uncertainpy".
    """
    logger = logging.getLogger(name)

    handler_exists = False
    for handler in logger.handlers:
        if isinstance(handler, TqdmLoggingHandler):
            handler_exists = True
            break

    if not handler_exists:
        console = TqdmLoggingHandler()
        console.setFormatter(MyFormatter())

        logger.addHandler(console)


def add_file_handler(name="uncertainpy", filename="uncertainpy.log"):
    """
    Add file handler to logger with `name`, if no file handler already
    exists for the given logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default name is "uncertainpy".
    filename : str
        Name of the logfile. If None, no logging to file is performed. Default is
        "uncertainpy.log".
    """
    logger = logging.getLogger(name)

    if filename is not None:
        handler_exists = False
        for handler in logger.handlers:
            if isinstance(handler, MultiprocessLoggingHandler):
                handler_exists = True
                file_handler = handler
                break

        if not handler_exists:
            multiprocess_file = MultiprocessLoggingHandler(filename=filename, mode="w")
            multiprocess_file.setFormatter(MyFormatter())

            logger.addHandler(multiprocess_file)

        else:
            current_dir = os.getcwd()
            old_filename = file_handler.handler.baseFilename.strip(current_dir)

            if old_filename != filename:
                # file_handler.close()
                logger.removeHandler(file_handler)

                multiprocess_file = MultiprocessLoggingHandler(filename=filename, mode="w")
                multiprocess_file.setFormatter(MyFormatter())

                logger.addHandler(multiprocess_file)



# def add_handlers(name="uncertainpy", filename="uncertainpy.log"):
#     """


#     Parameters
#     ----------
#     name : str
#         Name of the logger
#     level : {"info", "debug", "warning", "error", "critical", None}, optional
#         Set the threshold for the logging level. Logging messages less severe
#         than this level is ignored. If None, no logging is performed.
#         Default logger level is "info".
#     filename : str
#         Name of the logfile. If None, no logging to file is performed. Default is
#         "uncertainpy.log".
#     """
#     add_screen_handler(name=name)
#     add_file_handler(name=name, filename=filename)
