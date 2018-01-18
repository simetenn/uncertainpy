import logging
import tqdm
import sys


class MyFormatter(logging.Formatter):
    """
    The logging formater.
    """
    debug_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    info_format = "%(message)s"
    warning_format = "%(levelname)s - %(message)s"
    error_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"

    debug_fmt = logging.Formatter(debug_format)
    info_fmt = logging.Formatter(info_format)
    warning_fmt = logging.Formatter(warning_format)
    error_fmt = logging.Formatter(error_format)


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


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Set logging so logging to  stream works with Tqdm,
    logging now uses tqdm.write.
    """
    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)


def create_logger(level, filename=None, name="logger"):
    """
    Create a logger object.

    Parameters
    ----------
    level : {"info", "debug", "warning", "error", "critical"}
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored.
    filename : str
        Sets logging to a file with name `verbose_filename`.
        No logging to screen if set. Default is None.
    name : str
        Name of the logger.

    Returns
    -------
    logger : Logger object
        The logger object.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    logging.captureWarnings(True)

    # Delete possible handlers already existing
    logger.handlers = []

    if filename is None:
        # console = logging.StreamHandler(stream=sys.stdout)
        console = TqdmLoggingHandler()
        console.setLevel(numeric_level)
        console.setFormatter(MyFormatter())

        logger.addHandler(console)
    else:
        handler = logging.FileHandler(filename=filename, mode='w')
        handler.setLevel(numeric_level)
        handler.setFormatter(MyFormatter())

        logger.addHandler(handler)

    return logger
