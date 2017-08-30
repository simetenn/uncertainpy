import logging
import sys


class MyFormatter(logging.Formatter):

    debug_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    info_format = "%(message)s"
    warning_format = "%(levelname)s - %(module)s - %(message)s"
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


def create_logger(logger_level, logger_filename=None, logger_name="logger"):
    numeric_level = getattr(logging, logger_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % logger_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    logging.captureWarnings(True)

    # Delete possible handlers already existing
    logger.handlers = []

    if logger_filename is None:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(numeric_level)
        console.setFormatter(MyFormatter())

        logger.addHandler(console)
    else:
        handler = logging.FileHandler(filename=logger_filename, mode='w')
        handler.setLevel(numeric_level)
        handler.setFormatter(MyFormatter())

        logger.addHandler(handler)

    return logger
