import logging
import sys


class MyFormatter(logging.Formatter):

    debugg_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"
    info_format = "%(message)s"
    warning_format = "%(levelname)s - %(module)s - %(message)s"
    error_format = "%(levelname)s - %(module)s - %(filename)s - %(lineno)d - %(message)s"


    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)


    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.debugg_format

        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_format

        elif record.levelno == logging.WARNING:
            self._fmt = MyFormatter.warning_format

        elif record.levelno == logging.ERROR:
            self._fmt = MyFormatter.error_format

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


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
