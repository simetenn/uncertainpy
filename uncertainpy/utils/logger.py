import logging
import sys


def create_logger(logger_level, logger_filename=None, logger_name="logger"):
    numeric_level = getattr(logging, logger_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % logger_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter('%(levelname)s: %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if logger_filename is None:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(numeric_level)
        console.setFormatter(formatter)

        logger.addHandler(console)
    else:
        handler = logging.FileHandler(filename=logger_filename, mode='w')
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
