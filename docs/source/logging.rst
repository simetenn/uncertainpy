.. _logging:

Logging
=======

Uncertainpy uses the logging module to log to both file and to screen.
All loggers are named ``"uncertainpy."``.
The configuration that Uncertainpy sets by default is

.. literalinclude:: ../../src/uncertainpy/utils/logging.yaml
    :language: yaml

The logging can be customized by giving a yaml file as the `configuration_file`.
This should only be changed if you know what you are doing. Be warned that
logging is performed in parallel. If the``MultiprocessLoggingHandler()`` is not
used when trying to write to a single log file uncertainpy will hang. This
happens because several processes try to log to the same file.

Logging can easily be added to custom models and features by::

    # Import the functions and libraries needed
    from uncertainpy.utils import create_logger
    import logging

    # Set up a logger. This adds handlers to the "uncertainpy" logger if not
    # handlers already exists
    # The all log messages with level "info" or higher will be logged.
    create_logger("uncertainpy.coffee_cup", level="info")

    # Get the logger recently created
    logger = logging.getLogger("uncertainpy.coffee_cup")
    # Log a message with the level "info".
    logger.info("info logging message here")



API Reference
-------------

.. automodule:: uncertainpy.utils.logger
   :members: