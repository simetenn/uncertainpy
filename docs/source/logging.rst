.. _logging:

Logging
=======

Uncertainpy uses the logging module to log to both file and to screen.
All loggers are named
``class_instance.__module__ + "." +  class_instance.__class__.__name__``.
An example, the logger in a ``Data```object is named
``uncertainpy.data.Data``.
If the the module name does not start with "uncertainpy.", "uncertainpy."
as added as a prefix.

A file handler is only added to the logging by ``UncertaintyQuantification``.
If level is set to None, no logging in Uncertainpy is set up and the logging can
be customized as necessary by using the logging module.
This should only be done if you know what you are doing. Be warned that
logging is performed in parallel. If the ``MultiprocessLoggingHandler()`` is not
used when trying to write to a single log file, Uncertainpy will hang. This
happens because several processes try to log to the same file.

Logging can easily be added to custom models and features by::

    # Import the functions and libraries needed
    from uncertainpy.utils import create_logger
    import logging

    # Set up a logger. This adds a screen handlers to the "uncertainpy" logger
    # if it does not already exist
    # All log messages with level "info" or higher will be logged.
    setup_logger("uncertainpy.logger_name", level="info")

    # Get the logger recently created
    logger = logging.getLogger("uncertainpy.logger_name")

    # Log a message with the level "info".
    logger.info("info logging message here")

Note that if you want to use the logger setup in Uncertainpy, the name of your
loggers should start with ``uncertainpy.``.


API Reference
-------------

.. automodule:: uncertainpy.utils.logger
   :members: