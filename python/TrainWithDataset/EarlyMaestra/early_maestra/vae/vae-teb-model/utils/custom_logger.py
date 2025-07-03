
import sys
import logging as _logging
from typing import Optional

from loguru import logger


class InterceptHandler(_logging.Handler):
    """
    A logging.Handler that redirects stdlib logging records into Loguru,
    preserving levels and exception info.
    """
    def __init__(self):
        super().__init__()
        self.setLevel(_logging.NOTSET)  # capture all levels

    def emit(self, record: _logging.LogRecord):
        level = record.levelname
        # Find the call frame so Loguru reports the correct origin
        frame, depth = _logging.currentframe(), 2
        while frame and frame.f_code.co_filename == _logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    *,
    log_to_file: bool = True,
    log_to_console: bool = True,
    file_path: str = "app.log",
    file_level: str = "DEBUG",
    console_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: Optional[str] = "zip",
    serialize: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
):
    """
    Configure Loguru-based logging for the entire application.

    Parameters:
    -----------
    log_to_file: bool
        Enable logging to a file.
    log_to_console: bool
        Enable logging to stderr (console).
    file_path: str
        Path to the log file.
    file_level: str
        Minimum log level for file sink.
    console_level: str
        Minimum log level for console sink.
    rotation: str
        When to rotate the log file (e.g., "10 MB", "1 day").
    retention: str
        How long to keep old logs (e.g., "7 days", "10 files").
    compression: Optional[str]
        Compression for rotated files (e.g., "zip", "gz"), or None.
    serialize: bool
        If True, use built-in JSON serialization for sinks.
    backtrace: bool
        If True, include full Python stack backtraces on exceptions.
    diagnose: bool
        If True, show local variable values in tracebacks for deep debugging.
    """

    # 1) Remove any existing Loguru handlers
    logger.remove()

    # 2) Intercept all stdlib logging calls
    _logging.root.setLevel(_logging.DEBUG)
    _logging.root.handlers = [InterceptHandler()]

    # 3) File sink
    if log_to_file:
        if serialize:
            # JSON serialization: let Loguru handle format
            logger.add(
                file_path,
                level=file_level,
                rotation=rotation,
                retention=retention,
                compression=compression,
                serialize=True,
                backtrace=backtrace,
                diagnose=diagnose,
            )
        else:
            # Human-readable text format
            file_fmt = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "<green>{level: <8}</green> | "
                "{module}:{function}:{line} - {message}\n"
            )
            logger.add(
                file_path,
                level=file_level,
                rotation=rotation,
                retention=retention,
                compression=compression,
                serialize=False,
                format=file_fmt,
                backtrace=backtrace,
                diagnose=diagnose,
            )

    # 4) Console sink
    if log_to_console:
        if serialize:
            # JSON to console
            logger.add(
                sys.stderr,
                level=console_level,
                serialize=True,
                backtrace=backtrace,
                diagnose=diagnose,
            )
        else:
            # Colored, human-friendly console output
            console_fmt = (
                "{time:HH:mm:ss} | <level>{level: <8}</level> | "
                "{module}:{line} - {message}\n"
            )
            logger.add(
                sys.stderr,
                level=console_level,
                colorize=True,
                format=console_fmt,
                serialize=False,
                backtrace=backtrace,
                diagnose=diagnose,
            )


if __name__ == "__main__":
    # Example usage
    setup_logging(
        log_to_file=True,
        log_to_console=True,
        file_path="my_service.log",
        file_level="DEBUG",
        console_level="INFO",
        rotation="100 MB",
        retention="14 days",
        compression="zip",
        serialize=False,   # True → JSON output
        backtrace=True,    # include full stack backtraces
        diagnose=False,    # include local vars in tracebacks
    )

    # Now all logging goes through Loguru
    from loguru import logger as log
    import logging as std_logging

    log.debug("This is a debug message from Loguru")
    log.info("Service started successfully")
    std_logging.warning("This warning comes from the stdlib logging module!")

    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("Caught a division by zero error!")
