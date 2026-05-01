"""Logging utilities with colorful terminal output and file logging."""

import functools
import sys
import logging

from accelerate.logging import MultiProcessAdapter
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    """Formatter that adds color to warning/error messages."""

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def setup_logger(
    name="MUSE",
    log_level: str = None,
    color=True,
    use_accelerate=True,
    output_file=None
):
    """Set up a logger with colorful terminal output and optional file logging.

    Args:
        name: Logger name.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        color: Whether to use colorful formatting.
        use_accelerate: Whether to wrap with MultiProcessAdapter for DDP.
        output_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if log_level is None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level.upper())

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_file is not None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if use_accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger
