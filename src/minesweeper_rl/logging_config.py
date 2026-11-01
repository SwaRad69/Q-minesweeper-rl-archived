"""Small logging setup shared by experiments and browser code."""

import logging

_PACKAGE_LOGGER = "minesweeper_rl"


def get_logger(name: str = _PACKAGE_LOGGER) -> logging.Logger:
    """Return a package logger without changing the application's handlers."""

    return logging.getLogger(name)


def enable_diagnostics(level: int = logging.DEBUG) -> logging.Logger:
    """Attach a console handler to the package logger for development use.

    Diagnostics are gated independently by ``EnvironmentConfig.diagnostics``;
    this helper only makes the debug output visible in the terminal.
    """

    logger = get_logger(_PACKAGE_LOGGER)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
