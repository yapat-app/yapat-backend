"""
Centralized logging configuration for API and Celery workers.
"""

import logging
from logging.config import dictConfig

from app.config import settings


def configure_logging() -> None:
    """Apply a shared logging configuration once per process."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level = (settings.LOG_LEVEL or "INFO").upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )
