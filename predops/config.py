# m5a/config.py
# Configurations.

import logging
import logging.config
import sys
from pathlib import Path

import pretty_errors  # NOQA: F401 (imported but unused)
from rich.logging import RichHandler

# create directories
def init_config(PROJECT_KEY):
    global LOGS_DIR
    global RAW_DIR
    global DATA_DIR
    global STORES_DIR
    global logger

    # Directories
    BASE_DIR = Path(__file__).parent.parent.absolute()
    # CONFIG_DIR = Path(BASE_DIR, "config")
    DATA_DIR = Path(BASE_DIR, "data/processed", PROJECT_KEY)
    LOGS_DIR = Path(BASE_DIR, "data/logs", PROJECT_KEY)
    RAW_DIR = Path(BASE_DIR, "data/raw", PROJECT_KEY)
    
    # MODEL_DIR = Path(BASE_DIR, "model")
    STORES_DIR = Path(BASE_DIR, "data/stores", PROJECT_KEY)

    # # Local stores
    # BLOB_STORE = Path(STORES_DIR, "blob")
    # FEATURE_STORE = Path(STORES_DIR, "feature")
    # MODEL_REGISTRY = Path(STORES_DIR, "model")

    # Create dirs
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STORES_DIR.mkdir(parents=True, exist_ok=True)
    # BLOB_STORE.mkdir(parents=True, exist_ok=True)
    # FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    # MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

    # Logger
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "minimal": {"format": "%(message)s"},
            "detailed": {
                "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "minimal",
                "level": logging.DEBUG,
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(LOGS_DIR, "info.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.INFO,
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(LOGS_DIR, "error.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.ERROR,
            },
        },
        "loggers": {
            "root": {
                "handlers": ["console", "info", "error"],
                "level": logging.INFO,
                "propagate": True,
            },
            "alembic": {
                "handlers": ["console", "info", "error"],
                "level": logging.WARN,
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("root")
    logger.handlers[0] = RichHandler(markup=True)