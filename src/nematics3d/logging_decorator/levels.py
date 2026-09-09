"""Log levels and process-wide defaults for Nematics3D logging."""

import logging

DETAIL = 5
PROGRESS = 15
RECOVERY = 35

logging.addLevelName(DETAIL, "DETAIL")
logging.addLevelName(PROGRESS, "PROGRESS")
logging.addLevelName(RECOVERY, "RECOVERY")

GLOBAL_DEFAULTS = {
    "log_mode": "screen",
    "log_folder": "log",
    "show_timestamp": False,
    "log_level": PROGRESS,
}


def set_global_logging_defaults(**kwargs):
    """Update the process-wide defaults used by outermost decorated calls."""
    GLOBAL_DEFAULTS.update(kwargs)
