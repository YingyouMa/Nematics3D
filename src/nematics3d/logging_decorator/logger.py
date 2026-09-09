"""Logger object injected into decorated Nematics3D functions."""

import logging
import traceback

from .formatting import get_log_call_context
from .levels import DETAIL, PROGRESS, RECOVERY


class Logger:
    def __init__(self, safe_log):
        self._log = safe_log

    def debug(self, msg):
        self._log(logging.DEBUG, msg)

    def info(self, msg):
        self._log(logging.INFO, msg)

    def warning(self, msg):
        current_text, caller_text = get_log_call_context()
        parts = [">>> " + msg]
        if current_text is not None:
            parts.append(f"Current warning call: {current_text}")
        if caller_text is not None:
            parts.append(f"Caller: {caller_text}")
        self._log(logging.WARNING, "\n".join(parts))

    def error(self, msg):
        current_text, caller_text = get_log_call_context()
        parts = [msg]
        if current_text is not None:
            parts.append(f"Current error call: {current_text}")
        if caller_text is not None:
            parts.append(f"Caller: {caller_text}")
        self._log(logging.ERROR, "\n".join(parts))

    def critical(self, msg):
        self._log(logging.CRITICAL, msg)

    def recovery(self, msg):
        self._log(RECOVERY, msg)

    def detail(self, msg):
        self._log(DETAIL, msg)

    def progress(self, msg):
        self._log(PROGRESS, msg)

    def exception(self, msg, exc_info=None):
        current_text, caller_text = get_log_call_context()
        if exc_info is None:
            exc_text = traceback.format_exc()
        else:
            exc_text = "".join(traceback.format_exception(*exc_info))

        parts = [">>> " + msg]
        if current_text is not None:
            parts.append(f"Current exception call: {current_text}")
        if caller_text is not None:
            parts.append(f"Caller: {caller_text}")
        parts.append(exc_text)
        self._log(logging.ERROR, "\n".join(parts))
