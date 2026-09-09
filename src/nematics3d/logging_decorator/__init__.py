"""
Structured logging utilities with custom Nematics3D log levels.

Available levels, from most to least verbose, are DETAIL (5), DEBUG (10),
PROGRESS (15), INFO (20), WARNING (30), RECOVERY (35), ERROR (40), and
CRITICAL (50). The public API is intentionally identical to the former
``nematics3d.logging_decorator`` module.
"""

from . import context as _context
from . import decorator as _decorator
from . import formatting as _formatting
from . import levels as _levels
from .context import INDENT
from .decorator import dummy_logger, logging_and_warning_decorator
from .formatting import get_program_name
from .levels import DETAIL, PROGRESS, RECOVERY, set_global_logging_defaults
from .logger import Logger

_GLOBAL_DEFAULTS = _levels.GLOBAL_DEFAULTS
_current_logger = _context.current_logger
_current_file_handler = _context.current_file_handler
_current_indent_level = _context.current_indent_level
_current_log_mode = _context.current_log_mode
_current_show_timestamp = _context.current_show_timestamp
_current_log_level = _context.current_log_level
_current_filename = _context.current_filename
_current_owner_label = _context.current_owner_label
_get_method_logging_context = _formatting.get_method_logging_context
_describe_frame = _formatting._describe_frame
_get_log_call_context = _formatting.get_log_call_context
_decorate = _decorator._decorate

__all__ = [
    "DETAIL",
    "INDENT",
    "PROGRESS",
    "RECOVERY",
    "Logger",
    "dummy_logger",
    "get_program_name",
    "logging_and_warning_decorator",
    "set_global_logging_defaults",
]
