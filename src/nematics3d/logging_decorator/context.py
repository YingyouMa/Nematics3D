"""Context-local state used to propagate nested logging calls."""

import contextvars

current_logger = contextvars.ContextVar("current_logger", default=None)
current_file_handler = contextvars.ContextVar("current_file_handler", default=None)
current_indent_level = contextvars.ContextVar("current_indent_level", default=0)
current_log_mode = contextvars.ContextVar("current_log_mode", default=None)
current_show_timestamp = contextvars.ContextVar("current_show_timestamp", default=None)
current_log_level = contextvars.ContextVar("current_log_level", default=None)
current_filename = contextvars.ContextVar("current_filename", default=None)
current_owner_label = contextvars.ContextVar("current_owner_label", default=None)

INDENT = "    "
