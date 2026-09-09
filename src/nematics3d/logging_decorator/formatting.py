"""Naming, call-site inspection, and message formatting helpers."""

import datetime
import inspect
import logging
import os
import sys

from .context import (
    INDENT,
    current_file_handler,
    current_indent_level,
    current_log_mode,
    current_show_timestamp,
)
from .levels import RECOVERY


def get_program_name():
    return os.path.basename(sys.argv[0]) or "<interactive>"


def get_method_logging_context(func, args):
    base_display_name = func.__name__
    owner_label = None

    parts = getattr(func, "__qualname__", base_display_name).split(".")
    if len(parts) < 2 or not args:
        return base_display_name, owner_label

    cls_name_in_def = parts[-2]
    first_arg = args[0]
    if isinstance(first_arg, type):
        obj_cls = first_arg
        owner_target = first_arg
    else:
        obj_cls = getattr(first_arg, "__class__", None)
        owner_target = first_arg

    if obj_cls is None or obj_cls.__name__ != cls_name_in_def:
        return base_display_name, owner_label

    try:
        name_attr = getattr(owner_target, "name", None)
    except Exception:
        name_attr = None

    owner_label = (
        f"{cls_name_in_def}[name={name_attr!r}]"
        if name_attr is not None
        else cls_name_in_def
    )
    return f"{owner_label}.{func.__name__}", owner_label


def _describe_frame(frame, include_code):
    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    if not include_code:
        return location

    try:
        frame_info = inspect.getframeinfo(frame, context=1)
    except OSError:
        code_line = None
    else:
        code_line = (
            frame_info.code_context[0].strip() if frame_info.code_context else None
        )

    if code_line:
        return f"{location}\ncode: {code_line}"
    return f"{location}\ncode: <source unavailable>"


def get_log_call_context():
    frame = inspect.currentframe()
    if frame is None:
        return None, None

    try:
        logger_frame = frame.f_back
        current_frame = logger_frame.f_back if logger_frame is not None else None
        caller_frame = current_frame.f_back if current_frame is not None else None

        package_dir = os.path.dirname(__file__)
        while (
            caller_frame is not None
            and os.path.dirname(caller_frame.f_code.co_filename) == package_dir
        ):
            caller_frame = caller_frame.f_back

        current_text = (
            _describe_frame(current_frame, include_code=False)
            if current_frame is not None
            else None
        )
        caller_text = (
            _describe_frame(caller_frame, include_code=True)
            if caller_frame is not None
            else None
        )
        return current_text, caller_text
    finally:
        del frame


def make_safe_log(effective_log_mode, effective_log_level):
    """Build the low-level sink used by one outermost decorated call."""

    def safe_log(level, msg):
        if effective_log_mode == "none":
            return
        if level < effective_log_level and level != RECOVERY:
            return

        indent_str = INDENT * current_indent_level.get()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_str = f"[{logging.getLevelName(level)}]"
        indented_msg = "\n".join(
            f"{indent_str}{line}" for line in str(msg).splitlines()
        )

        if current_show_timestamp.get():
            text = f"{level_str} - {timestamp}\n{indented_msg}\n"
        else:
            text = f"{level_str}\n{indented_msg}\n"

        mode = current_log_mode.get()
        file_handler = current_file_handler.get()
        if mode == "screen":
            print(text, end="")
        elif mode == "file" and file_handler:
            file_handler.write(text)

    return safe_log
