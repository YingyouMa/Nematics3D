"""Decorator implementation for structured Nematics3D logging."""

import atexit
import datetime
import functools
import logging
import os
import time

from .context import (
    current_file_handler,
    current_filename,
    current_indent_level,
    current_log_level,
    current_log_mode,
    current_logger,
    current_owner_label,
    current_show_timestamp,
)
from .formatting import get_method_logging_context, get_program_name, make_safe_log
from .levels import GLOBAL_DEFAULTS
from .logger import Logger


def dummy_logger(level, msg):
    pass


def _resolve_setting(explicit_value, context_var, default_name):
    if explicit_value is not None:
        return explicit_value
    inherited = context_var.get()
    if inherited is not None:
        return inherited
    return GLOBAL_DEFAULTS[default_name]


def logging_and_warning_decorator(
    log_mode=None, show_timestamp=None, log_level=None, start_finish_level=logging.DEBUG
):
    if callable(log_mode):
        func = log_mode
        return _decorate(func, start_finish_level=start_finish_level)

    def wrapper(func):
        return _decorate(
            func,
            log_mode=log_mode,
            show_timestamp=show_timestamp,
            log_level=log_level,
            start_finish_level=start_finish_level,
        )

    return wrapper


def _decorate(
    func,
    log_mode=None,
    show_timestamp=None,
    log_level=None,
    start_finish_level=logging.DEBUG,
):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        display_name, method_owner_label = get_method_logging_context(func, args)
        inherited_owner_label = current_owner_label.get()
        effective_owner_label = method_owner_label or inherited_owner_label
        contextual_display_name = display_name
        if method_owner_label is None and effective_owner_label is not None:
            contextual_display_name = f"{effective_owner_label} -> {display_name}"

        call_log_mode = kwargs.pop("log_mode", log_mode)
        call_show_timestamp = kwargs.pop("show_timestamp", show_timestamp)
        call_log_level = kwargs.pop("log_level", log_level)

        effective_log_mode = _resolve_setting(
            call_log_mode, current_log_mode, "log_mode"
        )
        effective_show_timestamp = _resolve_setting(
            call_show_timestamp, current_show_timestamp, "show_timestamp"
        )
        effective_log_level = _resolve_setting(
            call_log_level, current_log_level, "log_level"
        )

        token_log_mode = current_log_mode.set(effective_log_mode)
        token_show_ts = current_show_timestamp.set(effective_show_timestamp)
        token_log_level = current_log_level.set(effective_log_level)
        token_owner_label = current_owner_label.set(effective_owner_label)
        token_indent = current_indent_level.set(current_indent_level.get() + 1)

        outer_logger = current_logger.get()
        outer_file_handler = current_file_handler.get()
        is_outermost = outer_logger is None and outer_file_handler is None

        file_handler = None
        if not is_outermost:
            safe_log = outer_logger
        elif effective_log_mode == "none":
            safe_log = dummy_logger
            current_logger.set(dummy_logger)
            current_file_handler.set(None)
            current_filename.set(None)
        else:
            if effective_log_mode == "file":
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                folder = GLOBAL_DEFAULTS["log_folder"]
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, f"{display_name}_{timestamp_str}.log")
                file_handler = open(filename, mode="w", encoding="utf-8")
                atexit.register(
                    lambda: file_handler
                    and not file_handler.closed
                    and file_handler.close()
                )
                current_file_handler.set(file_handler)
                current_filename.set(filename)

            safe_log = make_safe_log(effective_log_mode, effective_log_level)
            current_logger.set(safe_log)

        def bound_safe_log(level, msg):
            if msg is None:
                safe_log(level, None)
            else:
                safe_log(level, f"<{contextual_display_name}> \n{msg}")

        logger_obj = Logger(bound_safe_log)
        kwargs["logger"] = logger_obj

        if safe_log != dummy_logger:
            safe_log(
                start_finish_level,
                f"Function `{contextual_display_name}` STARTED in program `{get_program_name()}`",
            )

        start_time = time.time()
        try:
            return func(*args, **kwargs)
        except Exception:
            logger_obj.exception(
                f"Function `{contextual_display_name}` raised an exception"
            )
            raise
        finally:
            elapsed = time.time() - start_time
            if safe_log != dummy_logger:
                safe_log(
                    start_finish_level,
                    f"Function `{contextual_display_name}` FINISHED in program `{get_program_name()}`. "
                    f"Elapsed time: {elapsed:.3f} seconds.",
                )

            if is_outermost:
                active_file_handler = current_file_handler.get()
                if active_file_handler:
                    active_file_handler.close()
                current_logger.set(None)
                current_file_handler.set(None)
                current_filename.set(None)

            current_indent_level.reset(token_indent)
            current_log_mode.reset(token_log_mode)
            current_show_timestamp.reset(token_show_ts)
            current_log_level.reset(token_log_level)
            current_owner_label.reset(token_owner_label)

    return inner
