"""
logging_decorator: A structured logging utility with custom log levels.

Available Log Levels (from most verbose to least verbose):

  5  DETAIL
      Extremely fine-grained diagnostic output; step-by-step tracing,
      per-iteration states, and deep internal debugging.

 10  DEBUG
      Developer-oriented debugging information. Captures important internal
      states and intermediate values useful for diagnosing issues.

 15  PROGRESS
      High-level procedural updates describing what the program is currently
      doing. Suitable as the default level when users want workflow visibility
      without full DEBUG/DETAIL noise.

 20  INFO
      Key results and essential runtime messages. Concise summaries of important
      events, outcomes, and performance information. This level is also used to 
      report critical default parameters that were implicitly applied because the 
      user did not provide explicit values.

 30  WARNING
      A potentially incorrect or risky condition was detected. Execution continues,
      but the user should verify their input or configuration.

 35  RECOVERY
      An error occurred but the system automatically corrected or compensated
      for it. This message is always shown because the result differs from the
      originally intended operation.

 40  ERROR
      A failure occurred in the current operation. Execution may or may not
      continue depending on the context. If the error is automatically handled
      and safely redirected, that situation should be logged as RECOVERY instead.

 50  CRITICAL
      A severe, unrecoverable error that requires immediate termination of the
      program (e.g. out-of-memory, divergence, or corrupted state).

To view this documentation at any time:
    >>> import logging_decorator
    >>> help(logging_decorator)
"""

import functools
import os
import time
import datetime
import sys
import contextvars
import logging
import traceback

# customize log level
RECOVERY = 35
logging.addLevelName(RECOVERY, "RECOVERY")
PROGRESS = 15
logging.addLevelName(PROGRESS, "PROGRESS")
DETAIL = 5
logging.addLevelName(DETAIL, "DETAIL")

# default settings
_GLOBAL_DEFAULTS = {
    "log_mode": "screen",  # "none" | "screen" | "file"
    "log_folder": "log",
    "show_timestamp": False,
    "log_level": PROGRESS,
}

# contextvars
_current_logger = contextvars.ContextVar("current_logger", default=None)
_current_file_handler = contextvars.ContextVar("current_file_handler", default=None)
_current_indent_level = contextvars.ContextVar("current_indent_level", default=0)
_current_log_mode = contextvars.ContextVar("current_log_mode", default=None)
_current_show_timestamp = contextvars.ContextVar("current_show_timestamp", default=None)
_current_log_level = contextvars.ContextVar("current_log_level", default=None)
_current_filename = contextvars.ContextVar("current_filename", default=None)

INDENT = "    "


def set_global_logging_defaults(**kwargs):
    _GLOBAL_DEFAULTS.update(kwargs)


def get_program_name():
    return os.path.basename(sys.argv[0]) or "<interactive>"


def dummy_logger(level, msg):
    pass


class Logger:
    def __init__(self, safe_log):
        self._log = safe_log

    def debug(self, msg):
        self._log(logging.DEBUG, msg)

    def info(self, msg):
        self._log(logging.INFO, msg)

    def warning(self, msg):
        self._log(logging.WARNING, ">>> " + msg)

    def error(self, msg):
        self._log(logging.ERROR, msg)

    def critical(self, msg):
        self._log(logging.CRITICAL, msg)

    def recovery(self, msg):
        self._log(RECOVERY, msg)
        
    def detail(self, msg): 
        self._log(DETAIL, msg)
        
    def progress(self, msg):
        self._log(PROGRESS, msg)

    def exception(self, msg, exc_info=None):
        if exc_info is None:
            exc_text = traceback.format_exc()
        else:
            exc_text = "".join(traceback.format_exception(*exc_info))
        self._log(logging.ERROR, msg + "\n" + exc_text)


def logging_and_warning_decorator(
        log_mode=None, 
        show_timestamp=None, 
        log_level=None,
        start_finish_level=logging.DEBUG):
    
    if callable(log_mode):
        func = log_mode
        return _decorate(func, start_finish_level=start_finish_level)

    def wrapper(func):
        return _decorate(
            func, 
            log_mode=log_mode, 
            show_timestamp=show_timestamp, 
            log_level=log_level,
            start_finish_level=start_finish_level
        )

    return wrapper


def _decorate(func, 
              log_mode=None, 
              show_timestamp=None, 
              log_level=None,
              start_finish_level=logging.DEBUG):
    
    @functools.wraps(func)
    def inner(*args, **kwargs):
        
        # Base name is the plain function name
        display_name = func.__name__

        # Try to detect if this function is defined as a method of a class
        qualname = getattr(func, "__qualname__", display_name)
        parts = qualname.split(".")

        # If qualname looks like "ClassName.method_name"
        if len(parts) >= 2 and args:
            cls_name_in_def = parts[-2]          # e.g. "QFieldObject"
            first_arg = args[0]

            # Handle both instance methods (first_arg is instance)
            # and classmethods (first_arg is the class itself).
            obj_cls = None
            if isinstance(first_arg, type):
                obj_cls = first_arg
            else:
                obj_cls = getattr(first_arg, "__class__", None)

            if obj_cls is not None and obj_cls.__name__ == cls_name_in_def:
                # We are very likely inside a method of this class.
                name_attr = getattr(first_arg, "name", None)
            
                if name_attr is not None:
                    # e.g. QFieldObject[name='testQ'].act_lines_classify
                    display_name = f"{cls_name_in_def}[name={name_attr!r}].{func.__name__}"
                else:
                    # e.g. QFieldObject.act_lines_classify
                    display_name = f"{cls_name_in_def}.{func.__name__}"

        effective_log_mode = kwargs.pop("log_mode", log_mode)
        effective_show_timestamp = kwargs.pop("show_timestamp", show_timestamp)
        effective_log_level = kwargs.pop("log_level", log_level)

        if effective_log_mode is None:
            effective_log_mode = _current_log_mode.get()
        if effective_log_mode is None:
            effective_log_mode = _GLOBAL_DEFAULTS["log_mode"]

        if effective_show_timestamp is None:
            effective_show_timestamp = _current_show_timestamp.get()
        if effective_show_timestamp is None:
            effective_show_timestamp = _GLOBAL_DEFAULTS["show_timestamp"]

        if effective_log_level is None:
            effective_log_level = _current_log_level.get()
        if effective_log_level is None:
            effective_log_level = _GLOBAL_DEFAULTS["log_level"]

        token_log_mode = _current_log_mode.set(effective_log_mode)
        token_show_ts = _current_show_timestamp.set(effective_show_timestamp)
        token_log_level = _current_log_level.set(effective_log_level)

        current_indent = _current_indent_level.get()
        token_indent = _current_indent_level.set(current_indent + 1)

        outer_logger = _current_logger.get()
        outer_file_handler = _current_file_handler.get()
        is_outermost = outer_logger is None and outer_file_handler is None

        file_handler = None
        safe_log = None

        if not is_outermost:
            safe_log = outer_logger
        else:
            if effective_log_mode == "none":
                safe_log = dummy_logger
                _current_logger.set(dummy_logger)
                _current_file_handler.set(None)
                _current_filename.set(None)
            else:
                filename = None
                if effective_log_mode == "file":
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    folder = _GLOBAL_DEFAULTS["log_folder"]
                    os.makedirs(folder, exist_ok=True)
                    filename = os.path.join(
                        folder, f"{display_name}_{timestamp_str}.log"
                    )
                    file_handler = open(filename, mode="w", encoding="utf-8")

                    import atexit

                    atexit.register(
                        lambda: file_handler
                        and not file_handler.closed
                        and file_handler.close()
                    )

                    _current_file_handler.set(file_handler)
                    _current_filename.set(filename)

                def safe_log(level, msg):
                    if effective_log_mode == "none":
                        return
                    if level < effective_log_level and level != RECOVERY:
                        return
                
                    show_ts = _current_show_timestamp.get()
                    indent_level = _current_indent_level.get()
                    indent_str = INDENT * indent_level
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    level_str = f"[{logging.getLevelName(level)}]"
                
                    # 这里直接用 msg，不要加 display_name
                    indented_msg = "\n".join(
                        f"{indent_str}{line}" for line in str(msg).splitlines()
                    )
                
                    mode = _current_log_mode.get()
                    fh = _current_file_handler.get()
                
                    if show_ts:
                        text = f"{level_str} - {timestamp}\n{indented_msg}\n"
                    else:
                        text = f"{level_str}\n{indented_msg}\n"
                
                    if mode == "screen":
                        print(text, end="")
                    elif mode == "file" and fh:
                        fh.write(text)

                _current_logger.set(safe_log)
                       
 
        
        def bound_safe_log(level, msg):
            # 给每条 logger.* 输出前面加上当前函数名，
            # start/finish 那两句仍然用原始 safe_log，不受影响
            if msg is None:
                # 防御一下极端用法
                safe_log(level, None)
            else:
                safe_log(level, f"<{display_name}> \n{msg}")
        logger_obj = Logger(bound_safe_log)
        kwargs["logger"] = logger_obj

        if safe_log != dummy_logger:
            safe_log(
                start_finish_level,
                f"Function `{display_name}` STARTED in program `{get_program_name()}`",
            )
            
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        except Exception:
            logger_obj.exception(f"Function `{display_name}` raised an exception")
            raise
        finally:
            elapsed = time.time() - start_time
            if safe_log != dummy_logger:
                safe_log(
                    start_finish_level,
                    f"Function `{display_name}` FINISHED in program `{get_program_name()}`. "
                    f"Elapsed time: {elapsed:.3f} seconds.",
                )
            if is_outermost:
                fh = _current_file_handler.get()
                if fh:
                    fh.close()
                _current_logger.set(None)
                _current_file_handler.set(None)
                _current_filename.set(None)

            _current_indent_level.reset(token_indent)
            _current_log_mode.reset(token_log_mode)
            _current_show_timestamp.reset(token_show_ts)
            _current_log_level.reset(token_log_level)

    return inner
