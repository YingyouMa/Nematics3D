import functools
import os
import time
import datetime
import sys
import contextvars
import logging
import traceback

# 自定义日志等级
RECOVERY = 35
logging.addLevelName(RECOVERY, "RECOVERY")

# 默认配置
_GLOBAL_DEFAULTS = {
    "log_mode": "screen",  # "none" | "screen" | "file"
    "log_folder": "log",
    "show_timestamp": False,
    "log_level": logging.INFO,
}

# 上下文变量
_current_logger = contextvars.ContextVar("current_logger", default=None)
_current_file_handler = contextvars.ContextVar("current_file_handler", default=None)
_current_indent_level = contextvars.ContextVar("current_indent_level", default=0)
_current_log_mode = contextvars.ContextVar(
    "current_log_mode", default=_GLOBAL_DEFAULTS["log_mode"]
)
_current_show_timestamp = contextvars.ContextVar(
    "current_show_timestamp", default=_GLOBAL_DEFAULTS["show_timestamp"]
)
_current_log_level = contextvars.ContextVar(
    "current_log_level", default=_GLOBAL_DEFAULTS["log_level"]
)
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
        self._log(logging.WARNING, msg)

    def error(self, msg):
        self._log(logging.ERROR, msg)

    def critical(self, msg):
        self._log(logging.CRITICAL, msg)

    def recovery(self, msg):
        self._log(RECOVERY, msg)

    def exception(self, msg, exc_info=None):
        if exc_info is None:
            exc_text = traceback.format_exc()
        else:
            exc_text = "".join(traceback.format_exception(*exc_info))
        self._log(logging.ERROR, msg + "\n" + exc_text)


def logging_and_warning_decorator(log_mode=None, show_timestamp=None, log_level=None):
    if callable(log_mode):
        func = log_mode
        return _decorate(func)

    def wrapper(func):
        return _decorate(
            func, log_mode=log_mode, show_timestamp=show_timestamp, log_level=log_level
        )

    return wrapper


def _decorate(func, log_mode=None, show_timestamp=None, log_level=None):
    @functools.wraps(func)
    def inner(*args, **kwargs):

        display_name = func.__name__
        if display_name == "__init__":
            if args and hasattr(args[0], "__class__"):
                display_name = f"{args[0].__class__.__name__}.__init__"
            else:
                display_name = "object.__init__"

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

                    indented_msg = "\n".join(
                        f"{indent_str}{line}" for line in str(msg).splitlines()
                    )

                    if show_ts:
                        text = f"{level_str} - {timestamp}\n{indented_msg}\n"
                    else:
                        text = f"{level_str}\n{indented_msg}\n"

                    mode = _current_log_mode.get()
                    fh = _current_file_handler.get()

                    if mode == "screen":
                        print(text, end="")
                    elif mode == "file" and fh:
                        fh.write(text)

                _current_logger.set(safe_log)

        start_time = time.time()
        logger_obj = Logger(safe_log)
        kwargs["logger"] = logger_obj

        if safe_log != dummy_logger:
            safe_log(
                logging.DEBUG,
                f"Function `{display_name}` STARTED in program `{get_program_name()}`",
            )

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
                    logging.DEBUG,
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
