import logging

import pytest

import nematics3d.logging_decorator as logging_decorator
from nematics3d.logging_decorator import (
    DETAIL,
    INDENT,
    PROGRESS,
    RECOVERY,
    Logger,
    logging_and_warning_decorator,
)


def test_public_logging_api_is_preserved():
    assert DETAIL == 5
    assert PROGRESS == 15
    assert RECOVERY == 35
    assert INDENT == "    "
    assert logging_decorator.Logger is Logger
    assert (
        logging_decorator.logging_and_warning_decorator is logging_and_warning_decorator
    )
    assert logging.getLevelName(DETAIL) == "DETAIL"
    assert logging.getLevelName(PROGRESS) == "PROGRESS"
    assert logging.getLevelName(RECOVERY) == "RECOVERY"


def test_decorator_injects_logger_and_preserves_return_value(capsys):
    @logging_and_warning_decorator()
    def sample(value, logger=None):
        assert isinstance(logger, Logger)
        logger.progress("working")
        return value + 1

    assert sample(2) == 3
    output = capsys.readouterr().out
    assert "[PROGRESS]" in output
    assert "<sample>" in output
    assert "working" in output


def test_call_level_logging_options_are_consumed_by_decorator(capsys):
    @logging_and_warning_decorator()
    def sample(*, logger=None):
        logger.warning("hidden")

    sample(log_mode="none", log_level=DETAIL, show_timestamp=True)
    assert capsys.readouterr().out == ""


def test_nested_calls_share_logger_context_and_indent(capsys):
    @logging_and_warning_decorator(start_finish_level=DETAIL)
    def inner(*, logger=None):
        logger.progress("inner message")

    @logging_and_warning_decorator(start_finish_level=DETAIL)
    def outer(*, logger=None):
        logger.progress("outer message")
        inner()

    outer(log_level=PROGRESS)
    output = capsys.readouterr().out
    assert f"{INDENT}<outer>" in output
    assert f"{INDENT}{INDENT}<inner>" in output


def test_recovery_is_visible_below_threshold(capsys):
    @logging_and_warning_decorator()
    def sample(*, logger=None):
        logger.progress("hidden")
        logger.recovery("recovered")

    sample(log_level=logging.CRITICAL)
    output = capsys.readouterr().out
    assert "hidden" not in output
    assert "[RECOVERY]" in output
    assert "recovered" in output


def test_exception_is_logged_and_reraised(capsys):
    @logging_and_warning_decorator()
    def sample(*, logger=None):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        sample(log_level=logging.ERROR)

    output = capsys.readouterr().out
    assert "[ERROR]" in output
    assert "raised an exception" in output
    assert "ValueError: boom" in output
