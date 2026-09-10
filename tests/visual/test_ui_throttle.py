import pytest
from qtpy.QtCore import QCoreApplication

from nematics3d.visual.qt.ui_throttle import UIThrottle


@pytest.fixture(scope="module", autouse=True)
def qcore_application():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_interval_must_be_positive_integer():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            UIThrottle(value)


def test_schedule_keeps_latest_call_without_restarting_timer():
    throttle = UIThrottle(1000)
    calls = []
    throttle.schedule(calls.append, "first")
    remaining_before = throttle._timer.remainingTime()
    throttle.schedule(calls.append, "second")
    assert throttle._timer.isActive()
    assert throttle._timer.remainingTime() <= remaining_before
    throttle.flush()
    assert calls == ["second"]


def test_flush_executes_pending_call_once_and_stops_timer():
    throttle = UIThrottle(1000)
    calls = []
    throttle.schedule(calls.append, 3)
    throttle.flush()
    throttle.flush()
    assert calls == [3]
    assert not throttle._timer.isActive()


def test_cancel_discards_pending_call():
    throttle = UIThrottle(1000)
    calls = []
    throttle.schedule(calls.append, 3)
    throttle.cancel()
    throttle.flush()
    assert calls == []
    assert not throttle._timer.isActive()


def test_timeout_executes_latest_pending_call():
    throttle = UIThrottle(1000)
    calls = []
    throttle.schedule(calls.append, "first")
    throttle.schedule(calls.append, "latest")
    throttle._on_timeout()
    assert calls == ["latest"]
    throttle.cancel()


def test_set_interval_validates_and_restarts_active_timer():
    throttle = UIThrottle(1000)
    throttle.schedule(lambda: None)
    throttle.set_interval_ms(2000)
    assert throttle.interval_ms == 2000
    assert throttle._timer.isActive()
    assert throttle._timer.remainingTime() > 1000
    throttle.cancel()


def test_schedule_rejects_non_callable_without_disturbing_pending_call():
    throttle = UIThrottle(1000)
    calls = []
    throttle.schedule(calls.append, "kept")
    with pytest.raises(TypeError, match="callable"):
        throttle.schedule(None)
    throttle.flush()
    assert calls == ["kept"]
