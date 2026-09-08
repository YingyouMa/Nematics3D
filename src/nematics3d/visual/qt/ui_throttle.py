from __future__ import annotations

from typing import Any, Callable

from qtpy.QtCore import QObject, QTimer


def _as_positive_interval_ms(value: int) -> int:
    """Validate a Qt timer interval expressed as a positive integer in ms."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("`interval_ms` must be a positive integer.")
    return value


class UIThrottle(QObject):
    """Coalesce high-frequency Qt callbacks and execute the latest one.

    The first :meth:`schedule` call starts a single-shot timer. Further calls
    while that timer is active replace the pending callback and arguments
    without restarting the timer. When the interval expires, only the latest
    pending call is executed.

    Use :meth:`schedule` while UI values are changing, :meth:`flush` when the
    final value should be applied immediately, and :meth:`cancel` to discard a
    pending update.
    """

    def __init__(self, interval_ms: int = 40, parent: QObject | None = None):
        super().__init__(parent)
        self._interval_ms = _as_positive_interval_ms(interval_ms)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._pending_func: Callable[..., Any] | None = None
        self._pending_args: tuple[Any, ...] = ()
        self._pending_kwargs: dict[str, Any] = {}

    @property
    def interval_ms(self) -> int:
        """Return the current throttle interval in milliseconds."""
        return self._interval_ms

    def set_interval_ms(self, interval_ms: int) -> None:
        """Set the interval, restarting an active timer from this call."""
        self._interval_ms = _as_positive_interval_ms(interval_ms)
        if self._timer.isActive():
            self._timer.start(self._interval_ms)

    def schedule(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Schedule the latest callback without restarting an active timer."""
        if not callable(func):
            raise TypeError("`func` must be callable.")
        self._pending_func = func
        self._pending_args = args
        self._pending_kwargs = dict(kwargs)
        if not self._timer.isActive():
            self._timer.start(self._interval_ms)

    def flush(self) -> None:
        """Immediately execute the latest pending callback, if any."""
        if self._timer.isActive():
            self._timer.stop()
        self._run_pending()

    def cancel(self) -> None:
        """Discard the pending callback and stop the timer."""
        if self._timer.isActive():
            self._timer.stop()
        self._clear_pending()

    def _on_timeout(self) -> None:
        self._run_pending()

    def _run_pending(self) -> None:
        func = self._pending_func
        args = self._pending_args
        kwargs = self._pending_kwargs
        self._clear_pending()
        if func is not None:
            func(*args, **kwargs)

    def _clear_pending(self) -> None:
        self._pending_func = None
        self._pending_args = ()
        self._pending_kwargs = {}


__all__ = ["UIThrottle"]
