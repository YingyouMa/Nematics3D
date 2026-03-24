from __future__ import annotations

from typing import Any, Callable

from qtpy.QtCore import QObject, QTimer


class UIThrottle(QObject):
    """
    Lightweight Qt-side throttle helper for expensive UI-triggered callbacks.

    Typical usage:

    - call `schedule(func, *args, **kwargs)` from high-frequency UI signals
    - call `flush()` on slider/button release to apply the latest pending value
    - call `cancel()` if the pending update should be discarded
    """

    def __init__(self, interval_ms: int = 40, parent: QObject | None = None):
        super().__init__(parent)

        if interval_ms <= 0:
            raise ValueError("`interval_ms` must be positive.")

        self._interval_ms = int(interval_ms)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._pending_func: Callable[..., Any] | None = None
        self._pending_args: tuple[Any, ...] = ()
        self._pending_kwargs: dict[str, Any] = {}

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def set_interval_ms(self, interval_ms: int) -> None:
        if interval_ms <= 0:
            raise ValueError("`interval_ms` must be positive.")
        self._interval_ms = int(interval_ms)
        if self._timer.isActive():
            self._timer.start(self._interval_ms)

    def schedule(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if not callable(func):
            raise TypeError("`func` must be callable.")

        self._pending_func = func
        self._pending_args = args
        self._pending_kwargs = dict(kwargs)

        if not self._timer.isActive():
            self._timer.start(self._interval_ms)

    def flush(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._run_pending()

    def cancel(self) -> None:
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

        if func is None:
            return

        func(*args, **kwargs)

    def _clear_pending(self) -> None:
        self._pending_func = None
        self._pending_args = ()
        self._pending_kwargs = {}
