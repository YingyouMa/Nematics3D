from qtpy import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
import datetime
from typing import Callable, MutableMapping, Any
import math
import numpy as np

from nematics3d.datatypes import as_str, Vect
from nematics3d.geometry import (
    calc_vec_from_azimuth_polar,
    get_azimuth as geometry_get_azimuth,
    get_polar_angle as geometry_get_polar_angle,
)
from .ui_throttle import UIThrottle


@dataclass(slots=True)
class SliderItem:
    slider: QtWidgets.QSlider
    label: QtWidgets.QLabel
    tick_to_value: Callable[[int], float]
    value_to_tick: Callable[[float], int]
    state_key: str
    value_min: int
    value_max: int
    value_fmt: str = "{:.2f}"

    def get_value(self) -> float:
        return float(self.tick_to_value(int(self.slider.value())))

    def set_label(self, value: float | None = None) -> None:
        v = self.get_value() if value is None else float(value)
        self.label.setText(self.value_fmt.format(v))

    def sync_to_state(self, state: MutableMapping[str, Any]) -> float:
        v = self.get_value()
        self.set_label(v)
        state[self.state_key] = v
        return v

    def set_tick(self, value: float, *, is_block_signals: bool = True) -> None:
        tick = self.value_to_tick(value)
        if is_block_signals:
            self.slider.blockSignals(True)
        try:
            tick_max = self.value_to_tick(float(self.value_max))
            if tick > tick_max:
                tick_max = int(tick * 1.2)
                self.slider.setMaximum(tick_max)
            self.slider.setValue(int(tick))
            self.set_label()
        finally:
            if is_block_signals:
                self.slider.blockSignals(False)

    def set_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(bool(enabled))
        self.label.setEnabled(bool(enabled))


def make_labeled_slider_row(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    name: str,
    state_key: str | None = None,
    value_min: int,
    value_max: int,
    value_init: int,
    tick_to_value: Callable[[int], float] = float,
    value_to_tick: Callable[[float], int] = int,
    value_fmt: str = "{:.2f}",
    key_min_width: int = 120,
    val_min_width: int = 70,
    single_step: int = 1,
    page_step: int = 10,
    tracking: bool = True,
    spacing: int = 8,
) -> SliderItem:

    tick_min = value_to_tick(value_min)
    tick_max = value_to_tick(value_max)
    tick_init = value_to_tick(value_init)

    # ---- row container ----
    row_widget = QtWidgets.QWidget(parent)
    h = QtWidgets.QHBoxLayout(row_widget)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(int(spacing))

    # ---- key label ----
    lab_key = QtWidgets.QLabel(f"{name}:", row_widget)
    lab_key.setMinimumWidth(int(key_min_width))
    h.addWidget(lab_key)

    # ---- slider ----
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, row_widget)
    slider.setMinimum(int(tick_min))
    slider.setMaximum(int(tick_max))
    slider.setSingleStep(int(single_step))
    slider.setPageStep(int(page_step))
    slider.setTracking(bool(tracking))
    h.addWidget(slider, 1)

    # ---- value label ----
    lab_val = QtWidgets.QLabel("", row_widget)
    lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    lab_val.setMinimumWidth(int(val_min_width))
    h.addWidget(lab_val)

    # ---- init ----
    slider.setValue(int(tick_init))

    item = SliderItem(
        slider=slider,
        label=lab_val,
        tick_to_value=tick_to_value,
        value_to_tick=value_to_tick,
        state_key=(name if state_key is None else state_key),
        value_fmt=value_fmt,
        value_min=value_min,
        value_max=value_max,
    )
    item.set_label()  # initialize label text

    layout.addWidget(row_widget)
    return item


def make_RGB_slider(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    sliders: dict[str, SliderItem],
    prefix: str,
    init_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    value_min: int = 0,
    value_max: int = 1,
    value_fmt: str = "{:.2f}",
    single_step: int = 1,
    page_step: int = 10,
) -> None:

    def _t2v(t: int) -> float:
        return t / 1000.0

    def _v2t(v: float) -> int:
        return int(v * 1000.0)

    for i, ch in enumerate(("r", "g", "b")):
        dict_key = f"{prefix}_{ch}"  # key in `sliders` dict
        state_key = dict_key  # key in `state` dict

        sliders[dict_key] = make_labeled_slider_row(
            parent=parent,
            layout=layout,
            name=ch.upper(),  # label on UI
            state_key=state_key,  # where to write in state
            value_min=value_min,
            value_max=value_max,
            value_init=init_rgb[i],
            tick_to_value=_t2v,
            value_to_tick=_v2t,
            value_fmt=value_fmt,
            single_step=single_step,
            page_step=page_step,
        )


@dataclass(frozen=True, slots=True)
class LogTickMapper:
    value_min: float
    value_max: float
    tick_min: int = 0
    tick_max: int = 1000
    base: float = 10.0

    def __post_init__(self):
        if not (self.value_min > 0 and self.value_max > 0):
            raise ValueError("LogTickMapper: value_min/value_max must be > 0.")
        if not (self.tick_max > self.tick_min):
            raise ValueError("LogTickMapper: tick_max must be > tick_min.")
        if not (self.value_max > self.value_min):
            raise ValueError("LogTickMapper: value_max must be > value_min.")
        if not (self.base > 0 and self.base != 1.0):
            raise ValueError("LogTickMapper: base must be > 0 and != 1.")

    def tick_to_value(self, t: int) -> float:
        t = int(t)
        alpha = (t - self.tick_min) / float(self.tick_max - self.tick_min)
        alpha = min(1.0, max(0.0, alpha))
        log_min = math.log(self.value_min, self.base)
        log_max = math.log(self.value_max, self.base)
        log_v = log_min + alpha * (log_max - log_min)
        return float(self.base**log_v)

    def value_to_tick(self, v: float) -> int:
        v = float(v)
        v = min(self.value_max, max(self.value_min, v))
        log_min = math.log(self.value_min, self.base)
        log_max = math.log(self.value_max, self.base)
        log_v = math.log(v, self.base)
        alpha = (log_v - log_min) / float(log_max - log_min)
        t = self.tick_min + alpha * (self.tick_max - self.tick_min)
        return int(round(t))


@dataclass(slots=True, weakref_slot=True)
class PressHoldButtonItem:

    button: QtWidgets.QPushButton
    callback: Callable[[], None]
    long_press_ms: int = 450
    repeat_ms: int = 80

    _hold_active: bool = False
    _pressed: bool = False
    _timer_long: QtCore.QTimer | None = None
    _timer_repeat: QtCore.QTimer | None = None

    def __post_init__(self) -> None:
        # timer: detect long press
        self._timer_long = QtCore.QTimer(self.button)
        self._timer_long.setSingleShot(True)
        self._timer_long.timeout.connect(self._on_long_press_begin)

        # timer: repeat while holding
        self._timer_repeat = QtCore.QTimer(self.button)
        self._timer_repeat.setSingleShot(False)
        self._timer_repeat.timeout.connect(self._fire)

        self.button.pressed.connect(self._on_pressed)
        self.button.released.connect(self._on_released)

    def _fire(self) -> None:
        try:
            self.callback()
        except Exception:
            raise

    def _on_pressed(self) -> None:
        self._pressed = True
        self._hold_active = False
        self._fire()
        # start long-press detection
        assert self._timer_long is not None
        self._timer_long.start(int(self.long_press_ms))

    def _on_long_press_begin(self) -> None:
        # Only enter hold mode if still pressed.
        if not self._pressed:
            return
        self._hold_active = True
        # start repeating after the initial short-press fire
        assert self._timer_repeat is not None
        self._timer_repeat.start(int(self.repeat_ms))

    def _on_released(self) -> None:
        # Stop timers.
        self._pressed = False
        assert self._timer_long is not None
        assert self._timer_repeat is not None
        self._timer_long.stop()
        self._timer_repeat.stop()

        self._hold_active = False

    def set_enabled(self, enabled: bool) -> None:
        self.button.setEnabled(bool(enabled))


def make_press_hold_button(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    text: str,
    callback: Callable[[], None],
    long_press_ms: int = 450,
    repeat_ms: int = 80,
    min_width: int | None = None,
) -> PressHoldButtonItem:

    btn = QtWidgets.QPushButton(text, parent)
    if min_width is not None:
        btn.setMinimumWidth(int(min_width))
    layout.addWidget(btn)

    return PressHoldButtonItem(
        button=btn,
        callback=callback,
        long_press_ms=int(long_press_ms),
        repeat_ms=int(repeat_ms),
    )


@dataclass(slots=True)
class MovePointConsole:

    state: MutableMapping[str, Any]
    center_key: str
    step_key: str

    parent: QtWidgets.QWidget
    title: str = "Move Point"

    step_min: float = 0.01
    step_max: float = 100.0
    step_tick_max: int = 1000
    step_fmt: str = "{:.2f}"
    center_fmt: str = "{:.2f}"

    on_move: Callable[[np.ndarray], None] | None = None
    on_press: Callable[[Vect(3), np.ndarray], None] | None = None
    on_hold: Callable[[Vect(3), np.ndarray], None] | None = None
    on_release: Callable[[Vect(3), np.ndarray], None] | None = None

    long_press_ms: int = 450
    repeat_ms: int = 80

    # owned widgets/items
    group: QtWidgets.QGroupBox | None = None
    gl: QtWidgets.QVBoxLayout | None = None
    lab_center: QtWidgets.QLabel | None = None

    slider_step: SliderItem | None = None

    btn_x_neg: PressHoldButtonItem | None = None
    btn_x_pos: PressHoldButtonItem | None = None
    btn_y_neg: PressHoldButtonItem | None = None
    btn_y_pos: PressHoldButtonItem | None = None
    btn_z_neg: PressHoldButtonItem | None = None
    btn_z_pos: PressHoldButtonItem | None = None

    def __post_init__(self):
        # ---- group + layout ----
        self.group = QtWidgets.QGroupBox(self.title, self.parent)
        self.gl = QtWidgets.QVBoxLayout(self.group)

        # ---- step slider (log) ----
        step_map = LogTickMapper(
            value_min=float(self.step_min),
            value_max=float(self.step_max),
            tick_min=0,
            tick_max=int(self.step_tick_max),
            base=10.0,
        )

        # ensure step exists in state
        if self.step_key not in self.state:
            self.state[self.step_key] = 1.0
        step0 = float(self.state[self.step_key])

        self.slider_step = make_labeled_slider_row(
            parent=self.group,
            layout=self.gl,
            name="step",
            state_key=self.step_key,
            value_min=step_map.value_min,
            value_max=step_map.value_max,
            value_init=step0,
            tick_to_value=step_map.tick_to_value,
            value_to_tick=step_map.value_to_tick,
            value_fmt=self.step_fmt,
        )
        # sync label/state once
        self.slider_step.sync_to_state(self.state)

        # ---- center display + buttons ----
        grid_widget = QtWidgets.QWidget(self.group)
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.gl.addWidget(grid_widget)

        grid.addWidget(QtWidgets.QLabel("Location:", self.group), 0, 0, 1, 1)
        self.lab_center = QtWidgets.QLabel("", self.group)
        self.lab_center.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        grid.addWidget(self.lab_center, 0, 1, 1, 2)

        # ensure center exists in state
        if self.center_key not in self.state:
            self.state[self.center_key] = np.array([0.0, 0.0, 0.0], dtype=float)

        self._update_center_label()

        # helper: create a button in a grid cell (needs parent+layout for factory)
        def _cell_button(row: int, col: int, text: str, cb, release_cb):
            cell = QtWidgets.QWidget(self.group)
            v = QtWidgets.QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            item = make_press_hold_button(
                parent=cell,
                layout=v,
                text=text,
                callback=cb,
                long_press_ms=self.long_press_ms,
                repeat_ms=self.repeat_ms,
            )
            # Release hook (to expose on_release)
            if release_cb is not None:
                item.button.released.connect(release_cb)
            grid.addWidget(cell, row, col)
            return item

        def _move_once(dir_: Vect(3), *, item_ref: PressHoldButtonItem):
            # pull latest step from state (slider may have changed)
            step = float(self.state[self.step_key])
            c = np.array(self.state[self.center_key], dtype=float)
            d = np.array(dir_, dtype=float)
            c_new = c + step * d
            self.state[self.center_key] = c_new

            # callback classification
            if bool(item_ref._hold_active):
                # during long hold repeats
                if self.on_hold is not None:
                    self.on_hold(dir_, c_new)
            else:
                # short press single fire (happens on button press)
                if self.on_press is not None:
                    self.on_press(dir_, c_new)

            # always perform the move action
            if self.on_move is not None:
                self.on_move(c_new)
            self._update_center_label()

        def _on_release(dir_: Vect(3)):
            if self.on_release is None:
                return
            c_now = np.array(self.state[self.center_key], dtype=float)
            self.on_release(dir_, c_now)

        # Create 6 buttons.
        # Layout: row 1: X, row 2: Y, row 3: Z (neg left, pos right)
        self.btn_x_neg = _cell_button(
            1,
            0,
            "-X",
            cb=lambda: _move_once(
                (-1, 0, 0), item_ref=self.btn_x_neg
            ),  # placeholder, fixed below
            release_cb=lambda: _on_release((-1, 0, 0)),
        )
        self.btn_x_pos = _cell_button(
            1,
            1,
            "+X",
            cb=lambda: _move_once((+1, 0, 0), item_ref=self.btn_x_pos),
            release_cb=lambda: _on_release((+1, 0, 0)),
        )
        self.btn_y_neg = _cell_button(
            2,
            0,
            "-Y",
            cb=lambda: _move_once((0, -1, 0), item_ref=self.btn_y_neg),
            release_cb=lambda: _on_release((0, -1, 0)),
        )
        self.btn_y_pos = _cell_button(
            2,
            1,
            "+Y",
            cb=lambda: _move_once((0, +1, 0), item_ref=self.btn_y_pos),
            release_cb=lambda: _on_release((0, +1, 0)),
        )
        self.btn_z_neg = _cell_button(
            3,
            0,
            "-Z",
            cb=lambda: _move_once((0, 0, -1), item_ref=self.btn_z_neg),
            release_cb=lambda: _on_release((0, 0, -1)),
        )
        self.btn_z_pos = _cell_button(
            3,
            1,
            "+Z",
            cb=lambda: _move_once((0, 0, +1), item_ref=self.btn_z_pos),
            release_cb=lambda: _on_release((0, 0, +1)),
        )

    def _update_center_label(self) -> None:
        assert self.lab_center is not None
        c = np.array(self.state[self.center_key], dtype=float)
        fmt = self.center_fmt
        self.lab_center.setText(
            f"({fmt.format(c[0])}, {fmt.format(c[1])}, {fmt.format(c[2])})"
        )

    def set_enabled(self, enabled: bool) -> None:
        en = bool(enabled)
        assert self.group is not None
        self.group.setEnabled(en)


class PanelBase(QtWidgets.QWidget):

    # -------------------------------
    # Initialization
    # -------------------------------

    def __init__(
        self,
        host,
        figure,
        title: str = "Panel",
        slider_throttle_ms: int | None = None,
    ):

        title = as_str(title, name="The title of panel", replace="Panel")

        super().__init__()

        required_methods = (
            "act_save_opts",
            "act_attach_sync_task",
            "act_detach_sync_task",
            "act_commit",
        )
        missing_methods = [
            name for name in required_methods if not callable(getattr(host, name, None))
        ]
        missing_attrs = [name for name in ("opts_backup",) if not hasattr(host, name)]
        if missing_methods or missing_attrs:
            lines = [
                "PanelBase requires a host object compatible with the panel sync/reset workflow.",
            ]
            if missing_methods:
                lines.append(f"Missing methods: {missing_methods}")
            if missing_attrs:
                lines.append(f"Missing attributes: {missing_attrs}")
            raise TypeError("\n".join(lines))

        self.host = host
        self.fig = figure
        self.raw_name = "panel_unregistered"
        self.str_now = datetime.datetime.now().strftime("panel_%Y%m%d_%H%M%S_%f")[:-3]
        self.host.act_save_opts(self.str_now)
        self.str_now_live = self.str_now + "_live"
        self.host.act_save_opts(self.str_now_live)
        if hasattr(self.host, "_state_is_interactable"):
            object.__setattr__(self.host, "_state_is_interactable", False)

        self._is_block_chk_commit = False

        self.state: dict[str, object] = {}
        if slider_throttle_ms is None:
            pick_manager = (
                getattr(self.fig, "pick_manager", None)
                if self.fig is not None
                else None
            )
            slider_throttle_ms = getattr(
                getattr(pick_manager, "opts", None), "slider_throttle_ms", 20
            )
        self.sliders: dict[str, SliderItem] = {}
        if not hasattr(self, "_custom_sliders"):
            self._custom_sliders = []
        self.slider_throttle_ms = int(slider_throttle_ms)
        self._slider_throttle = UIThrottle(
            interval_ms=self.slider_throttle_ms,
            parent=self,
        )

        self.setWindowTitle(title)
        self.setObjectName("panel")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)

        if self.fig is not None and hasattr(self.fig, "act_register_interact"):
            interact_name = self.fig.act_register_interact(self)
            if interact_name is not None:
                self.setObjectName(interact_name)

        self.build_ui()
        self.act_wire_default_slider_connections()

        # ----------------------------
        # Reset Actions group
        # ----------------------------
        group_reset = QtWidgets.QGroupBox("Reset", self)
        hl_reset = QtWidgets.QHBoxLayout(group_reset)
        self.layout.addWidget(group_reset)

        self.btn_reset_live = QtWidgets.QPushButton("Reset to Live", group_reset)
        self.btn_reset_live.setToolTip(
            "Discard UI changes and revert to the current live baseline."
        )
        self.btn_reset_live.clicked.connect(self._on_reset_to_live)
        hl_reset.addWidget(self.btn_reset_live)

        self.btn_reset_orig = QtWidgets.QPushButton("Restore Original", group_reset)
        self.btn_reset_orig.setToolTip(
            "Discard all console overrides and restore the initial state."
        )
        self.btn_reset_orig.clicked.connect(self._on_reset_to_original)
        hl_reset.addWidget(self.btn_reset_orig)

        self.host.act_attach_sync_task(name=self.str_now_live, func=self._sync_func)

        console = getattr(self.fig, "console", None) if self.fig is not None else None
        if console is not None and self.name != "panel_unregistered":
            console.println(f"Opened the control panel for {self.host!s}.")
            console.println(
                "In the command line, the controlled object is also available as "
                f"the current figure's interacts[{self.name!r}].host."
            )
            console.println(
                "The red helper marker shows the first currently used point of "
                "this object."
            )

    # -------------------------------
    # Slider synchronization and scheduling
    # -------------------------------

    def _sync_from_host_slider(self, attr: str, value: float):
        s = self.sliders[attr]
        s.set_tick(value, is_block_signals=True)
        self.on_changed(0, is_commit=False)

    def _helper_sync_update_live_backup(self, kwargs, *, host=None) -> bool:
        """
        Update the panel live snapshot only for changes that come from outside
        the panel itself.

        Design note
        -----------
        In this repository, panel-originated commits and external command-line
        commits should share the same synchronization side effects such as
        label refreshes, helper-visual updates, and marker movement. The only
        behavior that should remain exclusive to external updates is advancing
        the panel's live backup, which powers ``Reset to Live``.
        """
        if getattr(self, "_is_gui_updating", False):
            return False
        if host is None:
            host = self.host
        host.opts_backup[self.str_now_live].update(kwargs)
        return True

    def on_changed(self, _v=0, is_commit=True):
        for item in self.sliders.values():
            item.sync_to_state(self.state)

        if is_commit:
            self.commit()

    def _helper_begin_slider_interaction(self, *_args):
        func = getattr(self, "_on_slider_pressed", None)
        if not callable(func):
            func = getattr(self, "_helper_begin_continuous_interaction", None)
        if callable(func):
            func(*_args)

    def _helper_end_slider_interaction(self, *_args):
        self._slider_throttle.flush()
        func = getattr(self, "_on_slider_released", None)
        if not callable(func):
            func = getattr(self, "_helper_end_continuous_interaction", None)
        if callable(func):
            func(*_args)

    def _schedule_on_changed(self, _value=0):
        self._slider_throttle.schedule(self.on_changed)

    def act_wire_default_slider_connections(self):
        custom_sliders = list(getattr(self, "_custom_sliders", []))
        for key, item in self.sliders.items():
            if item in custom_sliders:
                continue
            slider = item.slider
            try:
                slider.valueChanged.disconnect()
            except Exception:
                pass
            try:
                slider.sliderPressed.disconnect()
            except Exception:
                pass
            try:
                slider.sliderReleased.disconnect()
            except Exception:
                pass

            if str(key).endswith("_move_step"):
                slider.valueChanged.connect(
                    lambda _v=0: self.on_changed(is_commit=False)
                )
            else:
                slider.valueChanged.connect(self._schedule_on_changed)
                slider.sliderPressed.connect(self._helper_begin_slider_interaction)
                slider.sliderReleased.connect(self._helper_end_slider_interaction)

    # -------------------------------
    # Subclass hooks
    # -------------------------------

    def build_ui(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def _sync_func(self):
        raise NotImplementedError

    # -------------------------------
    # Public panel actions
    # -------------------------------

    def act_set_slider_throttle_ms(self, value: int):
        self.slider_throttle_ms = int(value)
        self._slider_throttle.set_interval_ms(self.slider_throttle_ms)

    def _on_reset_to_live(self):
        self.host.act_commit(**self.host.opts_backup[self.str_now_live])

    def _on_reset_to_original(self):
        original = self.host.opts_backup[self.str_now]
        self.host.act_commit(**original)
        self.host.opts_backup[self.str_now_live] = dict(original)

    # -------------------------------
    # Qt lifecycle
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PanelBase overrides QWidget.closeEvent so panel-specific cleanup always
    # runs before the window is accepted and removed.
    # ==================================================

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.on_close()
        finally:
            event.accept()

    def on_close(self):
        self._slider_throttle.cancel()
        if hasattr(self.host, "_state_is_interactable"):
            object.__setattr__(self.host, "_state_is_interactable", True)
        self.host.act_detach_sync_task(self.str_now_live)
        if self.fig is not None and hasattr(self.fig, "act_unregister_interact"):
            self.fig.act_unregister_interact(self)

    # -------------------------------
    # Geometry and naming helpers
    # -------------------------------

    @staticmethod
    def _vect_text(vect, name):
        text = f"{name}: ({vect[0]:.2f}, {vect[1]:.2f}, {vect[2]:.2f})"
        return text

    @staticmethod
    def _helper_calc_vec(azimuth, polar_angle):
        return calc_vec_from_azimuth_polar(azimuth, polar_angle)

    @staticmethod
    def get_azimuth(vec):
        return geometry_get_azimuth(vec)

    @staticmethod
    def get_polar_angle(vec):
        return geometry_get_polar_angle(vec)

    # -------------------------------
    # Readable identity
    # -------------------------------

    @property
    def name(self):
        return self.raw_name

    @name.setter
    def name(self, value):
        self.raw_name = as_str(value, name="Panel name", replace="panel")

    def __str__(self):
        return f"{self.name} -> {self.host!s}"

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r} -> {self.host!s})"
