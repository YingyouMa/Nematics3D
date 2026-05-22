from qtpy import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
import datetime
from typing import Callable, Literal, MutableMapping, Any
import math
import numpy as np

from nematics3d.datatypes import as_str, Vect
from nematics3d.geometry import (
    calc_vec_from_azimuth_polar,
    get_azimuth as geometry_get_azimuth,
    get_polar_angle as geometry_get_polar_angle,
)
from .ui_throttle import UIThrottle


@dataclass(slots=True, weakref_slot=True)
class SliderItem:
    slider: QtWidgets.QSlider
    value_box: QtWidgets.QDoubleSpinBox
    tick_to_value: Callable[[int], float]
    value_to_tick: Callable[[float], int]
    state_key: str
    value_min: float
    value_max: float
    value_fmt: str = "{:.2f}"
    input_out_of_range: Literal["clamp", "expand_max"] = "clamp"

    @property
    def label(self) -> QtWidgets.QDoubleSpinBox:
        return self.value_box

    def get_value(self) -> float:
        return float(self.tick_to_value(int(self.slider.value())))

    def set_label(self, value: float | None = None) -> None:
        v = self.get_value() if value is None else float(value)
        self.value_box.blockSignals(True)
        try:
            self._helper_extend_value_box_max(v)
            self.value_box.setValue(v)
        finally:
            self.value_box.blockSignals(False)

    def sync_to_state(self, state: MutableMapping[str, Any]) -> float:
        v = self.get_value()
        self.set_label(v)
        state[self.state_key] = v
        return v

    def set_tick(self, value: float, *, is_block_signals: bool = True) -> None:
        value = self._helper_normalize_input_value(float(value))
        tick = self.value_to_tick(value)
        if is_block_signals:
            self.slider.blockSignals(True)
        try:
            self._helper_extend_slider_max(int(tick))
            self.slider.setValue(int(tick))
            self.set_label()
        finally:
            if is_block_signals:
                self.slider.blockSignals(False)

    def apply_value_box_edit(self) -> None:
        value = self._helper_normalize_input_value(float(self.value_box.value()))
        self.set_tick(value, is_block_signals=False)

    def set_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(bool(enabled))
        self.value_box.setEnabled(bool(enabled))

    def _helper_normalize_input_value(self, value: float) -> float:
        value = max(float(self.value_min), float(value))
        if value <= float(self.value_max):
            return value
        if self.input_out_of_range == "expand_max":
            self.value_max = value
            self._helper_extend_value_box_max(value)
            return value
        return float(self.value_max)

    def _helper_extend_slider_max(self, tick: int) -> None:
        if tick <= int(self.slider.maximum()):
            return
        tick_max = max(tick, int(math.ceil(tick * 1.2)))
        self.slider.setMaximum(tick_max)

    def _helper_extend_value_box_max(self, value: float) -> None:
        if value <= float(self.value_box.maximum()):
            return
        value_max = max(value, abs(value) * 1.2)
        self.value_box.setMaximum(float(value_max))


def _helper_decimals_from_value_fmt(value_fmt: str) -> int:
    marker = "{:."
    if not value_fmt.startswith(marker) or not value_fmt.endswith("f}"):
        return 6
    raw = value_fmt[len(marker) : -2]
    try:
        return max(0, int(raw))
    except ValueError:
        return 6


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
    input_out_of_range: Literal["clamp", "expand_max"] = "clamp",
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

    # ---- editable value box ----
    value_box = QtWidgets.QDoubleSpinBox(row_widget)
    value_box.setDecimals(_helper_decimals_from_value_fmt(value_fmt))
    value_box.setKeyboardTracking(False)
    value_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    value_box.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    value_box.setMinimumWidth(int(val_min_width))
    value_box_max = float(value_max)
    if input_out_of_range == "expand_max":
        value_box_max = max(value_box_max, abs(value_box_max) * 1000.0, 1.0e12)
    value_box.setRange(float(value_min), value_box_max)
    h.addWidget(value_box)

    # ---- init ----
    slider.setValue(int(tick_init))

    item = SliderItem(
        slider=slider,
        value_box=value_box,
        tick_to_value=tick_to_value,
        value_to_tick=value_to_tick,
        state_key=(name if state_key is None else state_key),
        value_fmt=value_fmt,
        value_min=value_min,
        value_max=value_max,
        input_out_of_range=input_out_of_range,
    )
    item.set_label()  # initialize label text
    value_box.editingFinished.connect(item.apply_value_box_edit)

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
        alpha = max(0.0, alpha)
        log_min = math.log(self.value_min, self.base)
        log_max = math.log(self.value_max, self.base)
        log_v = log_min + alpha * (log_max - log_min)
        return float(self.base**log_v)

    def value_to_tick(self, v: float) -> int:
        v = float(v)
        v = max(self.value_min, v)
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
            input_out_of_range="expand_max",
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

    @classmethod
    def show_once(cls, *args, **kwargs):
        """Show one panel unless the same figure already has it for this host."""
        host = kwargs.get("host", args[0] if args else None)
        figure = kwargs.get("figure", args[1] if len(args) >= 2 else None)
        if figure is None and host is not None:
            figure = getattr(host, "fig", None)

        interacts = getattr(figure, "interacts", None) if figure is not None else None
        if interacts is not None:
            for panel in interacts:
                if isinstance(panel, cls) and getattr(panel, "host", None) is host:
                    return panel

        panel = cls(*args, **kwargs)
        panel.show()
        return panel

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
        self._snapshot_names_saved_from_panel: list[str] = []
        self._helper_save_snapshot(self.str_now, is_user_snapshot=False)
        if hasattr(self.host, "state_is_interactable"):
            object.__setattr__(self.host, "state_is_interactable", False)

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
        # Save / Load Actions group
        # ----------------------------
        group_reset = QtWidgets.QGroupBox("Save / Load", self)
        hl_reset = QtWidgets.QHBoxLayout(group_reset)
        self.layout.addWidget(group_reset)

        self.btn_save_current = QtWidgets.QPushButton("Save Current", group_reset)
        self.btn_save_current.setToolTip(
            "Save the current parameters using a default timestamped name."
        )
        self.btn_save_current.clicked.connect(self._on_save_current_snapshot)
        hl_reset.addWidget(self.btn_save_current)

        self.btn_reset_orig = QtWidgets.QPushButton("Restore Original", group_reset)
        self.btn_reset_orig.setToolTip(
            "Restore the state captured when this control panel was opened."
        )
        self.btn_reset_orig.clicked.connect(self._on_restore_original_snapshot)
        hl_reset.addWidget(self.btn_reset_orig)

        self.btn_load_latest = QtWidgets.QPushButton("Load Latest Save", group_reset)
        self.btn_load_latest.setToolTip(
            "Restore the most recent snapshot created from this control panel."
        )
        self.btn_load_latest.clicked.connect(self._on_load_latest_snapshot)
        hl_reset.addWidget(self.btn_load_latest)

        self.btn_load_choose = QtWidgets.QPushButton("Load Saved...", group_reset)
        self.btn_load_choose.setToolTip("Choose one available snapshot and restore it.")
        self.btn_load_choose.clicked.connect(self._on_choose_snapshot_to_restore)
        hl_reset.addWidget(self.btn_load_choose)

        self.host.act_attach_sync_task(name=self.str_now, func=self._sync_func)

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

    # -------------------------------
    # Snapshot helpers
    # -------------------------------

    def _helper_list_snapshot_hosts(self):
        return [self.host]

    def _helper_make_snapshot_name(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _helper_get_snapshot_names(self) -> list[str]:
        hosts = list(self._helper_list_snapshot_hosts())
        if not hosts:
            return []

        name_sets = [set(getattr(host, "opts_backup", {}).keys()) for host in hosts]
        names_common = set.intersection(*name_sets) if name_sets else set()
        names_primary = list(getattr(hosts[0], "opts_backup", {}).keys())
        return [name for name in names_primary if name in names_common]

    def _helper_get_snapshot_choice_entries(self) -> list[tuple[str, str]]:
        entries = []
        for name in self._helper_get_snapshot_names():
            label = f"{name} (initial)" if name == self.str_now else name
            entries.append((label, name))
        return entries

    def _helper_notify_snapshot(self, message: str) -> None:
        console = getattr(self.fig, "console", None) if self.fig is not None else None
        if console is not None:
            console.println(message)

    def _helper_save_snapshot(self, name: str, *, is_user_snapshot: bool) -> None:
        for host in self._helper_list_snapshot_hosts():
            host.act_save_opts(name)
        if is_user_snapshot:
            self._snapshot_names_saved_from_panel.append(name)

    def _helper_restore_snapshot(self, name: str) -> None:
        for host in self._helper_list_snapshot_hosts():
            snapshot = getattr(host, "opts_backup", {}).get(name)
            if snapshot is None:
                raise KeyError(f"Snapshot {name!r} was not found on {host!s}.")
            attrs_forbidden = set(getattr(host, "attrs_forbidden", ()))
            payload = {
                key: value
                for key, value in snapshot.items()
                if key not in attrs_forbidden
            }
            host.act_commit(**payload)

    def _helper_after_restore_snapshot(self, name: str) -> None:
        return None

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

    def _on_save_current_snapshot(self):
        name = self._helper_make_snapshot_name()
        self._helper_save_snapshot(name, is_user_snapshot=True)
        self._helper_notify_snapshot(f"Saved current parameters as {name!r}.")

    def _on_restore_original_snapshot(self):
        self._helper_restore_snapshot(self.str_now)
        self._helper_after_restore_snapshot(self.str_now)
        self._helper_notify_snapshot("Restored the original panel snapshot.")

    def _on_load_latest_snapshot(self):
        if not self._snapshot_names_saved_from_panel:
            QtWidgets.QMessageBox.information(
                self,
                "No Saved Snapshot",
                "No panel-created snapshot is available yet.",
            )
            return
        name = self._snapshot_names_saved_from_panel[-1]
        self._helper_restore_snapshot(name)
        self._helper_after_restore_snapshot(name)
        self._helper_notify_snapshot(f"Restored snapshot {name!r}.")

    def _on_choose_snapshot_to_restore(self):
        entries = self._helper_get_snapshot_choice_entries()
        if not entries:
            QtWidgets.QMessageBox.information(
                self,
                "No Saved Snapshot",
                "No snapshot is available for this panel.",
            )
            return

        labels = [label for label, _name in entries]
        label_selected, is_ok = QtWidgets.QInputDialog.getItem(
            self,
            "Load Saved Snapshot",
            "Choose one snapshot to restore:",
            labels,
            0,
            False,
        )
        if not is_ok or not label_selected:
            return

        lookup = dict(entries)
        name = lookup[str(label_selected)]
        self._helper_restore_snapshot(name)
        self._helper_after_restore_snapshot(name)
        self._helper_notify_snapshot(f"Restored snapshot {name!r}.")

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
        if hasattr(self.host, "state_is_interactable"):
            object.__setattr__(self.host, "state_is_interactable", True)
        self.host.act_detach_sync_task(self.str_now)
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
