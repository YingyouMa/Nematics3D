import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtWidgets, QtCore
from dataclasses import dataclass
from typing import Callable


# ============================================================
# UI helpers
# ============================================================
@dataclass
class SliderItem:
    slider: QtWidgets.QSlider
    label: QtWidgets.QLabel
    get_value: callable


def make_labeled_slider_row(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    name: str,
    tick_min: int,
    tick_max: int,
    tick_init: int,
    tick_to_value: Callable[[int], float],
    value_fmt: str = "{:.4g}",
    key_min_width: int = 120,
    val_min_width: int = 70,
    single_step: int = 1,
    page_step: int = 10,
) -> SliderItem:
    row_widget = QtWidgets.QWidget(parent)
    h = QtWidgets.QHBoxLayout(row_widget)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)

    lab_key = QtWidgets.QLabel(f"{name}:", row_widget)
    lab_key.setMinimumWidth(key_min_width)
    h.addWidget(lab_key)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, row_widget)
    slider.setMinimum(int(tick_min))
    slider.setMaximum(int(tick_max))
    slider.setSingleStep(single_step)
    slider.setPageStep(page_step)
    slider.setTracking(True)
    h.addWidget(slider, 1)

    lab_val = QtWidgets.QLabel("", row_widget)
    lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    lab_val.setMinimumWidth(val_min_width)
    h.addWidget(lab_val)

    slider.setValue(tick_init)

    def get_value() -> float:
        return float(tick_to_value(int(slider.value())))

    lab_val.setText(value_fmt.format(get_value()))
    layout.addWidget(row_widget)

    return SliderItem(slider=slider, label=lab_val, get_value=get_value)


@dataclass
class RepeatButtonItem:
    button: QtWidgets.QPushButton
    delay_timer: QtCore.QTimer
    repeat_timer: QtCore.QTimer


def make_repeat_button(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    name: str,
    on_trigger: Callable[[], None],
    repeat_delay_ms: int = 250,
    repeat_interval_ms: int = 40,
    min_width: int | None = None,
) -> RepeatButtonItem:
    """
    Create a QPushButton with press-and-hold auto-repeat behavior.

    Behavior
    --------
    - pressed  : trigger once immediately
    - hold     : after delay, trigger repeatedly at fixed interval
    - released : stop immediately
    """
    btn = QtWidgets.QPushButton(name, parent)
    if min_width is not None:
        btn.setMinimumWidth(int(min_width))
    layout.addWidget(btn)

    repeat_timer = QtCore.QTimer(btn)
    repeat_timer.setInterval(int(repeat_interval_ms))
    repeat_timer.timeout.connect(on_trigger)

    delay_timer = QtCore.QTimer(btn)
    delay_timer.setSingleShot(True)

    def _start_repeat() -> None:
        if btn.isDown():
            repeat_timer.start()

    delay_timer.timeout.connect(_start_repeat)

    def _pressed() -> None:
        on_trigger()
        delay_timer.start(int(repeat_delay_ms))

    def _released() -> None:
        delay_timer.stop()
        repeat_timer.stop()

    btn.pressed.connect(_pressed)
    btn.released.connect(_released)

    return RepeatButtonItem(button=btn, delay_timer=delay_timer, repeat_timer=repeat_timer)


# ============================================================
# Main window
# ============================================================
class MainWindow(QtWidgets.QMainWindow):
    # Step range: 1e-6 .. 1e0 on log10 scale
    LOG10_MIN = -6.0
    LOG10_MAX = 0.0

    # Slider ticks
    TICK_MIN = 0
    TICK_MAX = 6000
    TICK_INIT = 3900

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyVistaQt: +X Step (log) + Press-and-hold (Packed Controls)")
        self.resize(1100, 700)

        # ---- central layout ----
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- 3D view ----
        self.view = QtInteractor(central)
        root.addWidget(self.view, stretch=1)

        self.view.set_background("white")

        self.pos = np.array([0.0, 0.0, 0.0], dtype=float)
        sphere = pv.Sphere(radius=0.2, center=self.pos)
        self.actor = self.view.add_mesh(sphere, smooth_shading=True)

        # ---- control panel ----
        panel = QtWidgets.QWidget(central)
        panel.setFixedWidth(380)
        root.addWidget(panel, stretch=0)

        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        title = QtWidgets.QLabel("X controls")
        title.setStyleSheet("font-weight: 600;")
        v.addWidget(title)

        # +X repeat button (wrapped)
        self.btn_x_plus = make_repeat_button(
            parent=panel,
            layout=v,
            name="+X (hold to repeat)",
            on_trigger=self._step_x_plus_once,
            repeat_delay_ms=250,
            repeat_interval_ms=40,
        )

        # Step slider row (wrapped)
        self.step_item = make_labeled_slider_row(
            parent=panel,
            layout=v,
            name="step",
            tick_min=self.TICK_MIN,
            tick_max=self.TICK_MAX,
            tick_init=self.TICK_INIT,
            tick_to_value=self._tick_to_step,
            value_fmt="{:.3e}",
            key_min_width=60,
            val_min_width=160,
            single_step=1,
            page_step=50,
        )
        self.step_item.slider.valueChanged.connect(self._on_step_tick_changed)

        self.lbl_pos = QtWidgets.QLabel("")
        v.addWidget(self.lbl_pos)
        v.addStretch(1)

        # Init UI state
        self._on_step_tick_changed(self.step_item.slider.value())
        self._refresh_pos_label()

    # ---- mapping: tick -> log step ----
    def _tick_to_step(self, tick: int) -> float:
        t = float(tick) / float(self.TICK_MAX)
        log10_step = self.LOG10_MIN + t * (self.LOG10_MAX - self.LOG10_MIN)
        return 10.0 ** log10_step

    def _on_step_tick_changed(self, _tick: int) -> None:
        step = float(self.step_item.get_value())
        self.step_item.label.setText(f"{step:.3e} (log10={np.log10(step):.2f})")

    # ---- motion ----
    def _step_x_plus_once(self) -> None:
        step = float(self.step_item.get_value())
        self.pos[0] += step

        # Fast: move actor without rebuilding mesh
        self.actor.SetPosition(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        self._refresh_pos_label()

    def _refresh_pos_label(self) -> None:
        self.lbl_pos.setText(
            f"pos = ({self.pos[0]:.6f}, {self.pos[1]:.6f}, {self.pos[2]:.6f})"
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
