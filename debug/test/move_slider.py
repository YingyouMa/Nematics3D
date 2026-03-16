from qtpy import QtWidgets, QtCore
import numpy as np
import pyvistaqt as pvqt
from dataclasses import dataclass
from typing import Callable, MutableMapping, Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import Nematics3D

from Nematics3D.classes.visual.qt.panel_base import *

# class InteractMoveSphere(PanelBase):
#     """
#     Panel to move a single sphere center in +/-x, +/-y, +/-z directions
#     with a log-step slider controlling the step size.
#     """

#     def __init__(self, host):
#         super().__init__(host, title="Move Sphere")

#     def build_ui(self):
#         # ----------------------------
#         # initial state
#         # ----------------------------
#         self.state = {
#             "step": 1.0,
#             "center": np.array([0.0, 0.0, 0.0], dtype=float),
#         }
    
#         # ----------------------------
#         # Single group + single layout
#         # ----------------------------
#         group = QtWidgets.QGroupBox("Controls", self)
#         gl = QtWidgets.QVBoxLayout(group)
#         self.layout.addWidget(group)
    
#         # ----------------------------
#         # Step slider (log scale)
#         # ----------------------------
#         step_map = LogTickMapper(value_min=0.01, value_max=100.0, tick_min=0, tick_max=1000, base=10.0)
#         self.sliders["step"] = make_labeled_slider_row(
#             parent=group,
#             layout=gl,
#             name="step",
#             state_key="step",
#             tick_min=step_map.tick_min,
#             tick_max=step_map.tick_max,
#             tick_init=step_map.value_to_tick(self.state["step"]),
#             tick_to_value=step_map.tick_to_value,
#             value_fmt="{:.2f}",
#         )
#         self.sliders["step"].slider.valueChanged.connect(self.on_changed)
    
#         # ----------------------------
#         # Center display + move buttons (press/hold)
#         # ----------------------------
#         grid_widget = QtWidgets.QWidget(group)
#         grid = QtWidgets.QGridLayout(grid_widget)
#         grid.setContentsMargins(0, 0, 0, 0)
#         grid.setHorizontalSpacing(8)
#         grid.setVerticalSpacing(6)
#         gl.addWidget(grid_widget)
    
#         self.lab_center = QtWidgets.QLabel("", group)
#         self.lab_center.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
#         grid.addWidget(QtWidgets.QLabel("Center:", group), 0, 0, 1, 1)
#         grid.addWidget(self.lab_center, 0, 1, 1, 2)
    
#         def move(dx: float, dy: float, dz: float):
#             step = float(self.state["step"])
#             c = np.array(self.state["center"], dtype=float)
#             c += step * np.array([dx, dy, dz], dtype=float)
#             self.state["center"] = c
#             self.commit()
#             self._update_center_label()
    
#         def _cell_button(row: int, col: int, text: str, cb):
#             cell = QtWidgets.QWidget(group)
#             v = QtWidgets.QVBoxLayout(cell)
#             v.setContentsMargins(0, 0, 0, 0)
#             v.setSpacing(0)
#             item = make_press_hold_button(parent=cell, layout=v, text=text, callback=cb)
#             grid.addWidget(cell, row, col)
#             return item
    
#         # 6 buttons: -/+X, -/+Y, -/+Z
#         self.btn_x_neg = _cell_button(1, 0, "-X", lambda: move(-1, 0, 0))
#         self.btn_x_pos = _cell_button(1, 1, "+X", lambda: move(+1, 0, 0))
#         self.btn_y_neg = _cell_button(2, 0, "-Y", lambda: move(0, -1, 0))
#         self.btn_y_pos = _cell_button(2, 1, "+Y", lambda: move(0, +1, 0))
#         self.btn_z_neg = _cell_button(3, 0, "-Z", lambda: move(0, 0, -1))
#         self.btn_z_pos = _cell_button(3, 1, "+Z", lambda: move(0, 0, +1))
    
#         # Initialize label and commit once
#         self.on_changed(0, is_commit=False)
#         self._update_center_label()
#         self.commit()

#     def _update_center_label(self):
#         c = np.array(self.state["center"], dtype=float)
#         self.lab_center.setText(f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")

#     def commit(self):
#         c = np.array(self.state["center"], dtype=float)
#         # Host is the sphere. Update via act_commit with coords=[center].
#         self.host.act_commit(coords=[c])


# sphere = Nematics3D.PlotSphere(coords=[0,0,0])

# # Make the sphere the PanelBase host
# panel = InteractMoveSphere(sphere)
# panel.show()

@dataclass(slots=True)
class MovePointConsole:
    """
    A composite control console for moving a 3D point stored in `state[center_key]`.

    This console owns its own QGroupBox and layout, so panels can directly add `console.group`
    to any layout.

    Parameters
    ----------
    state : MutableMapping[str, Any]
        The panel state dict. The console reads/writes into it.
    center_key : str
        Key in `state` storing the point center. Must be array-like shape (3,).
    step_key : str
        Key in `state` storing the step size (float).
    parent : QtWidgets.QWidget
        Qt parent for widgets.
    title : str
        GroupBox title.
    step_min, step_max : float
        Step range in linear space (must be > 0). Slider uses log mapping.
    step_tick_max : int
        Slider tick resolution (0..step_tick_max).
    step_fmt : str
        Label format for step slider.
    center_fmt : str
        Label format for center display, e.g. "{:.2f}".
    on_move : Callable[[np.ndarray], None] | None
        Called after center is updated (typically host.act_commit(coords=[center])).
    on_press : Callable[[Vect(3), np.ndarray], None] | None
        Called when a direction action fires the "first time".
    on_hold : Callable[[Vect(3), np.ndarray], None] | None
        Called on each repeat tick during a long press.
    on_release : Callable[[Vect(3), np.ndarray], None] | None
        Called once when the button is released (regardless of short/long press).
    long_press_ms, repeat_ms : int
        Timing parameters forwarded to press/hold behavior.
    """

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
            tick_min=step_map.tick_min,
            tick_max=step_map.tick_max,
            tick_init=step_map.value_to_tick(step0),
            tick_to_value=step_map.tick_to_value,
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

        grid.addWidget(QtWidgets.QLabel("Center:", self.group), 0, 0, 1, 1)
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

        # Direction callbacks: we need to distinguish first fire vs repeat fire.
        # With your current PressHoldButtonItem, we can infer:
        # - For a long press: first fire occurs at long-press begin; repeats happen afterwards.
        # - For a short press: fire occurs on release (once).
        #
        # We implement:
        # - cb_first_or_repeat: called by PressHoldButtonItem for both "short press single fire"
        #   and "long press begin + repeats".
        # We track per-button whether hold mode has started by inspecting item._hold_active.
        #
        def _move_once(dir_: Vect(3), *, item_ref: PressHoldButtonItem):
            # pull latest step from state (slider may have changed)
            step = float(self.state[self.step_key])
            c = np.array(self.state[self.center_key], dtype=float)
            d = np.array(dir_, dtype=float)
            c_new = c + step * d
            self.state[self.center_key] = c_new

            # callback classification
            if bool(item_ref._hold_active):
                # during long hold repeats and also the first long-press fire
                if self.on_hold is not None:
                    self.on_hold(dir_, c_new)
            else:
                # short press single fire (happens on release inside PressHoldButtonItem)
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
            1, 0, "-X",
            cb=lambda: _move_once((-1, 0, 0), item_ref=self.btn_x_neg),  # placeholder, fixed below
            release_cb=lambda: _on_release((-1, 0, 0)),
        )
        self.btn_x_pos = _cell_button(
            1, 1, "+X",
            cb=lambda: _move_once((+1, 0, 0), item_ref=self.btn_x_pos),
            release_cb=lambda: _on_release((+1, 0, 0)),
        )
        self.btn_y_neg = _cell_button(
            2, 0, "-Y",
            cb=lambda: _move_once((0, -1, 0), item_ref=self.btn_y_neg),
            release_cb=lambda: _on_release((0, -1, 0)),
        )
        self.btn_y_pos = _cell_button(
            2, 1, "+Y",
            cb=lambda: _move_once((0, +1, 0), item_ref=self.btn_y_pos),
            release_cb=lambda: _on_release((0, +1, 0)),
        )
        self.btn_z_neg = _cell_button(
            3, 0, "-Z",
            cb=lambda: _move_once((0, 0, -1), item_ref=self.btn_z_neg),
            release_cb=lambda: _on_release((0, 0, -1)),
        )
        self.btn_z_pos = _cell_button(
            3, 1, "+Z",
            cb=lambda: _move_once((0, 0, +1), item_ref=self.btn_z_pos),
            release_cb=lambda: _on_release((0, 0, +1)),
        )

    def _update_center_label(self) -> None:
        assert self.lab_center is not None
        c = np.array(self.state[self.center_key], dtype=float)
        fmt = self.center_fmt
        self.lab_center.setText(f"({fmt.format(c[0])}, {fmt.format(c[1])}, {fmt.format(c[2])})")

    def set_enabled(self, enabled: bool) -> None:
        en = bool(enabled)
        assert self.group is not None
        self.group.setEnabled(en)
        
class InteractMoveSphere(PanelBase):
    """
    Move a single sphere center in +/-X, +/-Y, +/-Z using a MovePointConsole.

    Requirements satisfied:
    - sphere is the PanelBase host
    - center stored in self.state['center']
    - PlotSphere created with coords=[center]
    - move updates via sphere.act_commit(coords=[center])
    """

    def __init__(self, host):
        super().__init__(host, title="Move Sphere")

    def build_ui(self):
        # ----------------------------
        # initial state
        # ----------------------------
        center0 = np.array([0.0, 0.0, 0.0], dtype=float)
        self.state = {
            "center": center0,
            "step": 1.0,
        }

        # ----------------------------
        # console (owns its own group/layout)
        # ----------------------------
        def _commit_center(c_new: np.ndarray) -> None:
            # host is the sphere
            self.host.act_commit(coords=[np.array(c_new, dtype=float)])

        self.console = MovePointConsole(
            parent=self,
            state=self.state,
            center_key="center",
            step_key="step",
            title="Move Sphere",
            step_min=0.01,
            step_max=100.0,
            step_tick_max=1000,
            step_fmt="{:.2f}",
            center_fmt="{:.2f}",
            on_move=_commit_center,
            on_press=None,
            on_hold=None,
            on_release=None,
            long_press_ms=450,
            repeat_ms=80,
        )

        # User can directly add the console group into the panel layout
        self.layout.addWidget(self.console.group)

        # ----------------------------
        # integrate the console slider into PanelBase.on_changed
        # ----------------------------
        # This ensures PanelBase.on_changed() syncs 'step' into state.
        self.sliders["step"] = self.console.slider_step
        self.sliders["step"].slider.valueChanged.connect(self.on_changed)

        # ----------------------------
        # init
        # ----------------------------
        self.on_changed(0, is_commit=False)
        self.commit()

    def commit(self):
        # Panel-level commit keeps sphere in sync with state['center'].
        c = np.array(self.state["center"], dtype=float)
        self.host.act_commit(coords=[c])
        
sphere = Nematics3D.PlotSphere(coords=[0,0,0])

# Make the sphere the PanelBase host
panel = InteractMoveSphere(sphere)
panel.show()


    
