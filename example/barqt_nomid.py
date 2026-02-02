import numpy as np
import pyvista as pv
import datetime

import sys
sys.path.insert(0, "D:/Document/GitHub/")
import Nematics3D

from qtpy import QtWidgets, QtCore


# ============================================================
# Data
# ============================================================
index_max = 128
n = np.load("data/n_example_global.npy")[0:index_max, 0:index_max, 0:index_max]
S = np.load("data/S_example_global.npy")[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
pts = Q.lines[1]._raw_defect_indices[:]

# ============================================================
# Figure + static points
# ============================================================
figure = Nematics3D.PlotFigure()
p = figure.pl


# Nematics3D.PlotSphere(
#     pts,
#     figure=figure,
#     sides=12,
#     radius=0.2,
#     color=(0, 0, 0),
# )

# ============================================================
# PlotTube (your system) + SmoothedLine
# ============================================================
smooth = Nematics3D.SmoothedLine(pts, window_length=10)

radius_set = 0.5 # np.linspace(0.2, 1.5, 9000)


tube = Nematics3D.PlotTube(
    smooth.result,
    figure=figure,
    #paint_by='scalars',
    #color=lambda x: np.abs(x) / np.linalg.norm(x, axis=-1, keepdims=True),
    scalars=lambda x: np.max(x, axis=-1),
    radius=radius_set,
)



current_radius = tube.opts.radius
tube._helper_clear_silhouette()

state = dict(window_length=smooth.opts.window_length)
state["radius scale"] = 1.0


def _commit_plottube_full() -> None:
    """
    Recompute smoothed coords and commit to PlotTube immediately.
    """
    # update smoothing
    smooth.opts.window_length = int(state["window_length"])
    # tube.coords = smooth._entity

    # update radius (supports callable radius field)
    if callable(current_radius):
        radius_now = lambda x: float(state["radius scale"]) * current_radius(x)
    else:
        radius_now = float(state["radius scale"]) * current_radius

    tube.act_commit(
        coords=smooth.result,
        radius=radius_now,
        is_silhouette=False
    )



# ============================================================
# Standalone control window (Qt)
#   - while sliding: realtime update PlotTube
# ============================================================
class ControlsWindow(QtWidgets.QWidget):
    def __init__(
        self,
        max_window_length=200,
        parent=None,
    ):
        super().__init__(parent)
        self._is_closing = False
        self._is_dragging = False

        self.setWindowTitle("Line Controls (Realtime)")
        self.setObjectName("window_controls")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- window_length group ---
        group_w = QtWidgets.QGroupBox("Smoothing", self)
        gl_w = QtWidgets.QVBoxLayout(group_w)

        row_w = QtWidgets.QHBoxLayout()
        self.lab_w_key = QtWidgets.QLabel("window_length:", group_w)
        self.lab_w_val = QtWidgets.QLabel(str(int(state["window_length"])), group_w)
        self.lab_w_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row_w.addWidget(self.lab_w_key)
        row_w.addWidget(self.lab_w_val)
        gl_w.addLayout(row_w)

        self.slider_w = QtWidgets.QSlider(QtCore.Qt.Horizontal, group_w)
        self.slider_w.setMinimum(4)
        self.slider_w.setMaximum(max_window_length)
        self.slider_w.setValue(int(state["window_length"]))
        self.slider_w.setSingleStep(1)
        self.slider_w.setPageStep(5)
        self.slider_w.setTracking(True)  # valueChanged fires continuously
        gl_w.addWidget(self.slider_w)

        layout.addWidget(group_w)

        # --- radius group ---
        group_r = QtWidgets.QGroupBox("Tube", self)
        gl_r = QtWidgets.QVBoxLayout(group_r)

        row_r = QtWidgets.QHBoxLayout()
        self.lab_r_key = QtWidgets.QLabel("radius scale:", group_r)
        self.lab_r_val = QtWidgets.QLabel(f"{state['radius scale']:.4g}", group_r)
        self.lab_r_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row_r.addWidget(self.lab_r_key)
        row_r.addWidget(self.lab_r_val)
        gl_r.addLayout(row_r)

        self._t_min = 20   # 0.2x
        self._t_max = 500  # 5x

        self.slider_r = QtWidgets.QSlider(QtCore.Qt.Horizontal, group_r)
        self.slider_r.setMinimum(self._t_min)
        self.slider_r.setMaximum(self._t_max)
        self.slider_r.setSingleStep(1)
        self.slider_r.setPageStep(10)
        self.slider_r.setTracking(True)

        # default 1x
        self.slider_r.setValue(100)
        gl_r.addWidget(self.slider_r)

        hint = QtWidgets.QLabel("radius_resize", group_r)
        hint.setWordWrap(True)
        gl_r.addWidget(hint)

        layout.addWidget(group_r)

        # --- signals: realtime update ---
        self.slider_w.valueChanged.connect(self._on_window_changed)
        self.slider_r.valueChanged.connect(self._on_radius_changed)

        # --- optional: track drag state (not required, but kept clean) ---
        self.slider_w.sliderPressed.connect(self._on_any_slider_pressed)
        self.slider_r.sliderPressed.connect(self._on_any_slider_pressed)
        self.slider_w.sliderReleased.connect(self._on_any_slider_released)
        self.slider_r.sliderReleased.connect(self._on_any_slider_released)
        
        str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        smooth.opts._internal_sync_func['window_length'][str_now] = lambda: self._on_window_changed(smooth.opts.window_length, is_commit=False)
        

    def _on_any_slider_pressed(self):
        self._is_dragging = True

    def _on_any_slider_released(self):
        self._is_dragging = False
        # ensure final commit once more (in case last valueChanged missed)
        _commit_plottube_full()

    def _on_window_changed(self, v: int, is_commit: bool = True):
        v = int(v)
        self.lab_w_val.setText(str(v))
        state["window_length"] = v
        if is_commit:
            _commit_plottube_full()

    def _on_radius_changed(self, t: int):
        t = int(t)
        radius_scale = t / 100.0
        self.lab_r_val.setText(f"{radius_scale:.4g}")
        state["radius scale"] = radius_scale
        _commit_plottube_full()

    def closeEvent(self, event):
        if not getattr(self, "_is_closing", False):
            self._is_closing = True
        event.accept()


# ============================================================
# Boot
# ============================================================
_commit_plottube_full()

controls_window = ControlsWindow(parent=None)
controls_window.resize(380, 200)
controls_window.show()
