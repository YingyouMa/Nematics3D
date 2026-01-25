import numpy as np
import pyvista as pv

import sys
sys.path.insert(0, "D:/Document/GitHub/")
import Nematics3D

from qtpy import QtWidgets, QtCore


# exit: reset_camera back

# ============================================================
# Data
# ============================================================
index_max = 128
n = np.load("data/n_example_global.npy")[0:index_max, 0:index_max, 0:index_max]
S = np.load("data/S_example_global.npy")[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
pts = Q.lines[0]._raw_defect_indices[:2000]

# ============================================================
# Figure + static points
# ============================================================
figure = Nematics3D.PlotFigure()
p = figure.pl

Nematics3D.PlotSphere(
    pts,
    figure=figure,
    sides=12,
    radius=0.2,
    color=(0, 0, 0),
)

# ============================================================
# PlotTube (your system) + SmoothedLine
# ============================================================


smooth = Nematics3D.SmoothedLine(pts, window_length=10)

# Try to register as "line" if your PlotTube supports name/raw_name
tube = Nematics3D.PlotTube(
    smooth._entity,
    figure=figure,
    color=(1, 0, 0),
    radius=0.2,
    # name="line",
    # raw_name="line",
)




current_radius_mean = tube._calc_radius.mean()
current_radius = tube.opts.radius
current_camera_set = tube.opts.is_reset_camera
tube.opts.is_reset_camera = False
state = dict(window_length=smooth.opts.window_length)
state["radius scale"] = 1

# ============================================================
# Fast preview actor (pure PyVista) for dragging
# ============================================================
preview = {
    "is_active": False,
    "actor": None,          # pyvista.Actor
    "poly": None,           # pv.MultipleLines
    "r_avg": None,          # float
}

poly0 = pv.MultipleLines(tube.raw_coords)
r_avg = float(state["radius scale"]) * current_radius_mean
preview["r_avg"] = float(r_avg)

tube0 = poly0.tube(radius=float(preview["r_avg"]))
actor = p.add_mesh(
    tube0,
    name="__preview_tube__",
    color=(0, 0, 0),
    opacity=1.0,
    smooth_shading=True,
)
preview["actor"] = actor
preview["poly"] = poly0


def _ensure_preview_actor(coords: np.ndarray) -> None:
    preview["r_avg"] = float(state["radius scale"]) * current_radius_mean
    poly = pv.MultipleLines(coords)
    tube_mesh = poly.tube(radius=float(preview["r_avg"]))
    preview["actor"].mapper.SetInputData(tube_mesh)
    preview["actor"].mapper.Update()



def _begin_drag() -> None:

    if preview["is_active"]:
        return
    preview["is_active"] = True

    # hide PlotTube
    tube.opts.is_visible = False

    # ensure preview actor exists and visible
    _ensure_preview_actor(smooth._entity)
    preview['actor'].visibility = True

    # if hasattr(p, "render"):
    #     p.render()


def _end_drag() -> None:

    if not preview["is_active"]:
        return
    preview["is_active"] = False

    # hide preview
    preview['actor'].visibility = False

    _commit_plottube_full()

    # show PlotTube
    tube.opts.is_visible = True



def _commit_plottube_full() -> None:

    smooth.opts.window_length = int(state["window_length"])
    
    if callable(current_radius):
        radius_now = lambda x: float(state["radius scale"]) * current_radius(x)
    else:
        radius_now = float(state["radius scale"]) * current_radius
    
    tube.act_commit(
        coords=smooth._entity,
        radius=radius_now,
        )



def _preview_update_only() -> None:

    smooth.opts.window_length = int(state["window_length"])
    _ensure_preview_actor(smooth._entity)
    if hasattr(p, "render"):
        p.render()


# ============================================================
# Standalone control window (Qt)
#   - while sliding: preview updates only
#   - on slider release: one-time commit to PlotTube
# ============================================================
class ControlsWindow(QtWidgets.QWidget):
    def __init__(self, 
                 current_radius_mean,
                 max_window_length=200,
                 parent=None):
        
        super().__init__(parent)
        self._is_closing = False

        self.setWindowTitle("Line Controls (Preview + Commit)")
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

        hint = QtWidgets.QLabel(
            "radius_resize",
            group_r,
        )
        hint.setWordWrap(True)
        gl_r.addWidget(hint)

        layout.addWidget(group_r)

        # --- signals: continuous preview ---
        self.slider_w.valueChanged.connect(self._on_window_changed)
        self.slider_r.valueChanged.connect(self._on_radius_changed)

        # --- signals: press/release to begin/end drag ---
        self.slider_w.sliderPressed.connect(self._on_any_slider_pressed)
        self.slider_r.sliderPressed.connect(self._on_any_slider_pressed)

        self.slider_w.sliderReleased.connect(self._on_any_slider_released)
        self.slider_r.sliderReleased.connect(self._on_any_slider_released)

    def _on_any_slider_pressed(self):
        _begin_drag()

    def _on_any_slider_released(self):
        _end_drag()

    def _on_window_changed(self, v: int):
        v = int(v)
        self.lab_w_val.setText(str(v))
        state["window_length"] = v

        if preview["is_active"]:
            _preview_update_only()
        else:
            # If user clicks arrows without dragging, treat as immediate commit
            _commit_plottube_full()
            if hasattr(p, "render"):
                p.render()

    def _on_radius_changed(self, t: int):
        t = int(t)
        radius_scale = t / 100.0
        self.lab_r_val.setText(f"{radius_scale:.4g}")
        state["radius scale"] = radius_scale

        if preview["is_active"]:
            # Only update preview radius (fast): rebuild preview tube at new scalar radius
            _preview_update_only()
        else:
            _commit_plottube_full()
            
            
    def closeEvent(self, event):
        if not getattr(self, "_is_closing", False):
            self._is_closing = True
        event.accept()



# ============================================================
# Boot: hide preview, show PlotTube, render
# ============================================================
# make sure PlotTube is visible, preview hidden


# create & show independent control window (keep global ref)
controls_window = ControlsWindow(parent=None, current_radius_mean=current_radius_mean)
controls_window.resize(380, 200)
controls_window.show()
