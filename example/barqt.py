import numpy as np
import pyvista as pv

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
pts = Q.lines[0]._raw_defect_indices[:200]

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
state = dict(window_length=10)

smooth = Nematics3D.SmoothedLine(pts, window_length=int(state["window_length"]))

# Try to register as "line" if your PlotTube supports name/raw_name
tube = Nematics3D.PlotTube(
    smooth._entity,
    figure=figure,
    color=(1, 0, 0),
    radius=0.2,
    # name="line",
    # raw_name="line",
)

LINE_KEY = "line"


# --- base radius for scaling ---
try:
    r0 = float(figure[LINE_KEY].opts.radius)
except Exception:
    try:
        r0 = float(tube.opts.radius)
    except Exception:
        r0 = 0.2

state["radius"] = r0

# ============================================================
# Fast preview actor (pure PyVista) for dragging
# ============================================================
preview = {
    "is_active": False,
    "actor": None,          # pyvista.Actor
    "poly": None,           # pv.MultipleLines
    "r_avg": None,          # float
}

def _radius_average(radius_value) -> float:
    """
    Compute an average scalar radius from PlotTube's radius setting.

    Parameters
    ----------
    radius_value : Any
        Could be a float, a per-point array-like, or other supported forms.

    Returns
    -------
    r_avg : float
        A reasonable scalar "average" radius for preview usage.
    """
    if radius_value is None:
        return float(state["radius"])

    if isinstance(radius_value, (int, float, np.floating)):
        return float(radius_value)

    try:
        arr = np.asarray(radius_value, dtype=float)
        if arr.shape == ():
            return float(arr)
        if arr.size == 0:
            return float(state["radius"])
        return float(np.mean(arr))
    except Exception:
        # Fallback if it's callable or unsupported in preview
        return float(state["radius"])


def _ensure_preview_actor(coords: np.ndarray) -> None:
    """
    Ensure a black preview tube actor exists, and update its dataset to coords.

    Notes
    -----
    - The preview uses pv.MultipleLines + tube() then swaps mapper input data.
    - Actor is created once; subsequent updates are mapper input swaps.
    """
    if preview["actor"] is None:
        poly0 = pv.MultipleLines(coords)
        # Use average radius of PlotTube's current radius (best-effort)
        try:
            r_avg = _radius_average(figure[LINE_KEY].opts.radius)
        except Exception:
            r_avg = float(state["radius"])
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
    else:
        # update preview radius if user changes radius slider while dragging
        # (keep using r_avg, which we recompute from desired scalar state)
        preview["r_avg"] = float(state["radius"])

    # Update geometry by swapping mapper input
    poly = pv.MultipleLines(coords)
    tube_mesh = poly.tube(radius=float(preview["r_avg"]))

    preview["actor"].mapper.SetInputData(tube_mesh)
    preview["actor"].mapper.Update()


def _set_plottube_visible(is_visible: bool) -> None:
    """
    Toggle PlotTube visibility via its opts.

    Notes
    -----
    This should be fast (LEVEL_ACTOR), but your commit chain might still do work.
    If this is still slow, you should directly set actor.visibility.
    """
    obj = figure[LINE_KEY]
    try:
        obj.opts.is_reset_camera = False
    except Exception:
        pass

    # Preferred: go through your opts system
    try:
        obj.opts.is_visible = bool(is_visible)
        return
    except Exception:
        pass

    # Fallback: direct VTK actor toggle
    try:
        obj._entity.visibility = bool(is_visible)
    except Exception:
        pass


def _set_preview_visible(is_visible: bool) -> None:
    if preview["actor"] is None:
        return
    try:
        preview["actor"].visibility = bool(is_visible)
    except Exception:
        # If it isn't a VTK actor wrapper for some reason
        pass


def _begin_drag() -> None:
    """
    Enter preview mode: hide PlotTube, show preview actor.
    """
    if preview["is_active"]:
        return
    preview["is_active"] = True

    # hide PlotTube
    _set_plottube_visible(False)

    # ensure preview actor exists and visible
    _ensure_preview_actor(smooth._entity)
    _set_preview_visible(True)

    if hasattr(p, "render"):
        p.render()


def _end_drag(commit_to_plottube: bool = True) -> None:
    """
    Exit preview mode: hide preview actor, show PlotTube, optionally commit updates.
    """
    if not preview["is_active"]:
        return
    preview["is_active"] = False

    # hide preview
    _set_preview_visible(False)

    if commit_to_plottube:
        # Now do the expensive commit once
        _commit_plottube_full()

    # show PlotTube
    _set_plottube_visible(True)

    if hasattr(p, "render"):
        p.render()


def _commit_plottube_full() -> None:
    """
    Do the expensive update once (on slider release):
      - update smoothing window_length
      - update PlotTube coords
      - update PlotTube radius
    """
    smooth.opts.window_length = int(state["window_length"])

    obj = figure[LINE_KEY]
    try:
        obj.opts.is_reset_camera = False
    except Exception:
        pass

    # Update coords (likely triggers remesh in your system)
    obj.coords = smooth._entity

    # Update radius (likely triggers remesh in your system)
    try:
        obj.opts.radius = float(state["radius"])
    except Exception:
        # fallback direct
        try:
            obj.act_commit(radius=float(state["radius"]))
        except Exception:
            pass


def _preview_update_only() -> None:
    """
    Fast path during dragging:
      - recompute smooth coords
      - update preview mapper input
    """
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
    def __init__(self, parent=None):
        super().__init__(parent)

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
        self.slider_w.setMaximum(200)
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
        self.lab_r_key = QtWidgets.QLabel("radius:", group_r)
        self.lab_r_val = QtWidgets.QLabel(f"{state['radius']:.4g}", group_r)
        self.lab_r_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row_r.addWidget(self.lab_r_key)
        row_r.addWidget(self.lab_r_val)
        gl_r.addLayout(row_r)

        self._r0 = float(r0)
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
            f"range: {self._r0/5:.4g}  ...  {self._r0*5:.4g}   (relative to initial radius)",
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
        _end_drag(commit_to_plottube=True)

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
        radius = self._r0 * (t / 100.0)
        self.lab_r_val.setText(f"{radius:.4g}")
        state["radius"] = float(radius)

        if preview["is_active"]:
            # Only update preview radius (fast): rebuild preview tube at new scalar radius
            _preview_update_only()
        else:
            _commit_plottube_full()
            if hasattr(p, "render"):
                p.render()


# ============================================================
# Boot: hide preview, show PlotTube, render
# ============================================================
# make sure PlotTube is visible, preview hidden
_set_plottube_visible(True)
_set_preview_visible(False)
if hasattr(p, "render"):
    p.render()

# create & show independent control window (keep global ref)
controls_window = ControlsWindow(parent=None)
controls_window.resize(380, 200)
controls_window.show()
