import sys
from pathlib import Path
import types

import numpy as np
import pyvista as pv

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.smoothed_line import SmoothedLine


def build_noisy_line(num_points=120, noise_scale=0.18, seed=7):
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 4.0 * np.pi, num_points)
    x = t
    y = 1.4 * np.sin(t) + 0.25 * np.sin(3.0 * t)
    z = 0.6 * np.cos(0.5 * t) + 0.15 * np.sin(2.0 * t)

    pts = np.column_stack((x, y, z))
    pts += noise_scale * rng.normal(size=pts.shape)
    return pts


pts = build_noisy_line()

p = pv.Plotter()
p.add_axes()

p.add_mesh(
    pv.PolyData(pts),
    render_points_as_spheres=True,
    point_size=6,
    opacity=0.25,
    color="tomato",
)

txt = p.add_text("", position=(3, 1000), font_size=12)

state = {"window_length": 5}

smooth = SmoothedLine(pts, window_length=state["window_length"])
poly = pv.MultipleLines(smooth.result)
tube0 = poly.tube(radius=0.08)
sm_actor = p.add_mesh(
    tube0,
    name="spline_tube",
    show_edges=False,
    color="deepskyblue",
)


def rebuild():
    smooth.opts.window_length = state["window_length"]
    poly_local = pv.MultipleLines(smooth.result)
    tube = poly_local.tube(radius=0.08)

    sm_actor.mapper.SetInputData(tube)
    sm_actor.mapper.Update()

    txt.SetInput(
        "Spline reconstruction (real-time)\n"
        f"window_length={int(state['window_length'])}\n"
        f"calc_is_smoothed={smooth.calc_is_smoothed}\n"
        f"calc_status={smooth.calc_status}\n"
    )
    p.render()


def cb_nspline(v):
    state["window_length"] = int(round(v))
    rebuild()


p.add_slider_widget(
    cb_nspline,
    rng=[5, 31],
    value=state["window_length"],
    title="window_length (larger => smoother curve)",
    pointa=(0.22, 0.75),
    pointb=(0.22, 0.15),
    interaction_event="always",
)

rebuild()
p.show(interactive_update=True)
