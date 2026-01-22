import numpy as np
import pyvista as pv

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D


index_max =  128
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
pts = Q.lines[0]._raw_defect_indices[:200]

# ----------------------------
# 2) Plotter + actors
# ----------------------------
p = pv.Plotter()
p.add_axes()

p.add_mesh(pv.PolyData(pts), render_points_as_spheres=True, point_size=6, opacity=0.25)

txt = p.add_text("", position=(5, 1000), font_size=12)

state = dict(window_length=5)

# 初始化一条 spline 曲线
smooth = Nematics3D.SmoothedLine(pts, window_length=5)
poly = pv.MultipleLines(smooth._entity)
tube0 = poly.tube()
sm_actor = p.add_mesh(tube0, name="spline_tube", show_edges=False)

def rebuild():
    smooth.opts.window_length = state["window_length"]
    poly = pv.MultipleLines(smooth._entity)
    tube = poly.tube()

    sm_actor.mapper.SetInputData(tube)
    sm_actor.mapper.Update()
    
    # txt.input = (f"Spline reconstruction (real-time)\n"
    #                f"window_length={int(state['window_length'])}\n"
    #                )

    txt.SetInput(f"Spline reconstruction (real-time)\n"
                   f"window_length={int(state['window_length'])}\n"
                   )
    p.render()

def cb_nspline(v):
    print("cb_nspline:", v)
    state["window_length"] = int(round(v))
    rebuild()

# ----------------------------
# 3) Slider：拖动实时更新
# ----------------------------
p.add_slider_widget(
    cb_nspline,
    rng=[5, 30],
    value=state["window_length"],
    title="window_length (larger => smoother curve)",
    pointa=(0.22, 0.75),
    pointb=(0.22, 0.15),
    interaction_event="always",  # 实时触发 :contentReference[oaicite:3]{index=3}
)

rebuild()
p.show(interactive_update=True)
