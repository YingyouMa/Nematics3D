import numpy as np
import pyvista as pv

import sys
# sys.path.insert(0, 'D:/Document/GitHub/3D-active-nematics/simulation')
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

# rng = np.random.default_rng(0)

# # ----------------------------
# # 1) 生成 noisy 3D ring 点集
# # ----------------------------
# n_pts = 200
# R = 5.0
# t = np.linspace(0, 2*np.pi, n_pts, endpoint=False)

# base = np.c_[R*np.cos(t), R*np.sin(t), 0.8*np.sin(3*t)]
# noise_sigma = 0.8
# pts = base + rng.normal(scale=noise_sigma, size=base.shape)
# pts_closed = np.vstack([pts, pts[0]])

index_max =  128
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=index_max >= 128, name="testQ")
pts = Q.lines[0]._raw_defect_indices
pts_closed = np.vstack([pts, pts[0]])

# ----------------------------
# 2) Plotter + actors
# ----------------------------
p = pv.Plotter()
p.add_axes()

# 原始 noisy 点（半透明点云，作为对照）
p.add_mesh(pv.PolyData(pts), render_points_as_spheres=True, point_size=6, opacity=0.25)

txt = p.add_text("", position="upper_left", font_size=12)

state = dict(
    n_spline=800,   # spline 输出采样点数（越多越平滑/越细）
    radius=0.10,    # tube 半径
)

# 初始化一条 spline 曲线
spline0 = pv.Spline(pts_closed, state["n_spline"])
tube0 = spline0.tube(radius=state["radius"], n_sides=24)

sm_actor = p.add_mesh(tube0, name="spline_tube", show_edges=False)

def rebuild():
    # 重新生成 spline -> tube
    spline = pv.Spline(pts_closed, int(state["n_spline"]))
    tube = spline.tube(radius=state["radius"], n_sides=24)

    # 更新 actor
    sm_actor.mapper.SetInputData(tube)
    sm_actor.mapper.Update()

    txt.SetText(0, f"Spline reconstruction (real-time)\n"
                   f"n_spline={int(state['n_spline'])}\n"
                   )
    p.render()

def cb_nspline(v):
    print("cb_nspline:", v)
    state["n_spline"] = int(round(v))
    rebuild()

# ----------------------------
# 3) Slider：拖动实时更新
# ----------------------------
p.add_slider_widget(
    cb_nspline,
    rng=[5000, 9000],
    value=state["n_spline"],
    title="n_spline (larger => smoother curve)",
    pointa=(0.02, 0.05),
    pointb=(0.55, 0.05),
    interaction_event="always",  # 实时触发 :contentReference[oaicite:3]{index=3}
)

rebuild()
p.show()
