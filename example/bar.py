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
pts = Q.lines[0]._raw_defect_indices[:1000]
figure = Nematics3D.PlotFigure()

Nematics3D.PlotSphere(pts, figure=figure, sides=12, radius=0.2, color=(0,0,0))

p = figure.pl
# p = pv.Plotter()
state = dict(window_length=10)


smooth = Nematics3D.SmoothedLine(pts, window_length=20)
# smooth = Nematics3D.SmoothedLine(pts, opts=)
Nematics3D.PlotTube(smooth._entity, figure=figure, color=(1,0,0), radius=0.2)

def rebuild():
    smooth.opts.window_length = int(state["window_length"])
    figure['line'].opts.is_reset_camera = False
    figure['line'].coords = smooth._entity

def cb_nspline(v):
    state["window_length"] = int(round(v))
    rebuild()

# ----------------------------
# 3) Slider：拖动实时更新
# ----------------------------
p.add_slider_widget(
    cb_nspline,
    rng=[4, 100],
    value=state['window_length'],
    title="window_length",
    pointa=(0.52, 0.15),
    pointb=(0.95, 0.15),
    interaction_event="always",  # 实时触发 :contentReference[oaicite:3]{index=3}
)

# p.show(in)
