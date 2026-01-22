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
Nematics3D.PlotTube(smooth._entity, figure=figure, color=(1,0,0), radius=0.2)

def rebuild():
    smooth.opts.window_length = int(state["window_length"])
    figure['line'].opts.is_reset_camera = False
    figure['line'].coords = smooth._entity

def cb_nspline(v):
    state["window_length"] = int(round(v))
    rebuild()

from qtpy import QtWidgets, QtCore  # pyvistaqt 通常依赖 qtpy，兼容 PyQt5/PySide6

# 你的 p = figure.pl
# 对 BackgroundPlotter：p.app_window 是 QMainWindow
# 对某些嵌入式场景：p 本身可能就是 QMainWindow
main_window = getattr(p, "app_window", None)
if main_window is None and isinstance(p, QtWidgets.QMainWindow):
    main_window = p

if main_window is None:
    raise RuntimeError(
        "Cannot find a QMainWindow to attach a dock widget. "
        "Expected a pyvistaqt.BackgroundPlotter with .app_window or a QMainWindow."
    )

# --- Dock 容器 ---
dock = QtWidgets.QDockWidget("Controls", main_window)
dock.setObjectName("dock_controls")
dock.setAllowedAreas(
    QtCore.Qt.LeftDockWidgetArea
    | QtCore.Qt.RightDockWidgetArea
    | QtCore.Qt.BottomDockWidgetArea
    | QtCore.Qt.TopDockWidgetArea
)

panel = QtWidgets.QWidget(dock)
layout = QtWidgets.QVBoxLayout(panel)
layout.setContentsMargins(8, 8, 8, 8)

# --- 标题 + 数值显示 ---
row = QtWidgets.QHBoxLayout()
label = QtWidgets.QLabel("window_length:", panel)
value_label = QtWidgets.QLabel(str(state["window_length"]), panel)
value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
row.addWidget(label)
row.addWidget(value_label)
layout.addLayout(row)

# --- QSlider ---
slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, panel)
slider.setMinimum(4)
slider.setMaximum(100)
slider.setValue(int(state["window_length"]))
slider.setSingleStep(1)
slider.setPageStep(5)
slider.setTracking(True)  # 拖动时实时发 valueChanged（类似 interaction_event="always"）
layout.addWidget(slider)

dock.setWidget(panel)
main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

def _on_slider_value_changed(v: int):
    v = int(v)
    value_label.setText(str(v))
    state["window_length"] = v
    rebuild()
    # 确保刷新渲染
    if hasattr(p, "render"):
        p.render()

slider.valueChanged.connect(_on_slider_value_changed)
