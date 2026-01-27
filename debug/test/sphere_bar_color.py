import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

import numpy as np
import pyvista as pv
from qtpy import QtWidgets, QtCore
import datetime


# --- 准备测试数据和函数 ---
def get_path(offset_y=0):
    z = np.linspace(0, 10, 50)
    line1 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    z = np.linspace(15, 20, 25)
    line2 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    return np.concatenate([line1, line2])


figure = Nematics3D.PlotFigure()

spheres = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=0),
    name="solid_blue",
    color=(0, 0, 1),
    radius=0.3,
    sides=12,
)


class SphereColorControlsWindow(QtWidgets.QWidget):
    def __init__(self, glyph):
        self.glyph = glyph
        self.str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        self.glyph.act_save_opts(self.str_now)

        # 只管 color
        c0 = getattr(self.glyph.opts, "color", (0.0, 0.0, 0.0))
        c0 = tuple(float(x) for x in c0)

        self.state = dict(
            color=c0,
            use_control_color=True, 
        )
        
        super().__init__(None)
        
        # --- helpers ---
        def _make_rgb_row(parent, key, init_val):
            row = QtWidgets.QHBoxLayout()
            lab_key = QtWidgets.QLabel(f"{key}:", parent)
            lab_val = QtWidgets.QLabel(f"{init_val:.4g}", parent)
            lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(lab_key)
            row.addWidget(lab_val)
            gl_color.addLayout(row)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent)
            slider.setMinimum(0)
            slider.setMaximum(1000)       # 0..1 映射到 0..1000
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.setTracking(True)
            slider.setValue(int(round(init_val * 1000)))
            gl_color.addWidget(slider)

            return lab_val, slider

        self.setWindowTitle("Sphere Color Controls (Realtime)")
        self.setObjectName("window_controls_color")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        group_color = QtWidgets.QGroupBox("Color (RGB 0..1)", self)
        gl_color = QtWidgets.QVBoxLayout(group_color)
        
        self.chk_use_color = QtWidgets.QCheckBox("Use controlled color", group_color)
        self.chk_use_color.setChecked(self.state["use_control_color"])
        gl_color.addWidget(self.chk_use_color)
        
        self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)

        self.lab_r_val, self.slider_r = _make_rgb_row(group_color, "R", self.state["color"][0])
        self.lab_g_val, self.slider_g = _make_rgb_row(group_color, "G", self.state["color"][1])
        self.lab_b_val, self.slider_b = _make_rgb_row(group_color, "B", self.state["color"][2])

        hint = QtWidgets.QLabel("Drag to update sphere.opts.color in realtime.", group_color)
        hint.setWordWrap(True)
        gl_color.addWidget(hint)

        layout.addWidget(group_color)

        # --- signals ---
        self.slider_r.valueChanged.connect(self._on_changed)
        self.slider_g.valueChanged.connect(self._on_changed)
        self.slider_b.valueChanged.connect(self._on_changed)

        for s in (self.slider_r, self.slider_g, self.slider_b):
            s.sliderPressed.connect(self.glyph._helper_clear_silhouette)
            s.sliderReleased.connect(self.glyph._helper_add_silhouette)

        self.glyph.opts._internal_sync_func["color"][self.str_now] = self._sync_from_glyph
        
    def _on_toggle_use_color(self, _state: int):
        self.state["use_control_color"] = self.chk_use_color.isChecked()
        self.commit()
        

    def _sync_from_glyph(self):
        c = tuple(float(x) for x in self.glyph.opts.color)
        # 防止递归触发：简单起见先 setValue，再统一 _on_changed(commit=False)
        self.slider_r.setValue(int(round(c[0] * 1000)))
        self.slider_g.setValue(int(round(c[1] * 1000)))
        self.slider_b.setValue(int(round(c[2] * 1000)))
        self._on_changed(0, is_commit=False)

    def _on_changed(self, _v: int, is_commit: bool = True):
        r = int(self.slider_r.value()) / 1000.0
        g = int(self.slider_g.value()) / 1000.0
        b = int(self.slider_b.value()) / 1000.0

        self.lab_r_val.setText(f"{r:.4g}")
        self.lab_g_val.setText(f"{g:.4g}")
        self.lab_b_val.setText(f"{b:.4g}")

        self.state["color"] = (r, g, b)

        if is_commit:
            self.commit()

    def commit(self):
        if self.state["use_control_color"]:
            color_now = self.state["color"]
        else:
            color_now = self.glyph._internal_opts_backup[self.str_now]["color"]
    
        self.glyph.act_commit(
            color=color_now,
            is_silhouette=False,
        )


controls_window = SphereColorControlsWindow(spheres)
controls_window.resize(420, 180)
controls_window.show()
