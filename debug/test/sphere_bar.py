import numpy as np
import pyvista as pv
from qtpy import QtWidgets, QtCore, QtGui
import datetime

import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

from Nematics3D.classes.visual.qt.panel_base import PanelBase, make_labeled_slider_row, make_RGB_slider
from Nematics3D.datatypes import as_ColorRGB



# --- 准备测试数据和函数 ---
def get_path(offset_y=0):
    z = np.linspace(0, 10, 50)
    line1 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    z = np.linspace(15, 20, 25)
    line2 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    return np.concatenate([line1, line2])

def radius_wave(coords):
    return 0.1 + 0.2 * np.abs(np.sin(coords[:, 2]))

def color_func(coords):
    z_norm = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    return np.column_stack((z_norm, np.zeros_like(z_norm), 1 - z_norm))

def opacity_func(coords):
    opacity = np.abs(np.sin(coords[:, 2]))
    return opacity

figure = Nematics3D.PlotFigure()

line_index = np.ones(75)
line_index[-25:] = 2

spheres = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=0),  
    name="solid_blue",
    color=(0,0,1), # 蓝色
    radius=0.3,
    sides=12,
)

class SphereControlsWindow(PanelBase):
    def __init__(self, glyph):
        super().__init__(glyph, title="Sphere Controls (Realtime)")

    def build_ui(self):
        # ----------------------------
        # initial state
        # ----------------------------
        self.state = {
            "radius_rescale":           1.0,
            "sides":                    int(self.glyph.opts.sides),
            "is_use_control_color":     False,
            "is_use_control_opacity":   False,
            "color":                    (1,1,1),
            "opacity":                  1,
            }

        # ----------------------------
        # Geometry group
        # ----------------------------
        group_geometry = QtWidgets.QGroupBox("Geometry", self)
        gl_geometry = QtWidgets.QVBoxLayout(group_geometry)
        self.layout.addWidget(group_geometry)

        
        self.sliders["radius_rescale"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="radius_rescale",
            tick_min=20,
            tick_max=500,
            tick_init=100,     
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )

    
        self.sliders["sides"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="sides",
            tick_min=4,
            tick_max=40,
            tick_init=int(self.glyph.opts.sides),   
            tick_to_value=lambda t: float(int(t)),
            value_fmt="{:.0f}",
        )
        
        # ----------------------------
        # RGB group
        # ----------------------------
        
        group_RGB = QtWidgets.QGroupBox("Color (RGB 0..1)", self)
        gl_RGB = QtWidgets.QVBoxLayout(group_RGB)
        self.layout.addWidget(group_RGB)
        
        try:
            init_rgb = as_ColorRGB(self.glyph.opts.color) 
        except:
            init_rgb = (0.5, 0.5, 0.5)
        make_RGB_slider(
            parent=group_RGB,
            layout=gl_RGB,
            sliders=self.sliders,
            prefix="color",
            init_rgb=init_rgb
            )
        
        self.chk_use_color = QtWidgets.QCheckBox("Use controlled color", group_RGB)
        self.chk_use_color.setChecked(self.state["is_use_control_color"])
        gl_RGB.addWidget(self.chk_use_color)
        self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)
        self._apply_color_enabled()
        
        # ----------------------------
        # Opacity group
        # ----------------------------
        
        group_opacity = QtWidgets.QGroupBox("Opacity (0..1)", self)
        gl_opacity = QtWidgets.QVBoxLayout(group_opacity)
        self.layout.addWidget(group_opacity)
        
        self.sliders["opacity"] = make_labeled_slider_row(
            parent=group_opacity,
            layout=gl_opacity,
            name="opacity",
            tick_min=0,
            tick_max=100,
            tick_init=100,   
            tick_to_value=lambda t: float(t/100.0),
            value_fmt="{:.2f}",
        )
        
        self.chk_use_opacity = QtWidgets.QCheckBox("Use controlled opacity", group_opacity)
        self.chk_use_opacity.setChecked(self.state["is_use_control_opacity"])
        gl_opacity.addWidget(self.chk_use_opacity)
        self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
        self._apply_opacity_enabled()


        for item in self.sliders.values():
            item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self.glyph._helper_clear_silhouette)
            item.slider.sliderReleased.connect(self.glyph._helper_add_silhouette)

        self.on_changed(0, is_commit=False)
        
        self.glyph.opts._internal_sync_func["sides"][self.str_now] = lambda: self._sync_sides_from_glyph("sides", self.glyph.opts.sides)

    def on_changed(self, _v: int = 0, is_commit: bool = True):
        # ---- radius_rescale ----
        rr = float(self.sliders["radius_rescale"].get_value())
        self.sliders["radius_rescale"].label.setText(f"{rr:.4g}")
        self.state["radius_rescale"] = rr
    
        # ---- sides ----
        sides_f = float(self.sliders["sides"].get_value())
        sides = int(round(sides_f))
        self.sliders["sides"].label.setText(f"{sides:d}")
        self.state["sides"] = sides
    
        # ---- color (RGB) + checkbox ----
        self.state["is_use_control_color"] = bool(self.chk_use_color.isChecked())
    
        r = float(self.sliders["color_r"].get_value())
        g = float(self.sliders["color_g"].get_value())
        b = float(self.sliders["color_b"].get_value())
    
        self.sliders["color_r"].label.setText(f"{r:.4g}")
        self.sliders["color_g"].label.setText(f"{g:.4g}")
        self.sliders["color_b"].label.setText(f"{b:.4g}")
        
        self.state["color"] = (r, g, b)
        
        # ---- opacity + checkbox ----
        self.state["is_use_control_opacity"] = bool(self.chk_use_opacity.isChecked())
        opacity = float(self.sliders["opacity"].get_value())
        self.sliders["opacity"].label.setText(f"{opacity:4g}")
        self.state["opacity"] = opacity

        if is_commit:
            self.commit()

    def commit(self):
        # ---- radius ----
        current_radius = self.glyph._internal_opts_backup[self.str_now]["radius"]
        scale = float(self.state["radius_rescale"])
        if callable(current_radius):
            radius_now = lambda x: scale * current_radius(x)
        else:
            radius_now = scale * float(current_radius)
            
        # ---- color (controlled or restore) ----
        if bool(self.state.get("is_use_control_color", False)):
            color_now = tuple(float(x) for x in self.state["color"])
            paint_by_now = 'color'
        else:
            color_now = self.glyph._internal_opts_backup[self.str_now]["color"]
            paint_by_now = self.glyph._internal_opts_backup[self.str_now]["paint_by"]
            
        # ---- opacity (controlled or restore) ----
        if bool(self.state.get("is_use_control_opacity", False)):
            opacity_now = self.state["opacity"]
        else:
            opacity_now = self.glyph._internal_opts_backup[self.str_now]["opacity"]

        self.glyph.act_commit(
            radius=radius_now,
            color=color_now,
            opacity=opacity_now,
            paint_by=paint_by_now,
            sides=int(self.state["sides"]),
            is_silhouette=False,
        )

    def _on_toggle_use_color(self, _state: int):
        self.state["is_use_control_color"] = self.chk_use_color.isChecked()
        self._apply_color_enabled()
        self.commit()
        
    def _on_toggle_use_opacity(self, _state: int):
        self.state["is_use_control_opacity"] = self.chk_use_opacity.isChecked()
        self._apply_opacity_enabled()
        self.commit()
        
    def _apply_color_enabled(self):
        color_enabled = bool(self.chk_use_color.isChecked())
        for k in ["color_r", "color_g", "color_b"]:
            item = self.sliders[k]
            item.slider.setEnabled(color_enabled)
            item.label.setEnabled(color_enabled)
            
    def _apply_opacity_enabled(self):
        opacity_enabled = bool(self.chk_use_opacity.isChecked())
        item = self.sliders['opacity']
        item.slider.setEnabled(opacity_enabled)
        item.label.setEnabled(opacity_enabled)
        



controls_window = SphereControlsWindow(spheres)
controls_window.show()









# class SphereControlsWindow(QtWidgets.QWidget):
#     def __init__(
#         self,
#         glyph,
#     ):
        
#         self.glyph = glyph
#         self.str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
#         self.glyph.act_save_opts(self.str_now)
        
#         self.state = dict(
#             radius_rescale=1,
#             sides=self.glyph.opts.sides,
#             )
        
#         super().__init__(None)
#         self._is_closing = False
#         self._is_dragging = False
        
#         self.setWindowTitle("Sphere Controls (Realtime)")
#         self.setObjectName("window_controls")
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.setContentsMargins(10, 10, 10, 10)
#         layout.setSpacing(10)
        
#         # --- geometry group ---
#         group_geometry = QtWidgets.QGroupBox("Geometry", self)
#         gl_geometry = QtWidgets.QVBoxLayout(group_geometry)

#         # --- radius slider ---
#         row_radius = QtWidgets.QHBoxLayout()
#         self.lab_radius_key = QtWidgets.QLabel("radius_rescale:", group_geometry)
#         self.lab_radius_val = QtWidgets.QLabel(f"{self.state['radius_rescale']:.4g}", group_geometry)
#         self.lab_radius_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
#         row_radius.addWidget(self.lab_radius_key)
#         row_radius.addWidget(self.lab_radius_val)
#         gl_geometry.addLayout(row_radius)

#         self._radius_min = 20   # 0.2x
#         self._radius_max = 500  # 5x

#         self.slider_radius = QtWidgets.QSlider(QtCore.Qt.Horizontal, group_geometry)
#         self.slider_radius.setMinimum(self._radius_min)
#         self.slider_radius.setMaximum(self._radius_max)
#         self.slider_radius.setSingleStep(1)
#         self.slider_radius.setPageStep(10)
#         self.slider_radius.setTracking(True)
        
#         self.slider_radius.setValue(100)
#         gl_geometry.addWidget(self.slider_radius)

#         hint = QtWidgets.QLabel("radius_resize", group_geometry)
#         hint.setWordWrap(True)
#         gl_geometry.addWidget(hint)

#         layout.addWidget(group_geometry)

#         self.slider_radius.valueChanged.connect(self._on_changed)
#         self.slider_radius.sliderReleased.connect(self.glyph._helper_add_silhouette)
#         self.slider_radius.sliderPressed.connect(self.glyph._helper_clear_silhouette)
        
#         # --- sides slider ---
#         row_sides = QtWidgets.QHBoxLayout()
#         self.lab_sides_key = QtWidgets.QLabel("sides:", group_geometry)
#         self.lab_sides_val = QtWidgets.QLabel(f"{self.state['sides']}", group_geometry)
#         self.lab_sides_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
#         row_sides.addWidget(self.lab_sides_key)
#         row_sides.addWidget(self.lab_sides_val)
#         gl_geometry.addLayout(row_sides)

#         self._sides_min = 4   
#         self._sides_max = 40  

#         self.slider_sides = QtWidgets.QSlider(QtCore.Qt.Horizontal, group_geometry)
#         self.slider_sides.setMinimum(self._sides_min)
#         self.slider_sides.setMaximum(self._sides_max)
#         self.slider_sides.setSingleStep(1)
#         self.slider_sides.setPageStep(10)
#         self.slider_sides.setTracking(True)
        
#         self.slider_sides.setValue(self.glyph.opts.sides)
#         gl_geometry.addWidget(self.slider_sides)

#         hint = QtWidgets.QLabel("sides number", group_geometry)
#         hint.setWordWrap(True)
#         gl_geometry.addWidget(hint)

#         layout.addWidget(group_geometry)

#         self.slider_sides.valueChanged.connect(self._on_changed)
#         self.slider_sides.sliderReleased.connect(self.glyph._helper_add_silhouette)
#         self.slider_sides.sliderPressed.connect(self.glyph._helper_clear_silhouette)
        
#         self.glyph.opts._internal_sync_func['sides'][self.str_now] = lambda: self.slider_sides.setValue(self.glyph.opts.sides)
        
#     def _on_changed(self, _v:int, is_commit=True):
#         t = int(self.slider_radius.value())
#         radius_rescale = t / 100.0
#         self.lab_radius_val.setText(f"{radius_rescale:.4g}")
#         self.state["radius_rescale"] = radius_rescale
        
#         t = int(self.slider_sides.value())
#         self.lab_sides_val.setText(str(t))
#         self.state["sides"] = t
        
#         if is_commit:
#             self.commit()
            
        
#     def commit(self):
#         current_radius = self.glyph._internal_opts_backup[self.str_now]['radius']
#         if callable(current_radius):
#             radius_now = lambda x: float(self.state["radius_rescale"]) * current_radius(x)
#         else:
#             radius_now = float(self.state["radius_rescale"]) * current_radius
#         self.glyph.act_commit(
#             radius=radius_now,
#             sides=self.state["sides"],
#             is_silhouette=False,
#             )
        
# controls_window = SphereControlsWindow(spheres)
# controls_window.resize(380, 100)
# controls_window.show()
