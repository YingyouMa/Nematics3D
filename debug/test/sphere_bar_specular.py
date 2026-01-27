import sys
sys.path.insert(0, 'D:/Document/GitHub/')
import Nematics3D

import numpy as np
import pyvista as pv
from qtpy import QtWidgets, QtCore
import datetime


# ============================================================
# Data
# ============================================================
def get_path(offset_y=0):
    z = np.linspace(0, 10, 50)
    line1 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    z = np.linspace(15, 20, 25)
    line2 = np.column_stack((np.sin(z), np.cos(z) + offset_y, z))
    return np.concatenate([line1, line2])


figure = Nematics3D.PlotFigure()

# spheres 生成不变
spheres = Nematics3D.PlotSphere(
    figure=figure,
    coords=get_path(offset_y=0),
    name="solid_blue",
    color=(0, 0, 1),
    radius=0.3,
    sides=12,
)


# ============================================================
# Controls Window
# ============================================================
class SphereShadingControlsWindow(QtWidgets.QWidget):
    """
    Two panels:
      - Phong: ambient, diffuse, specular, specular_power, specular_color (RGB)
      - PBR: metallic, roughness

    Mutually exclusive checkboxes:
      - Check Phong => uncheck PBR
      - Check PBR   => uncheck Phong
      - Uncheck one => auto-check the other (never both off)
    """

    def __init__(self, glyph):
        self.glyph = glyph
        self.str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        self.glyph.act_save_opts(self.str_now)

        # ---- initial state (best-effort pull from opts if present) ----
        def _get_opt(name, default):
            return getattr(self.glyph.opts, name, default)

        # Phong
        ambient0 = float(_get_opt("ambient", 0.2))
        diffuse0 = float(_get_opt("diffuse", 1.0))
        specular0 = float(_get_opt("specular", 0.0))
        specular_power0 = float(_get_opt("specular_power", _get_opt("specular_power", 10.0)))
        sc0 = _get_opt("specular_color", (1.0, 1.0, 1.0))
        specular_color0 = tuple(float(x) for x in sc0)

        # PBR
        metallic0 = float(_get_opt("metallic", 0.0))
        roughness0 = float(_get_opt("roughness", 0.5))

        self.state = dict(
            use_phong=True,
            use_pbr=False,
            phong=dict(
                ambient=ambient0,
                diffuse=diffuse0,
                specular=specular0,
                specular_power=specular_power0,
                specular_color=specular_color0,
            ),
            pbr=dict(
                metallic=metallic0,
                roughness=roughness0,
            ),
        )

        super().__init__(None)

        self.setWindowTitle("Sphere Shading Controls (Phong / PBR)")
        self.setObjectName("window_controls_shading")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ----------------------------
        # Helper widgets
        # ----------------------------
        def _make_float_slider(parent, label, init_val, vmin, vmax, steps=1000, decimals=4):
            """
            One-row layout:  [label] [slider] [value]
            Slider maps [0..steps] -> [vmin..vmax].
            Returns (row_widget, lab_val, slider, getter()).
            """
            roww = QtWidgets.QWidget(parent)
            h = QtWidgets.QHBoxLayout(roww)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
        
            lab_key = QtWidgets.QLabel(f"{label}", roww)
            lab_key.setMinimumWidth(110)  # 你可以调小/调大
            h.addWidget(lab_key)
        
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, roww)
            slider.setMinimum(0)
            slider.setMaximum(int(steps))
            slider.setSingleStep(1)
            slider.setPageStep(max(1, int(steps // 20)))
            slider.setTracking(True)
            h.addWidget(slider, 1)
        
            lab_val = QtWidgets.QLabel("", roww)
            lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lab_val.setMinimumWidth(70)   # 你可以调小/调大
            h.addWidget(lab_val)
        
            def _val_to_tick(x):
                x = float(x)
                x = max(vmin, min(vmax, x))
                if vmax == vmin:
                    return 0
                return int(round((x - vmin) / (vmax - vmin) * steps))
        
            def _tick_to_val(t):
                t = int(t)
                return vmin + (vmax - vmin) * (t / steps)
        
            slider.setValue(_val_to_tick(init_val))
        
            def getter():
                return float(_tick_to_val(slider.value()))
        
            lab_val.setText(f"{getter():.{decimals}g}")
            return roww, lab_val, slider, getter

        def _make_rgb_sliders(parent, title, init_rgb):
            """
            Compact RGB rows, each row: [R/G/B] [slider] [value]
            """
            container = QtWidgets.QGroupBox(title, parent)
            vbox = QtWidgets.QVBoxLayout(container)
            vbox.setContentsMargins(8, 8, 8, 8)
            vbox.setSpacing(6)
        
            def _one(channel, init_val):
                roww = QtWidgets.QWidget(container)
                h = QtWidgets.QHBoxLayout(roww)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(8)
        
                lab_key = QtWidgets.QLabel(channel, roww)
                lab_key.setMinimumWidth(20)
                h.addWidget(lab_key)
        
                s = QtWidgets.QSlider(QtCore.Qt.Horizontal, roww)
                s.setMinimum(0)
                s.setMaximum(1000)
                s.setSingleStep(1)
                s.setPageStep(10)
                s.setTracking(True)
                s.setValue(int(round(float(init_val) * 1000)))
                h.addWidget(s, 1)
        
                lab_val = QtWidgets.QLabel("", roww)
                lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                lab_val.setMinimumWidth(70)
                h.addWidget(lab_val)
        
                def _get():
                    return float(s.value()) / 1000.0
        
                lab_val.setText(f"{_get():.4g}")
        
                vbox.addWidget(roww)
                return lab_val, s, _get
        
            lr, sr, gr = _one("R", init_rgb[0])
            lg, sg, gg = _one("G", init_rgb[1])
            lb, sb, gb = _one("B", init_rgb[2])
        
            def getter():
                return (gr(), gg(), gb())
        
            return container, (lr, lg, lb), (sr, sg, sb), getter

        # ----------------------------
        # Phong Panel
        # ----------------------------
        group_phong = QtWidgets.QGroupBox("Phong", self)
        v_phong = QtWidgets.QVBoxLayout(group_phong)

        self.chk_phong = QtWidgets.QCheckBox("Enable Phong shading", group_phong)
        self.chk_phong.setChecked(True)
        v_phong.addWidget(self.chk_phong)

        box_amb, self.lab_amb, self.slider_amb, self.get_amb = _make_float_slider(
            group_phong, "ambient", self.state["phong"]["ambient"], 0.0, 1.0, steps=1000, decimals=4
        )
        box_dif, self.lab_dif, self.slider_dif, self.get_dif = _make_float_slider(
            group_phong, "diffuse", self.state["phong"]["diffuse"], 0.0, 1.0, steps=1000, decimals=4
        )
        box_spe, self.lab_spe, self.slider_spe, self.get_spe = _make_float_slider(
            group_phong, "specular", self.state["phong"]["specular"], 0.0, 1.0, steps=1000, decimals=4
        )
        box_pow, self.lab_pow, self.slider_pow, self.get_pow = _make_float_slider(
            group_phong, "specular_power", self.state["phong"]["specular_power"], 1.0, 200.0, steps=1990, decimals=4
        )

        v_phong.addWidget(box_amb)
        v_phong.addWidget(box_dif)
        v_phong.addWidget(box_spe)
        v_phong.addWidget(box_pow)

        self.box_spec_rgb, self.labs_spec_rgb, self.sliders_spec_rgb, self.get_spec_rgb = _make_rgb_sliders(
            group_phong, "specular_color (RGB 0..1)", self.state["phong"]["specular_color"]
        )
        v_phong.addWidget(self.box_spec_rgb)

        layout.addWidget(group_phong)

        # ----------------------------
        # PBR Panel
        # ----------------------------
        group_pbr = QtWidgets.QGroupBox("PBR", self)
        v_pbr = QtWidgets.QVBoxLayout(group_pbr)

        self.chk_pbr = QtWidgets.QCheckBox("Enable PBR shading", group_pbr)
        self.chk_pbr.setChecked(False)
        v_pbr.addWidget(self.chk_pbr)

        box_met, self.lab_met, self.slider_met, self.get_met = _make_float_slider(
            group_pbr, "metallic", self.state["pbr"]["metallic"], 0.0, 1.0, steps=1000, decimals=4
        )
        box_rou, self.lab_rou, self.slider_rou, self.get_rou = _make_float_slider(
            group_pbr, "roughness", self.state["pbr"]["roughness"], 0.0, 1.0, steps=1000, decimals=4
        )

        v_pbr.addWidget(box_met)
        v_pbr.addWidget(box_rou)

        layout.addWidget(group_pbr)

        # ----------------------------
        # Signals
        # ----------------------------
        self.chk_phong.stateChanged.connect(self._on_toggle_mode)
        self.chk_pbr.stateChanged.connect(self._on_toggle_mode)

        # Sliders -> update state + commit
        for s in (
            self.slider_amb, self.slider_dif, self.slider_spe, self.slider_pow,
            *self.sliders_spec_rgb,
            self.slider_met, self.slider_rou
        ):
            s.valueChanged.connect(self._on_changed)

            # 你的 silhouette 行为：按下清理，松手加回
            s.sliderPressed.connect(self.glyph._helper_clear_silhouette)
            s.sliderReleased.connect(self.glyph._helper_add_silhouette)

        # 初始化：关掉 PBR 控件区（因为默认 Phong）
        self._apply_enabled_state()
        self.commit()

    # ----------------------------
    # UI logic
    # ----------------------------
    def _apply_enabled_state(self):
        use_phong = self.chk_phong.isChecked()
        use_pbr = self.chk_pbr.isChecked()

        # Phong 控件启用/禁用
        for w in (self.slider_amb, self.slider_dif, self.slider_spe, self.slider_pow, *self.sliders_spec_rgb):
            w.setEnabled(use_phong)

        # PBR 控件启用/禁用
        for w in (self.slider_met, self.slider_rou):
            w.setEnabled(use_pbr)

    def _on_toggle_mode(self, _state: int):
        # sender-based mutual exclusion with "never both off"
        sender = self.sender()

        if sender is self.chk_phong:
            if self.chk_phong.isChecked():
                self.chk_pbr.blockSignals(True)
                self.chk_pbr.setChecked(False)
                self.chk_pbr.blockSignals(False)
            else:
                # 如果 Phong 被关掉，则强制打开 PBR
                self.chk_pbr.blockSignals(True)
                self.chk_pbr.setChecked(True)
                self.chk_pbr.blockSignals(False)

        elif sender is self.chk_pbr:
            if self.chk_pbr.isChecked():
                self.chk_phong.blockSignals(True)
                self.chk_phong.setChecked(False)
                self.chk_phong.blockSignals(False)
            else:
                # 如果 PBR 被关掉，则强制打开 Phong
                self.chk_phong.blockSignals(True)
                self.chk_phong.setChecked(True)
                self.chk_phong.blockSignals(False)

        self.state["use_phong"] = self.chk_phong.isChecked()
        self.state["use_pbr"] = self.chk_pbr.isChecked()

        self._apply_enabled_state()
        self.commit()

    def _on_changed(self, _v: int, is_commit: bool = True):
        # Update labels + state from sliders
        amb = self.get_amb(); self.lab_amb.setText(f"{amb:.4g}")
        dif = self.get_dif(); self.lab_dif.setText(f"{dif:.4g}")
        spe = self.get_spe(); self.lab_spe.setText(f"{spe:.4g}")
        powv = self.get_pow(); self.lab_pow.setText(f"{powv:.4g}")

        rgb = self.get_spec_rgb()
        self.labs_spec_rgb[0].setText(f"{rgb[0]:.4g}")
        self.labs_spec_rgb[1].setText(f"{rgb[1]:.4g}")
        self.labs_spec_rgb[2].setText(f"{rgb[2]:.4g}")

        met = self.get_met(); self.lab_met.setText(f"{met:.4g}")
        rou = self.get_rou(); self.lab_rou.setText(f"{rou:.4g}")

        self.state["phong"].update(
            ambient=amb,
            diffuse=dif,
            specular=spe,
            specular_power=powv,
            specular_color=rgb,
        )
        self.state["pbr"].update(
            metallic=met,
            roughness=rou,
        )

        if is_commit:
            self.commit()

    # ----------------------------
    # Commit
    # ----------------------------
    def commit(self):

        if self.state["use_phong"]:
            p = self.state["phong"]
            self.glyph.act_commit(
                # mode
                shading_type='phong',  # 若你实现里是 "is_pbr" / "use_pbr" 之类，改这里
                # phong params
                ambient=float(p["ambient"]),
                diffuse=float(p["diffuse"]),
                specular=float(p["specular"]),
                specular_power=float(p["specular_power"]),     # 若你用 specular_pow 就改名
                specular_color=tuple(float(x) for x in p["specular_color"]),
                is_silhouette=False,
            )
        else:
            p = self.state["pbr"]
            self.glyph.act_commit(
                shading_type='pbr',  # 若你实现里不是这个字段，改这里
                metallic=float(p["metallic"]),
                roughness=float(p["roughness"]),
                is_silhouette=False,
            )


controls_window = SphereShadingControlsWindow(spheres)
controls_window.resize(520, 560)
controls_window.show()
