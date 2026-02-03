from qtpy import QtWidgets, QtCore, QtGui
from dataclasses import dataclass
import datetime
from typing import Callable

from Nematics3D.datatypes import as_str


@dataclass
class SliderItem:
    slider: QtWidgets.QSlider
    label: QtWidgets.QLabel
    get_value: callable
    

def make_labeled_slider_row(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    name: str,
    tick_min: int,
    tick_max: int,
    tick_init: int,
    tick_to_value: Callable[[int], float],
    value_fmt: str = "{:.4g}",
    key_min_width: int = 120,
    val_min_width: int = 70,
    single_step: int = 1,
    page_step: int = 10,
) -> SliderItem:

    # ---- row container ----
    row_widget = QtWidgets.QWidget(parent)
    h = QtWidgets.QHBoxLayout(row_widget)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)

    lab_key = QtWidgets.QLabel(f"{name}:", row_widget)
    lab_key.setMinimumWidth(key_min_width)
    h.addWidget(lab_key)

    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, row_widget)
    slider.setMinimum(int(tick_min))
    slider.setMaximum(int(tick_max))
    slider.setSingleStep(single_step)
    slider.setPageStep(page_step)                 
    slider.setTracking(True)
    h.addWidget(slider, 1)

    lab_val = QtWidgets.QLabel("", row_widget)
    lab_val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    lab_val.setMinimumWidth(val_min_width)
    h.addWidget(lab_val)

    slider.setValue(tick_init)

    def get_value() -> float:
        return float(tick_to_value(int(slider.value())))

    lab_val.setText(value_fmt.format(get_value()))

    layout.addWidget(row_widget)

    return SliderItem(slider=slider, label=lab_val, get_value=get_value)


def make_RGB_slider(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QBoxLayout,
    sliders: dict[str, SliderItem],
    prefix: str,
    init_rgb: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    tick_min: int = 0,
    tick_max: int = 1000,
    value_fmt: str = "{:.3f}",
    single_step: int = 1,
    page_step: int = 10,
) -> None:

    def _t2v(t: int) -> float:
        return t / 1000.0

    names = ("r", "g", "b")
    for i, ch in enumerate(names):
        key = f"{prefix}_{ch}"
        init_val = float(init_rgb[i])
        t0 = int(round(init_val * 1000))

        sliders[key] = make_labeled_slider_row(
            parent=parent,
            layout=layout,
            name=ch.upper(),
            tick_min=tick_min,
            tick_max=tick_max,
            tick_init=t0,
            tick_to_value=_t2v,
            value_fmt=value_fmt,
            single_step=single_step,
            page_step=page_step,
        )
    



class PanelBase(QtWidgets.QWidget):
    
    def __init__(self, glyph, title: str = "Panel"):
        
        title = as_str(title, name="The title of panel", replace="Panel")
        
        super().__init__()
        
        self.glyph = glyph
        self.str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        self.glyph.act_save_opts(self.str_now)
        object.__setattr__(self.glyph, "_state_is_interactable", False)
        
        self.state: dict[str, object] = {}
        self.sliders: dict[str, SliderItem] = {}
        
        self.setWindowTitle(title)
        self.setObjectName('panel')
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        
        self.build_ui()
        
    def _sync_sides_from_glyph(self, attr: str, value: int):
        s = self.sliders[attr].slider
        try:
            s.blockSignals(True)
            s.setValue(int(value))
        finally:
            s.blockSignals(False)
        self.on_changed(0, is_commit=False)
        
    def build_ui(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def on_changed(self, _v: int = 0):
        raise NotImplementedError
        
    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.on_close()
        finally:
            event.accept()
            
    def on_close(self):
        object.__setattr__(self.glyph, "_state_is_interactable", True)
        sync = getattr(self.glyph.opts, "_impl_sync_func", None)
        for k, sub in sync.items():
            sub.pop(self.str_now, None)