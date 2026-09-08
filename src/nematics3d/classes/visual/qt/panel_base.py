"""Compatibility import for the migrated Qt panel infrastructure."""

from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    MovePointConsole,
    PanelBase,
    PressHoldButtonItem,
    SliderItem,
    make_RGB_slider,
    make_labeled_slider_row,
    make_press_hold_button,
)

__all__ = [
    "SliderItem",
    "make_labeled_slider_row",
    "make_RGB_slider",
    "LogTickMapper",
    "PressHoldButtonItem",
    "make_press_hold_button",
    "MovePointConsole",
    "PanelBase",
]
