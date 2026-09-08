"""Compatibility import for the migrated pick manager."""

from nematics3d.visual.pick_manager import (
    OptsPickManager,
    PickManager,
    _ClickTracker as _ClickTracker,
    _Marker as _Marker,
)

__all__ = ["OptsPickManager", "PickManager"]
