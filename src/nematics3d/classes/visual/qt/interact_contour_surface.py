import numpy as np
from qtpy import QtWidgets

from .interact_glyph_base import InteractGlyphBase
from .panel_base import make_labeled_slider_row


class InteractContourSurface(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractContourSurface extends the shared glyph panel with one contour-
    # specific level control that operates on the owning ContourSurface rather
    # than on the visual opts object.
    # ==================================================
    def __init__(self, host, figure):
        self.surface = getattr(host, "owner", None)
        self.surface_set = getattr(self.surface, "owner", None)
        if self.surface is None or self.surface_set is None:
            raise RuntimeError(
                "InteractContourSurface requires a PlotContourSurface with live "
                "ContourSurface and ContourSurfaceSet owners."
            )

        values = np.asarray(self.surface_set.raw_values, dtype=float)
        self.level_min = float(np.min(values))
        self.level_max = float(np.max(values))
        self._level_original = float(host.calc_level)
        self._snapshot_levels: dict[str, float] = {}

        super().__init__(
            host,
            figure,
            title=f"Contour Controls of {host.name!r}",
            is_radius=False,
            is_sides=False,
            is_geometry=False,
            is_color=True,
            is_opacity=True,
        )

    def _build_extra_group(self):
        self.state["level"] = float(self.host.calc_level)
        group_level = QtWidgets.QGroupBox("Level", self)
        layout_level = QtWidgets.QVBoxLayout(group_level)
        self.layout.addWidget(group_level)

        span = max(self.level_max - self.level_min, 1.0)
        decimals = 6 if span < 1e-3 else 4 if span < 1e-1 else 3
        value_fmt = "{:." + str(decimals) + "f}"

        self.sliders["level"] = make_labeled_slider_row(
            parent=group_level,
            layout=layout_level,
            name="level",
            state_key="level",
            value_min=self.level_min,
            value_max=self.level_max,
            value_init=self.state["level"],
            tick_to_value=lambda t: self.level_min
            + (self.level_max - self.level_min) * float(t) / 1000.0,
            value_to_tick=lambda v: int(
                round(
                    1000.0
                    * (float(v) - self.level_min)
                    / max(self.level_max - self.level_min, 1.0e-12)
                )
            ),
            value_fmt=value_fmt,
        )

    # ==================== OVERRIDE ====================
    # Level lives on the owning ContourSurface, not on host.opts, so contour
    # commits must update the owner first, then apply any glyph-style visual
    # controls such as color and opacity.
    # ==================================================
    def commit(self):
        params = self._helper_build_commit_params()
        level = float(self.state["level"])

        self._is_gui_updating = True
        try:
            if not np.isclose(level, float(self.surface.raw_level)):
                self.surface.act_set_level(level)
            self._helper_run_commit(params)
        finally:
            self._is_gui_updating = False

    def _sync_func(self, **kwargs):
        super()._sync_func(**kwargs)
        if getattr(self, "_is_gui_updating", False):
            return

        level_current = float(self.host.calc_level)
        self._sync_from_host_slider("level", level_current)

    def _helper_save_snapshot(self, name: str, *, is_user_snapshot: bool) -> None:
        super()._helper_save_snapshot(name, is_user_snapshot=is_user_snapshot)
        self._snapshot_levels[name] = float(self.host.calc_level)

    def _helper_restore_snapshot(self, name: str) -> None:
        level = self._snapshot_levels.get(name)
        if level is None:
            if name == self.str_now:
                level = float(self._level_original)
            else:
                raise KeyError(f"Snapshot {name!r} has no saved contour level.")
        self.surface.act_set_level(float(level))
        super()._helper_restore_snapshot(name)
