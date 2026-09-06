"""Legacy PlotGlyph silhouette behavior before the lazy-silhouette optimization.

This backup preserves the eager silhouette lifecycle used by PlotGlyph at
commit 06c3d74c9d898f7b455b7b98b2c000a3c9ee92ea.  It is intentionally kept as
a compact executable compatibility shim rather than a second full copy of the
~1100-line glyph module: all behavior is inherited from the production
PlotGlyph except the silhouette lifecycle, which is restored to the previous
eager semantics here.
"""

from nematics3d.classes.visual.glyph import PlotGlyph


class PlotGlyphBeforeLazySilhouette(PlotGlyph):
    """PlotGlyph variant that eagerly creates/recreates silhouettes."""

    def _helper_make_figure(self, logger=None):
        super()._helper_make_figure(logger=logger)
        if (
            self.state_is_silhouette
            and getattr(self, "entity_actor", None) is not None
            and not self.calc_is_empty
        ):
            self._helper_add_silhouette()

    def _helper_refresh_silhouette_after_remesh(self):
        if self.state_is_silhouette:
            self._helper_add_silhouette()
        else:
            self._helper_clear_silhouette()
