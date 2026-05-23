from .interact_glyph_base import InteractGlyphBase


class InteractDelaunay(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractDelaunay overrides InteractGlyphBase.__init__ because Delaunay
    # surface control only needs color and opacity, while radius,
    # sides, and extra geometry controls are not meaningful here.
    # ==================================================
    def __init__(self, host, figure):
        super().__init__(
            host,
            figure,
            title=f"Delaunay Controls of {host.name!r}",
            is_radius=False,
            is_sides=False,
            is_geometry=False,
            is_color=True,
            is_opacity=True,
        )
