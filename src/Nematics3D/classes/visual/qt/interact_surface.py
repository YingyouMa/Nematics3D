from .interact_glyph_base import InteractGlyphBase


class InteractSurface(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractSurface overrides InteractGlyphBase.__init__ because
    # surface control only needs color and opacity, while radius,
    # sides, and extra geometry controls are not meaningful here.
    # ==================================================
    def __init__(self, host, figure):
        super().__init__(
            host,
            figure,
            title=f"Surface Controls of {host.name!r}",
            is_radius=False,
            is_sides=False,
            is_geometry=False,
            is_color=True,
            is_opacity=True,
        )