import numpy as np

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    Tensor,
    as_Tensor,
)
from .plot_tube import PlotTube, OptsTube
from .plot_figure import PlotFigure


class PlotExtent(PlotTube):
    """
    Plot a 3D bounding frame defined by 8 corner points.

    This class is a thin wrapper around PlotTube, managing the semantic
    concept of an "extent" (bounding box / frame) while reusing the full
    tube rendering and update pipeline.
    """
    
    DEFAULT_VAL_TUBE = {
        "color":   (0.0, 0.0, 0.0),  # black
        "radius":  0.15,             # thinner than PlotTube default
        "opacity": 1.0,              # fully opaque
        "scalars": None,             # no scalars by default
    }

    _EDGES = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (4, 7),
        (3, 5),
        (3, 6),
        (5, 7),
        (6, 7),
    ]

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        corners: Tensor,
        Figure: PlotFigure | None = None,
        opts: OptsTube | None = None,
        logger=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        corners : (8, 3) array-like
            Coordinates of the eight corners defining the extent.
        Figure : PlotFigure, optional
            Target figure.
        opts : OptsTube, optional
            Rendering options. Defaults are overridden for extent style.
        """
    
        corners = as_Tensor(corners, (8,3), name='The original 8 corner points defining the extent.')

        coords, line_index = self._helper_build_edges_from_corners(corners)

        if opts is None:
            opts = OptsTube(
                name = "extent",
                category = "extent"
                )

        super().__init__(
            coords=coords,
            Figure=Figure,
            opts=opts,
            line_index=line_index,
            logger=logger,
            **kwargs
        )

        self.act_add_attr(
            name="_raw_corners",
            doc="The original 8 corner points defining the extent.",
            default=corners,
        )

    @staticmethod
    def _helper_build_edges_from_corners(corners: np.ndarray):
        """
        Convert corner points and edge definitions into PlotTube-compatible
        coords and line_index.
        """
        coords = []
        line_index = []

        for i, (a, b) in enumerate(PlotExtent._EDGES):
            coords.append(corners[a])
            coords.append(corners[b])
            line_index.extend([i, i])

        return np.asarray(coords), np.asarray(line_index, dtype=int)
