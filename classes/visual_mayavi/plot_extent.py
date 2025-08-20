from mayavi import mlab
import numpy as np
from typing import List, Tuple

from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..opts import Vect, as_Vect, ColorRGB, as_ColorRGB

class PlotExtent:
    """
    Represents a 3D rectangular box (wireframe) in Mayavi,
    allowing unified control over its appearance and lifecycle.
    """

    def __init__(
        self,
        corners: np.ndarray,
        radius: float = 1,
        sides: int = 6,
        color: ColorRGB = (0, 0, 0),
        opacity: float = 1,
    ):
        """
        Create an Extent object and draw it in the Mayavi scene.

        Parameters
        ----------
        corners : np.ndarray, (8,3),
            Coordinates of the 8 corners of the box.

        radius : float, optional
            Thickness of the box edges.

        sides : int, optional
            Number of sides of each tube (smoothness).

        color : tuple of 3 floats, optional
            RGB color of the box edges, values in [0, 1].
            Default is black

        opacity : float in [0,1], optional
            The opacity of extent
        """
        color = as_ColorRGB(color)
        self.corners = corners
        self.items = self.draw_box(
            corners, radius=radius, sides=sides, color=color, opacity=opacity
        )

    @staticmethod
    def draw_box(
        corners: np.ndarray,
        radius: float = 0.05,
        sides: int = 6,
        color: ColorRGB = (0, 0, 0),
        opacity: float = 1,
    ):
        """Draw the box edges and store the actors."""
        color = as_ColorRGB(color)
        edges = [
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
        result = []
        for i, j in edges:
            p1, p2 = corners[i], corners[j]
            coords = np.array([p1, (p1 + p2) / 2, p2])
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            actor = mlab.plot3d(
                x,
                y,
                z,
                tube_radius=radius,
                tube_sides=sides,
                color=color,
                opacity=opacity,
            )
            result.append(actor)

        return result

    def hide(self):
        self.is_visible = False

    def show(self):
        self.is_visible = True

    def remove(self):
        for item in self.items:
            item.remove()

    @logging_and_warning_decorator()
    def log_properties(self, logger=None) -> None:
        """
        Log all current tube properties using logger.info().

        This will include all attributes defined in the @auto_properties mapping,
        as well as the number of points in the coordinates.
        """

        print_lines = []
        print_lines.append("=== PlotTube Properties ===")

        for attr_name in self.__class__._auto_properties.keys():
            if attr_name in {"x", "y", "z"}:
                continue
            try:
                value = getattr(self, attr_name)
                print_lines.append(f"{attr_name}: {value}")
            except Exception as e:
                logger.warning(f"Could not retrieve '{attr_name}': {e}")

        print_lines.append("===========================")

        logger.info("\n".join(print_lines))
