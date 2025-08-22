from mayavi import mlab
import numpy as np
from dataclasses import fields

from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..opts import OptsExtent, auto_opts_tubes

@auto_opts_tubes(
    {
        "opts_color": "actor.property.diffuse_color",
        "opts_opacity": "actor.property.opacity",
        "opts_radius": "parent.parent.filter.radius",
        "opts_sides": "parent.parent.filter.number_of_sides",
        "opts_specular": "actor.property.specular",
        "opts_specular_color": "actor.property.specular_color",
        "opts_specular_power": "actor.property.specular_power",
        "_state__state_is_visible": "actor.visible",
    }
)
class PlotExtent:
    """
    Represents a 3D rectangular box (wireframe) in Mayavi,
    allowing unified control over its appearance and lifecycle.
    """

    def __init__(
        self,
        opts=OptsExtent()
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
        if opts.corners is None:
            raise ValueError("The array \'corners\', which stores the positions of the 8 points, are not inputted (the value is None)")
        self._data_corners = opts.corners
        self._items = self.draw_box(opts)
        
        self._internal = opts
        
    # @property
    # def opts_color(self):
    #     return self._internal.color

    # @opts_color.setter
    # def opts_color(self, value):
    #     self._internal.color = value
    #     for item in self._items:
    #         item.actor.property.diffuse_color = self._internal.color

    def draw_box(self, opts):
        """Draw the box edges and store the actors."""
        corners = self._data_corners
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
                tube_radius=opts.radius,
                tube_sides=opts.sides,
                color=opts.color,
                opacity=opts.opacity,
            )
            result.append(actor)

        return result

    def hide(self):
        self._state_is_visible = False

    def show(self):
        self._state_is_visible = True

    def remove(self):
        for item in self._items:
            item.remove()

    # @logging_and_warning_decorator()
    # def log_properties(self, logger=None) -> None:
    #     """
    #     Log all current tube properties using logger.info().

    #     This will include all attributes defined in the @auto_properties mapping,
    #     as well as the number of points in the coordinates.
    #     """

    #     print_lines = []
    #     print_lines.append("=== PlotTube Properties ===")

    #     for attr_name in self.__class__._auto_properties.keys():
    #         if attr_name in {"x", "y", "z"}:
    #             continue
    #         try:
    #             value = getattr(self, attr_name)
    #             print_lines.append(f"{attr_name}: {value}")
    #         except Exception as e:
    #             logger.warning(f"Could not retrieve '{attr_name}': {e}")

    #     print_lines.append("===========================")

    #     logger.info("\n".join(print_lines))
