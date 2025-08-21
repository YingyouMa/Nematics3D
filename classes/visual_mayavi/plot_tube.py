from mayavi import mlab
import numpy as np
from typing import Optional
from dataclasses import replace

from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..opts import OptsTube
from .visual_decorator import auto_properties

@auto_properties(
    {
        "opts_color": "actor.property.diffuse_color",
        "opts_opacity": "actor.property.opacity",
        "opts_radius": "parent.parent.filter.radius",
        "opts_sides": "parent.parent.filter.number_of_sides",
        "opts_specular": "actor.property.specular",
        "opts_specular_color": "actor.property.specular_color",
        "opts_specular_power": "actor.property.specular_power",
        "opts_is_visible": "actor.visible",
        "opts_name": "name",
    }
)
class PlotTube:
    """
    A utility class to create and manage a 3D tube-like curve in Mayavi.

    Features:
    - Draws a smooth tube along given coordinates.
    - Provides dynamic attribute access to key visual properties (color, opacity, radius, etc.).
    - Supports scalar coloring for gradient effects.
    - Includes convenience methods for updating coordinates, hiding/showing, and removing the tube.

    Attributes:
        actor: The Mayavi pipeline object returned by mlab.plot3d().
    """

    @logging_and_warning_decorator()
    def __init__(
        self,
        coords_all: np.ndarray,
        scalars_all: Optional[np.ndarray] = None,
        opts = OptsTube(),
        logger=None,
    ) -> None:
        """
        Initialize and draw the tube.

        Args:
            coords (np.ndarray): (num_sublines, N, 3) array of 3D coordinates defining the tube path.
            color (Vect3D): RGB color tuple in [0, 1]. Ignored if 'scalars' is provided.
            radius (float): Tube radius.
            opacity (float): Opacity in [0, 1].
            sides (int): Number of sides for the tube cross-section (higher = smoother).
            specular (float): Specular reflection coefficient.
            specular_color (Vect3D): RGB color tuple for specular highlight.
            specular_power (float): Shininess exponent for specular highlight.
            scalars (Optional[np.ndarray]): Optional scalar values for each vertex
                (enables gradient coloring). If provided, overrides 'color'.
            logger: Optional logger instance used for warnings.
        """

        self.items = []
        self.coords_all = coords_all
        self.opts = opts
        self.scalars_all = scalars_all
        
        if self.opts.color is None:
            logger.warning("The color input of tube is None. Changed it into (1,1,1).")
            self.opts.color = (1,1,1)

        num_sublines = len(self.coords_all)
        if self.scalars_all is not None:
            logger.debug(">>> The scalars of tube is input")
            logger.debug(">>> The color of tube will be ignored")
        else:
            self.scalars_all = [None for i in range(num_sublines)]
            
        for coords, scalars in zip(self.coords_all, self.scalars_all):
        
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            if scalars is not None and len(scalars) != len(x):
                msg = f">>> The length of this subline {len(x)} does not match with scalars {len(scalars)}.\n"
                msg += ">>> Ignore scalars in the following"
                logger.warning(msg)
                scalars = None

            if scalars is not None:

                item = mlab.plot3d(
                    x,
                    y,
                    z,
                    scalars,
                    tube_radius=self.opts.radius,
                    tube_sides=self.opts.sides,
                    opacity=self.opts.opacity,
                )
            else:
                item = mlab.plot3d(
                    x,
                    y,
                    z,
                    color=self.opts.color,
                    tube_radius=self.opts.radius,
                    tube_sides=self.opts.sides,
                    opacity=self.opts.opacity,
                )
                

            prop = item.actor.property
            prop.specular = self.opts.specular
            prop.specular_color = self.opts.specular_color
            prop.specular_power = self.opts.specular_power
            
            self.items.append(item)
            

    def hide(self):
        self.is_visible = False

    def show(self):
        self.is_visible = True

    def remove(self):
        for item in self.items:
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
