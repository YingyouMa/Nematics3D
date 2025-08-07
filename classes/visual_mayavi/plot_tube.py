from mayavi import mlab
import numpy as np
from typing import Optional

from Nematics3D.datatypes import Vect3D
from .visual_decorator import auto_properties
from Nematics3D.logging_decorator import logging_and_warning_decorator


@auto_properties(
    # Dynamically create property getters/setters for common visualization attributes.
    # These map directly to Mayavi actor attributes so you can write:
    #   lineObj.color = (1, 0, 0)   instead of    lineObj.actor.actor.property.color = (1, 0, 0)
    {
        "color": "actor.actor.property.color",
        "opacity": "actor.actor.property.opacity",
        "radius": "actor.parent.parent.filter.radius",
        "sides": "actor.parent.parent.filter.number_of_sides",
        "specular": "actor.actor.property.specular",
        "specular_color": "actor.actor.property.specular_color",
        "specular_power": "actor.actor.property.specular_power",
        "x": "actor.mlab_source.x",
        "y": "actor.mlab_source.y",
        "z": "actor.mlab_source.z",
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
        coords: np.ndarray,
        color: Vect3D = (0, 0, 0),
        radius: float = 1,
        opacity: float = 1,
        sides: int = 6,
        specular: float = 1,
        specular_color: Vect3D = (1.0, 1.0, 1.0),
        specular_power: float = 11,
        scalars: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        logger=None,
    ) -> None:
        """
        Initialize and draw the tube.

        Args:
            coords (np.ndarray): Nx3 array of 3D coordinates defining the tube path.
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
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        if scalars is not None:
            logger.warning(">>> The scalars of tube is input")
            logger.warning(">>> The color of tube will be ignored")
            self.actor = mlab.plot3d(
                x, y, z, scalars,
                tube_radius=radius,
                tube_sides=sides,
                opacity=opacity,
            )
        else:
            color = tuple(color)
            self.actor = mlab.plot3d(
                x, y, z,
                color=color,
                tube_radius=radius,
                tube_sides=sides,
                opacity=opacity,
            )
        

        prop = self.actor.actor.property
        prop.specular = specular
        prop.specular_color = specular_color
        prop.specular_power = specular_power
        
        self.name = name

    def update_coords(self, coords: np.ndarray) -> None:
        self.x, self.y, self.z = coords[:, 0], coords[:, 1], coords[:, 2]

    def hide(self):
        self.actor.visible = False

    def show(self):
        self.actor.visible = True

    def remove(self):
        self.actor.remove()

    @property
    def coords(self) -> np.array:
        return np.column_stack((self.x, self.y, self.z))
    
    @logging_and_warning_decorator()
    def log_properties(self, logger=None) -> None:
        """
        Log all current tube properties using logger.info().

        This will include all attributes defined in the @auto_properties mapping,
        as well as the number of points in the coordinates.
        """
        
        print_lines = []
        print_lines.append(" ")
        print_lines.append("=== PlotTube Properties ===")
        
        for attr_name in self.__class__._auto_properties.keys():
            if attr_name in {"x", "y", "z"}:
                continue
            try:
                value = getattr(self, attr_name)
                print_lines.append(f"{attr_name}: {value}")
            except Exception as e:
                logger.warning(f"Could not retrieve '{attr_name}': {e}")

        # Additional info about coordinates
        print_lines.append(f"Number of points: {len(self.x)}")
        print_lines.append("===========================")
        
        logger.info("\n".join(print_lines))
