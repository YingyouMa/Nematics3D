import numpy as np
from mayavi import mlab
from collections import defaultdict

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import as_ColorRGB
from .scene_wrapper import SceneWrapper
from ..opts import OptsScene


class PlotScene:
    """
    A scene manager that holds and manages multiple Mayavi plot objects,
    sharing a single Mayavi figure as the drawing canvas.
    """
    
    @logging_and_warning_decorator
    def __init__(
        self,
        is_new: bool = True,
        opts = OptsScene(),
        logger=None
    ):
        """
        Initialize the scene with a new Mayavi figure.

        Args:
            size (tuple): Window size in pixels (width, height).
            bgcolor (tuple): Background color as RGB in [0, 1].
            fgcolor (tuple): Foreground color (axes, labels) as RGB in [0, 1].
            is_new: If create a now scene.
                If not, use the current scene and all other arguments are ignored.
        """
        if is_new:
            size = opts.fig_size
            bgcolor = opts.bgcolor
            fgcolor = opts.fgcolor
            self._fig = mlab.figure(size=size, bgcolor=bgcolor, fgcolor=fgcolor)
            self.scene = SceneWrapper(self._fig.scene)
        else:
            self._fig = mlab.gcf()
            self.scene = SceneWrapper(self._fig.scene)
            self.scene.background = opts.bgcolor
            self.scene.foreground = opts.fgcolor

        # Store objects in categories: { "tubes": [obj1, obj2], "surfaces": [...] }
        self.objects = defaultdict(list)
        
        if opts.name is not None:
            if hasattr(self, 'name') and self.name is not None:
                logger.warning("The figure already has its name {self.name}. Now it is re-named as {opts.name}")
            self.name = opts.name
        if opts.name is None:
            if not hasattr(self, 'name'):
                self.name = opts.name
            
        self.scene._set_angles(
            opts.azimuth,
            opts.elevation,
            opts.roll, 
            opts.distance, 
            opts.focal_point)
        
        

    @logging_and_warning_decorator
    def add_object(self, obj, category: str = "default", logger=None) -> None:
        """
        Add a plot object to a category in the scene.
        Ensures the object has a unique 'name' within the category.
        If no 'name' exists, one is assigned automatically.
        """
        # Ensure the category exists
        if category not in self.objects:
            self.objects[category] = []

        # Step 1: Determine the base name
        if hasattr(obj, "name"):
            base_name = obj.name
        else:
            base_name = category + "_0"  # fallback: use category name

        # Step 2: Ensure uniqueness in this category
        existing_names = {getattr(o, "name") for o in self.objects[category]}
        new_name = base_name
        counter = 1
        while new_name in existing_names:
            new_name = f"{base_name}_{counter}"
            counter += 1
        if counter != 1:
            logger.warning(
                f">>> {base_name} already exists. Changed it into {base_name}_{counter}"
            )

        # Step 3: Assign the final name
        setattr(obj, "name", new_name)

        # Step 4: Add to storage
        self.objects[category].append(obj)

    @logging_and_warning_decorator
    def find_object(self, category: str, name: str, logger=None):
        """
        Find an object in a category by its name.

        Args:
            category (str): Category name.
            name (str): Object name to search.

        Returns:
            The matching object, or None if not found.
        """
        for obj in self.objects.get(category, []):
            if getattr(obj, "name") == name:
                return obj
        logger.warning(f">>> No {name} found in {category}")
        return None

    def remove_object(self, category: str, name: str) -> None:
        """Remove a specific object from a category."""
        obj = self.find_object(category, name)
        if obj != None:
            self.objects.get(category, []).remove(obj)
            obj.remove()

    def clear_category(self, category: str) -> None:
        """Remove all objects in a specific category."""
        for obj in self.objects.get(category, []):
            if hasattr(obj, "remove"):
                obj.remove()
        self.objects.pop(category, None)  # 删除这个类别键

    def clear_all(self) -> None:
        """Remove all objects in all categories."""
        for category in list(self.objects.keys()):
            self.clear_category(category)

    def show_category(self, category: str) -> None:
        """Show all objects in a specific category."""
        for obj in self.objects.get(category, []):
            if hasattr(obj, "show"):
                obj.show()

    def hide_category(self, category: str) -> None:
        """Hide all objects in a specific category."""
        for obj in self.objects.get(category, []):
            if hasattr(obj, "hide"):
                obj.hide()

    def show_all(self):
        """Show all objects in all categories."""
        for category in list(self.objects.keys()):
            self.show_category(category)

    def hide_all(self):
        """Hide all objects in all categories."""
        for category in list(self.objects.keys()):
            self.hide_category(category)

    def save(self, address: str):
        mlab.savefig(address, figure=self._fig)

    @logging_and_warning_decorator()
    def log_info(self, mode: str = "all", logger=None) -> None:
        """
        Log scene parameters, category objects, or both.

        Args:
            mode (str):
                "scene"  -> Only log scene parameters (size, bgcolor, fgcolor).
                category name -> Log all objects in this category.
                "all"    -> Log scene parameters and all objects in all categories.
        """
        lines = []

        def log_scene_params():
            lines.append("=== Scene Parameters ===")
            lines.append(f"Size: {self._fig.scene.get_size()}")
            lines.append(f"Background color: {self._fig.scene.background}")
            lines.append(f"Foreground color: {self._fig.scene.foreground}")
            lines.append("========================")

        if mode == "scene":
            log_scene_params()

        elif mode == "all":
            log_scene_params()
            for category, objs in self.objects.items():
                for obj in objs:
                    # Each object has its own print function
                    # We capture its printed text into logger
                    obj.log_properties(logger=logger)

        elif mode in self.objects:
            for obj in self.objects.get(mode, []):
                obj.log_properties(logger=logger)

        else:
            logger.warning(
                f"Invalid mode '{mode}'. Use 'scene', 'all', or a valid category name."
            )
            return

        logger.info("\n".join(lines))
