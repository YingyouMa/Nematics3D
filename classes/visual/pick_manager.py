import weakref
import time
import vtk
import numpy as np
from dataclasses import dataclass, field

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import find_nearest_point, closest_point_on_polyline
from Nematics3D.datatypes import (
    as_Number,
    ColorRGB,
    as_ColorRGB,
    )
from ..opts import merge_opts_all


@dataclass(slots=True)
class OptsPickManager:
    double_click_threshold:             float = 0.3
    marker_proximity_threshold:         float = 0.5
    marker_size:                        int = 14
    marker_color:                       ColorRGB = (1, 1, 0)
    marker_font_size:                   int = 14
    sil_color:                          ColorRGB = (0,0,0)
    sil_opacity:                        float = 0.8
    sil_width:                          float = 3
    
    _impl_owner_ref: weakref.ReferenceType | None = field(default=None, init=False, repr=False)
    
    __descriptions__ = {
        "double_click_threshold":       ("The maximum time interval (in seconds)"
                                         " between two consecutive clicks to be registered as a double-click."),
        "marker_proximity_threshold":   ("The minimum distance (in meters) required between two markers"
                                         " to distinguish them as separate locations."),
        "marker_size":                  "Screen-space size (in pixels) of the marker point.",
        "marker_color":                 "RGB color of the marker point",
        "marker_font_size":             "Font size (in pixels) of the numeric label on top of the marker.",
        "sil_color":                    "RGB color of silhouette.",
        "sil_opacity":                  "Opacity of silhouette.",
        "sil_width":                    "Line width of silhouette."
        }
    
    _validators = {
        "double_click_threshold":       lambda v, d: as_Number(v, name=d, replace=0.3),
        "marker_proximity_threshold":   lambda v, d: as_Number(v, name=d, replace=0.5),
        "marker_size":                  lambda v, d: as_Number(v, name=d, replace=14),
        "marker_color":                 lambda v, d: as_ColorRGB(v, name=d, replace=(1, 1, 0)),
        "marker_font_size":             lambda v, d: as_Number(v, name=d, replace=14),
        "sil_color":                    lambda v, d: as_ColorRGB(v, name=d, replace=(0, 0, 0)),
        "sil_opacity":                  lambda v, d: as_Number(v, name=d, value_range=(0,1)),
        "sil_width":                    lambda v, d: as_Number(v, name=d, value_range=(0,np.inf)),
        }
    
    def __setattr__(self, key, value):
        if key in self._validators:
            desc = f'{key!r}: {self.__descriptions__.get(key)}'
            value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)
        
        owner = getattr(self, "_impl_owner_ref", None)
        if owner:
            owner = owner()
            markers = owner._entity_markers
            
            if key == "marker_size":
                for pack in markers:
                    pack["actor"].GetProperty().SetPointSize(value)
                    
            elif key == "marker_color":
                for pack in markers:
                    pack["actor"].GetProperty().SetColor(*value)
                    
            elif key == "marker_font_size":
                for pack in markers:
                    pack["text_actor"].GetTextProperty().SetFontSize(value)
            
            else:
                for glyph in self._impl_registry.values():
                    if hasattr(glyph, '_entity_silhouette') and glyph._entity_silhouette.visibility:
                        if key == "sil_color":
                            glyph._entity_silhouette.prop.color = value
                        if key == "sil_opacity":
                            glyph._entity_silhouette.prop.opacity = value
                        if key == "sil_width":
                            glyph._entity_silhouette.prop.line_width = value
            
            owner.owner.pl.render()
                
            
    



class PickManager:
    """
    A minimal pick manager supporting:
      - Single click: print owner.name only
      - Double click (time-based):
          * If a marker is near the picked point -> delete the nearest marker (no new marker)
          * Else -> add a new numbered marker at the resolved position
            (PlotTube: picked point; PlotSphere/PlotRod: nearest point in owner.raw_coords)
      - Marker labels (2D text) are hidden during camera interaction and shown again after.
    """

    __descriptions__ = {
        "opts": "The OptsPickManager instance controlling behavior.",
        "_impl_owner_ref": (
            "A weak reference to the PlotFigure that owns this pick manager."
        ),
        "_impl_registry": "A registry dict: actor -> visual object",
        "_impl_active_glyphs": "Set of currently active glyphs (multi-selection).",
        
        "_state_pick_count": "Monotonic counter for marker numbering (never decreases).",
        
        "_state_last_click_time": "Last click timestamp (monotonic time) for double-click detection.",
        "_state_last_click_actor": "Last clicked actor for double-click detection.",
        "_state_last_rclick_time": "Last RIGHT click timestamp for right-double-click detection.",
        "_state_last_rclick_actor": "Last RIGHT clicked actor for right-double-click detection.",

        "_entity_markers": (
            "A list of marker packs; each pack holds VTK actors for one overlay point marker."
        ),
    }

    __slots__ = tuple(__descriptions__.keys())+ ("__weakref__",)
    
    def __init__(self, figure, opts: OptsPickManager | None = None, **kwargs):

        object.__setattr__(self, "_impl_owner_ref", weakref.ref(figure))
        object.__setattr__(self, "_impl_registry", {})
        object.__setattr__(self, "_state_pick_count", 0)
        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)
        object.__setattr__(self, "_state_last_rclick_time", None)
        object.__setattr__(self, "_state_last_rclick_actor", None)
        object.__setattr__(self, "_entity_markers", [])
        object.__setattr__(self, "_impl_active_glyphs", [])
        
        if opts is None:
            opts = OptsPickManager()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)
        
        fig = self.owner
        if fig is not None:
            iren = fig.pl.iren.interactor  
            iren.AddObserver("RightButtonPressEvent", self._vtk_on_right_button_press)

    @property
    def owner(self):
        return self._impl_owner_ref()

    # ---------------------------------------------------------------------
    # Registry: actor -> owner (PlotTube / PlotSphere / PlotRod / ...)
    # ---------------------------------------------------------------------
    def act_register(self, actor, owner):
        self._impl_registry[actor] = owner

    def act_unregister(self, actor, logger=None):
        if actor in self._impl_registry:
            del self._impl_registry[actor]


    # ---------------------------------------------------------------------
    # Picking callback
    # ---------------------------------------------------------------------
    def _helper_callback(self, point, picker):

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return

        owner = self._impl_registry[actor]

        now = time.monotonic()
        last_t = self._state_last_click_time
        last_a = self._state_last_click_actor

        # Detect double-click: same actor within a short time window.
        is_double = (
            last_t is not None
            and (actor is last_a)
            and ((now - last_t) <= self.opts.double_click_threshold)
        )

        # Always update last-click state after printing.
        object.__setattr__(self, "_state_last_click_time", now)
        object.__setattr__(self, "_state_last_click_actor", actor)

        # Single click: do nothing.
        if not is_double:
            return

        # Double click: delete nearest marker if close; otherwise add a new marker.
        resolved, msg, _ = owner._helper_resolve_pick(point)
    
        nearest_pack, nearest_d2 = self._helper_find_nearest_marker_pack(resolved)

        # World-space threshold (tune as needed)
        thr = self.opts.marker_proximity_threshold
        if nearest_pack is not None and nearest_d2 is not None and nearest_d2 <= thr:
            self._helper_remove_marker_pack(nearest_pack)
            pos = nearest_pack['world_xyz']
            self.owner.console.println(
                f"remove point #{nearest_pack['id']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
                f"on {str(owner)}"
            )
            self.owner.console.println(msg)

            object.__setattr__(self, "_state_last_click_time", None)
            object.__setattr__(self, "_state_last_click_actor", None)
            return

        # No nearby marker -> add a new marker at resolved position.
        self._helper_add_marker(resolved)
        self.owner.console.println(
            f"picked point #{self._state_pick_count}: ({resolved[0]:.2f}, {resolved[1]:.2f}, {resolved[2]:.2f}) "
            f"on {owner.name!r}"
            )
        self.owner.console.println(msg)

        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)

    # ---------------------------------------------------------------------
    # Marker creation / removal
    # ---------------------------------------------------------------------
    def _helper_create_marker_pack(self):

        fig = self.owner
        if fig is None:
            return None

        # Expect PlotFigure to have overlay renderer prepared (layer=1)
        overlay = getattr(fig, "_entity_overlay", None)
        if overlay is None:
            return None

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(1)
        pts.SetPoint(0, 0.0, 0.0, 0.0)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)

        verts = vtk.vtkCellArray()
        verts.InsertNextCell(1)
        verts.InsertCellPoint(0)
        poly.SetVerts(verts)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetRenderPointsAsSpheres(True)
        actor.GetProperty().SetPointSize(self.opts.marker_size)  # fixed for now
        actor.GetProperty().SetColor(*self.opts.marker_color)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        actor.SetVisibility(False)
        overlay.AddActor(actor)

        text = vtk.vtkTextActor()
        text.GetTextProperty().SetColor(0.0, 0.0, 0.0)  # black digits
        text.GetTextProperty().SetFontSize(self.opts.marker_font_size)          # tune with point size
        text.GetTextProperty().BoldOn()
        text.GetTextProperty().SetJustificationToCentered()
        text.GetTextProperty().SetVerticalJustificationToCentered()
        text.SetVisibility(False)
        overlay.AddActor2D(text)

        pack = {
            "overlay": overlay,
            "pts": pts,
            "poly": poly,
            "actor": actor,
            "text_actor": text,
            "world_xyz": None,
            "id": None,
        }
        return pack

    def _helper_add_marker(self, xyz, marker_id=None):

        pack = self._helper_create_marker_pack()
        if pack is None:
            return

        fig = self.owner
        if fig is None:
            return

        xyz = np.asarray(xyz, dtype=float).reshape(3,)
        pack["world_xyz"] = xyz

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        pack["pts"].SetPoint(0, x, y, z)
        pack["pts"].Modified()
        pack["poly"].Modified()
        pack["actor"].SetVisibility(True)

        if marker_id is None:
            object.__setattr__(self, "_state_pick_count", self._state_pick_count + 1)
            k = self._state_pick_count
            pack["id"] = k
        else:
            pack["id"] = marker_id

        text = pack["text_actor"]
        text.SetInput(str(k))

        self._helper_update_one_marker_label_position(pack)
        text.SetVisibility(True)

        self._entity_markers.append(pack)

        fig.pl.render()

    def _helper_remove_marker_pack(self, pack):

        fig = self.owner
        if fig is None:
            return

        overlay = pack["overlay"]
        overlay.RemoveActor(pack["actor"])
        overlay.RemoveActor2D(pack["text_actor"])
        self._entity_markers.remove(pack)
        
        fig.pl.render()

    # ---------------------------------------------------------------------
    # Marker spatial query (world-space)
    # ---------------------------------------------------------------------
    def _helper_find_nearest_marker_pack(self, p):

        if not self._entity_markers:
            return None, None

        nearest_pack = None
        nearest_d2 = None

        for pack in self._entity_markers:
            xyz0 = pack.get("world_xyz", None)
            if xyz0 is None:
                continue
            d = p - xyz0
            d2 = float(np.dot(d, d))
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest_pack = pack

        return nearest_pack, nearest_d2

    # ---------------------------------------------------------------------
    # Label update / interaction hooks
    # ---------------------------------------------------------------------
    def _helper_update_one_marker_label_position(self, pack):

        xyz = pack.get("world_xyz", None)
        if xyz is None:
            return

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        overlay = pack["overlay"]
        overlay.SetWorldPoint(x, y, z, 1.0)
        overlay.WorldToDisplay()
        dx, dy, _ = overlay.GetDisplayPoint()

        text = pack.get("text_actor", None)
        if text is None:
            return

        text.SetDisplayPosition(int(dx), int(dy))

    def _helper_update_all_marker_labels_position(self):

        for pack in self._entity_markers:
            self._helper_update_one_marker_label_position(pack)

    def _helper_hide_marker_label_during_interaction(self):

        for pack in self._entity_markers:
            text = pack.get("text_actor", None)
            if text is not None and text.GetVisibility():
                text.SetVisibility(False)

    def _helper_show_marker_label_after_interaction(self):

        self._helper_update_all_marker_labels_position()

        for pack in self._entity_markers:
            text = pack.get("text_actor", None)
            if text is not None:
                text.SetVisibility(True)   


    def _vtk_on_right_button_press(self, vtk_iren, _evt):

        fig = self.owner
        if fig is None:
            return

        # 1) pick (actor + world point)
        x, y = vtk_iren.GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)  
        picker.Pick(x, y, 0.0, fig.pl.renderer)

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return
        owner = self._impl_registry[actor]

        # 2) right-double-click detect (time + same actor)
        now = time.monotonic()
        last_t = self._state_last_rclick_time
        last_a = self._state_last_rclick_actor

        is_double = (
            last_t is not None
            and (actor is last_a)
            and ((now - last_t) <= self.opts.double_click_threshold)
        )

        object.__setattr__(self, "_state_last_rclick_time", now)
        object.__setattr__(self, "_state_last_rclick_actor", actor)


        # Once clicked, switch the highlight status
        if hasattr(owner, '_entity_silhouette'):
            if owner._entity_silhouette.visibility == True:
                owner.act_dehighlight()
            else:
                owner.act_highlight(
                    color=self.opts.sil_color,
                    opacity=self.opts.sil_opacity,
                    width=self.opts.sil_width
                    )

        # Single click: print only.
        if not is_double:
            self.owner.console.println(str(owner))
            return

        # 3) on right-double-click
        
        if getattr(owner, "_state_is_interactable", False):
            owner.act_interact()

        # if type(owner).__name__ == "PlotSphere" and getattr(owner, "_state_is_interactable", False):
            
        #     from .qt.interact_sphere import InteractSphere
        #     control = InteractSphere(owner, owner.fig)
        #     control.show()
            
        # elif type(owner).__name__ == "PlotTube" and getattr(owner, "_state_is_interactable", False):
            
        #     owner = owner.owner
        #     if type(owner).__name__ == "DisclinationLineSmoothPlot":
        #         from .qt.interact_disclination_line import InteractDisclinationLine
        #         control = InteractDisclinationLine(owner)
        #         control.show()
                
        # if type(owner).__name__ == "PlotRod" and getattr(owner, "_state_is_interactable", False):
            
        #     from .qt.interact_rod import InteractRod
        #     control = InteractRod(owner, owner.fig)
        #     control.show()
            
        #     figure = owner.fig
        #     owner = owner.owner
        #     if type(owner).__name__ == "QPlane" and getattr(owner, "_state_is_interactable", False):
        #         from .qt.interact_plane import InteractPlane
        #         control = InteractPlane(owner, figure)
        #         control.show()
        #     elif type(owner).__name__ == "QPlanePolar" and getattr(owner, "_state_is_interactable", False):
        #         defectPlane = owner.plane.owner
        #         if type(defectPlane).__name__ == "DefectPlane":
        #             from .qt.interact_defect_plane import InteractDefectPlane
        #             control = InteractDefectPlane(owner, figure)
        #             control.show()
        #             # DEFECT_PLANE_GUI_ONLY_WARNING = (
        #             #     "WARNING!!! \n"
        #             #     "Due to incomplete implementation of the related functionality, "
        #             #     "please modify the DefectPlane ONLY through the GUI during interactive adjustments. "
        #             #     "Do NOT manually change its properties from the command line, "
        #             #     "as this may cause the internal state and the GUI to become out of sync."
        #             # )
        #             # self.owner.console.println(DEFECT_PLANE_GUI_ONLY_WARNING)
        #             # print(DEFECT_PLANE_GUI_ONLY_WARNING)
                
        # if type(owner).__name__ == "PlotSurface" and getattr(owner, "_state_is_interactable", False):
            
        #     from .qt.interact_surface import InteractSurface
        #     control = InteractSurface(owner)
        #     control.show()
            
        #     figure = owner.fig
        #     owner = owner.owner
        #     if type(owner).__name__ in ["QPlane", "InterpolatePlane"] and getattr(owner, "_state_is_interactable", False):
        #         from .qt.interact_plane import InteractPlane
        #         control = InteractPlane(owner, figure)
        #         control.show()
            
            

        # reset to avoid triple-trigger
        object.__setattr__(self, "_state_last_rclick_time", None)
        object.__setattr__(self, "_state_last_rclick_actor", None)




