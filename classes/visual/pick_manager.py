import weakref
import time
import vtk
import pyvista as pv
import numpy as np

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import find_nearest_point


class PickManager:
    """
    A minimal pick manager supporting:
      - Single click: print owner.opts.name only
      - Double click (time-based):
          * If a marker is near the picked point -> delete the nearest marker (no new marker)
          * Else -> add a new numbered marker at the resolved position
            (PlotTube: picked point; PlotSphere/PlotRod: nearest point in owner.raw_coords)
      - Marker labels (2D text) are hidden during camera interaction and shown again after.
    """

    __descriptions__ = {
        "_internal_owner_ref": (
            "A weak reference to the PlotFigure that owns this pick manager."
        ),
        "_internal_registry": "A registry dict: actor -> visual object",

        "_state_pick_count": "Monotonic counter for marker numbering (never decreases).",
        "_state_last_click_time": "Last click timestamp (monotonic time) for double-click detection.",
        "_state_last_click_actor": "Last clicked actor for double-click detection.",

        "_entity_markers": (
            "A list of marker packs; each pack holds VTK actors for one overlay point marker."
        ),
    }

    __slots__ = tuple(__descriptions__.keys())
    
    def __init__(self, figure, logger=None):

        object.__setattr__(self, "_internal_owner_ref", weakref.ref(figure))
        object.__setattr__(self, "_internal_registry", {})
        object.__setattr__(self, "_state_pick_count", 0)
        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)
        object.__setattr__(self, "_entity_markers", [])

    @property
    def figure(self):
        return self._internal_owner_ref()

    # ---------------------------------------------------------------------
    # Registry: actor -> owner (PlotTube / PlotSphere / PlotRod / ...)
    # ---------------------------------------------------------------------
    def act_register(self, actor, owner):
        self._internal_registry[actor] = owner

    def act_unregister(self, actor, logger=None):
        if actor in self._internal_registry:
            del self._internal_registry[actor]


    # ---------------------------------------------------------------------
    # Picking callback
    # ---------------------------------------------------------------------
    def _helper_callback(self, point, picker):

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._internal_registry:
            return

        owner = self._internal_registry[actor]

        now = time.monotonic()
        last_t = self._state_last_click_time
        last_a = self._state_last_click_actor

        # Detect double-click: same actor within a short time window.
        is_double = (
            last_t is not None
            and (actor is last_a)
            and ((now - last_t) <= 0.30)
        )

        # Always update last-click state after printing.
        object.__setattr__(self, "_state_last_click_time", now)
        object.__setattr__(self, "_state_last_click_actor", actor)

        # Single click: print only.
        if not is_double:
            print(owner.opts.name)
            return

        # Double click: delete nearest marker if close; otherwise add a new marker.
        resolved = self._helper_resolve_marker_pos(owner, point)
        if resolved is None:
            return
        
        nearest_pack, nearest_d2 = self._helper_find_nearest_marker_pack(resolved)

        # World-space threshold (tune as needed)
        thr = 0.5
        if nearest_pack is not None and nearest_d2 is not None and nearest_d2 <= thr:
            self._helper_remove_marker_pack(nearest_pack)

            # Clear state to avoid chained double-click detections.
            object.__setattr__(self, "_state_last_click_time", None)
            object.__setattr__(self, "_state_last_click_actor", None)
            return

        # No nearby marker -> add a new marker at resolved position.
        self._helper_add_marker(resolved)

        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)

    # ---------------------------------------------------------------------
    # Marker creation / removal
    # ---------------------------------------------------------------------
    def _helper_create_marker_pack(self):

        fig = self.figure
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
        actor.GetProperty().SetPointSize(14)  # fixed for now
        actor.GetProperty().SetColor(*pv.Color("yellow").float_rgb)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        actor.SetVisibility(False)
        overlay.AddActor(actor)

        text = vtk.vtkTextActor()
        text.GetTextProperty().SetColor(0.0, 0.0, 0.0)  # black digits
        text.GetTextProperty().SetFontSize(14)          # tune with point size
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

    def _helper_add_marker(self, xyz):

        pack = self._helper_create_marker_pack()
        if pack is None:
            return

        fig = self.figure
        if fig is None:
            return

        xyz = np.asarray(xyz, dtype=float).reshape(3,)
        pack["world_xyz"] = xyz

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        pack["pts"].SetPoint(0, x, y, z)
        pack["pts"].Modified()
        pack["poly"].Modified()
        pack["actor"].SetVisibility(True)

        object.__setattr__(self, "_state_pick_count", self._state_pick_count + 1)
        k = self._state_pick_count
        pack["id"] = k

        text = pack["text_actor"]
        text.SetInput(str(k))

        self._helper_update_one_marker_label_position(pack)
        text.SetVisibility(True)

        self._entity_markers.append(pack)

        fig.pl.render()

    def _helper_remove_marker_pack(self, pack):

        fig = self.figure
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
                
                
    def _helper_resolve_marker_pos(self, owner, picked_point):
        p = np.asarray(picked_point, dtype=float).reshape(3,)
    
        name = type(owner).__name__
    
        if name in ("PlotTube",):
            return p
    
        if name in ("PlotSphere", "PlotRod"):
            return find_nearest_point(p, owner.raw_coords)
    
        return None
