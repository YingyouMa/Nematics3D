import weakref
import vtk
import pyvista as pv
import numpy as np

from Nematics3D.logging_decorator import logging_and_warning_decorator


class PickManager:


    __descriptions__ = {
        "_internal_owner_ref": (
            "A weak reference to the PlotFigure that owns this pick manager."
        ),
        "_internal_registry": "A registry dict: actor -> visual object",
        "_state_active_actor": "The currently active (selected) actor.",
        "_entity_marker_pack": (
            "A packed dict holding VTK objects for a single overlay point marker."
        ),
    }

    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(self, figure, logger=None):

        object.__setattr__(self, "_internal_owner_ref", weakref.ref(figure))
        object.__setattr__(self, "_internal_registry", {})
        object.__setattr__(self, "_state_active_actor", None)
        object.__setattr__(self, "_entity_marker_pack", None)

    @property
    def figure(self):
        return self._internal_owner_ref()

    def act_register(self, actor, owner):
        self._internal_registry[actor] = owner

    @logging_and_warning_decorator(start_finish_level=5)
    def act_unregister(self, actor, logger=None):
        if actor in self._internal_registry:
            del self._internal_registry[actor]
        if actor is self._state_active_actor:
            object.__setattr__(self, "_state_active_actor", None)

    def act_clear_active(self):
        object.__setattr__(self, "_state_active_actor", None)
        self._helper_hide_marker()

    def act_get_owner(self, actor):
        return self._internal_registry.get(actor, None)
    
        

    def _helper_select_actor(self, actor, picked_point):
        object.__setattr__(self, "_state_active_actor", actor)

        owner = self._internal_registry.get(actor, None)
        if owner is None:
            return

        # Only PlotTube leaves a marker (no extra behaviors for others yet)
        if type(owner).__name__ == "PlotTube":
            p = np.asarray(picked_point, dtype=float).reshape(3,)
            self._helper_show_marker(p)
        else:
            self._helper_hide_marker()
            

    @logging_and_warning_decorator(start_finish_level=0)
    def _helper_callback(self, point, picker, logger=None):
        
        actor = picker.GetActor() if picker is not None else None

        if actor is None or actor not in self._internal_registry:
            return

        if actor is self._state_active_actor:
            self.act_clear_active()
            return

        if self._state_active_actor is not None:
            self.act_clear_active()

        self._helper_select_actor(actor, point)
    
    
    
    # ---------------------------------------------------------------------
    # Marker (overlay point) - created lazily, single instance
    # ---------------------------------------------------------------------
    def _helper_get_marker_pack(self):
        pack = self._entity_marker_pack
        if pack is not None:
            return pack

        fig = self.figure
        if fig is None:
            return None

        # Expect PlotFigure to have overlay renderer prepared (layer=1)
        ren = getattr(fig, "_entity_overlay", None)
        if ren is None:
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

        ren.AddActor(actor)

        pack = {"ren": ren, "pts": pts, "poly": poly, "actor": actor}
        object.__setattr__(self, "_entity_marker_pack", pack)
        return pack
    
    def _helper_show_marker(self, xyz):
        pack = self._helper_get_marker_pack()
        if pack is None:
            return

        x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        pack["pts"].SetPoint(0, x, y, z)
        pack["pts"].Modified()
        pack["poly"].Modified()
        pack["actor"].SetVisibility(True)

        fig = self.figure
        if fig is not None:
            fig.pl.render()

    def _helper_hide_marker(self):
        pack = self._entity_marker_pack
        if pack is None:
            return
        pack["actor"].SetVisibility(False)

        fig = self.figure
        if fig is not None:
            fig.pl.render()
