import pyvista as pv
import numpy as np
import vtk


def closest_point_on_polyline(query_pt: np.ndarray, poly_pts: np.ndarray) -> np.ndarray:
    """
    Compute the closest point on a polyline to a specific query point in 3D.

    The algorithm treats the polyline as a series of independent segments,
    projects the query point onto each segment, clips the projection to the
    segment boundaries, and identifies the globally closest result.

    Parameters
    ----------
    query_pt : (3,) array
        Coordinates of the query point (x, y, z).
    poly_pts : (N, 3) array
        Ordered vertices defining the polyline.

    Returns
    -------
    closest : (3,) array
        The coordinates of the point on the polyline closest to query_pt.
    """
    q = np.asarray(query_pt, dtype=float)
    pts = np.asarray(poly_pts, dtype=float)

    if pts.shape[0] == 1:
        return pts[0].copy()

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    aq = q - a

    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.where(ab2 <= 1e-30, 1e-30, ab2)

    t = np.einsum("ij,ij->i", aq, ab) / ab2
    t = np.clip(t, 0.0, 1.0)

    proj = a + ab * t[:, None]
    diff = proj - q
    d2 = np.einsum("ij,ij->i", diff, diff)

    idx = int(np.argmin(d2))
    return proj[idx]


class SingleSilhouette:
    def __init__(self, pl: pv.Plotter, *, color="black", line_width=6, opacity=1.0):
        self.pl = pl

        self._sil = vtk.vtkPolyDataSilhouette()

        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._sil.GetOutputPort())
        self._mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self._mapper)
        self.actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        self.actor.GetProperty().SetLineWidth(line_width)
        self.actor.GetProperty().SetOpacity(opacity)
        self.actor.GetProperty().LightingOff()
        self.actor.SetPickable(False)
        self.actor.SetVisibility(False)

        pl.renderer.AddActor(self.actor)

    def show_for(self, polydata: pv.PolyData):
        # silhouette 对 triangulated surface 更可靠
        surf = polydata.extract_surface().triangulate().clean()

        self._sil.SetCamera(self.pl.renderer.GetActiveCamera())
        self._sil.SetInputData(surf)
        self._sil.Update()

        self.actor.SetVisibility(True)
        self.pl.camera.reset_clipping_range()
        self.pl.render()

    def hide(self):
        self.actor.SetVisibility(False)
        self.pl.render()


def _make_overlay_renderer(pl: pv.Plotter) -> vtk.vtkRenderer:
    """
    Create a foreground overlay renderer (layer=1) sharing the main camera.

    This overlay is drawn after the main renderer, so its actors are not
    occluded by 3D geometry from the main layer.
    """
    rw = pl.render_window

    # Ensure we have at least two layers
    rw.SetNumberOfLayers(2)

    # Main renderer in layer 0 (PyVista uses pl.renderer as the active one)
    pl.renderer.SetLayer(0)

    # Foreground overlay renderer in layer 1, sharing camera
    overlay = vtk.vtkRenderer()
    overlay.SetLayer(1)
    overlay.SetActiveCamera(pl.renderer.GetActiveCamera())
    overlay.SetInteractive(False)

    rw.AddRenderer(overlay)
    return overlay


class OverlayPointMarker:
    """
    A single point marker rendered in an overlay renderer (always on top).

    IMPORTANT:
      VTK will NOT render points unless the PolyData has vertex cells (Verts).
    """

    def __init__(
        self,
        overlay_renderer: vtk.vtkRenderer,
        *,
        point_size: int = 14,
        color="yellow",
    ):
        self.ren = overlay_renderer
        self.point_size = int(point_size)
        self.color = pv.Color(color).float_rgb
        self._actor: vtk.vtkActor | None = None

    def show(self, xyz: np.ndarray):
        xyz = np.asarray(xyz, dtype=float).reshape(3,)

        # 1) vtkPoints
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(1)
        pts.SetPoint(0, float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # 2) vtkPolyData with a single VERTEX cell (this is the key)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)

        verts = vtk.vtkCellArray()
        verts.InsertNextCell(1)
        verts.InsertCellPoint(0)
        poly.SetVerts(verts)

        # 3) mapper/actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetRenderPointsAsSpheres(True)
        actor.GetProperty().SetPointSize(self.point_size)
        actor.GetProperty().SetColor(*self.color)
        actor.GetProperty().LightingOff()
        actor.PickableOff()

        # Replace previous marker actor
        if self._actor is not None:
            self.ren.RemoveActor(self._actor)

        self._actor = actor
        self.ren.AddActor(actor)

    def hide(self):
        if self._actor is not None:
            self.ren.RemoveActor(self._actor)
            self._actor = None

class TogglePickManager:
    """
    Single-selection toggle with a single silhouette actor.

    Adds an always-visible point marker (overlay layer) for:
      - tube: marker at snapped point on centerline, and prints snapped point.
      - sphere: marker at sphere center, and prints center.

    Toggle rules:
      - click selected object again: deselect (hide silhouette + marker)
      - click another registered object: switch selection
      - click blank/unregistered: deselect
    """

    def __init__(
        self,
        pl: pv.Plotter,
        *,
        sil_color="black",
        sil_line_width=6,
        marker_point_size=14,
        marker_color="yellow",
    ):
        self.pl = pl
        self.sil = SingleSilhouette(pl, color=sil_color, line_width=sil_line_width)

        self._registry: dict[object, dict] = {}   # actor -> record
        self._active_actor = None

        # Foreground overlay marker (never occluded)
        self._overlay = _make_overlay_renderer(pl)
        self.marker = OverlayPointMarker(
            self._overlay,
            point_size=marker_point_size,
            color=marker_color,
        )

    def register_tube(self, actor, tube_mesh: pv.PolyData, *, centerline_points: np.ndarray):
        self._registry[actor] = {
            "kind": "tube",
            "mesh": tube_mesh,  # for silhouette
            "centerline_points": np.asarray(centerline_points, dtype=float),
        }

    def register_sphere(self, actor, sphere_mesh: pv.PolyData):
        self._registry[actor] = {
            "kind": "sphere",
            "mesh": sphere_mesh,  # for silhouette
        }

    def clear(self):
        self._active_actor = None
        self.sil.hide()
        self.marker.hide()
        self.pl.render()

    def _select(self, actor, picked_point):
        rec = self._registry[actor]
        self._active_actor = actor

        # Always show silhouette for the selected object
        self.sil.show_for(rec["mesh"])

        kind = rec["kind"]

        if kind == "tube":
            snapped = closest_point_on_polyline(
                query_pt=np.asarray(picked_point, dtype=float),
                poly_pts=rec["centerline_points"],
            )

            # Marker is rendered in overlay layer -> never occluded
            self.marker.show(snapped)

            print(f"Snapped point on centerline: {tuple(float(x) for x in snapped)}")

        elif kind == "sphere":
            c = rec["mesh"].center
            self.marker.show(np.asarray(c, dtype=float))
            print(f"Sphere center: {tuple(float(x) for x in c)}")

        self.pl.render()

    def callback(self, point, picker):
        actor = picker.GetActor() if picker is not None else None

        # Click blank or non-registered -> clear
        if actor is None or actor not in self._registry:
            # self.clear()
            return

        # Click the same active object -> toggle off
        if actor is self._active_actor:
            self.clear()
            return

        # Switch to the new object
        self._select(actor, point)


# ---------------- demo ----------------
pl = pv.Plotter()
pl.add_axes()

pm = TogglePickManager(
    pl,
    sil_color="black",
    sil_line_width=6,
    marker_point_size=18,
    marker_color="yellow",
)

# Tube object: need BOTH tube mesh and its centerline points for snapping
center_pts = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]], dtype=float)
tube = pv.MultipleLines(center_pts).tube(radius=0.03, n_sides=24)
a_tube = pl.add_mesh(tube, color="tomato", name="tube1")
pm.register_tube(a_tube, tube, centerline_points=center_pts)

# Sphere object: print center and mark center
sph = pv.Sphere(radius=0.12, center=(1.2, 0.3, 0.2))
a_sph = pl.add_mesh(sph, color="deepskyblue", name="sphere1")
pm.register_sphere(a_sph, sph)

pl.enable_point_picking(
    callback=pm.callback,
    left_clicking=True,
    pickable_window=True,
    use_picker=True,
    show_point=False,
    picker="cell",
    tolerance=0.03,
    show_message=False,
)

pl.show(interactive_update=True)
