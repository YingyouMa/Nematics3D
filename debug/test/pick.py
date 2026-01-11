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
    # Ensure inputs are floating-point numpy arrays for calculation
    q = np.asarray(query_pt, dtype=float)
    pts = np.asarray(poly_pts, dtype=float)

    # Handle edge case: polyline consists of only a single point
    if pts.shape[0] == 1:
        return pts[0].copy()

    # 1. Define segments
    # a: Start points of all segments (indices 0 to N-2)
    # b: End points of all segments (indices 1 to N-1)
    a = pts[:-1]
    b = pts[1:]
    
    # Vector from segment start to end (ab) and from start to query point (aq)
    ab = b - a
    aq = q - a

    # 2. Compute squared length of each segment (||ab||^2)
    # Uses Einstein summation to compute row-wise dot products efficiently
    ab2 = np.einsum("ij,ij->i", ab, ab)
    
    # Avoid division by zero for degenerate segments (where distance is effectively zero)
    ab2 = np.where(ab2 <= 1e-30, 1e-30, ab2)

    # 3. Calculate the projection parameter 't'
    # t represents the normalized distance along the vector ab:
    # t = (aq · ab) / ||ab||^2
    t = np.einsum("ij,ij->i", aq, ab) / ab2
    
    # 4. Clamp t to the range [0.0, 1.0]
    # If t < 0, the closest point is the start (a).
    # If t > 1, the closest point is the end (b).
    t = np.clip(t, 0.0, 1.0)

    # 5. Calculate the candidate points on each segment
    # proj = a + t * (b - a)
    proj = a + ab * t[:, None]
    
    # 6. Find the candidate point with the minimum distance to the query point
    # Calculate squared Euclidean distance: d^2 = ||proj - q||^2
    diff = proj - q
    d2 = np.einsum("ij,ij->i", diff, diff)
    
    # Select the point index with the smallest distance and return its coordinates
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
        
class PointMarker:
    def __init__(self, pl: pv.Plotter, *, name="pick_marker", point_size=14, color="yellow"):
        self.pl = pl
        self.name = name
        self.point_size = int(point_size)
        self.color = color
        self._actor = None

    def show(self, xyz):
        xyz = np.asarray(xyz, dtype=float).reshape(1, 3)
        poly = pv.PolyData(xyz)

        # 移除旧 marker（保证只有一个）
        self.pl.remove_actor(self.name, reset_camera=False)

        self._actor = self.pl.add_points(
            poly,
            color=self.color,
            point_size=self.point_size,
            render_points_as_spheres=True,  # 更显眼
            pickable=False,
            name=self.name,
        )

        # ⭐ 关键：关闭深度测试，保证永远在最前
        self._actor.GetProperty().DepthTestOff()

        self.pl.render()

    def hide(self):
        self.pl.remove_actor(self.name, reset_camera=False)
        self.pl.render()


class TogglePickManager:
    """
    Single-selection toggle with a single silhouette actor.
    Adds a snapped point marker only for tubes.

    Registration supports two kinds:
      - kind="tube": requires `centerline_points` for snapping.
      - kind="sphere": prints sphere center.
    """
    def __init__(
        self,
        pl: pv.Plotter,
        *,
        sil_color="black",
        sil_line_width=6,
        marker_name="pick_marker",
        marker_radius=0.06,
        marker_color="yellow",
    ):
        self.pl = pl
        self.sil = SingleSilhouette(pl, color=sil_color, line_width=sil_line_width)

        self._registry: dict[object, dict] = {}   # actor -> record
        self._active_actor = None

        # self.marker_name = marker_name
        # self.marker_radius = float(marker_radius)
        # self.marker_color = marker_color
        
        self.marker = PointMarker(pl, name="pick_marker", point_size=marker_radius, color=marker_color)

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

    # def _remove_marker(self):
    #     self.pl.remove_actor(self.marker_name, reset_camera=False)

    def clear(self):
        self._active_actor = None
        self.sil.hide()
        # self._remove_marker()
        self.marker.hide()
        self.pl.render()

    def _select(self, actor, picked_point):
        rec = self._registry[actor]
        self._active_actor = actor

        # Always show silhouette for the selected object
        self.sil.show_for(rec["mesh"])

        # Per-kind behavior
        kind = rec["kind"]

        if kind == "tube":
            # Snap picked point to the centerline polyline
            snapped = closest_point_on_polyline(
                query_pt=np.asarray(picked_point, dtype=float),
                poly_pts=rec["centerline_points"],
            )

            # Draw a marker at the snapped point
            self.marker.show(snapped)
            # self._remove_marker()
            # marker = pv.Sphere(radius=self.marker_radius, center=snapped)
            # self.pl.add_mesh(marker, color=self.marker_color, name=self.marker_name)

            print(f"Snapped point on centerline: {tuple(float(x) for x in snapped)}")

        else:
            # Not a tube -> no marker
            # self._remove_marker()

            if kind == "sphere":
                center = rec["mesh"].center
                self.marker.show(center)
                print(f"Sphere center: {tuple(float(x) for x in center)}")

        self.pl.render()

    def callback(self, point, picker):
        actor = picker.GetActor() if picker is not None else None

        # Click blank or non-registered -> clear
        if actor is None or actor not in self._registry:
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

pm = TogglePickManager(pl, sil_color="black", sil_line_width=6, marker_radius=0.06)

# Tube object: need BOTH tube mesh and its centerline points for snapping
center_pts = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]], dtype=float)
tube = pv.MultipleLines(center_pts).tube(radius=0.03, n_sides=24)
a_tube = pl.add_mesh(tube, color="tomato", name="tube1")
pm.register_tube(a_tube, tube, centerline_points=center_pts)

# Sphere object: print center
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
