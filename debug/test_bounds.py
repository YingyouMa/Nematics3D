import numpy as np
import pyvista as pv

from Nematics3D.classes.bounds import Bounds, OptsBounds


def test_bounds_clip_geometry():
    bounds = Bounds(
        name="debug-bounds",
        opts=OptsBounds(
            origin=(0.0, 0.0, 0.0),
            axis1=(1.0, 1.0, 0.2),
            axis2=None,
            length1=4.0,
            length2=2.6,
            length3=2.0,
            alignment="center",
        ),
    )

    clip_surface = bounds._entity_clip_geometry

    t = np.linspace(-6.0, 6.0, 400)
    centerline_points = np.column_stack(
        [
            t,
            1.4 * np.sin(1.2 * t),
            0.7 * np.cos(0.7 * t) + 0.35 * np.sin(2.1 * t),
        ]
    )
    centerline = pv.Spline(centerline_points, n_points=1600)
    tube = centerline.tube(radius=0.14, n_sides=24)
    tube_inside = tube.clip_surface(clip_surface, invert=False)
    tube_outside = tube.clip_surface(clip_surface, invert=True)

    sphere = pv.Sphere(
        radius=1.75,
        center=(1.2, 0.4, 0.15),
        theta_resolution=120,
        phi_resolution=120,
    )
    sphere_inside = sphere.clip_surface(clip_surface, invert=False)
    sphere_outside = sphere.clip_surface(clip_surface, invert=True)

    plotter = pv.Plotter()
    plotter.add_mesh(tube_outside, color="#5b6c8f", smooth_shading=True)
    plotter.add_mesh(tube_inside, color="#e85d04", smooth_shading=True)
    plotter.add_mesh(sphere_outside, color="#7a8da3", smooth_shading=True, opacity=0.45)
    plotter.add_mesh(sphere_inside, color="#ff6b35", smooth_shading=True, opacity=0.92)
    plotter.add_mesh(
        clip_surface,
        color="white",
        opacity=0.12,
        show_edges=True,
        edge_color="black",
        line_width=2,
        label="bounds",
    )
    plotter.add_points(
        bounds._entity_corners,
        color="black",
        point_size=12,
        render_points_as_spheres=True,
    )
    plotter.add_legend(
        [
            ["tube outside", "#5b6c8f"],
            ["tube inside", "#e85d04"],
            ["sphere outside", "#7a8da3"],
            ["sphere inside", "#ff6b35"],
            ["bounds", "white"],
        ]
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    test_bounds_clip_geometry()
