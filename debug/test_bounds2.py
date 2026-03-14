import numpy as np

from Nematics3D.classes.bounds import Bounds, OptsBounds
from Nematics3D.classes.visual.plot_figure import PlotFigure
from Nematics3D.classes.visual.plot_sphere import OptsSphere, PlotSphere
from Nematics3D.classes.visual.plot_tube import OptsTube, PlotTube


def test_bounds_bind_sequence():
    bounds = Bounds(
        name="debug-bounds-2",
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

    figure = PlotFigure()

    t = np.linspace(-6.0, 6.0, 160)
    tube_coords = np.column_stack(
        [
            t,
            1.4 * np.sin(1.2 * t),
            0.7 * np.cos(0.7 * t) + 0.35 * np.sin(2.1 * t),
        ]
    )
    PlotTube(
        coords=tube_coords,
        name="bounds-tube",
        figure=figure,
        bounds=bounds,
        opts=OptsTube(
            color=(0.88, 0.36, 0.06),
            radius=0.14,
            sides=24,
        ),
    )

    sphere = PlotSphere(
        coords=np.array([[1.2, 0.4, 0.15]], dtype=float),
        name="bounds-sphere",
        figure=figure,
        opts=OptsSphere(
            color=(0.45, 0.56, 0.64),
            radius=1.75,
            sides=48,
        ),
    )
    sphere.act_bind_bounds(bounds)

    figure.pl.add_mesh(
        bounds.clip_geometry,
        color="white",
        opacity=0.12,
        show_edges=True,
        edge_color="black",
        line_width=2,
    )
    figure.pl.add_points(
        bounds.corners,
        color="black",
        point_size=12,
        render_points_as_spheres=True,
    )
    figure.pl.add_axes()
    figure.pl.show_grid()
    figure.pl.show()


if __name__ == "__main__":
    test_bounds_bind_sequence()
