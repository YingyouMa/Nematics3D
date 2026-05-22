"""Minimal PyVista demo for testing interactive scalar-bar behavior."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def build_demo_mesh() -> pv.PolyData:
    """Create one simple mesh with a smooth scalar field for the demo."""
    mesh = pv.Sphere(
        radius=1.0,
        theta_resolution=96,
        phi_resolution=96,
    )
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]
    mesh["demo_scalars"] = np.sin(3.0 * x) + np.cos(4.0 * y) + 0.5 * z
    return mesh


def main() -> None:
    """Launch one interactive PyVista window with an interactive scalar bar."""
    mesh = build_demo_mesh()

    plotter = pv.Plotter(window_size=(1280, 900))
    actor = plotter.add_mesh(
        mesh,
        scalars="demo_scalars",
        cmap="viridis",
        smooth_shading=True,
        show_scalar_bar=False,
    )

    plotter.add_scalar_bar(
        title="demo scalar bar",
        mapper=actor.mapper,
        interactive=True,
        vertical=True,
        width=0.12,
        height=0.7,
        position_x=0.82,
        position_y=0.12,
        fmt="%.3f",
        n_labels=6,
        outline=False,
        unconstrained_font_size=True,
    )

    # widget = plotter.scalar_bars._scalar_bar_widgets["demo scalar bar"]
    # representation = widget.GetRepresentation()
    # representation.SetShowBorderToOff()
    # representation.SetShowPolygonToOff()
    # representation.SetShowHorizontalBorder(0)
    # representation.SetShowVerticalBorder(0)

    plotter.add_text(
        "Try dragging and resizing the scalar bar with the widget frame hidden.",
        position="upper_left",
        font_size=12,
    )
    plotter.show()


if __name__ == "__main__":
    main()
