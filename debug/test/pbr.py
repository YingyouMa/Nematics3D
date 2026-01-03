"""
PyVista tube + cinematic lighting demo (PBR, shadows, anti-aliasing).

Dependencies
------------
- pyvista
- numpy

Notes
-----
- This example creates a parametric 3D curve, converts it to a tube, then uses
  PBR material properties (metallic/roughness) plus a multi-light rig to achieve
  a more realistic look.
- Some lighting options require an OpenGL-capable backend (typical desktop GPUs).
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


def make_curve(n: int = 800) -> pv.PolyData:
    """
    Create a smooth 3D parametric curve.

    Parameters
    ----------
    n : int, default=800
        Number of sample points along the curve.

    Returns
    -------
    pv.PolyData
        PolyData containing points and a single polyline cell.
    """
    t = np.linspace(0.0, 10.0 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = 0.15 * t + 0.25 * np.sin(2.0 * t)
    pts = np.c_[x, y, z]

    # Build a single polyline cell
    lines = np.hstack(([n], np.arange(n))).astype(np.int64)
    poly = pv.PolyData(pts)
    poly.lines = lines
    return poly


def add_cinematic_lights(plotter: pv.Plotter) -> None:
    """
    Add a simple three-point light rig + a soft rim light.

    Parameters
    ----------
    plotter : pv.Plotter
        The plotter to which lights are added.

    Returns
    -------
    None
    """
    # Key light (main)
    key = pv.Light(position=(3.0, 3.0, 6.0), focal_point=(0.0, 0.0, 2.0))
    key.intensity = 1.0
    key.positional = True

    # Fill light (soften shadows)
    fill = pv.Light(position=(-6.0, 2.0, 3.5), focal_point=(0.0, 0.0, 2.0))
    fill.intensity = 0.45
    fill.positional = True

    # Back/Rim light (edge highlight)
    rim = pv.Light(position=(0.0, -7.0, 6.0), focal_point=(0.0, 0.0, 2.0))
    rim.intensity = 0.65
    rim.positional = True

    # Subtle top light for specular shaping
    top = pv.Light(position=(0.0, 0.0, 12.0), focal_point=(0.0, 0.0, 2.5))
    top.intensity = 0.35
    top.positional = True

    plotter.add_light(key)
    plotter.add_light(fill)
    plotter.add_light(rim)
    plotter.add_light(top)


def main() -> None:
    """
    Render a PBR tube with cinematic lighting.

    Returns
    -------
    None
    """
    pv.set_plot_theme("document")  # neutral defaults

    # --- Geometry: polyline -> tube ---
    curve = make_curve(n=900)
    tube = curve.tube(radius=0.04, n_sides=48)

    # Add some scalar variation to modulate roughness visually (optional)
    # Here we just use z to drive a colormap; you can disable scalars if you want a uniform material.
    z = tube.points[:, 2]
    tube["z"] = (z - z.min()) / (z.max() - z.min() + 1e-12)

    # --- Plotter ---
    pl = pv.Plotter(window_size=(1200, 800))

    # Camera and background (dark background tends to amplify specular realism)
    pl.set_background("black")

    # Lighting: disable default headlight-style lighting and add a rig
    pl.remove_all_lights()
    add_cinematic_lights(pl)

    # Shadows and anti-aliasing
    pl.enable_shadows()
    pl.enable_anti_aliasing("ssaa")  # try "msaa" if ssaa is slow on your GPU

    # PBR material knobs:
    # - metallic: 0 (plastic) -> 1 (metal)
    # - roughness: 0 (mirror) -> 1 (matte)
    # - specular/specular_power also affect highlight behavior (non-PBR-ish but still useful)
    actor = pl.add_mesh(
        tube,
        scalars="z",
        cmap="viridis",
        smooth_shading=True,
        pbr=True,
        metallic=0.35,
        roughness=0.18,
        specular=0.6,
        specular_power=80,
    )

    # Optional: add a faint floor plane to receive shadows (greatly increases perceived realism)
    floor = pv.Plane(center=(0.0, 0.0, -0.5), direction=(0.0, 0.0, 1.0), i_size=8.0, j_size=8.0)
    pl.add_mesh(
        floor,
        color="white",
        pbr=True,
        metallic=0.0,
        roughness=0.9,
        opacity=0.08,
        smooth_shading=True,
    )

    # Camera pose: a mildly oblique view works well for specular cues
    pl.camera_position = [
        (3.5, -4.2, 6.0),  # camera location
        (0.0, 0.0, 2.0),   # focal point
        (0.0, 0.0, 1.0),   # view up
    ]

    pl.show(title="PyVista PBR Tube Lighting Demo", interactive_update=True)
    
    return pl


if __name__ == "__main__":
    pl = main()
