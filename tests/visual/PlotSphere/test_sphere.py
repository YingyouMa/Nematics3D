import sys
from pathlib import Path
import types

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_sphere import OptsSphere, PlotSphere

#!!! current version intentionally avoids scalar bar rendering while the scalar bar API is still being cleaned up

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PATH_IMAGE_OFFSCREEN = OUTPUT_DIR / "test_sphere_offscreen.png"
PATH_IMAGE_INTERACTIVE = OUTPUT_DIR / "test_sphere_interactive.png"


def build_path(offset_y=0.0):
    z_lower = np.linspace(0, 10, 50)
    z_upper = np.linspace(15, 20, 25)

    lower_segment = np.column_stack(
        (np.sin(z_lower), np.cos(z_lower) + offset_y, z_lower)
    )
    upper_segment = np.column_stack(
        (np.sin(z_upper), np.cos(z_upper) + offset_y, z_upper)
    )

    return np.concatenate([lower_segment, upper_segment])


def radius_wave(coords):
    return 0.1 + 0.2 * np.abs(np.sin(coords[:, 2]))


def color_gradient(coords):
    z_coords = coords[:, 2]
    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    return np.column_stack((z_norm, np.zeros_like(z_norm), 1 - z_norm))


def opacity_wave(coords):
    return np.abs(np.sin(coords[:, 2]))


def radius_percent(u_percent):
    phase = np.pi * u_percent / 100.0
    return 0.12 + 0.08 * (0.5 + 0.5 * np.sin(phase))


def color_percent(u_percent):
    t = u_percent / 100.0
    return np.column_stack((0.15 + 0.75 * t, 0.35 + 0.25 * np.cos(np.pi * t), 1.0 - t))


def manual_color_palette(coords):
    z_coords = coords[:, 2]
    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    return np.column_stack(
        (
            0.25 + 0.55 * z_norm,
            0.85 - 0.45 * z_norm,
            0.35 + 0.25 * np.cos(z_coords / 3.0),
        )
    )


def scalar_profile(coords):
    return 0.3 + 0.15 * np.cos(coords[:, 2] / 2.0)


def opacity_profile(coords):
    z_coords = coords[:, 2]
    return 0.35 + 0.55 * (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())


def build_sphere_gallery(figure):
    coords_baseline = build_path(offset_y=0.0)
    coords_function = build_path(offset_y=5.0)
    coords_palette = build_path(offset_y=10.0)
    coords_scalar_function = build_path(offset_y=15.0)
    coords_scalar_array = build_path(offset_y=20.0)

    spheres1 = PlotSphere(
        figure=figure,
        coords=coords_baseline,
        name="spheres_solid_blue",
        color=(0, 0, 1),
        radius=0.3,
        sides=12,
    )
    spheres1.act_commit(color=(0.1, 0.35, 0.95), radius=0.24, opacity=0.85, sides=14)

    spheres2 = PlotSphere(
        figure=figure,
        coords=coords_function,
        name="spheres_function_driven",
        color=color_gradient,
        radius=radius_wave,
        opacity=opacity_wave,
        resolver_source="coords",
    )
    spheres2.opts.specular = 0.55
    spheres2.opts.specular_power = 35
    spheres2.opts.diffuse = 0.85

    opts3 = OptsSphere(
        color=manual_color_palette(coords_palette),
        radius=np.linspace(0.08, 0.22, len(coords_palette)),
        opacity=opacity_profile(coords_palette),
        shading_type="pbr",
        metallic=1,
        roughness=0.4,
        sides=18,
    )
    spheres3 = PlotSphere(
        figure=figure,
        coords=coords_palette,
        name="spheres_manual_palette",
        opts=opts3,
    )

    opts3b = OptsSphere(
        color=np.column_stack(
            (
                np.linspace(0.9, 0.2, len(coords_palette)),
                np.linspace(0.2, 0.7, len(coords_palette)),
                np.full(len(coords_palette), 0.45),
            )
        ),
        radius=0.12,
        opacity=0.95,
        shading_type="phong",
        ambient=0.25,
        diffuse=0.9,
        specular=0.1,
    )
    spheres3.act_commit(opts=opts3b)

    opts4 = OptsSphere(
        resolver_source="u_percent",
        color=color_percent,
        radius=radius_percent,
        opacity=lambda u: 0.4 + 0.6 * np.sin(np.pi * u / 100.0) ** 2,
        sides=10,
    )
    spheres4 = PlotSphere(
        figure=figure,
        coords=coords_scalar_function,
        name="spheres_scalar_function",
        opts=opts4,
        sides=22,
    )
    spheres4.opts.opacity = lambda u: 0.25 + 0.75 * (u / 100.0)
    spheres4.act_commit(radius=lambda u: 0.1 + 0.06 * np.cos(np.pi * u / 100.0) ** 2)

    opts5 = OptsSphere(
        paint_by="scalars",
        scalars=scalar_profile(coords_scalar_array),
        scalars_cmap="plasma",
        scalars_clim=(0.15, 0.45),
        scalar_bar_title="scalar_profile",
        is_scalar_bar=False,
        radius=0.22,
        opacity=np.linspace(0.45, 1.0, len(coords_scalar_array)),
    )
    spheres5 = PlotSphere(
        figure=figure,
        coords=coords_scalar_array,
        name="spheres_scalar_array",
        category="sphere_scalar_demo",
        opts=opts5,
        shading_type="pbr",
        metallic=1,
        roughness=0.35,
        is_pickable=False,
        opts_defaults_override={"is_reset_camera": False, "sides": 20},
    )
    spheres5.act_commit(scalars=lambda pts: 0.2 + 0.18 * np.sin(pts[:, 2] / 2.5))
    spheres5.opts.scalars_clim = (0.02, 0.38)
    spheres5.opts.opacity = np.linspace(0.55, 0.95, len(coords_scalar_array))

    assert len(figure.pl.scalar_bars) == 0

    return {
        "figure": figure,
        "spheres1": spheres1,
        "spheres2": spheres2,
        "spheres3": spheres3,
        "spheres4": spheres4,
        "spheres5": spheres5,
    }


def save_gallery_snapshot(figure, path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.act_view_isometric()
    window_size = tuple(int(x) for x in figure.opts.size)
    figure.act_savefig(path, window_size=window_size)
    return path


def render_gallery_variants():
    figure_offscreen = PlotFigure(
        is_off_screen=True,
        name="test_sphere_gallery_offscreen",
        size=(2200, 1200),
    )
    scene_offscreen = build_sphere_gallery(figure_offscreen)
    path_offscreen = save_gallery_snapshot(figure_offscreen, PATH_IMAGE_OFFSCREEN)

    figure_interactive = PlotFigure(
        name="test_sphere_gallery_interactive",
        size=(2200, 1200),
    )
    scene_interactive = build_sphere_gallery(figure_interactive)
    path_interactive = save_gallery_snapshot(figure_interactive, PATH_IMAGE_INTERACTIVE)

    return {
        "offscreen": {
            "path": path_offscreen,
            "scene": scene_offscreen,
        },
        "interactive": {
            "path": path_interactive,
            "scene": scene_interactive,
        },
    }


if __name__ == "__main__":
    GALLERIES = render_gallery_variants()
