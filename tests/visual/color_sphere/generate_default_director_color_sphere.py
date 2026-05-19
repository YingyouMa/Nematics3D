import sys
from pathlib import Path
import types

import numpy as np
from PIL import Image, ImageChops

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_surface import PlotSurface
from nematics3d.classes.visual.plot_vector import PlotVector
from nematics3d.field import n_color_immerse

OUTPUT_DIR = Path(__file__).resolve().parent
PATH_IMAGE = OUTPUT_DIR / "default_director_color_sphere.png"


def _normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def director_colors_from_coords(coords):
    director = _normalize_rows(coords)
    return np.asarray(n_color_immerse(director), dtype=float)


def crop_white_margins(path, *, padding=80):
    image = Image.open(path).convert("RGB")
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, white_bg)
    bbox = diff.getbbox()
    if bbox is None:
        image.save(path)
        return path

    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.size[0], bbox[2] + padding)
    bottom = min(image.size[1], bbox[3] + padding)
    image.crop((left, top, right, bottom)).save(path)
    return path


def build_cube_sphere_patch(face_name, *, radius=1.0, samples=33):
    u = np.linspace(-1.0, 1.0, samples)
    v = np.linspace(-1.0, 1.0, samples)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    if face_name == "+x":
        xyz = np.stack([np.ones_like(uu), uu, vv], axis=-1)
    elif face_name == "-x":
        xyz = np.stack([-np.ones_like(uu), uu, vv], axis=-1)
    elif face_name == "+y":
        xyz = np.stack([uu, np.ones_like(uu), vv], axis=-1)
    elif face_name == "-y":
        xyz = np.stack([uu, -np.ones_like(uu), vv], axis=-1)
    elif face_name == "+z":
        xyz = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    elif face_name == "-z":
        xyz = np.stack([uu, vv, -np.ones_like(uu)], axis=-1)
    else:
        raise ValueError(f"Unsupported face name: {face_name!r}")

    xyz = xyz.reshape(-1, 3)
    xyz = radius * _normalize_rows(xyz)
    return xyz


def build_director_color_sphere(
    *,
    radius=1.0,
    patch_samples=33,
    axis_length=1.55,
):
    figure = PlotFigure(
        is_off_screen=True,
        name="default_director_color_sphere",
        size=(1800, 1800),
    )

    face_names = ("+x", "-x", "+y", "-y", "+z", "-z")
    surfaces = []
    for face_name in face_names:
        coords = build_cube_sphere_patch(
            face_name,
            radius=radius,
            samples=patch_samples,
        )
        surface = PlotSurface(
            coords=coords,
            figure=figure,
            name=f"sphere_patch_{face_name}",
            category="director_color_sphere",
            color=director_colors_from_coords(coords),
            opacity=1.0,
            ambient=0.32,
            diffuse=0.82,
            specular=0.18,
            specular_power=18,
            is_pickable=False,
            is_reset_camera=False,
        )
        surfaces.append(surface)

    axis_orient = axis_length * np.eye(3)
    axis_colors = np.asarray(n_color_immerse(np.eye(3)), dtype=float)
    axes = PlotVector(
        coords=np.zeros((3, 3), dtype=float),
        orient=axis_orient,
        figure=figure,
        name="director_axes",
        category="director_color_axes",
        color=axis_colors,
        opacity=1.0,
        length=lambda orient_length: orient_length,
        radius=0.04 * radius,
        tip_length_fraction=0.24,
        tip_radius_ratio=2.9,
        anchor="tail",
        sides=20,
        ambient=0.25,
        diffuse=0.88,
        specular=0.12,
        is_pickable=False,
        is_reset_camera=False,
    )

    figure.pl.view_vector((1.7, 1.55, 1.2), viewup=(0.0, 0.0, 1.0))
    figure.pl.camera.zoom(0.85)
    figure.pl.render()

    return {
        "figure": figure,
        "surfaces": surfaces,
        "axes": axes,
    }


def save_director_color_sphere(path=PATH_IMAGE):
    scene = build_director_color_sphere()
    figure = scene["figure"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.act_savefig(
        path, scale=2, window_size=tuple(int(x) for x in figure.opts.size)
    )
    figure.act_close()
    crop_white_margins(path)

    return path


if __name__ == "__main__":
    path = save_director_color_sphere()
    print(f"Saved figure to: {path}")
