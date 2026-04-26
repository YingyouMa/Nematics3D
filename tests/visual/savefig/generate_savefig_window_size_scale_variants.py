import json
import sys
from itertools import product
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_DIR / "src"
DATA_DIR = REPO_DIR / "example" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

sys.path.insert(0, str(SRC_DIR))

import nematics3d


INITIAL_WINDOW_SIZES = (
    (900, 600),
    (1200, 800),
)
SAVE_WINDOW_SIZES = (
    (900, 600),
    (1400, 900),
)
SAVE_SCALES = (
    1,
    1.5,
    2,
    3,
)
IS_OFF_SCREEN_OPTIONS = (
    False,
    True,
)


def load_informative_qfield():
    n = np.load(DATA_DIR / "n_example_global.npy")[:60, :60, :60]
    s = np.load(DATA_DIR / "S_example_global.npy")[:60, :60, :60]

    q_field = nematics3d.QFieldObject(S=s, n=n, name="savefig_test_q")
    q_field.act_lines_smooth()
    return q_field


def build_informative_figure(q_field, *, initial_window_size, is_off_screen):
    mode = "offscreen" if is_off_screen else "interactive"
    size_label = format_size(initial_window_size)
    figure = nematics3d.PlotFigure(
        name=f"savefig_{mode}_{size_label}",
        is_off_screen=is_off_screen,
        size=initial_window_size,
    )

    q_field.act_visualize_disclination_lines(
        figure=figure,
        line_color=(0.5, 0.5, 0.5),
        line_radius=0.3,
    )

    spacing = 3
    trans = 6
    q_field.act_visualize_n_plane(
        figure=figure,
        is_extent=False,
        grid_normal=(1, 1, 1),
        grid_origin=(
            30 - trans,
            30 - trans,
            30 - trans,
        ),
        grid_size=100,
        grid_spacing=spacing,
        n_length=spacing,
        plane_name=f"n-plane-{mode}-{size_label}",
    )

    figure.act_commit(elevation=0, azimuth=90, distance=150)
    return figure


def format_size(size):
    width, height = size
    return f"{width}x{height}"


def format_scale(scale):
    return str(scale).replace(".", "p")


def render_savefig_variants():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    q_field = load_informative_qfield()
    manifest = []

    figure_configs = product(INITIAL_WINDOW_SIZES, IS_OFF_SCREEN_OPTIONS)
    save_configs = tuple(product(SAVE_WINDOW_SIZES, SAVE_SCALES))

    for initial_window_size, is_off_screen in figure_configs:
        figure = build_informative_figure(
            q_field,
            initial_window_size=initial_window_size,
            is_off_screen=is_off_screen,
        )

        try:
            for save_window_size, save_scale in save_configs:
                mode = "offscreen" if is_off_screen else "interactive"
                filename = (
                    f"savefig_{mode}"
                    f"_initial_{format_size(initial_window_size)}"
                    f"_save_{format_size(save_window_size)}"
                    f"_scale_{format_scale(save_scale)}.png"
                )
                path = OUTPUT_DIR / filename

                figure.act_savefig(
                    path,
                    window_size=save_window_size,
                    scale=save_scale,
                )

                manifest.append(
                    {
                        "path": str(path.relative_to(REPO_DIR)),
                        "is_off_screen": is_off_screen,
                        "initial_window_size": list(initial_window_size),
                        "save_window_size": list(save_window_size),
                        "save_scale": save_scale,
                    }
                )
        finally:
            figure.act_close()

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


if __name__ == "__main__":
    variants = render_savefig_variants()
    print(f"Saved {len(variants)} images to {OUTPUT_DIR}")
