import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nematics3d


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    level = 0.4
    S = np.load(DATA_DIR / "S_example_global.npy")

    contour = nematics3d.ContourSurfaceSet(
        S,
        levels=(level,),
        name="S_example_global_contour",
    )
    mesh = contour.act_extract_surface_by_level(level)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    focal_point = mesh.center
    extent = np.array(
        [
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ],
        dtype=float,
    )
    distance = 1.8 * float(np.linalg.norm(extent))

    figure = nematics3d.PlotFigure(
        name="S contour surface",
        is_off_screen=False,
        bg_color=(0.98, 0.98, 0.99),
    )

    contour.act_plot_surface_by_level(
        level,
        figure=figure,
        color=(0.15, 0.45, 0.85),
        opacity=0.92,
        ambient=0.35,
        diffuse=0.7,
        specular=0.1,
        line_width=1.0,
        is_show_edges=False,
    )

    figure.act_commit(
        focal_point=focal_point,
        azimuth=35,
        elevation=20,
        distance=distance,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "S_example_global_contour_level_0p2.png"
    figure.act_savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
