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

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.classes.plane_grid import OptsPlaneGrid
from nematics3d.classes.vector_plane import VectorPlane
from nematics3d.classes.visual.plot_figure import PlotFigure


DATA_DIR = Path(__file__).resolve().parent
VELOCITY_SHAPE = (128, 128, 128)
VELOCITY_STRIDE_XY = 4

VELOCITY_DATASET = None
VELOCITY_FIELD = None
VELOCITY_INTERPOLATOR = None
VELOCITY_PLANE = None
VELOCITY_FIGURE = None
VELOCITY_VISUAL = None
VELOCITY_VALUES = None
VELOCITY_SPEED = None
VELOCITY_SCENE_MID_Z = None


def _load_velocity_components():
    ux = np.fromfile(DATA_DIR / "ux_1.dat", dtype=np.float64).reshape(VELOCITY_SHAPE)
    uy = np.fromfile(DATA_DIR / "uy_1.dat", dtype=np.float64).reshape(VELOCITY_SHAPE)
    uz = np.fromfile(DATA_DIR / "uz_1.dat", dtype=np.float64).reshape(VELOCITY_SHAPE)
    return ux, uy, uz


def _build_velocity_values():
    ux, uy, uz = _load_velocity_components()
    values = np.stack((ux, uy, uz), axis=-1)
    return values


def _speed_to_length(orient_length):
    orient_length = np.asarray(orient_length, dtype=float)
    speed_max = float(np.max(orient_length))
    if speed_max <= 1e-12:
        return np.full_like(orient_length, 2.0)
    return 1.5 + 10.5 * orient_length / speed_max


def render_velocity_mid_z_plane():
    """
    Build one interactive VectorPlane for the middle z-plane velocity field.

    This is intentionally a foreground/manual test. It constructs the complete
    shared-grid path:
    GridFieldDataset -> velocity FieldData -> GridInterpolator -> VectorPlane.
    """

    velocity_values = _build_velocity_values()
    nx, ny, nz, _ = velocity_values.shape
    z_index = nz // 2

    dataset = GridFieldDataset(
        inputValue=InputGridField(
            shape=(nx, ny, nz),
            box_periodic_flag=(False, False, False),
        ),
        name="velocity dataset",
    )
    field = dataset.act_add_field("velocity", velocity_values)
    interpolator = field.act_add_interpolator()

    plane = VectorPlane(
        interpolator=interpolator,
        name=f"velocity mid-z plane (z={z_index})",
        opts=OptsPlaneGrid(
            normal=(0.0, 0.0, 1.0),
            axis1=(1.0, 0.0, 0.0),
            origin=(0.0, 0.0, float(z_index)),
            alignment="bottom-left",
            spacing=float(VELOCITY_STRIDE_XY),
            size=float(nx - 1),
            size_extra=float(ny - 1),
        ),
    )

    figure = PlotFigure(
        name=f"velocity mid-z plane figure (z={z_index})",
        size=(1800, 1200),
        bg_color=(1, 1, 1),
    )
    visual = plane.act_visualize_vector(
        figure=figure,
        resolver_source="orient_length",
        paint_by="scalars",
        scalars=lambda orient_length: orient_length,
        scalars_cmap="turbo",
        scalar_bar_title=f"|u| on z={z_index}",
        length=_speed_to_length,
        radius=0.22,
        tip_length_fraction=0.3,
        tip_radius_ratio=2.8,
        anchor="center",
        sides=12,
    )

    figure.act_view_xy()

    speed = plane.calc_magnitude

    assert dataset is not None
    assert field is not None
    assert interpolator is not None
    assert plane is not None
    assert figure is not None
    assert visual is not None
    assert not figure.pl.off_screen
    assert plane.grid.opts.origin[2] == float(z_index)
    assert plane.result.shape[1] == 3
    assert len(speed) == len(plane.grid())

    return {
        "dataset": dataset,
        "field": field,
        "interpolator": interpolator,
        "plane": plane,
        "figure": figure,
        "visual": visual,
        "values": velocity_values,
        "speed": speed,
        "z_index": z_index,
    }


VELOCITY_SCENE_MID_Z = render_velocity_mid_z_plane()
VELOCITY_DATASET = VELOCITY_SCENE_MID_Z["dataset"]
VELOCITY_FIELD = VELOCITY_SCENE_MID_Z["field"]
VELOCITY_INTERPOLATOR = VELOCITY_SCENE_MID_Z["interpolator"]
VELOCITY_PLANE = VELOCITY_SCENE_MID_Z["plane"]
VELOCITY_FIGURE = VELOCITY_SCENE_MID_Z["figure"]
VELOCITY_VISUAL = VELOCITY_SCENE_MID_Z["visual"]
VELOCITY_VALUES = VELOCITY_SCENE_MID_Z["values"]
VELOCITY_SPEED = VELOCITY_SCENE_MID_Z["speed"]
