import numpy as np
import pytest

from nematics3d.classes.visual.plot_figure import OptsFigure
from nematics3d.datatypes import UNSET
from nematics3d.visual.camera import (
    camera_pose_from_vectors,
    camera_vectors_from_pose,
)


@pytest.mark.parametrize(
    "pose",
    [
        (0.0, 0.0, 0.0, 5.0),
        (90.0, 20.0, 35.0, 12.0),
        (275.0, -45.0, -80.0, 2.5),
        (180.0, 89.999, 120.0, 7.0),
    ],
)
def test_camera_pose_round_trip_away_from_poles(pose):
    focal = np.array([1.5, -2.0, 4.0])
    position, focal_out, up = camera_vectors_from_pose(*pose, focal)
    recovered = camera_pose_from_vectors(position, focal_out, up)

    np.testing.assert_allclose(recovered, pose, atol=1e-9)


@pytest.mark.parametrize("elevation", [-90.0, 90.0])
@pytest.mark.parametrize("roll", [-120.0, 0.0, 47.0])
def test_camera_pole_convention_is_stable(elevation, roll):
    position, focal, up = camera_vectors_from_pose(
        123.0, elevation, roll, 3.0, [0.0, 0.0, 0.0]
    )
    azimuth, recovered_elevation, recovered_roll, distance = camera_pose_from_vectors(
        position, focal, up
    )

    assert azimuth == pytest.approx(0.0)
    assert recovered_elevation == pytest.approx(elevation)
    assert recovered_roll == pytest.approx(roll)
    assert distance == pytest.approx(3.0)


@pytest.mark.parametrize("distance", [0.0, -1.0])
def test_camera_pose_rejects_nonpositive_distance(distance):
    with pytest.raises(ValueError, match="strictly positive"):
        camera_vectors_from_pose(0.0, 0.0, 0.0, distance, [0.0, 0.0, 0.0])


def test_camera_vectors_reject_coincident_position_and_focal_point():
    with pytest.raises(ValueError, match="must differ"):
        camera_pose_from_vectors([1, 2, 3], [1, 2, 3], [0, 0, 1])


def test_camera_vectors_reject_parallel_view_up():
    with pytest.raises(ValueError, match="must not be parallel"):
        camera_pose_from_vectors([1, 0, 0], [0, 0, 0], [-1, 0, 0])


def test_figure_opts_reject_zero_camera_distance():
    opts = OptsFigure(distance=0.0)

    opts.act_finalize(is_allow_unset=True)

    assert opts.distance is UNSET
