import numpy as np
import pytest

from nematics3d.datatypes import as_axes


def test_identity_frame_is_accepted_and_returned_as_float_copy():
    source = np.eye(3, dtype=int)

    result = as_axes(source)

    np.testing.assert_array_equal(result, np.eye(3))
    assert result.dtype == float
    assert result is not source


def test_result_does_not_share_storage_with_input():
    source = np.eye(3, dtype=float)

    result = as_axes(source)
    result[0, 0] = 0.0

    np.testing.assert_array_equal(source, np.eye(3))


def test_general_right_handed_orthonormal_frame_is_accepted():
    theta = np.deg2rad(37.0)
    frame = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = as_axes(frame)

    np.testing.assert_allclose(result, frame)
    np.testing.assert_allclose(result.T @ result, np.eye(3))
    assert np.linalg.det(result) > 0.0


def test_left_handed_frame_is_made_right_handed_by_flipping_last_axis():
    frame = np.diag([1.0, 1.0, -1.0])

    result = as_axes(frame)

    np.testing.assert_array_equal(result[:, :2], frame[:, :2])
    np.testing.assert_array_equal(result[:, 2], -frame[:, 2])
    assert np.linalg.det(result) > 0.0


def test_left_handed_frame_is_preserved_when_requested():
    frame = np.diag([1.0, 1.0, -1.0])

    result = as_axes(frame, is_right_handed=False)

    np.testing.assert_array_equal(result, frame)
    assert np.linalg.det(result) < 0.0


def test_atol_controls_orthonormality_tolerance():
    frame = np.eye(3)
    frame[0, 1] = 5e-9

    result = as_axes(frame, atol=1e-8)
    np.testing.assert_array_equal(result, frame)

    with pytest.raises(ValueError, match="orthonormal"):
        as_axes(frame, atol=1e-10)


@pytest.mark.parametrize(
    "value",
    [
        np.ones((3,)),
        np.ones((1, 3)),
        np.ones((3, 1)),
        np.ones((2, 3)),
        np.ones((3, 2)),
        np.ones((3, 3, 1)),
    ],
)
def test_wrong_shapes_are_rejected(value):
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        as_axes(value)


@pytest.mark.parametrize(
    "value",
    [
        [["x", 0, 0], [0, 1, 0], [0, 0, 1]],
        np.eye(3, dtype=complex) * (1.0 + 1.0j),
    ],
)
def test_non_real_numeric_frames_are_rejected(value):
    with pytest.raises(TypeError, match="real numeric"):
        as_axes(value)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(bad_value):
    frame = np.eye(3)
    frame[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        as_axes(frame)


@pytest.mark.parametrize(
    "frame",
    [
        np.diag([1.0, 1.0, 2.0]),
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.zeros((3, 3)),
    ],
)
def test_non_orthonormal_frames_are_rejected(frame):
    with pytest.raises(ValueError, match="orthonormal"):
        as_axes(frame)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, "small"])
def test_invalid_atol_is_rejected(atol):
    expected_error = TypeError if isinstance(atol, str) else ValueError
    with pytest.raises(expected_error, match="atol"):
        as_axes(np.eye(3), atol=atol)


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_is_right_handed_option_must_be_boolean(value):
    with pytest.raises(TypeError, match="is_right_handed"):
        as_axes(np.eye(3), is_right_handed=value)


def test_custom_name_is_used_in_validation_error():
    with pytest.raises(ValueError, match="local frame"):
        as_axes(np.zeros((3, 3)), name="local frame")
