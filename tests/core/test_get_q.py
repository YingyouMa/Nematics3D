import numpy as np
import pytest

import nematics3d as n3d
from nematics3d.analysis.q_diagonalization import q_diagonalize
from nematics3d.field import get_q


def test_get_q_constructs_uniaxial_tensor_and_broadcasts_fields():
    directors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    scalar_order = np.array([[0.6], [0.3]])

    result = get_q(directors[:, None, :], S=scalar_order)

    assert result.shape == (2, 1, 3, 3)
    np.testing.assert_allclose(np.trace(result, axis1=-2, axis2=-1), 0.0, atol=1e-15)
    np.testing.assert_allclose(result, np.swapaxes(result, -1, -2))
    np.testing.assert_allclose(result[0, 0], np.diag([0.4, -0.2, -0.2]))
    np.testing.assert_allclose(result[1, 0], np.diag([-0.1, 0.2, -0.1]))


def test_get_q_defaults_to_unit_scalar_order():
    expected = np.diag([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    np.testing.assert_allclose(get_q([1.0, 0.0, 0.0]), expected)


def test_get_q_can_return_compact_q5_without_changing_values():
    directors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scalar_order = np.array([0.6, 0.3])

    q9 = get_q(directors, S=scalar_order, output="q9")
    q5 = get_q(directors, S=scalar_order, output="q5")

    assert q5.shape == (2, 5)
    np.testing.assert_allclose(
        q5,
        q9[..., (0, 0, 0, 1, 1), (0, 1, 2, 1, 2)],
    )


def test_get_q_q5_preserves_float32_when_all_inputs_are_float32():
    directors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scalar_order = np.array([0.75], dtype=np.float32)

    result = get_q(directors, S=scalar_order, output="q5")

    assert result.dtype == np.float32


@pytest.mark.parametrize("output", ["compact", "Q5", 5])
def test_get_q_rejects_invalid_output_representation(output):
    with pytest.raises((TypeError, ValueError)):
        get_q([1.0, 0.0, 0.0], output=output)


@pytest.mark.parametrize("biaxial_order", [-0.2, 0.2])
def test_get_q_constructs_signed_biaxial_tensor(biaxial_order):
    result = get_q(
        [1.0, 0.0, 0.0],
        S=0.6,
        m=[0.0, 1.0, 0.0],
        P=biaxial_order,
    )

    expected = np.diag([0.4, -0.2 + biaxial_order, -0.2 - biaxial_order])
    np.testing.assert_allclose(result, expected, atol=1e-15)
    np.testing.assert_allclose(np.trace(result), 0.0, atol=1e-15)

    compact = get_q(
        [1.0, 0.0, 0.0],
        S=0.6,
        m=[0.0, 1.0, 0.0],
        P=biaxial_order,
        output="q5",
    )
    np.testing.assert_allclose(
        compact,
        result[(0, 0, 0, 1, 1), (0, 1, 2, 1, 2)],
    )


def test_get_q_round_trips_complete_diagonalization():
    axes, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    n = axes[:, 0]
    m = axes[:, 1]
    original = get_q(n, S=0.75, m=m, P=0.2)

    result = q_diagonalize(original, is_biaxial=True, log_mode="none")
    recovered_p = (result.eigenvalues[..., 1] - result.eigenvalues[..., 2]) / 2.0
    reconstructed = get_q(
        result.n,
        S=result.S,
        m=result.eigenvectors[..., :, 1],
        P=recovered_p,
    )

    np.testing.assert_allclose(reconstructed, original, atol=1e-12)


def test_get_q_is_invariant_to_director_signs():
    positive = get_q([1.0, 0.0, 0.0], S=0.7, m=[0.0, 1.0, 0.0], P=0.1)
    negative = get_q([-1.0, 0.0, 0.0], S=0.7, m=[0.0, -1.0, 0.0], P=0.1)
    np.testing.assert_allclose(positive, negative)


def test_get_q_broadcasts_biaxial_fields_without_modifying_inputs():
    n = np.array([[[2.0, 0.0, 0.0]], [[0.0, 0.0, 3.0]]])
    m = np.array([0.0, 4.0, 0.0])
    scalar_order = np.array([[0.6], [0.3]])
    biaxial_order = np.array([0.1])
    inputs_before = tuple(value.copy() for value in (n, m, scalar_order, biaxial_order))

    result = get_q(n, S=scalar_order, m=m, P=biaxial_order)

    assert result.shape == (2, 1, 3, 3)
    assert result.dtype == float
    np.testing.assert_allclose(result, np.swapaxes(result, -1, -2))
    np.testing.assert_allclose(np.trace(result, axis1=-2, axis2=-1), 0.0, atol=1e-15)
    for value, expected in zip((n, m, scalar_order, biaxial_order), inputs_before):
        np.testing.assert_array_equal(value, expected)


def test_get_q_is_available_from_the_top_level_package():
    assert n3d.get_q is get_q


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"m": [0.0, 1.0, 0.0]}, "must be supplied together"),
        ({"P": 0.1}, "must be supplied together"),
        ({"m": [1.0, 1.0, 0.0], "P": 0.1}, "must be orthogonal"),
    ],
)
def test_get_q_rejects_invalid_biaxial_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_q([1.0, 0.0, 0.0], **kwargs)


def test_get_q_rejects_zero_directors_and_incompatible_shapes():
    with pytest.raises(ValueError, match="zero directors"):
        get_q([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="must be broadcastable"):
        get_q(np.ones((2, 3)), S=np.ones(3))
