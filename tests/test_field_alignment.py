import numpy as np
import pytest

from nematics3d.misc import align_director_stack, align_directors


def test_align_directors_flips_target_without_modifying_inputs():
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference_before = reference.copy()
    target_before = target.copy()

    aligned = align_directors(reference, target)

    np.testing.assert_allclose(aligned, reference)
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(target, target_before)
    assert not np.shares_memory(aligned, target)


def test_align_directors_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        align_directors(np.ones((2, 3)), np.ones((1, 3)))


def test_align_director_stack_propagates_signs_and_does_not_modify_input():
    stack = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    stack_before = stack.copy()

    aligned = align_director_stack(stack)

    np.testing.assert_allclose(aligned, np.tile([1.0, 0.0, 0.0], (4, 1)))
    np.testing.assert_array_equal(stack, stack_before)
    assert not np.shares_memory(aligned, stack)


def test_align_director_stack_single_slice_returns_independent_copy():
    stack = np.array([[1.0, 0.0, 0.0]])
    aligned = align_director_stack(stack)

    np.testing.assert_array_equal(aligned, stack)
    assert not np.shares_memory(aligned, stack)
