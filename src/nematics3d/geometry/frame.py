"""Helpers for comparing and sign-aligning orthonormal 3D frames."""

import numpy as np

from ..datatypes import as_axes

__all__ = ["align_axes_to_reference", "axes_angle_changes_deg"]


def axes_angle_changes_deg(axes, reference_axes, *, is_unsigned: bool = True):
    """Return column-wise angle changes between two 3D orthonormal frames."""
    axes = as_axes(axes, name="axes")
    reference_axes = as_axes(reference_axes, name="reference_axes")
    cosines = np.sum(axes * reference_axes, axis=0)
    if is_unsigned:
        cosines = np.abs(cosines)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def align_axes_to_reference(axes, reference_axes, *, is_right_handed: bool = True):
    """Resolve column-wise sign ambiguity against a reference frame."""
    axes = as_axes(axes, name="axes", is_right_handed=False)
    reference_axes = as_axes(
        reference_axes,
        name="reference_axes",
        is_right_handed=False,
    )

    signs = np.where(np.sum(axes * reference_axes, axis=0) < 0, -1.0, 1.0)
    aligned_axes = axes * signs
    if is_right_handed and np.linalg.det(aligned_axes) < 0:
        aligned_axes[:, -1] = -aligned_axes[:, -1]
    return aligned_axes
