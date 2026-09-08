"""Field-level utilities for structured Q-tensor and director data."""

import numpy as np

from .datatypes import (
    DimensionInfo,
    GeneralField,
    as_dimension_info,
    as_director_field,
    nField,
)


def add_periodic_boundary(
    data: GeneralField, is_boundary_periodic: DimensionInfo = 0
) -> GeneralField:
    """Append one periodic image slice along selected spatial axes.

    The first three axes are interpreted as spatial dimensions. For every
    periodic axis, one extra slice is appended to the high-index side and is
    copied from the first slice on that axis. Any trailing component axes are
    preserved unchanged.

    The returned array is always independent of the input, including when no
    periodic axis is selected.

    Parameters
    ----------
    data : GeneralField
        Input physical field of shape (Nx, Ny, Nz, ...), where (Nx, Ny, Nz) are spatial dimensions,
        and the remaining axes represent vector/tensor components or other per-voxel data.

    is_boundary_periodic : DimensionInfo, optional
        A 3-element flag indicating which spatial dimensions are periodic.
        - Can be a scalar (broadcasted), or
        - A list/tuple/array of booleans with shape (3,)
        - Default is 0 (all dimensions non-periodic)

    Returns
    -------
    output : numpy.ndarray
        Independent extended copy. Its first three dimensions are
        ``(Nx + px, Ny + py, Nz + pz)`` where ``px``, ``py``, and ``pz`` are
        the periodic-axis flags.
    """
    data = np.asarray(data)
    if data.ndim < 3:
        raise ValueError(
            "`data` must have at least three spatial dimensions; "
            f"got shape {data.shape}."
        )

    is_boundary_periodic = as_dimension_info(
        is_boundary_periodic,
        name="is_boundary_periodic",
        is_bool=True,
    )

    output = data.copy()
    for axis, is_periodic in enumerate(is_boundary_periodic):
        if is_periodic:
            output = np.concatenate(
                (output, np.take(output, [0], axis=axis)), axis=axis
            )
    return output


def align_directors(n_reference: nField, n_target: nField) -> nField:
    """Flip target directors to the nematic branch nearest a reference field.

    ``n`` and ``-n`` represent the same nematic director.  This helper chooses,
    point by point, the sign of ``n_target`` whose dot product with
    ``n_reference`` is non-negative.  Inputs are validated as director fields
    and are never modified in place.

    Parameters
    ----------
    n_reference, n_target : array_like, shape (..., 3)
        Director fields with matching shapes.

    Returns
    -------
    numpy.ndarray
        A new array with the same shape as ``n_target``.
    """
    n_reference = as_director_field(n_reference, name="n_reference")
    n_target = as_director_field(n_target, name="n_target")
    if n_reference.shape != n_target.shape:
        raise ValueError(
            "`n_reference` and `n_target` must have the same shape; "
            f"got {n_reference.shape} and {n_target.shape}."
        )

    dots = np.einsum("...i,...i->...", n_reference, n_target)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    return n_target * signs[..., np.newaxis]


def align_director_stack(stack: nField) -> nField:
    """Align an ordered stack of directors by propagating nematic signs.

    The first director slice is kept fixed.  Each subsequent slice is compared
    with the previous *original* slice, and cumulative sign flips are then
    applied so neighboring aligned slices lie on a consistent nematic branch.
    The input is not modified.

    Parameters
    ----------
    stack : array_like, shape (N, ..., 3)
        Ordered director stack. ``N`` must be at least one.

    Returns
    -------
    numpy.ndarray
        A new aligned array with the same shape as ``stack``.
    """
    stack = as_director_field(stack, name="stack")
    if stack.shape[0] == 0:
        raise ValueError("`stack` must contain at least one director slice.")

    if stack.shape[0] == 1:
        return stack.copy()

    dots = np.einsum("...i,...i->...", stack[:-1], stack[1:])
    flips = np.ones(stack.shape[:-1], dtype=np.int8)
    flips[1:] = np.where(dots < 0.0, -1, 1).astype(np.int8, copy=False)
    accumulated_flips = np.cumprod(flips, axis=0)
    return stack * accumulated_flips[..., np.newaxis]
