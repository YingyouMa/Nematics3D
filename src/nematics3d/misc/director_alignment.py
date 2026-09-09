"""Nematic-director branch alignment utilities."""

import numpy as np

from ..datatypes import as_director_field, nField


def align_directors(n_reference: nField, n_target: nField) -> nField:
    """Flip target directors to the nematic branch nearest a reference field."""
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
    """Align an ordered director stack by propagating nematic sign choices."""
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


__all__ = ["align_director_stack", "align_directors"]
