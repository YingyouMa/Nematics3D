"""General helpers that do not yet have a more specific module home."""

from ..geometry.points import points_membership_mask


__all__ = ["mark_points_membership"]


def mark_points_membership(points1, points2):
    """Deprecated internal alias for :func:`points_membership_mask`."""
    return points_membership_mask(points1, points2)
