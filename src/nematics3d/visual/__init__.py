"""Visualization helpers for Nematics3D.

This package is the destination for visualization code currently being
migrated out of ``nematics3d.classes.visual``.  New independent visual helpers
should live here rather than extending the legacy ``classes`` package.
"""

from .camera import camera_pose_from_vectors, camera_vectors_from_pose

__all__ = ["camera_pose_from_vectors", "camera_vectors_from_pose"]
