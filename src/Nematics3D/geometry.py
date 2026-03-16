"""
Geometry helpers for vector parameterization, angle wrapping, and local frame conversions.
"""

import numpy as np

from Nematics3D.general import rotation_matrix_from_vectors


def calc_vec_from_azimuth_polar(azimuth, polar_angle):
    x = np.sin(polar_angle) * np.cos(azimuth)
    y = np.sin(polar_angle) * np.sin(azimuth)
    z = np.cos(polar_angle)
    return np.array((x, y, z), dtype=float)


def get_azimuth(vec):
    vec = np.asarray(vec, dtype=float)
    az_rad = np.arctan2(vec[1], vec[0])
    return np.degrees(az_rad) % 360


def get_polar_angle(vec):
    vec = np.asarray(vec, dtype=float).copy()
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    polar = np.arccos(vec[2])
    return np.degrees(polar)


def get_axis1_azimuth(axis1, normal):
    axis1 = np.asarray(axis1, dtype=float).copy()
    axis1 /= np.linalg.norm(axis1, axis=-1, keepdims=True)
    rotation = rotation_matrix_from_vectors((0, 0, 1), normal)
    axisx = rotation @ np.array([1.0, 0.0, 0.0])
    axisy = rotation @ np.array([0.0, 1.0, 0.0])
    az_rad = np.arctan2(axis1 @ axisy, axis1 @ axisx)
    return np.degrees(az_rad) % 360


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
