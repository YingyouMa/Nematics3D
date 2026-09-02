"""Surface-smoothing object model and Taubin smoothing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from ...core.class_base import AttrDef
from ...core.host_base import HostBase, OptsBase
from ...core.opts import cover_value
from ...datatypes import Number, UNSET, Unset, as_number, as_readonly_array
from ...logging_decorator import logging_and_warning_decorator
from ..polydata import as_polydata_input, copy_polydata_geometry


@dataclass(slots=True, repr=False)
class OptsSmoothedSurface(OptsBase):
    """Options controlling wavelength-based Taubin smoothing of a surface.

    ``cutoff_wavelength`` is the main geometric smoothing parameter. It is a
    physical wavelength in the same length units as the surface coordinates,
    not a mesh spacing or displacement distance. With the discrete
    Laplace--Beltrami convention ``L phi = -kappa phi``, define

        kappa_c = (2*pi / cutoff_wavelength)**2.

    Nematics3D defines ``cutoff_wavelength`` as the -3 dB amplitude cutoff of
    the complete smoothing pass:

        G_N(kappa_c) = 1/sqrt(2).

    A Taubin pair applies ``lambda > 0`` followed by ``mu < 0``. Their relative
    magnitude is parameterized by

        taubin_ratio = -mu / lambda.

    The iteration count is not a public option. For the requested wavelength,
    ratio, and current mesh spectrum, ``SmoothedSurface`` selects the smallest
    positive pair count whose highest resolved mode is not amplified.
    """

    cutoff_wavelength: Number | Unset = UNSET
    taubin_ratio: Number | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "cutoff_wavelength": (
            "positive physical wavelength defining the -3 dB amplitude cutoff "
            "of the complete surface-smoothing pass"
        ),
        "taubin_ratio": (
            "dimensionless Taubin coefficient ratio -mu/lambda; values greater "
            "than 1 give the usual slightly stronger negative step"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "cutoff_wavelength": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(0.0, 1.0), np.inf),
        ),
        "taubin_ratio": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(1.0, np.inf), np.inf),
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "tag": "smoothed surface options",
            "taubin_ratio": 1.0674,
        }
    )


class SurfaceSmoothingConfigError(ValueError):
    """Configuration or geometry error preventing Taubin surface smoothing."""


class SmoothedSurface(HostBase):
    """Smooth a triangle surface using a fixed-operator Taubin spectral filter.

    The input is normalized to ``pyvista.PolyData`` and copied as geometry only.
    A triangulated initial surface defines a fixed cotangent stiffness matrix K
    and lumped mass M. With

        L = -M**(-1) K,

    the implementation resolves the smallest stable Taubin pair count N and
    corresponding lambda/mu coefficients, then repeatedly applies

        V <- V + lambda L V
        V <- V + mu     L V.

    The operator remains fixed to the initial geometry during one smoothing
    pass, so the stated spectral transfer function remains literal. Changing
    only smoothing opts reuses that operator; replacing the raw surface rebuilds
    it before smoothing again.

    ``result`` is the smoothed geometry-only ``pyvista.PolyData``. ``vertices``
    exposes the same final point coordinates as a read-only NumPy array.
    Surface-function sampling/interpolation is intentionally handled separately.
    """

    __attr_defs__ = {
        "raw_surface": AttrDef(
            doc="Raw input surface normalized to pyvista.PolyData.",
            kind="raw",
            validator=lambda v, d: as_polydata_input(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "calc_surface_initial": AttrDef(
            doc="Geometry-only triangulated copy defining the fixed smoothing operator.",
            kind="calc",
        ),
        "calc_vertices_initial": AttrDef(
            doc="Read-only initial surface vertices used by the smoothing operator.",
            kind="calc",
        ),
        "calc_faces": AttrDef(
            doc="Read-only triangle connectivity with shape (n_faces, 3).",
            kind="calc",
        ),
        "calc_mass_lumped": AttrDef(
            doc="Read-only lumped vertex masses of the cotangent discretization.",
            kind="calc",
        ),
        "calc_stiffness_matrix": AttrDef(
            doc="Fixed cotangent stiffness matrix K for K phi = kappa M phi.",
            kind="calc",
        ),
        "calc_kappa_cutoff": AttrDef(
            doc="Cutoff Laplace--Beltrami eigenvalue (2*pi/cutoff_wavelength)^2.",
            kind="calc",
        ),
        "calc_kappa_max": AttrDef(
            doc="Largest resolved eigenvalue of the fixed discrete Laplace--Beltrami operator.",
            kind="calc",
        ),
        "calc_iterations": AttrDef(
            doc="Smallest positive Taubin pair count satisfying high-frequency stability.",
            kind="calc",
        ),
        "calc_lambda": AttrDef(
            doc="Resolved positive Taubin lambda coefficient, in length squared.",
            kind="calc",
        ),
        "calc_mu": AttrDef(
            doc="Resolved negative Taubin mu coefficient, in length squared.",
            kind="calc",
        ),
        "calc_vertices_result": AttrDef(
            doc="Read-only smoothed vertex coordinates with shape (n_points, 3).",
            kind="calc",
        ),
        "calc_surface_result": AttrDef(
            doc="Geometry-only smoothed PolyData with topology inherited from the initial mesh.",
            kind="calc",
        ),
        "calc_is_smoothed": AttrDef(
            doc="Whether the current result was successfully produced by Taubin smoothing.",
            kind="calc",
        ),
        "calc_status": AttrDef(
            doc="Human-readable status of the current surface-smoothing pipeline.",
            kind="calc",
        ),
        "result": AttrDef(
            doc="Read-only: final smoothed PolyData.",
            kind="property",
            is_public_settable=False,
        ),
        "vertices": AttrDef(
            doc="Read-only: final smoothed vertex coordinates.",
            kind="property",
            is_public_settable=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
    )

    # -------------------------------
    # Initialization
    # -------------------------------

    def __init__(
        self,
        surface,
        name: str | None = None,
        opts: OptsSmoothedSurface | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsSmoothedSurface,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="surface",
            raw_surface=surface,
            **kwargs,
        )

        try:
            self.opts.act_finalize(self.opts_defaults)
        except KeyError as error:
            if "cutoff_wavelength" in str(error):
                raise SurfaceSmoothingConfigError(
                    "`cutoff_wavelength` is required for SmoothedSurface because "
                    "there is no geometry-independent default smoothing scale."
                ) from error
            raise

        self._helper_commit_apply_opts(is_reapply_opts=True)

    # -------------------------------
    # Fixed mesh operator
    # -------------------------------

    @staticmethod
    def _helper_extract_triangles(poly):
        if poly.n_points < 3 or poly.n_cells < 1:
            raise SurfaceSmoothingConfigError(
                "Surface smoothing requires at least three points and one face."
            )

        poly_tri = poly.triangulate(inplace=False)
        if poly_tri.n_cells < 1 or not poly_tri.is_all_triangles:
            raise SurfaceSmoothingConfigError(
                "Surface could not be converted to a non-empty triangle mesh."
            )

        faces_flat = np.asarray(poly_tri.faces, dtype=np.int64)
        if faces_flat.size != 4 * poly_tri.n_cells:
            raise SurfaceSmoothingConfigError(
                "Unexpected triangle connectivity layout after triangulation."
            )
        faces = faces_flat.reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise SurfaceSmoothingConfigError(
                "Unexpected non-triangle cell after triangulation."
            )
        return poly_tri, faces[:, 1:].copy()

    @staticmethod
    def _helper_build_cotangent_operator(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Build cotangent stiffness K and barycentric lumped mass diagonal."""
        n_vertices = len(vertices)
        mass = np.zeros(n_vertices, dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []

        eps = np.finfo(float).eps
        for i, j, k in faces:
            vi = vertices[i]
            vj = vertices[j]
            vk = vertices[k]

            e_ij = vj - vi
            e_ik = vk - vi
            double_area = float(np.linalg.norm(np.cross(e_ij, e_ik)))
            edge_scale_sq = max(
                float(np.dot(e_ij, e_ij)),
                float(np.dot(e_ik, e_ik)),
                float(np.dot(vk - vj, vk - vj)),
            )
            if (
                not np.isfinite(double_area)
                or edge_scale_sq <= 0.0
                or double_area <= eps * edge_scale_sq
            ):
                raise SurfaceSmoothingConfigError(
                    "Surface contains a degenerate or numerically singular triangle; "
                    "cotangent smoothing requires non-degenerate faces."
                )

            mass[[i, j, k]] += (0.5 * double_area) / 3.0

            cot_i = float(np.dot(vj - vi, vk - vi) / double_area)
            cot_j = float(np.dot(vi - vj, vk - vj) / double_area)
            cot_k = float(np.dot(vi - vk, vj - vk) / double_area)

            for a, b, weight in (
                (j, k, 0.5 * cot_i),
                (i, k, 0.5 * cot_j),
                (i, j, 0.5 * cot_k),
            ):
                rows.extend((a, b))
                cols.extend((b, a))
                weights.extend((weight, weight))

        if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise SurfaceSmoothingConfigError(
                "Every surface vertex must have a positive finite lumped mass. "
                "Check for unused points or invalid triangles."
            )

        adjacency = coo_matrix(
            (np.asarray(weights, dtype=float), (rows, cols)),
            shape=(n_vertices, n_vertices),
        ).tocsr()
        adjacency.sum_duplicates()
        diagonal = np.asarray(adjacency.sum(axis=1)).ravel()
        stiffness = (diags(diagonal, format="csr") - adjacency).tocsr()
        stiffness.sum_duplicates()
        return stiffness, mass

    @staticmethod
    def _helper_largest_generalized_eigenvalue(
        stiffness: csr_matrix,
        mass: np.ndarray,
    ) -> float:
        """Return largest kappa of K phi = kappa M phi via symmetric scaling."""
        inv_sqrt_mass = 1.0 / np.sqrt(mass)
        scale = diags(inv_sqrt_mass, format="csr")
        operator = (scale @ stiffness @ scale).tocsr()
        n = operator.shape[0]

        if n <= 64:
            kappa_max = float(np.linalg.eigvalsh(operator.toarray())[-1])
        else:
            try:
                kappa_max = float(
                    eigsh(
                        operator,
                        k=1,
                        which="LA",
                        return_eigenvectors=False,
                    )[0]
                )
            except Exception as error:
                raise SurfaceSmoothingConfigError(
                    "Failed to estimate the largest discrete Laplace--Beltrami eigenvalue."
                ) from error

        if not np.isfinite(kappa_max) or kappa_max <= 0.0:
            raise SurfaceSmoothingConfigError(
                "Could not obtain a positive finite maximum Laplace--Beltrami eigenvalue."
            )
        return kappa_max

    def _helper_prepare_initial_operator(self) -> None:
        poly_geometry = copy_polydata_geometry(self.raw_surface)
        poly_tri, faces = self._helper_extract_triangles(poly_geometry)
        vertices = np.asarray(poly_tri.points, dtype=float).copy()

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise SurfaceSmoothingConfigError(
                "Surface vertices must have shape (n_points, 3)."
            )
        if np.any(~np.isfinite(vertices)):
            raise SurfaceSmoothingConfigError(
                "Surface vertices must contain only finite coordinates."
            )

        stiffness, mass = self._helper_build_cotangent_operator(vertices, faces)
        kappa_max = self._helper_largest_generalized_eigenvalue(stiffness, mass)

        object.__setattr__(self, "calc_surface_initial", poly_tri)
        object.__setattr__(
            self,
            "calc_vertices_initial",
            as_readonly_array(vertices, dtype=None, copy=False),
        )
        object.__setattr__(
            self,
            "calc_faces",
            as_readonly_array(faces, dtype=None, copy=False),
        )
        object.__setattr__(
            self,
            "calc_mass_lumped",
            as_readonly_array(mass, dtype=None, copy=False),
        )
        object.__setattr__(self, "calc_stiffness_matrix", stiffness)
        object.__setattr__(self, "calc_kappa_max", kappa_max)

    # -------------------------------
    # Taubin parameter resolution
    # -------------------------------

    @staticmethod
    def _helper_coefficients_for_iterations(
        kappa_cutoff: float,
        taubin_ratio: float,
        iterations: int,
    ) -> tuple[float, float]:
        q = 2.0 ** (-1.0 / (2.0 * iterations))
        r = taubin_ratio
        x = (
            (r - 1.0)
            + np.sqrt((r - 1.0) ** 2 + 4.0 * r * (1.0 - q))
        ) / (2.0 * r)
        lambda_ = float(x / kappa_cutoff)
        mu = float(-r * lambda_)
        return lambda_, mu

    @classmethod
    def _helper_resolve_taubin_parameters(
        cls,
        *,
        kappa_cutoff: float,
        kappa_max: float,
        taubin_ratio: float,
    ) -> tuple[int, float, float]:
        """Resolve the minimum stable N and its lambda/mu coefficients."""

        def edge_gain(lambda_: float, mu: float) -> float:
            return float(
                (1.0 - lambda_ * kappa_max)
                * (1.0 - mu * kappa_max)
            )

        lambda_1, mu_1 = cls._helper_coefficients_for_iterations(
            kappa_cutoff,
            taubin_ratio,
            1,
        )
        gain_1 = edge_gain(lambda_1, mu_1)
        tol = 64.0 * np.finfo(float).eps
        if abs(gain_1) <= 1.0 + tol:
            return 1, lambda_1, mu_1

        # lambda_N decreases monotonically toward a positive limit. If N=1
        # already places kappa_max in the low-frequency amplified branch, larger
        # N cannot move that mode into the attenuating branch.
        if gain_1 > 1.0:
            raise SurfaceSmoothingConfigError(
                "No stable Taubin iteration count exists for this combination "
                "of cutoff_wavelength, taubin_ratio, and mesh spectrum: the "
                "highest resolved mode lies in the amplified pass-band branch."
            )

        r = taubin_ratio
        y_cross = (r - 1.0) / r
        y_stable_max = (
            (r - 1.0) + np.sqrt((r - 1.0) ** 2 + 8.0 * r)
        ) / (2.0 * r)
        spectral_ratio = kappa_max / kappa_cutoff
        y_infinite = y_cross * spectral_ratio

        if y_infinite >= y_stable_max * (1.0 - tol):
            raise SurfaceSmoothingConfigError(
                "No finite stable Taubin iteration count exists for this mesh. "
                "The requested cutoff_wavelength is too large relative to the "
                "highest resolved mesh frequency for the selected taubin_ratio."
            )

        x_limit = y_stable_max / spectral_ratio
        q_required = (1.0 - x_limit) * (1.0 + r * x_limit)
        if not (0.0 < q_required < 1.0):
            raise SurfaceSmoothingConfigError(
                "Internal Taubin parameter resolution produced an invalid "
                "stability-boundary gain."
            )

        n_float = (0.5 * np.log(2.0)) / (-np.log(q_required))
        iterations = max(2, int(np.ceil(n_float)))

        # Protect against floating-point rounding at an integer boundary.
        while True:
            lambda_, mu = cls._helper_coefficients_for_iterations(
                kappa_cutoff,
                r,
                iterations,
            )
            if abs(edge_gain(lambda_, mu)) <= 1.0 + tol:
                return iterations, lambda_, mu
            iterations += 1

    def _helper_resolve_filter_parameters(self) -> None:
        cutoff_wavelength = float(self.opts.cutoff_wavelength)
        taubin_ratio = float(self.opts.taubin_ratio)
        kappa_cutoff = float((2.0 * np.pi / cutoff_wavelength) ** 2)

        iterations, lambda_, mu = self._helper_resolve_taubin_parameters(
            kappa_cutoff=kappa_cutoff,
            kappa_max=float(self.calc_kappa_max),
            taubin_ratio=taubin_ratio,
        )

        object.__setattr__(self, "calc_kappa_cutoff", kappa_cutoff)
        object.__setattr__(self, "calc_iterations", iterations)
        object.__setattr__(self, "calc_lambda", lambda_)
        object.__setattr__(self, "calc_mu", mu)

    # -------------------------------
    # Smoothing
    # -------------------------------

    def _helper_apply_laplacian(self, vertices: np.ndarray) -> np.ndarray:
        """Apply the fixed negative-semidefinite L = -M^-1 K to coordinates."""
        return -(
            self.calc_stiffness_matrix @ vertices
        ) / self.calc_mass_lumped[:, None]

    def _helper_smooth_vertices(self) -> np.ndarray:
        """Apply the resolved number of fixed-operator Taubin pairs."""
        vertices = np.asarray(self.calc_vertices_initial, dtype=float).copy()
        lambda_ = float(self.calc_lambda)
        mu = float(self.calc_mu)

        for _ in range(int(self.calc_iterations)):
            vertices += lambda_ * self._helper_apply_laplacian(vertices)
            vertices += mu * self._helper_apply_laplacian(vertices)

        if np.any(~np.isfinite(vertices)):
            raise RuntimeError(
                "Taubin smoothing produced non-finite vertex coordinates despite "
                "the resolved spectral stability condition."
            )
        return vertices

    def _helper_set_result(self, vertices: np.ndarray) -> None:
        vertices_readonly = as_readonly_array(vertices, dtype=None, copy=False)
        surface_result = copy_polydata_geometry(self.calc_surface_initial)
        surface_result.points = np.asarray(vertices_readonly).copy()

        object.__setattr__(self, "calc_vertices_result", vertices_readonly)
        object.__setattr__(self, "calc_surface_result", surface_result)
        object.__setattr__(self, "calc_is_smoothed", True)
        object.__setattr__(self, "calc_status", "Success")

    # -------------------------------
    # Host commit pipeline
    # -------------------------------

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self,
        is_reapply_opts=False,
        logger=None,
        **kwargs,
    ):
        if not is_reapply_opts and not kwargs:
            return

        if kwargs:
            with self.opts.act_internal_update():
                cover_value(
                    self.opts,
                    is_allow_cover_target_set=True,
                    is_allow_unset_source=False,
                    **kwargs,
                )

        if is_reapply_opts or getattr(self, "calc_kappa_max", None) is None:
            self._helper_prepare_initial_operator()

        self._helper_resolve_filter_parameters()

        logger.debug(
            "Smoothing surface %r with %d vertices and %d faces; "
            "cutoff_wavelength=%s, taubin_ratio=%s, iterations=%d, "
            "lambda=%g, mu=%g, kappa_max=%g.",
            self.name,
            len(self.calc_vertices_initial),
            len(self.calc_faces),
            self.opts.cutoff_wavelength,
            self.opts.taubin_ratio,
            self.calc_iterations,
            self.calc_lambda,
            self.calc_mu,
            self.calc_kappa_max,
        )

        try:
            vertices = self._helper_smooth_vertices()
        except (FloatingPointError, RuntimeError, ValueError) as error:
            object.__setattr__(self, "calc_is_smoothed", False)
            object.__setattr__(self, "calc_status", f"Failed: {error}")
            raise

        self._helper_set_result(vertices)

    # -------------------------------
    # Readable properties
    # -------------------------------

    @property
    def result(self):
        return self.calc_surface_result

    @property
    def vertices(self):
        return self.calc_vertices_result


__all__ = [
    "OptsSmoothedSurface",
    "SmoothedSurface",
    "SurfaceSmoothingConfigError",
]
