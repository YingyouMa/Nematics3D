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
from ...datatypes import Number, UNSET, Unset, as_number, as_readonly_array
from ..polydata import as_polydata_input, copy_polydata_geometry


@dataclass(slots=True, repr=False)
class OptsSmoothedSurface(OptsBase):
    """Options controlling wavelength-based Taubin smoothing of a surface.

    ``cutoff_wavelength`` is the main geometric smoothing parameter. It is a
    physical wavelength in the same length units as the surface coordinates,
    not a mesh spacing or a displacement distance. With the discrete
    Laplace--Beltrami convention ``L phi = -kappa phi``, define

        kappa_c = (2*pi / cutoff_wavelength)**2.

    Nematics3D defines ``cutoff_wavelength`` as the -3 dB amplitude cutoff of
    the *complete* smoothing pass:

        G_N(kappa_c) = 1/sqrt(2).

    Thus a surface mode whose wavelength equals ``cutoff_wavelength`` retains
    1/sqrt(2) of its original amplitude after smoothing. Longer-wavelength
    modes are intended to be preserved more strongly, while shorter-wavelength
    modes are intended to be suppressed more strongly.

    A Taubin iteration pair applies coefficients ``lambda > 0`` and ``mu < 0``.
    This interface parameterizes their relative magnitude by

        taubin_ratio = -mu / lambda,

    so ``mu = -taubin_ratio * lambda``. The default ratio is 1.0674, close to
    the ratio of the widely used Taubin coefficients 0.6307 and -0.6732. The
    ratio is dimensionless; unlike those raw coefficients, it remains meaningful
    when a dimensional Laplace--Beltrami operator is used.

    The iteration count is intentionally not a public smoothing option. For a
    fixed ``cutoff_wavelength`` and ``taubin_ratio``, the implementation chooses
    the smallest positive integer N for which the resolved Taubin filter is
    stable at the highest resolved discrete surface frequency.

    Important readable attributes
    -----------------------------
    cutoff_wavelength
        Required positive physical wavelength defining the -3 dB amplitude
        cutoff of the complete smoothing pass. No universal default exists
        because the appropriate value depends on the surface length scale.
    taubin_ratio
        Positive dimensionless ratio ``-mu/lambda``. Values greater than 1
        correspond to the usual Taubin choice in which the negative step has a
        slightly larger magnitude than the positive step. The library default
        is 1.0674.
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
    """Configuration error preventing a stable Taubin surface filter."""


class SmoothedSurface(HostBase):
    """Triangle surface together with its resolved Taubin filter parameters.

    This first implementation stage initializes a geometry-only triangulated
    copy of the input surface, builds a fixed cotangent Laplace--Beltrami
    discretization on that initial geometry, estimates its largest eigenvalue,
    and resolves the smallest stable Taubin iteration count together with the
    corresponding lambda and mu coefficients.

    The actual vertex-update smoothing loop is intentionally not implemented in
    this stage. The fixed initial operator and all resolved filter parameters are
    cached for that next stage.
    """

    __attr_defs__ = {
        "raw_surface": AttrDef(
            doc="Raw input surface normalized to pyvista.PolyData.",
            kind="raw",
            validator=lambda v, d: as_polydata_input(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "calc_surface_initial": AttrDef(
            doc="Geometry-only triangulated copy used to define the fixed smoothing operator.",
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
            doc="Smallest positive Taubin pair count satisfying the high-frequency stability criterion.",
            kind="calc",
        ),
        "calc_lambda": AttrDef(
            doc="Resolved positive Taubin lambda coefficient, with units of length squared.",
            kind="calc",
        ),
        "calc_mu": AttrDef(
            doc="Resolved negative Taubin mu coefficient, with units of length squared.",
            kind="calc",
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
    )

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

    @staticmethod
    def _helper_extract_triangles(poly) -> np.ndarray:
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
            cross = np.cross(e_ij, e_ik)
            double_area = float(np.linalg.norm(cross))
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

            area = 0.5 * double_area
            mass[[i, j, k]] += area / 3.0

            cot_i = float(np.dot(vj - vi, vk - vi) / double_area)
            cot_j = float(np.dot(vi - vj, vk - vj) / double_area)
            cot_k = float(np.dot(vi - vk, vj - vk) / double_area)

            # Each cotangent contributes half its value to the opposite edge.
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
        stiffness = diags(diagonal, format="csr") - adjacency
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
            kappa_max = float(
                eigsh(
                    operator,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                )[0]
            )

        if not np.isfinite(kappa_max) or kappa_max <= 0.0:
            raise SurfaceSmoothingConfigError(
                "Could not obtain a positive finite maximum Laplace--Beltrami eigenvalue."
            )
        return kappa_max

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

        # lambda_N decreases monotonically with N. If N=1 already puts the
        # highest resolved mode in the amplified low-frequency branch, larger N
        # cannot repair it.
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

        # Equality is reached only in the N -> infinity limit, so a finite
        # stable filter requires a strict inequality here.
        if y_infinite >= y_stable_max * (1.0 - tol):
            raise SurfaceSmoothingConfigError(
                "No finite stable Taubin iteration count exists for this mesh. "
                "The requested cutoff_wavelength is too large relative to the "
                "highest resolved mesh frequency for the selected taubin_ratio."
            )

        # At the stability boundary lambda*kappa_max = y_stable_max. Convert
        # that lambda to the corresponding single-pair cutoff gain q_required,
        # then solve q_N = 2**(-1/(2N)) >= q_required analytically for N.
        x_limit = y_stable_max / spectral_ratio
        q_required = (1.0 - x_limit) * (1.0 + r * x_limit)

        if not (0.0 < q_required < 1.0):
            # This should only occur in a regime where N=1 was already stable;
            # keep it explicit rather than hiding a parameter-design bug.
            raise SurfaceSmoothingConfigError(
                "Internal Taubin parameter resolution produced an invalid "
                "stability-boundary gain."
            )

        n_float = (0.5 * np.log(2.0)) / (-np.log(q_required))
        iterations = max(2, int(np.ceil(n_float)))

        # Protect against floating-point rounding exactly at the integer boundary.
        while True:
            lambda_, mu = cls._helper_coefficients_for_iterations(
                kappa_cutoff,
                r,
                iterations,
            )
            if abs(edge_gain(lambda_, mu)) <= 1.0 + tol:
                return iterations, lambda_, mu
            iterations += 1

    def _helper_prepare_initial_operator(self) -> None:
        poly_geometry = copy_polydata_geometry(self.raw_surface)
        poly_tri, faces = self._helper_extract_triangles(poly_geometry)
        vertices = np.asarray(poly_tri.points, dtype=float).copy()

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

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        if not is_reapply_opts and not kwargs:
            return

        if kwargs:
            with self.opts.act_internal_update():
                for key, value in kwargs.items():
                    if key in type(self.opts).__attrs__:
                        setattr(self.opts, key, value)

        # The operator is fixed to the current raw geometry. Rebuild it only
        # when this is a full reapplication, which includes initialization and
        # raw-surface changes. Pure cutoff/ratio edits reuse the same spectrum.
        if is_reapply_opts or getattr(self, "calc_kappa_max", None) is None:
            self._helper_prepare_initial_operator()

        self._helper_resolve_filter_parameters()


__all__ = [
    "OptsSmoothedSurface",
    "SmoothedSurface",
    "SurfaceSmoothingConfigError",
]
