"""Shared registration options, input validation and preview-only box distances.

Ladder fitting lives in labeled_registration and uses fixed sensor facets only.
The box evaluator below never optimizes a pose or establishes sensor labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .nominal_geometry import NominalLadder, rigid_matrix


@dataclass(frozen=True)
class RegistrationOptions:
    uncertainty_sigma: float = 3.0
    containment_tolerance_mm: float = 1e-6
    max_translation_mm: float = 5.0
    max_rotation_deg: float = 5.0
    max_iterations: int = 300

    def __post_init__(self) -> None:
        for name in (
            "uncertainty_sigma",
            "containment_tolerance_mm",
            "max_translation_mm",
            "max_rotation_deg",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"registration.{name} must be finite and positive.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError("registration.max_iterations must be a positive integer.")


def validate_registration_points(points, covariances):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(
            "Registration requires at least three points with shape (N, 3)."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Registration points must be finite.")
    if covariances is not None:
        covariances = np.asarray(covariances, dtype=float)
        if covariances.shape != (len(points), 3, 3) or not np.all(
            np.isfinite(covariances)
        ):
            raise ValueError(
                "Registration covariances must be finite with shape (N, 3, 3)."
            )
        if not np.allclose(
            covariances, covariances.swapaxes(1, 2), atol=1e-12, rtol=0
        ):
            raise ValueError("Registration covariances must be symmetric.")
        if np.any(np.linalg.eigvalsh(covariances) < -1e-12):
            raise ValueError(
                "Registration covariances must be positive semidefinite."
            )
    return points, covariances


@dataclass
class RegistrationEvaluation:
    residuals_mm: np.ndarray
    sensor_indices: np.ndarray
    containment_margin_mm: np.ndarray

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_mm**2)))


class SensorBoxEvaluator:
    """Preview-only distances to boxes; no fitting or verified correspondence.

    Marginal k-sigma extents are projected onto each sensor axis; this is a box
    envelope, not an exact ellipsoid-intersection confidence test. These nearest
    box diagnostics must not be used as registration labels or acceptance tests.
    """

    def __init__(self, options: RegistrationOptions | None = None):
        self.options = options or RegistrationOptions()

    def evaluate(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        rotation: np.ndarray,
        translation_mm: np.ndarray,
        covariances: np.ndarray | None = None,
    ) -> RegistrationEvaluation:
        """Evaluate validated inputs, retaining every measured point."""
        best_residual = np.full(len(points), np.inf)
        assignments = np.zeros(len(points), dtype=int)
        union_margin = np.full(len(points), -np.inf)
        for index, sensor in enumerate(geometry.sensors):
            axes = rotation @ sensor.rotation
            center = rotation @ sensor.center_mm + translation_mm
            local = (points - center) @ axes
            q = np.abs(local) - sensor.dimensions_mm / 2
            # Standard signed distance to an oriented box; zero only on its surface.
            sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(
                np.max(q, axis=1), 0
            )
            nearer = np.abs(sdf) < np.abs(best_residual)
            best_residual[nearer] = sdf[nearer]
            assignments[nearer] = index
            padding = 0.0
            if covariances is not None:
                variances = np.einsum("ia,nij,ja->na", axes, covariances, axes)
                padding = self.options.uncertainty_sigma * np.sqrt(
                    np.maximum(variances, 0)
                )
            margin = np.min(padding - q, axis=1)
            union_margin = np.maximum(union_margin, margin)
        return RegistrationEvaluation(best_residual, assignments, union_margin)

    def evaluate_initial(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        covariances: np.ndarray | None = None,
        initial_transform: np.ndarray | None = None,
    ) -> RegistrationEvaluation:
        """Evaluate a supplied pose without optimizing or computing a Jacobian."""
        points, covariances = validate_registration_points(points, covariances)
        transform = rigid_matrix(
            np.eye(4) if initial_transform is None else initial_transform,
            "registration.initial_transform",
        )
        return self.evaluate(
            geometry, points, transform[:3, :3], transform[:3, 3], covariances
        )
