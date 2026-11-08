"""Local rigid ladder registration with all-point sensor-box containment."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

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


@dataclass
class RegistrationEvaluation:
    residuals_mm: np.ndarray
    sensor_indices: np.ndarray
    containment_margin_mm: np.ndarray

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_mm**2)))


@dataclass
class LadderRegistrationResult:
    rotation: np.ndarray
    translation_mm: np.ndarray
    pivot_mm: np.ndarray
    pivot_displacement_mm: np.ndarray
    initial: RegistrationEvaluation
    final: RegistrationEvaluation
    feasible: bool
    optimizer_success: bool
    optimizer_message: str
    iterations: int
    surface_jacobian_rank: int
    surface_jacobian_singular_values: np.ndarray
    surface_nullspace_scaled: np.ndarray
    rotation_scale_mm: float
    warnings: list[str]
    options: RegistrationOptions

    @property
    def accepted(self) -> bool:
        return self.feasible and self.optimizer_success

    @property
    def homogeneous(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation_mm
        return matrix

    def summary(self, geometry: NominalLadder) -> dict:
        def metrics(evaluation):
            return {
                "surface_rms_mm": evaluation.rms_mm,
                "surface_max_abs_mm": float(np.max(np.abs(evaluation.residuals_mm))),
                "outside_points": int(
                    np.count_nonzero(
                        evaluation.containment_margin_mm
                        < -self.options.containment_tolerance_mm
                    )
                ),
                "max_containment_violation_mm": float(
                    max(0.0, -np.min(evaluation.containment_margin_mm))
                ),
            }

        return {
            "convention": "p_as_built = R @ p_nominal + T; both in measurement_frame",
            "transform_kind": "reference-frame rigid correction, not ladder-local pose",
            "measurement_frame": geometry.frame,
            "nominal_ladder_node": geometry.node_path,
            "rotation_matrix": self.rotation.tolist(),
            "rotation_vector_rad": Rotation.from_matrix(self.rotation)
            .as_rotvec()
            .tolist(),
            "translation_vector_mm": self.translation_mm.tolist(),
            "homogeneous_matrix": self.homogeneous.tolist(),
            "nominal_pivot_mm": self.pivot_mm.tolist(),
            "pivot_displacement_mm": self.pivot_displacement_mm.tolist(),
            "accepted": self.accepted,
            "feasible": self.feasible,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "iterations": self.iterations,
            "total_points": len(self.final.residuals_mm),
            "uncertainty_sigma": self.options.uncertainty_sigma,
            "containment_tolerance_mm": self.options.containment_tolerance_mm,
            "initial": metrics(self.initial),
            "final": metrics(self.final),
            "sensor_surface_assignments": [
                {
                    "node_path": sensor.node_path,
                    "points": int(np.count_nonzero(self.final.sensor_indices == index)),
                }
                for index, sensor in enumerate(geometry.sensors)
            ],
            "observability": {
                "locally_surface_identifiable": self.surface_jacobian_rank == 6,
                "surface_jacobian_rank": self.surface_jacobian_rank,
                "singular_values": self.surface_jacobian_singular_values.tolist(),
                "nullspace_scaled": self.surface_nullspace_scaled.tolist(),
                "parameter_order": [
                    "scale_rx",
                    "scale_ry",
                    "scale_rz",
                    "dx",
                    "dy",
                    "dz",
                ],
                "rotation_scale_mm": self.rotation_scale_mm,
                "scope": "local surface residuals only; excludes containment inequalities and global ambiguities",
            },
            "warnings": self.warnings,
        }


class LadderRegistrar:
    """Fit one rigid correction without moving measured points or rejecting any.

    Objective: mean squared signed distance to the nearest sensor box surface.
    Constraint: every point belongs to the union of uncertainty-expanded oriented
    sensor boxes. Marginal k-sigma extents are projected onto each sensor axis;
    this is a box envelope, not an exact ellipsoid-intersection confidence test.
    """

    def __init__(self, options: RegistrationOptions | None = None):
        self.options = options or RegistrationOptions()

    @staticmethod
    def _validate_points(points, covariances):
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
        points, covariances = self._validate_points(points, covariances)
        transform = rigid_matrix(
            np.eye(4) if initial_transform is None else initial_transform,
            "registration.initial_transform",
        )
        return self.evaluate(
            geometry, points, transform[:3, :3], transform[:3, 3], covariances
        )

    def fit(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        covariances: np.ndarray | None = None,
        initial_transform: np.ndarray | None = None,
    ) -> LadderRegistrationResult:
        points, covariances = self._validate_points(points, covariances)
        initial_transform = rigid_matrix(
            np.eye(4) if initial_transform is None else initial_transform,
            "registration.initial_transform",
        )
        initial_rotation = initial_transform[:3, :3]
        initial_translation = initial_transform[:3, 3]
        pivot = np.mean([s.center_mm for s in geometry.sensors], axis=0)
        transformed_pivot = initial_rotation @ pivot + initial_translation
        scale = max(
            1.0, float(np.max(np.linalg.norm(geometry.corners_mm - pivot, axis=1)))
        )
        limits = np.array(
            [scale * np.deg2rad(self.options.max_rotation_deg)] * 3
            + [self.options.max_translation_mm] * 3
        )

        def pose(parameters):
            rotation = (
                Rotation.from_rotvec(parameters[:3] / scale).as_matrix()
                @ initial_rotation
            )
            translation = transformed_pivot + parameters[3:] - rotation @ pivot
            return rotation, translation

        def evaluate(parameters):
            rotation, translation = pose(parameters)
            return self.evaluate(geometry, points, rotation, translation, covariances)

        start = np.zeros(6)
        initial = evaluate(start)

        def warm_residual(parameters):
            evaluation = evaluate(parameters)
            return np.concatenate(
                (
                    evaluation.residuals_mm,
                    10 * np.minimum(evaluation.containment_margin_mm, 0),
                )
            )

        # A penalty-only fit supplies a starting point; acceptance is always based
        # on the subsequent hard constrained fit and an independent all-point check.
        warm = least_squares(
            warm_residual,
            start,
            bounds=(-limits, limits),
            max_nfev=self.options.max_iterations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        optimization = minimize(
            lambda p: np.mean(evaluate(p).residuals_mm ** 2),
            warm.x,
            method="SLSQP",
            bounds=list(zip(-limits, limits)),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda p: (
                        evaluate(p).containment_margin_mm
                        + self.options.containment_tolerance_mm
                    ),
                }
            ],
            options={"maxiter": self.options.max_iterations, "ftol": 1e-12},
        )
        rotation, translation = pose(optimization.x)
        final = evaluate(optimization.x)
        feasible = bool(
            np.all(
                final.containment_margin_mm >= -self.options.containment_tolerance_mm
            )
        )
        # Scale rotation by the ladder lever arm so rank does not depend simply
        # on choosing radians versus millimetres. This is not a pose covariance.
        step = 1e-4
        jacobian = np.column_stack(
            [
                (
                    evaluate(optimization.x + np.eye(6)[axis] * step).residuals_mm
                    - evaluate(optimization.x - np.eye(6)[axis] * step).residuals_mm
                )
                / (2 * step)
                for axis in range(6)
            ]
        )
        # Reduced SVD avoids an N x N allocation for dense measured clouds.
        _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
        rank = int(
            np.count_nonzero(singular_values > max(1e-8, singular_values[0] * 1e-6))
        )
        if len(points) < 6:
            # Complete only the tiny right-singular-vector matrix in this case.
            _, _, vh = np.linalg.svd(jacobian, full_matrices=True)
        warnings = [
            "Unlabelled sensor surfaces may admit alternative sensor/face matches "
            "(including front/back). A local fit does not establish global uniqueness."
        ]
        if rank < 6:
            warnings.append(
                f"Surface residuals constrain only {rank}/6 local pose directions. "
                "Containment may bound but does not uniquely measure the remaining directions."
            )
        if np.any(np.abs(optimization.x) >= limits * 0.999):
            warnings.append(
                "Fit reaches a configured search bound; inspect initialization and bounds."
            )
        if not feasible:
            warnings.append(
                "All-point containment failed: candidate transform must not be used as an accepted registration."
            )
        if not optimization.success:
            warnings.append(
                "Constrained optimizer did not converge; candidate transform is diagnostic only."
            )
        if covariances is None:
            warnings.append(
                "No point covariance supplied: containment uses unexpanded sensor boxes."
            )
        return LadderRegistrationResult(
            rotation,
            translation,
            pivot,
            rotation @ pivot + translation - pivot,
            initial,
            final,
            feasible,
            bool(optimization.success),
            str(optimization.message),
            int(optimization.nit),
            rank,
            singular_values,
            vh[rank:],
            scale,
            warnings,
            self.options,
        )
