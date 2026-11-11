"""Rigid ladder fit using assigned, optically corrected sensor facets.

All computations are in the declared C-frame-aligned measurement frame.  Sensor
geometry is never optimized independently.  Inlier filtering changes only the
registration objective; source rows are retained for diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .ladder_registration import RegistrationOptions, validate_registration_points
from .nominal_geometry import NominalLadder, rigid_matrix


@dataclass
class LabeledEvaluation:
    residuals_mm: np.ndarray
    sensor_indices: np.ndarray
    containment_margin_mm: np.ndarray
    inlier_mask: np.ndarray
    footprint_margin_mm: np.ndarray

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_mm[self.inlier_mask] ** 2)))


@dataclass
class LabeledRegistrationResult:
    rotation: np.ndarray
    translation_mm: np.ndarray
    pivot_mm: np.ndarray
    pivot_displacement_mm: np.ndarray
    initial: LabeledEvaluation
    final: LabeledEvaluation
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
    performance: dict

    @property
    def accepted(self) -> bool:
        return bool(self.feasible and self.optimizer_success)

    @property
    def homogeneous(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation_mm
        return matrix

    def summary(self, geometry: NominalLadder) -> dict:
        def metrics(e, sensor_index=None):
            selected = e.inlier_mask.copy()
            if sensor_index is not None:
                selected &= e.sensor_indices == sensor_index
            margin = e.containment_margin_mm[selected]
            residual = e.residuals_mm[selected]
            footprint = e.footprint_margin_mm[selected]
            violation = _footprint_violation(
                footprint, self.options.containment_tolerance_mm
            )
            return {
                "surface_rms_mm": float(np.sqrt(np.mean(residual**2))),
                "surface_max_abs_mm": float(np.max(np.abs(residual))),
                "outside_points": int(
                    np.count_nonzero(margin < -self.options.containment_tolerance_mm)
                ),
                "max_containment_violation_mm": float(max(0.0, -np.min(margin))),
                "outside_footprint_points": int(np.count_nonzero(violation)),
                "max_footprint_excess_mm": float(np.max(violation)),
                "footprint_penalty_sum_sq_mm2": float(
                    self.options.footprint_penalty_weight * np.dot(violation, violation)
                ),
            }

        return {
            "convention": "p_as_built = R @ p_nominal + T; both in measurement_frame",
            "transform_kind": "single rigid ladder correction in C-frame coordinates",
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
            "total_points": len(self.final.inlier_mask),
            "inlier_points": int(np.count_nonzero(self.final.inlier_mask)),
            "outlier_points": int(np.count_nonzero(~self.final.inlier_mask)),
            "uncertainty_sigma": self.options.uncertainty_sigma,
            "containment_tolerance_mm": self.options.containment_tolerance_mm,
            "footprint_penalty_weight": self.options.footprint_penalty_weight,
            "objective": "sum squared face distances + weight * sum squared worst-axis XY excess beyond tolerance",
            "initial": metrics(self.initial),
            "final": metrics(self.final),
            "sensor_surface_assignments": [
                {
                    "node_path": sensor.node_path,
                    "points": int(np.count_nonzero(self.final.sensor_indices == i)),
                    "inliers": int(
                        np.count_nonzero(
                            (self.final.sensor_indices == i) & self.final.inlier_mask
                        )
                    ),
                    "initial": metrics(self.initial, i),
                    "final": metrics(self.final, i),
                }
                for i, sensor in enumerate(geometry.sensors)
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
                "scope": "inlier surface residuals only; excludes containment inequalities",
            },
            "warnings": self.warnings,
            "performance": self.performance,
        }


def _plane_inliers(
    points: np.ndarray, threshold_sigma: float, approximate_normal: np.ndarray
) -> np.ndarray:
    """Iterated robust PCA; prevents remote scan points controlling the plane."""
    approximate_normal = approximate_normal / np.linalg.norm(approximate_normal)
    heights = points @ approximate_normal
    median = np.median(heights)
    sigma = 1.4826 * np.median(np.abs(heights - median))
    mask = np.abs(heights - median) <= threshold_sigma * max(float(sigma), 1e-6)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Too few points remain after initial robust height filtering")
    for _ in range(4):
        data = points[mask]
        if len(data) < 3:
            raise ValueError("Too few points remain in sensor plane fit.")
        center = np.mean(data, axis=0)
        centered = data - center
        scatter = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(scatter)
        distance = (points - center) @ eigenvectors[:, 0]
        med = np.median(distance[mask])
        sigma = 1.4826 * np.median(np.abs(distance[mask] - med))
        # Numerical floor avoids throwing away noiseless points.
        limit = threshold_sigma * max(float(sigma), 1e-4)
        updated = np.abs(distance - med) <= limit
        if np.count_nonzero(updated) < 3:
            raise ValueError("Plane outlier filtering rejected nearly all points.")
        if np.array_equal(updated, mask):
            break
        mask = updated
    return mask


@dataclass(frozen=True)
class _SensorStatistics:
    """Sufficient statistics for exact unweighted squared face-plane distances."""

    counts: np.ndarray
    means: np.ndarray
    scatter_roots: np.ndarray


def _sensor_statistics(
    points: np.ndarray, ids: np.ndarray, inliers: np.ndarray, n_sensors: int
) -> _SensorStatistics:
    counts = np.zeros(n_sensors, dtype=int)
    means = np.empty((n_sensors, 3))
    roots = np.empty((n_sensors, 3, 3))
    for index in range(n_sensors):
        selected = points[(ids == index) & inliers]
        counts[index] = len(selected)
        if not len(selected):
            raise ValueError("Each sensor requires inliers for compressed registration")
        mean = selected.mean(axis=0)
        centered = selected - mean
        scatter = centered.T @ centered
        values, vectors = np.linalg.eigh(scatter)
        means[index] = mean
        # ||roots @ n||^2 = n.T @ scatter @ n.  Unlike a raw second moment,
        # centered scatter avoids cancellation for large detector-frame offsets.
        roots[index] = np.sqrt(np.maximum(values, 0))[:, None] * vectors.T
    return _SensorStatistics(counts, means, roots)


def _compressed_residual(
    statistics: _SensorStatistics, normals: np.ndarray, face_centers: np.ndarray
) -> np.ndarray:
    """Exactly preserves sum of squared point-to-plane residuals for fixed labels.

    Four residuals per sensor replace an arbitrarily dense point cloud. This
    equivalence applies to the surface term only. Full-point footprint penalties
    are appended separately and cannot be compressed using these statistics.
    """
    offsets = np.einsum("ij,ij->i", statistics.means - face_centers, normals)
    variation = np.einsum("nij,nj->ni", statistics.scatter_roots, normals)
    return np.column_stack((np.sqrt(statistics.counts) * offsets, variation)).ravel()


def _footprint_violation(margin: np.ndarray, tolerance: float) -> np.ndarray:
    """Worst-axis XY excess in mm beyond the same allowance used for acceptance."""
    return np.maximum(0.0, -margin - tolerance)


class LabeledFaceRegistrar:
    """Optimize one six-parameter pose against preassigned outward sensor faces.

    `sensor_indices` are **ROOT geometry sensor-list indices**, not unverified
    manufacturing names. Caller must supply optically corrected geometry and
    establish the mapping before fitting; raw ROOT geometry is not a fit target.
    """

    def __init__(
        self,
        options: RegistrationOptions | None = None,
        *,
        facet_z_mm: float = 0.0,
        outlier_sigma: float = 3.0,
        logger: logging.Logger | None = None,
    ):
        self.options = options or RegistrationOptions()
        if not np.isfinite(facet_z_mm):
            raise ValueError("facet_z_mm must be finite")
        if not np.isfinite(outlier_sigma) or outlier_sigma <= 0:
            raise ValueError("outlier_sigma must be positive and finite")
        self.facet_z_mm = float(facet_z_mm)
        self.outlier_sigma = float(outlier_sigma)
        self.logger = logger or logging.getLogger(__name__)

    def fit(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        sensor_indices: np.ndarray,
        covariances: np.ndarray | None = None,
        initial_transform: np.ndarray | None = None,
    ) -> LabeledRegistrationResult:
        started = perf_counter()
        points, covariances = validate_registration_points(points, covariances)
        ids = np.asarray(sensor_indices)
        if ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("sensor_indices must contain one integer per point")
        if np.any(ids < 0) or np.any(ids >= len(geometry.sensors)):
            raise ValueError("sensor_indices outside nominal sensor inventory")
        if len(np.unique(ids)) != len(geometry.sensors):
            raise ValueError("Every nominal sensor must have a labeled measured cloud")
        initial = rigid_matrix(
            np.eye(4) if initial_transform is None else initial_transform,
            "registration.initial_transform",
        )

        # Precompute exactly one outward face per sensor, once per registration.
        centers = np.stack([sensor.center_mm for sensor in geometry.sensors])
        halfsize = np.stack([sensor.dimensions_mm / 2 for sensor in geometry.sensors])
        axes = np.stack([sensor.rotation for sensor in geometry.sensors])
        normal_z = axes[:, 2, 2]
        if np.any(np.abs(normal_z) < 0.99):
            raise ValueError(
                "Sensor local Z must align with C-frame Z for labeled-face fit"
            )
        offsets = centers[:, 2] - self.facet_z_mm
        if np.any(np.abs(offsets) < 1e-8):
            raise ValueError(
                "Cannot determine outward sensor face at the C-frame facet"
            )
        signs = np.sign(offsets * normal_z)
        normals = axes[:, :, 2] * signs[:, None]
        face_centers = centers + normals * halfsize[:, 2, None]
        sensor_masks = [ids == i for i in range(len(geometry.sensors))]
        inliers = np.zeros(len(points), dtype=bool)
        filtering_started = perf_counter()
        for group in sensor_masks:
            if np.count_nonzero(group) < 3:
                raise ValueError(
                    "Each sensor must contain at least three measured points"
                )
            sensor_index = int(ids[np.flatnonzero(group)[0]])
            inliers[group] = _plane_inliers(
                points[group], self.outlier_sigma, normals[sensor_index]
            )
            self.logger.info(
                "Ladder registration | phase=plane_filter | sensor=%s | points=%d | inliers=%d | outliers=%d",
                geometry.sensors[sensor_index].node_path,
                int(group.sum()),
                int(inliers[group].sum()),
                int(group.sum() - inliers[group].sum()),
            )
        filtering_seconds = perf_counter() - filtering_started
        statistics_started = perf_counter()
        statistics = _sensor_statistics(points, ids, inliers, len(geometry.sensors))
        statistics_seconds = perf_counter() - statistics_started
        self.logger.info(
            "Ladder registration | phase=preprocess | points=%d | sensors=%d | inliers=%d | outliers=%d | filtering_s=%.3f | statistics_s=%.3f | optimizer_residuals=%d",
            len(points),
            len(geometry.sensors),
            int(inliers.sum()),
            int((~inliers).sum()),
            filtering_seconds,
            statistics_seconds,
            4 * len(geometry.sensors) + int(inliers.sum()),
        )

        pivot = centers.mean(axis=0)
        start_r, start_t = initial[:3, :3], initial[:3, 3]
        start_pivot = start_r @ pivot + start_t
        scale = max(
            1.0, float(np.max(np.linalg.norm(geometry.corners_mm - pivot, axis=1)))
        )
        limits = np.array(
            [scale * np.deg2rad(self.options.max_rotation_deg)] * 3
            + [self.options.max_translation_mm] * 3
        )

        def pose(p):
            rotation = Rotation.from_rotvec(p[:3] / scale).as_matrix() @ start_r
            translation = start_pivot + p[3:] - rotation @ pivot
            return rotation, translation

        def face_margins(rotation, translation, measured, assigned, covariance):
            moved_axes = np.einsum("ij,njk->nik", rotation, axes)[assigned]
            local = np.einsum(
                "ni,nij->nj",
                measured - (centers @ rotation.T + translation)[assigned],
                moved_axes,
            )
            # The selected outward face, never the back or nearest box side.
            face_residual = signs[assigned] * local[:, 2] - halfsize[assigned, 2]
            padding = np.zeros_like(local)
            if covariance is not None:
                padding = self.options.uncertainty_sigma * np.sqrt(
                    np.maximum(
                        0,
                        np.einsum(
                            "nia,nij,nja->na", moved_axes, covariance, moved_axes
                        ),
                    )
                )
            lateral = np.min(
                halfsize[assigned, :2] + padding[:, :2] - np.abs(local[:, :2]), axis=1
            )
            normal = padding[:, 2] - np.abs(face_residual)
            return face_residual, lateral, normal

        fit_points, fit_ids = points[inliers], ids[inliers]
        fit_covariances = None if covariances is None else covariances[inliers]
        penalty_scale = np.sqrt(self.options.footprint_penalty_weight)
        surface_residual_count = 4 * len(geometry.sensors)

        def residual(p):
            rotation, translation = pose(p)
            moved_normals = normals @ rotation.T
            moved_faces = face_centers @ rotation.T + translation
            surface = _compressed_residual(statistics, moved_normals, moved_faces)
            _, lateral, _ = face_margins(
                rotation, translation, fit_points, fit_ids, fit_covariances
            )
            penalty = penalty_scale * _footprint_violation(
                lateral, self.options.containment_tolerance_mm
            )
            return np.concatenate((surface, penalty))

        optimizer_started = perf_counter()
        optimization = least_squares(
            residual,
            np.zeros(6),
            bounds=(-limits, limits),
            max_nfev=self.options.max_iterations,
            method="trf",
        )
        optimizer_seconds = perf_counter() - optimizer_started
        self.logger.info(
            "Ladder registration | phase=optimizer | success=%s | nfev=%d | njev=%s | cost=%.9g | elapsed_s=%.3f | message=%s",
            bool(optimization.success),
            int(optimization.nfev),
            str(optimization.njev),
            float(optimization.cost),
            optimizer_seconds,
            str(optimization.message).replace("\n", " "),
        )
        r, t = pose(optimization.x)

        def evaluation(p):
            rr, tt = pose(p)
            face_residual, lateral, normal = face_margins(
                rr, tt, points, ids, covariances
            )
            margin = np.minimum(lateral, normal)
            return LabeledEvaluation(
                face_residual, ids.copy(), margin, inliers.copy(), lateral
            )

        validation_started = perf_counter()
        initial_eval, final_eval = evaluation(np.zeros(6)), evaluation(optimization.x)
        feasible = bool(
            np.all(
                final_eval.containment_margin_mm[inliers]
                >= -self.options.containment_tolerance_mm
            )
        )
        # Only the leading surface rows describe surface observability. Including
        # footprint rows here would mislabel containment information as plane rank.
        jac = optimization.jac[:surface_residual_count]
        _, singular, vh = np.linalg.svd(jac, full_matrices=True)
        threshold = max(1e-8, singular[0] * 1e-6) if len(singular) else 1e-8
        rank = int(np.count_nonzero(singular > threshold))
        warnings = []
        excluded = int(np.count_nonzero(~inliers))
        if excluded:
            warnings.append(
                f"Excluded {excluded} sensor-plane outliers from fit and containment; original rows retained"
            )
        if rank < 6:
            warnings.append(
                f"Surface residuals constrain only {rank}/6 local pose directions"
            )
        if not feasible:
            warnings.append(
                "Inlier face-band / footprint containment failed; fit is diagnostic only"
            )
        if not optimization.success:
            warnings.append("Optimizer did not converge; fit is diagnostic only")
        if covariances is None:
            warnings.append(
                "No measurement covariance: face band has zero uncertainty width"
            )
        validation_seconds = perf_counter() - validation_started
        elapsed_seconds = perf_counter() - started
        performance = {
            "filtering_seconds": filtering_seconds,
            "statistics_seconds": statistics_seconds,
            "optimizer_seconds": optimizer_seconds,
            "validation_seconds": validation_seconds,
            "total_seconds": elapsed_seconds,
            "optimizer_residual_count": surface_residual_count + int(inliers.sum()),
            "surface_residual_count": surface_residual_count,
            "footprint_residual_count": int(inliers.sum()),
            "optimizer_nfev": int(optimization.nfev),
            "optimizer_njev": int(optimization.njev)
            if optimization.njev is not None
            else None,
        }
        self.logger.info(
            "Ladder registration | phase=validation | accepted=%s | feasible=%s | inlier_rms_mm=%.9g | outside_inliers=%d | rank=%d/6 | elapsed_s=%.3f | total_s=%.3f",
            bool(feasible and optimization.success),
            feasible,
            final_eval.rms_mm,
            int(
                np.count_nonzero(
                    final_eval.containment_margin_mm[inliers]
                    < -self.options.containment_tolerance_mm
                )
            ),
            rank,
            validation_seconds,
            elapsed_seconds,
        )
        return LabeledRegistrationResult(
            r,
            t,
            pivot,
            r @ pivot + t - pivot,
            initial_eval,
            final_eval,
            feasible,
            bool(optimization.success),
            str(optimization.message),
            int(optimization.nfev),
            rank,
            singular,
            vh[rank:],
            scale,
            warnings,
            self.options,
            performance,
        )
