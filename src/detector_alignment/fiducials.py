from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .models import PointCloudData


@dataclass(frozen=True)
class FiducialMarkerResult:
    """Repeated measurements and statistics for one physical fiducial marker."""

    marker_id: str
    measurement_indices: np.ndarray
    measurement_points: np.ndarray
    point_covariances: np.ndarray
    mean: np.ndarray
    measurement_covariance: np.ndarray | None
    mean_covariance: np.ndarray

    @property
    def measurement_stddev(self) -> np.ndarray | None:
        if self.measurement_covariance is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.measurement_covariance), 0.0))

    @property
    def mean_uncertainty_stddev(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.mean_covariance), 0.0))

    def transformed(
        self,
        linear: np.ndarray,
        translation: np.ndarray,
    ) -> FiducialMarkerResult:
        """Apply a finite affine transformation, including optional scaling."""
        A, t = _validate_affine_transform(linear, translation)
        measurement_covariance = None
        if self.measurement_covariance is not None:
            measurement_covariance = A @ self.measurement_covariance @ A.T

        return FiducialMarkerResult(
            marker_id=self.marker_id,
            measurement_indices=self.measurement_indices.copy(),
            measurement_points=self.measurement_points @ A.T + t,
            point_covariances=np.einsum(
                "ij,njk,lk->nil",
                A,
                self.point_covariances,
                A,
            ),
            mean=A @ self.mean + t,
            measurement_covariance=measurement_covariance,
            mean_covariance=A @ self.mean_covariance @ A.T,
        )


@dataclass(frozen=True)
class FiducialMeasurementResult:
    """Clustered fiducial measurements in one coordinate frame."""

    source: Path
    fiducial_min_distance: float
    markers: tuple[FiducialMarkerResult, ...]

    @property
    def means(self) -> np.ndarray:
        return np.vstack([marker.mean for marker in self.markers])

    def transformed(
        self,
        linear: np.ndarray,
        translation: np.ndarray,
    ) -> FiducialMeasurementResult:
        A, t = _validate_affine_transform(linear, translation)
        return FiducialMeasurementResult(
            source=self.source,
            fiducial_min_distance=self.fiducial_min_distance,
            markers=tuple(marker.transformed(A, t) for marker in self.markers),
        )


class FiducialProcessor:
    """Cluster repeated measurements and calculate per-marker statistics."""

    def process(
        self,
        data: PointCloudData,
        fiducial_min_distance: float,
    ) -> FiducialMeasurementResult:
        distance = _validate_min_distance(fiducial_min_distance)
        points = np.asarray(data.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("Fiducial points must have non-empty shape (N, 3).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Fiducial points must be finite.")
        if data.covariances is None:
            raise ValueError(
                "Fiducial uncertainty is required. Configure fiducials.uncertainty "
                "or the global uncertainty block."
            )

        covariances = np.asarray(data.covariances, dtype=float)
        if covariances.shape != (len(points), 3, 3):
            raise ValueError(
                "Fiducial covariances must have shape "
                f"({len(points)}, 3, 3); got {covariances.shape}."
            )
        if not np.all(np.isfinite(covariances)):
            raise ValueError("Fiducial covariances must be finite.")

        labels = self._cluster_labels(points, distance)
        grouped_indices = [
            np.flatnonzero(labels == label)
            for label in np.unique(labels)
        ]
        grouped_indices.sort(key=lambda indices: int(indices[0]))

        markers = tuple(
            self._summarize_marker(
                marker_number,
                indices,
                points,
                covariances,
            )
            for marker_number, indices in enumerate(grouped_indices, start=1)
        )
        return FiducialMeasurementResult(
            source=Path(data.source),
            fiducial_min_distance=distance,
            markers=markers,
        )

    @staticmethod
    def _cluster_labels(points: np.ndarray, distance: float) -> np.ndarray:
        if len(points) == 1:
            return np.ones(1, dtype=int)

        hierarchy = linkage(points, method="complete", metric="euclidean")
        strict_threshold = np.nextafter(distance, -np.inf)
        return fcluster(
            hierarchy,
            t=strict_threshold,
            criterion="distance",
        )

    @staticmethod
    def _summarize_marker(
        marker_number: int,
        indices: np.ndarray,
        points: np.ndarray,
        covariances: np.ndarray,
    ) -> FiducialMarkerResult:
        marker_points = points[indices]
        marker_covariances = covariances[indices]
        count = len(indices)

        measurement_covariance = None
        if count > 1:
            measurement_covariance = np.asarray(
                np.cov(marker_points, rowvar=False, ddof=1),
                dtype=float,
            )

        return FiducialMarkerResult(
            marker_id=f"fiducial_{marker_number:03d}",
            measurement_indices=indices.copy(),
            measurement_points=marker_points.copy(),
            point_covariances=marker_covariances.copy(),
            mean=np.mean(marker_points, axis=0),
            measurement_covariance=measurement_covariance,
            mean_covariance=np.sum(marker_covariances, axis=0) / (count**2),
        )


def _validate_min_distance(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("fiducial_min_distance must be a finite positive number.")
    try:
        distance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fiducial_min_distance must be a finite positive number."
        ) from exc
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("fiducial_min_distance must be a finite positive number.")
    return distance


def _validate_affine_transform(
    linear: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(linear, dtype=float)
    t = np.asarray(translation, dtype=float)
    if A.shape != (3, 3):
        raise ValueError(f"Transformation linear matrix must have shape (3, 3); got {A.shape}.")
    if t.shape != (3,):
        raise ValueError(f"Transformation translation must have shape (3,); got {t.shape}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(t)):
        raise ValueError("Transformation contains NaN or infinite values.")
    return A, t
