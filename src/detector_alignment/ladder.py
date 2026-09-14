from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .models import PlaneFitResult


@dataclass
class SensorPointCloud:
    """Point cloud belonging to one ladder sensor."""

    name: str
    points: np.ndarray
    covariances: np.ndarray | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"Sensor {self.name!r}: points must have shape (N, 3); "
                f"got {self.points.shape}."
            )

        if len(self.points) < 3:
            raise ValueError(f"Sensor {self.name!r}: at least 3 points are required.")

        if not np.all(np.isfinite(self.points)):
            raise ValueError(
                f"Sensor {self.name!r}: points contain NaN or infinite values."
            )

        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances, dtype=float)

            expected = (len(self.points), 3, 3)
            if self.covariances.shape != expected:
                raise ValueError(
                    f"Sensor {self.name!r}: covariances must have shape "
                    f"{expected}; got {self.covariances.shape}."
                )

            if not np.all(np.isfinite(self.covariances)):
                raise ValueError(
                    f"Sensor {self.name!r}: covariances contain NaN or infinite values."
                )


@dataclass
class SensorPlaneResult:
    """Plane-fit result for one ladder sensor."""

    name: str
    points: np.ndarray
    plane: PlaneFitResult
    source: Path | None = None

    @property
    def residuals_mm(self) -> np.ndarray:
        """Signed orthogonal distance of every point to the fitted plane."""
        return self.points @ self.plane.normal + self.plane.d

    @property
    def inlier_residuals_mm(self) -> np.ndarray:
        return self.residuals_mm[self.plane.inlier_mask]


AutomaticSegmenter = Callable[
    [np.ndarray, np.ndarray | None, dict],
    Iterable[SensorPointCloud],
]


class LadderProcessor:
    """
    Ladder point-cloud processing.

    Two segmentation modes are supported:

    1. ``segmented_files``
       One CloudCompare-exported file per sensor.

    2. ``automatic``
       One complete ladder point cloud, segmented by a callable supplied
       through ``automatic_segmenter``.

    Everything after segmentation is shared: transformation to detector
    coordinates and per-sensor plane fitting.
    """

    def __init__(
        self,
        loader,
        plane_fitter,
        resolve_path: Callable[[str | Path], Path] | None = None,
        automatic_segmenter: AutomaticSegmenter | None = None,
    ) -> None:
        self.loader = loader
        self.plane_fitter = plane_fitter
        self.resolve_path = resolve_path or (lambda p: Path(p))
        self.automatic_segmenter = automatic_segmenter

    def load_segmented_files(
        self,
        sensor_configs: list[dict],
        default_uncertainty: dict | None = None,
    ) -> list[SensorPointCloud]:
        """
        Load one ASCII point-cloud file per sensor.

        Expected configuration entry::

            - name: S0
              file: data/ladder/S0.xyz
              uncertainty:   # optional, overrides default
                mode: none
        """
        if not sensor_configs:
            raise ValueError(
                "segmented_files mode requires a non-empty 'sensors' list."
            )

        sensors: list[SensorPointCloud] = []
        names: set[str] = set()

        for index, cfg in enumerate(sensor_configs):
            if not isinstance(cfg, dict):
                raise ValueError(f"Sensor entry {index} must be a mapping/dictionary.")

            name = str(cfg.get("name", "")).strip()
            if not name:
                raise ValueError(f"Sensor entry {index} requires a non-empty 'name'.")

            if name in names:
                raise ValueError(f"Duplicate sensor name: {name!r}.")
            names.add(name)

            filename = cfg.get("file")
            if not filename:
                raise ValueError(f"Sensor {name!r} requires a 'file' entry.")

            path = Path(self.resolve_path(filename))

            uncertainty = cfg.get("uncertainty", default_uncertainty)
            cloud = self.loader.load(path, uncertainty)

            sensors.append(
                SensorPointCloud(
                    name=name,
                    points=cloud.points,
                    covariances=cloud.covariances,
                    source=cloud.source,
                )
            )

        return sensors

    def load_and_segment_automatically(
        self,
        filename: str | Path,
        segmentation_config: dict | None = None,
        uncertainty: dict | None = None,
    ) -> list[SensorPointCloud]:
        """
        Load a complete ladder cloud and run an injected automatic segmenter.

        The automatic segmenter callable must have this signature::

            segmenter(points, covariances, config)
                -> Iterable[SensorPointCloud]

        This deliberately keeps cbm_sts_tools-specific code outside the
        generic ladder-processing module.
        """
        if self.automatic_segmenter is None:
            raise RuntimeError(
                "Automatic ladder segmentation was requested, but no "
                "automatic_segmenter was configured. Wire the "
                "cbm_sts_tools.metrology segmenter through the adapter."
            )

        path = Path(self.resolve_path(filename))
        cloud = self.loader.load(path, uncertainty)

        cfg = segmentation_config or {}
        segmented = list(
            self.automatic_segmenter(
                cloud.points,
                cloud.covariances,
                cfg,
            )
        )

        if not segmented:
            raise RuntimeError(
                f"Automatic segmentation produced no sensors for {path}."
            )

        names: set[str] = set()
        validated: list[SensorPointCloud] = []

        for item in segmented:
            if not isinstance(item, SensorPointCloud):
                raise TypeError(
                    "Automatic segmenter must return SensorPointCloud objects."
                )

            if item.name in names:
                raise ValueError(
                    f"Automatic segmenter returned duplicate sensor name {item.name!r}."
                )
            names.add(item.name)

            if item.source is None:
                item.source = path

            validated.append(item)

        return validated

    def get_sensor_clouds(
        self,
        ladder_config: dict,
        default_uncertainty: dict | None = None,
    ) -> list[SensorPointCloud]:
        """Dispatch to either supported segmentation mode."""
        segmentation = ladder_config.get("segmentation", {})
        mode = str(segmentation.get("mode", "")).strip()

        if mode == "segmented_files":
            return self.load_segmented_files(
                ladder_config.get("sensors", []),
                default_uncertainty=default_uncertainty,
            )

        if mode == "automatic":
            filename = ladder_config.get("file")
            if not filename:
                raise ValueError(
                    "automatic ladder segmentation requires 'ladder_analysis.file'."
                )

            return self.load_and_segment_automatically(
                filename=filename,
                segmentation_config=segmentation.get("config", {}),
                uncertainty=ladder_config.get(
                    "uncertainty",
                    default_uncertainty,
                ),
            )

        raise ValueError(
            "Unknown ladder segmentation mode "
            f"{mode!r}. Expected 'segmented_files' or 'automatic'."
        )

    @staticmethod
    def transform_sensor(
        sensor: SensorPointCloud,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> SensorPointCloud:
        """Transform one sensor cloud into detector coordinates."""
        R = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float)

        if R.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3); got {R.shape}.")
        if t.shape != (3,):
            raise ValueError(f"translation must have shape (3,); got {t.shape}.")

        points = sensor.points @ R.T + t

        covariances = None
        if sensor.covariances is not None:
            # C' = R C R^T for every point.
            covariances = np.einsum(
                "ij,njk,lk->nil",
                R,
                sensor.covariances,
                R,
            )

        return SensorPointCloud(
            name=sensor.name,
            points=points,
            covariances=covariances,
            source=sensor.source,
        )

    def transform_sensors(
        self,
        sensors: Iterable[SensorPointCloud],
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> list[SensorPointCloud]:
        return [
            self.transform_sensor(sensor, rotation, translation) for sensor in sensors
        ]

    def fit_sensor_planes(
        self,
        sensors: Iterable[SensorPointCloud],
    ) -> list[SensorPlaneResult]:
        """Fit the existing RobustPlaneFitter independently to each sensor."""
        results: list[SensorPlaneResult] = []

        for sensor in sensors:
            plane = self.plane_fitter.fit(
                sensor.points,
                sensor.covariances,
            )

            results.append(
                SensorPlaneResult(
                    name=sensor.name,
                    points=sensor.points,
                    plane=plane,
                    source=sensor.source,
                )
            )

        if not results:
            raise ValueError("No sensor point clouds were supplied.")

        return results

    @staticmethod
    def result_summary(result: SensorPlaneResult) -> dict:
        """JSON-serializable summary of one fitted sensor."""
        residuals = result.residuals_mm
        inlier_residuals = result.inlier_residuals_mm

        return {
            "name": result.name,
            "source_file": (str(result.source) if result.source is not None else None),
            "total_points": int(len(result.points)),
            "inliers": int(np.count_nonzero(result.plane.inlier_mask)),
            "plane": {
                "normal": result.plane.normal.tolist(),
                "d_mm": float(result.plane.d),
                "centroid_mm": result.plane.centroid.tolist(),
                "rms_mm": float(result.plane.rms_mm),
            },
            "residuals": {
                "mean_mm": float(np.mean(residuals)),
                "std_mm": float(np.std(residuals)),
                "max_abs_mm": float(np.max(np.abs(residuals))),
                "inlier_mean_mm": float(np.mean(inlier_residuals)),
                "inlier_std_mm": float(np.std(inlier_residuals)),
                "inlier_max_abs_mm": float(np.max(np.abs(inlier_residuals))),
            },
        }

    @classmethod
    def results_summary(
        cls,
        results: Iterable[SensorPlaneResult],
    ) -> list[dict]:
        return [cls.result_summary(result) for result in results]
