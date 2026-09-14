from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .reference import MeasuredReferenceResult


@dataclass
class PointCloudData:
    points: np.ndarray
    covariances: np.ndarray | None
    header_lines: list[str]
    source: Path

@dataclass
class SensorPointCloud:
    name: str
    points: np.ndarray
    covariances: np.ndarray | None
    source: Path | None = None

@dataclass
class PlaneFitResult:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_mask: np.ndarray
    rms_mm: float
    params: np.ndarray  # [theta, phi, d]
    covariance_params: np.ndarray
    rotation_to_xy: np.ndarray


@dataclass
class SensorPlaneResult:
    name: str
    points: np.ndarray
    plane: PlaneFitResult
    source: Path | None = None
    

@dataclass
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    homogeneous: np.ndarray
    covariance_transform_params: np.ndarray  # [rx, ry, rz, tx, ty, tz]
    covariance_joint: np.ndarray
    plane: PlaneFitResult
    references: list[MeasuredReferenceResult]
    yaw_rad: float
