from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .math3d import normal_from_angles, rotation_align_normal_to_z


@dataclass
class MeasuredReferenceResult:
    """Directly measured scanner reference point represented in leveled XY."""

    source: Path
    measured_xyz: np.ndarray
    nominal_center_xy: np.ndarray
    center_xy: np.ndarray
    covariance_point_xyz: np.ndarray
    covariance_center_conditional: np.ndarray
    jacobian_center_wrt_plane: np.ndarray


class MeasuredReferenceProcessor:
    """Propagate a directly measured 3D point into the leveled reference state."""

    def __init__(self, derivative_step_rad: float = 1.0e-7):
        self.derivative_step_rad = float(derivative_step_rad)
        if self.derivative_step_rad <= 0.0:
            raise ValueError("derivative_step_rad must be positive.")

    @staticmethod
    def _rotation_from_plane_params(plane_params: np.ndarray) -> np.ndarray:
        plane_params = np.asarray(plane_params, dtype=float)
        if plane_params.shape != (3,):
            raise ValueError("plane_params must have shape (3,).")
        if not np.all(np.isfinite(plane_params)):
            raise ValueError("plane_params must be finite.")
        normal = normal_from_angles(float(plane_params[0]), float(plane_params[1]))
        return rotation_align_normal_to_z(normal)

    @classmethod
    def _center_xy(cls, point_xyz: np.ndarray, plane_params: np.ndarray) -> np.ndarray:
        return (cls._rotation_from_plane_params(plane_params) @ point_xyz)[:2]

    def process(
        self,
        point_xyz: np.ndarray,
        point_covariance_xyz: np.ndarray | None,
        plane_params: np.ndarray,
        nominal_center_xy: np.ndarray,
        source: str | Path,
    ) -> MeasuredReferenceResult:
        point_xyz = np.asarray(point_xyz, dtype=float)
        plane_params = np.asarray(plane_params, dtype=float)
        nominal_center_xy = np.asarray(nominal_center_xy, dtype=float)

        if point_xyz.shape != (3,) or not np.all(np.isfinite(point_xyz)):
            raise ValueError("point_xyz must be a finite vector with shape (3,).")
        if plane_params.shape != (3,):
            raise ValueError("plane_params must have shape (3,).")
        if nominal_center_xy.shape != (2,) or not np.all(np.isfinite(nominal_center_xy)):
            raise ValueError("nominal_center_xy must be a finite vector with shape (2,).")

        if point_covariance_xyz is None:
            point_covariance_xyz = np.zeros((3, 3), dtype=float)
        else:
            point_covariance_xyz = np.asarray(point_covariance_xyz, dtype=float)
        if point_covariance_xyz.shape != (3, 3) or not np.all(np.isfinite(point_covariance_xyz)):
            raise ValueError("point_covariance_xyz must be finite with shape (3, 3).")
        point_covariance_xyz = 0.5 * (point_covariance_xyz + point_covariance_xyz.T)

        R = self._rotation_from_plane_params(plane_params)
        center_xy = (R @ point_xyz)[:2]
        A = R[:2, :]
        covariance_center_conditional = A @ point_covariance_xyz @ A.T
        covariance_center_conditional = 0.5 * (
            covariance_center_conditional + covariance_center_conditional.T
        )

        J = np.zeros((2, 3), dtype=float)
        h = self.derivative_step_rad
        for parameter_index in (0, 1):
            p_plus = plane_params.copy()
            p_minus = plane_params.copy()
            p_plus[parameter_index] += h
            p_minus[parameter_index] -= h
            J[:, parameter_index] = (
                self._center_xy(point_xyz, p_plus) - self._center_xy(point_xyz, p_minus)
            ) / (2.0 * h)

        return MeasuredReferenceResult(
            source=Path(source),
            measured_xyz=point_xyz.copy(),
            nominal_center_xy=nominal_center_xy.copy(),
            center_xy=center_xy,
            covariance_point_xyz=point_covariance_xyz.copy(),
            covariance_center_conditional=covariance_center_conditional,
            jacobian_center_wrt_plane=J,
        )
