from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def normal_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def angles_from_normal(n: np.ndarray) -> tuple[float, float]:
    n = np.asarray(n, float)
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normal must be finite and non-zero.")
    n = n / norm
    theta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    phi = float(np.arctan2(n[1], n[0]))
    return theta, phi


def rotation_align_normal_to_z(n: np.ndarray) -> np.ndarray:
    """Return the minimal rotation mapping unit normal *n* onto +Z."""
    n = np.asarray(n, float)
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normal must be finite and non-zero.")
    n = n / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(n, z), -1.0, 1.0))
    if dot > 1.0 - 1e-14:
        return np.eye(3)
    if dot < -1.0 + 1e-14:
        return Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
    axis = np.cross(n, z)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3).")
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) @ np.asarray(R, float).T + np.asarray(t, float).reshape(1, 3)


def rotate_covariances(covs: np.ndarray, R: np.ndarray) -> np.ndarray:
    covs = np.asarray(covs, float)
    R = np.asarray(R, float)
    if covs.ndim != 3 or covs.shape[1:] != (3, 3):
        raise ValueError("covs must have shape (N, 3, 3).")
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3).")
    return np.einsum("ij,njk,lk->nil", R, covs, R)


def rigid_2d_registration(
    measured_xy: np.ndarray,
    nominal_xy: np.ndarray,
    weights: np.ndarray | None = None,
):
    """Weighted rigid registration mapping measured XY to nominal XY."""
    P = np.asarray(measured_xy, float)
    Q = np.asarray(nominal_xy, float)
    if P.ndim != 2 or P.shape[1] != 2 or Q.shape != P.shape:
        raise ValueError("measured_xy and nominal_xy must both have shape (N, 2).")
    if len(P) < 2:
        raise ValueError("At least two references are required for in-plane registration.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
        raise ValueError("Reference coordinates must be finite.")

    if weights is None:
        w = np.ones(len(P), dtype=float)
    else:
        w = np.asarray(weights, float)
        if w.shape != (len(P),) or np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite, non-negative, and have shape (N,).")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must contain at least one positive value.")
    w = w / total

    pc = np.sum(P * w[:, None], axis=0)
    qc = np.sum(Q * w[:, None], axis=0)
    X = P - pc
    Y = Q - qc
    S = (X * w[:, None]).T @ Y
    U, _, Vt = np.linalg.svd(S)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T
    t2 = qc - R2 @ pc
    yaw = float(np.arctan2(R2[1, 0], R2[0, 0]))
    return R2, t2, yaw


def transform_params(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rotvec = Rotation.from_matrix(np.asarray(R, float)).as_rotvec()
    return np.r_[rotvec, np.asarray(t, float).reshape(3)]
