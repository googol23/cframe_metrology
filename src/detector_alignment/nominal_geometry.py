"""ROOT-independent reader for geo-inspector ladder JSON exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


def rigid_matrix(value, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
    ):
        raise ValueError(f"{label} must be a finite rigid 4x4 homogeneous matrix.")
    validate_rotation(matrix[:3, :3], label)
    return matrix.copy()


def validate_rotation(rotation: np.ndarray, label: str) -> None:
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError(f"{label} rotation must be a proper orthonormal 3x3 matrix.")


@dataclass(frozen=True)
class NominalSensor:
    """Oriented box: frame point = center_mm + rotation @ local point."""

    node_path: str
    center_mm: np.ndarray
    dimensions_mm: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_mm, dtype=float)
        dimensions = np.asarray(self.dimensions_mm, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        if not isinstance(self.node_path, str) or not self.node_path.strip():
            raise ValueError("Sensor node_path must be a non-empty string.")
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(
                f"{self.node_path}: center must contain three finite lengths."
            )
        if (
            dimensions.shape != (3,)
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions <= 0)
        ):
            raise ValueError(
                f"{self.node_path}: dimensions must be three positive lengths."
            )
        validate_rotation(rotation, self.node_path)
        object.__setattr__(self, "center_mm", center.copy())
        object.__setattr__(self, "dimensions_mm", dimensions.copy())
        object.__setattr__(self, "rotation", rotation.copy())

    @property
    def corners_mm(self) -> np.ndarray:
        signs = np.asarray(list(product((-1, 1), repeat=3)))
        return (signs * self.dimensions_mm / 2) @ self.rotation.T + self.center_mm


@dataclass(frozen=True)
class NominalLadder:
    node_path: str
    frame: str
    sensors: tuple[NominalSensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_path, str) or not self.node_path.strip():
            raise ValueError("Ladder node_path must be a non-empty string.")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("Ladder frame must be a non-empty string.")
        if not self.sensors or not all(
            isinstance(s, NominalSensor) for s in self.sensors
        ):
            raise ValueError("Nominal ladder requires at least one NominalSensor.")
        paths = [sensor.node_path for sensor in self.sensors]
        if len(set(paths)) != len(paths):
            raise ValueError("Nominal ladder contains duplicate sensor node paths.")

    @property
    def corners_mm(self) -> np.ndarray:
        return np.concatenate([sensor.corners_mm for sensor in self.sensors])

    @classmethod
    def from_json(cls, path: str | Path, *, expected_frame: str) -> "NominalLadder":
        """Require callers to declare which geometry frame matches measured XYZ.

        Node-frame v2 data are already expressed in reference_node coordinates.
        reference_to_global is deliberately not applied: that would silently mix
        global ROOT and C-frame coordinates.
        """
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Nominal geometry JSON must contain an object.")
        version = payload.get("schema_version")
        coordinate_frame = payload.get("coordinate_frame")
        if version == 1 and coordinate_frame == "global":
            frame = "global"
        elif version == 2 and coordinate_frame == "node":
            frame = payload.get("reference_node")
            if not isinstance(frame, str) or not frame.strip():
                raise ValueError("Node-frame geometry requires reference_node.")
            reference = payload.get("reference_to_global")
            if not isinstance(reference, dict):
                raise ValueError("Node-frame geometry requires reference_to_global.")
            matrix = np.eye(4)
            matrix[:3, :3] = reference["rotation"]
            matrix[:3, 3] = reference["translation"]
            rigid_matrix(matrix, "reference_to_global")
        else:
            raise ValueError(
                "Expected geo-inspector global v1 or node-frame v2 geometry."
            )
        if not isinstance(expected_frame, str) or frame != expected_frame:
            raise ValueError(
                f"Geometry frame {frame!r} does not match measurement_frame {expected_frame!r}."
            )
        units = payload.get("units")
        if units not in {"cm", "mm"}:
            raise ValueError("Geometry units must be explicitly 'cm' or 'mm'.")
        scale = 10.0 if units == "cm" else 1.0
        rows = payload.get("sensors")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Nominal geometry requires a non-empty sensors list.")
        if payload.get("sensor_count") != len(rows):
            raise ValueError("Geometry sensor_count does not match sensors list.")
        sensors = tuple(
            NominalSensor(
                node_path=row["node_path"],
                center_mm=np.asarray(row["center"], dtype=float) * scale,
                dimensions_mm=np.asarray(row["dimensions"], dtype=float) * scale,
                rotation=np.asarray(row["rotation"], dtype=float),
            )
            for row in rows
        )
        return cls(payload["node_path"], frame, sensors)
