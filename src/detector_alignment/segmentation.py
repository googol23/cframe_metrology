from __future__ import annotations

"""
Adapter boundary for cbm_sts_tools automatic ladder segmentation.

This file intentionally contains no guessed cbm_sts_tools imports.

Once the exact segmentation class/function in
cbm_sts_tools/metrology is identified, only this module should need to
know its API. The rest of cframe_metrology continues to operate on
SensorPointCloud objects.

Expected public callable:

    segment_with_cbm_sts_tools(points, covariances, config)
        -> list[SensorPointCloud]
"""

import numpy as np

from .ladder import SensorPointCloud

def segment_ladder(
    points: np.ndarray,
    covariances: np.ndarray | None,
    config: dict,
) -> list[SensorPointCloud]:

    z_gap_mm = float(config.get("z_gap_mm", 0.7))
    min_points = int(config.get("min_points_per_sensor", 100))
    y_center_cfg = config.get("y_center_mm")
    
    if y_center_cfg is None:
        y_center_mm = float(np.median(points[:, 1]))
    else:
        y_center_mm = float(y_center_cfg)

    print(f"Using mid line y_center_mm = {y_center_mm}")

    # Sort complete transformed ladder cloud in Z.
    order = np.argsort(points[:, 2])
    dz = np.diff(points[order, 2])

    # Separate mechanically distinct Z layers.
    split_at = np.where(dz > z_gap_mm)[0] + 1
    z_groups = np.split(order, split_at)

    print(f"After spliting by z-axis {len(z_groups)} groups where found")
    print([np.mean(g) for g in z_groups])

    sensors = []

    for z_index, indices in enumerate(z_groups):
        layer = points[indices]

        for side, mask in (
            ("bottom", layer[:, 1] < y_center_mm),
            ("top", layer[:, 1] > y_center_mm),
        ):
            selected = indices[mask]

            if len(selected) < min_points:
                continue

            sensor_cov = (
                None
                if covariances is None
                else covariances[selected]
            )

            sensors.append(
                SensorPointCloud(
                    name=f"Z{z_index}_{side}",
                    points=points[selected],
                    covariances=sensor_cov,
                )
            )

    # Sensor order is geometrical: bottom -> top.
    sensors.sort(key=lambda s: np.mean(s.points[:, 1]))

    # Give stable sensor names.
    for i, sensor in enumerate(sensors):
        sensor.name = f"S{i}"

    return sensors