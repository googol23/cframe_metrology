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


def segment_with_cbm_sts_tools(
    points: np.ndarray,
    covariances: np.ndarray | None,
    config: dict,
) -> list[SensorPointCloud]:
    """
    Convert one complete ladder point cloud into per-sensor clouds.

    Parameters
    ----------
    points
        Complete ladder XYZ point cloud in scanner coordinates.

    covariances
        Optional per-point XYZ covariance matrices.

    config
        Automatic-segmentation configuration from YAML.

    Returns
    -------
    list[SensorPointCloud]
        One object per identified sensor.

    Notes
    -----
    Replace the RuntimeError below with the actual cbm_sts_tools
    metrology call. Keep all cbm_sts_tools-specific imports and data
    conversions inside this file.
    """
    raise RuntimeError(
        "cbm_sts_tools automatic segmentation adapter is not wired yet. "
        "Provide the exact cbm_sts_tools.metrology segmentation class or "
        "function, then implement it in detector_alignment/"
        "cbm_sts_segmenter.py."
    )
