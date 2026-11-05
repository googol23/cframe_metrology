from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from detector_alignment.fiducials import FiducialProcessor
from detector_alignment.models import PointCloudData


def _point_cloud(points: ArrayLike, covariances: np.ndarray | None) -> PointCloudData:
    return PointCloudData(
        points=np.asarray(points, dtype=float),
        covariances=covariances,
        header_lines=[],
        source=Path("fiducials.xyz"),
    )


def test_clusters_repeated_measurements_and_propagates_mean_uncertainty():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.2, 0.1],
            [10.0, 0.0, 0.0],
        ]
    )
    covariances = np.array(
        [
            np.diag([0.01, 0.04, 0.09]),
            np.diag([0.04, 0.09, 0.16]),
            np.diag([0.25, 0.36, 0.49]),
        ]
    )

    result = FiducialProcessor().process(
        _point_cloud(points, covariances),
        fiducial_min_distance=1.0,
    )

    assert [marker.marker_id for marker in result.markers] == [
        "fiducial_001",
        "fiducial_002",
    ]
    first, second = result.markers
    np.testing.assert_array_equal(first.measurement_indices, [0, 1])
    np.testing.assert_allclose(first.mean, [0.1, -0.1, 0.05])
    assert first.measurement_stddev is not None
    np.testing.assert_allclose(
        first.measurement_stddev,
        np.std(points[:2], axis=0, ddof=1),
    )
    np.testing.assert_allclose(
        first.mean_covariance,
        (covariances[0] + covariances[1]) / 4.0,
    )
    np.testing.assert_allclose(
        first.mean_uncertainty_stddev,
        np.sqrt(np.diag(first.mean_covariance)),
    )
    assert second.measurement_stddev is None


def test_complete_linkage_does_not_chain_distant_measurements():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0],
        ]
    )
    covariances = np.repeat(np.eye(3)[None, :, :], len(points), axis=0)

    result = FiducialProcessor().process(
        _point_cloud(points, covariances),
        fiducial_min_distance=1.0,
    )

    assert len(result.markers) == 2
    for marker in result.markers:
        distances = np.linalg.norm(
            marker.measurement_points[:, None, :]
            - marker.measurement_points[None, :, :],
            axis=2,
        )
        assert np.all(distances < 1.0)


def test_measurements_at_minimum_distance_are_separate_markers():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    covariances = np.repeat(np.eye(3)[None, :, :], len(points), axis=0)

    result = FiducialProcessor().process(
        _point_cloud(points, covariances),
        fiducial_min_distance=1.0,
    )

    assert len(result.markers) == 2


def test_affine_transformation_allows_scaling_and_propagates_covariances():
    points = np.array([[1.0, 2.0, 3.0], [1.2, 1.8, 3.1]])
    covariance = np.diag([0.01, 0.04, 0.09])
    covariances = np.repeat(covariance[None, :, :], len(points), axis=0)
    result = FiducialProcessor().process(
        _point_cloud(points, covariances),
        fiducial_min_distance=1.0,
    )
    marker = result.markers[0]
    linear = np.diag([2.0, 3.0, 4.0])
    translation = np.array([10.0, -5.0, 1.0])

    transformed = result.transformed(linear, translation)
    transformed_marker = transformed.markers[0]

    np.testing.assert_allclose(
        transformed_marker.measurement_points,
        points @ linear.T + translation,
    )
    np.testing.assert_allclose(
        transformed_marker.mean,
        linear @ marker.mean + translation,
    )
    np.testing.assert_allclose(
        transformed_marker.mean_covariance,
        linear @ marker.mean_covariance @ linear.T,
    )
    assert transformed_marker.measurement_covariance is not None
    assert marker.measurement_covariance is not None
    np.testing.assert_allclose(
        transformed_marker.measurement_covariance,
        linear @ marker.measurement_covariance @ linear.T,
    )


def test_fiducial_uncertainty_is_required():
    with pytest.raises(ValueError, match="Fiducial uncertainty is required"):
        FiducialProcessor().process(
            _point_cloud([[0.0, 0.0, 0.0]], None),
            fiducial_min_distance=1.0,
        )
