from pathlib import Path

import numpy as np

from detector_alignment.reference import MeasuredReferenceProcessor


def test_measured_reference_identity_plane():
    processor = MeasuredReferenceProcessor()
    covariance = np.diag([0.01, 0.04, 0.09])
    result = processor.process(
        point_xyz=np.array([3.0, 4.0, 5.0]),
        point_covariance_xyz=covariance,
        plane_params=np.array([0.0, 0.0, -5.0]),
        nominal_center_xy=np.array([10.0, 20.0]),
        source=Path("ref.xyz"),
    )
    np.testing.assert_allclose(result.center_xy, [3.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(result.covariance_center_conditional, covariance[:2, :2])
    np.testing.assert_allclose(result.jacobian_center_wrt_plane[:, 2], 0.0)
