from pathlib import Path

import numpy as np

from detector_alignment.reference import MeasuredReferenceResult
from detector_alignment.uncertainty import build_joint_covariance


def _ref(jacobian, conditional):
    return MeasuredReferenceResult(
        source=Path("r.xyz"),
        measured_xyz=np.zeros(3),
        nominal_center_xy=np.zeros(2),
        center_xy=np.zeros(2),
        covariance_point_xyz=np.zeros((3, 3)),
        covariance_center_conditional=np.asarray(conditional, float),
        jacobian_center_wrt_plane=np.asarray(jacobian, float),
    )


def test_joint_covariance_contains_shared_plane_cross_terms():
    plane_cov = np.diag([4.0, 9.0, 16.0])
    J1 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    J2 = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    refs = [_ref(J1, np.eye(2)), _ref(J2, 2.0 * np.eye(2))]
    C = build_joint_covariance(plane_cov, refs)
    np.testing.assert_allclose(C[3:5, 5:7], J1 @ plane_cov @ J2.T)
    np.testing.assert_allclose(C[3:5, :3], J1 @ plane_cov)
    np.testing.assert_allclose(C, C.T)
