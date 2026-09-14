import numpy as np

from detector_alignment.math3d import rigid_2d_registration
from detector_alignment.uncertainty import final_transform_from_q


def test_rigid_2d_registration_recovers_known_transform():
    measured = np.array([[1.0, 2.0], [4.0, 2.0], [1.0, 5.0]])
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    expected_R = np.array([[c, -s], [s, c]])
    expected_t = np.array([7.0, -3.0])
    nominal = measured @ expected_R.T + expected_t

    R, t, yaw = rigid_2d_registration(measured, nominal)
    np.testing.assert_allclose(R, expected_R, atol=1e-12)
    np.testing.assert_allclose(t, expected_t, atol=1e-12)
    assert np.isclose(yaw, angle)


def test_final_transform_plane_offset_sign_and_arbitrary_reference_order():
    # Plane z=10 -> d=-10, so tz must be -10 to map that plane to z=0.
    measured = np.array([[1.0, 7.0], [1.0, 2.0]])
    nominal = np.array([[0.0, 5.0], [0.0, 0.0]])  # decreasing nominal Y is valid
    q = np.r_[0.0, 0.0, -10.0, measured.reshape(-1)]
    R, t, _ = final_transform_from_q(q, nominal)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, [-1.0, -2.0, -10.0], atol=1e-12)
