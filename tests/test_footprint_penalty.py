"""Regression tests for the soft XY constraint and unchanged hard acceptance gate."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import detector_alignment.labeled_registration as registration
from detector_alignment.ladder_registration import RegistrationOptions
from detector_alignment.nominal_geometry import NominalLadder, NominalSensor


def competing_cloud():
    sensors = tuple(
        NominalSensor(f"/ladder/s{i}", np.array([0., y, 2.]),
                      np.array([20., 50., 0.3]),
                      Rotation.from_rotvec(tilt).as_matrix())
        for i, (y, tilt) in enumerate(zip((-250., 0., 250.),
            ([0.01, 0.02, 0.], [-0.02, 0.01, 0.], [0.02, -0.015, 0.])))
    )
    geometry = NominalLadder("/ladder", "global", sensors)
    rotation = Rotation.from_euler("z", 4, degrees=True).as_matrix()
    translation = np.array([2., 2., 0.])
    clouds = []
    for sensor in sensors:
        # XY stays over the original sensor, but Z lies on a displaced plane.
        # Plane-only fitting therefore prefers a laterally implausible pose.
        xy = np.array([[x, y] for x in np.linspace(-9.8, 9.8, 7)
                       for y in np.linspace(-24.8, 24.8, 7)])
        xy += sensor.center_mm[:2]
        normal = rotation @ sensor.rotation[:, 2]
        face = rotation @ (sensor.center_mm + sensor.rotation[:, 2] * 0.15) + translation
        z = face[2] - ((xy - face[:2]) @ normal[:2]) / normal[2]
        clouds.append(np.column_stack((xy, z)))
    points = np.vstack(clouds)
    covariance = np.repeat((np.eye(3) * 0.05**2)[None], len(points), axis=0)
    return geometry, points, np.repeat(np.arange(3), 49), covariance


def test_penalty_suppresses_plane_driven_lateral_escape():
    geometry, points, ids, covariance = competing_cloud()
    baseline = registration.LabeledFaceRegistrar(
        RegistrationOptions(footprint_penalty_weight=0)).fit(geometry, points, ids, covariance)
    result = registration.LabeledFaceRegistrar().fit(geometry, points, ids, covariance)
    before = baseline.summary(geometry)["final"]
    after = result.summary(geometry)["final"]
    assert before["outside_footprint_points"] > 50
    assert after["max_footprint_excess_mm"] < before["max_footprint_excess_mm"] / 100
    assert after["outside_footprint_points"] < before["outside_footprint_points"]
    assert baseline.final.rms_mm < result.final.rms_mm < 0.1
    assert abs(Rotation.from_matrix(result.rotation).as_rotvec()[2]) < 0.02
    np.testing.assert_array_equal(result.final.inlier_mask, baseline.final.inlier_mask)
    # A soft penalty must never silently relax the original acceptance test.
    assert result.feasible == bool(np.all(result.final.containment_margin_mm[
        result.final.inlier_mask] >= -result.options.containment_tolerance_mm))


def test_feasible_footprint_recovery_preserves_surface_only_rank():
    sensor = NominalSensor("/ladder/s0", np.array([0., 0., 2.]),
                           np.array([20., 20., 0.3]), np.eye(3))
    geometry = NominalLadder("/ladder", "global", (sensor,))
    points = np.array([[x + 1., y, 2.15] for x in (-9.8, 0., 9.8)
                       for y in (-9.8, 0., 9.8)])
    result = registration.LabeledFaceRegistrar().fit(geometry, points, np.zeros(9, dtype=int))
    assert result.summary(geometry)["initial"]["outside_footprint_points"] == 3
    assert result.accepted
    assert result.summary(geometry)["final"]["outside_footprint_points"] == 0
    assert result.final.rms_mm < 1e-7
    assert result.surface_jacobian_rank == 3
    assert result.surface_nullspace_scaled.shape == (3, 6)


@pytest.mark.parametrize("weight", [0., 1., 100.])
def test_objective_uses_every_inlier_depth_and_shared_covariance_margin(monkeypatch, weight):
    sensor = NominalSensor("/ladder/s0", np.array([0., 0., 2.]),
                           np.array([20., 20., 0.3]),
                           Rotation.from_euler("z", 25, degrees=True).as_matrix())
    geometry = NominalLadder("/ladder", "global", (sensor,))
    local = np.array([[x, y, 0.15] for x in (-10.5, 0., 10.8)
                      for y in (-9., 0., 9.)] + [[50., 50., 5.]])
    points = local @ sensor.rotation.T + sensor.center_mm
    covariance = np.repeat((sensor.rotation @ np.diag([0.1**2, 0.02**2, 0.03**2])
                            @ sensor.rotation.T)[None], len(points), axis=0)
    options = RegistrationOptions(footprint_penalty_weight=weight, containment_tolerance_mm=0.1)
    observed = {}
    real_solver = registration.least_squares

    def capture(fun, x0, **kwargs):
        observed["initial"] = fun(x0)
        shifted = x0.copy()
        shifted[3] = 1.0
        assert fun(shifted).shape == observed["initial"].shape
        solution = real_solver(fun, x0, **kwargs)
        observed["jac"] = solution.jac
        return solution

    monkeypatch.setattr(registration, "least_squares", capture)
    result = registration.LabeledFaceRegistrar(options).fit(geometry, points, np.zeros(10, dtype=int), covariance)
    assert not result.final.inlier_mask[-1]
    expected = np.repeat([0.1, 0., 0.4], 3)  # Width allowance = 3 * 0.1 + 0.1 mm.
    np.testing.assert_allclose(observed["initial"][4:], np.sqrt(weight) * expected, atol=1e-12)
    np.testing.assert_allclose(result.initial.footprint_margin_mm[:9],
                               [-0.2, -0.2, -0.2, 1.06, 10.06, 1.06, -0.5, -0.5, -0.5], atol=1e-12)
    assert result.performance["optimizer_residual_count"] == 4 + 9
    assert result.performance["footprint_residual_count"] == 9
    summary = result.summary(geometry)
    assert summary["footprint_penalty_weight"] == weight
    assert summary["initial"]["footprint_penalty_sum_sq_mm2"] == pytest.approx(weight * np.dot(expected, expected))
    singular = np.linalg.svd(observed["jac"][:4], compute_uv=False)
    np.testing.assert_allclose(result.surface_jacobian_singular_values, singular)
    if weight:
        assert np.linalg.matrix_rank(observed["jac"], tol=1e-6) > result.surface_jacobian_rank


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), True, "100", None])
def test_invalid_footprint_weight(weight):
    with pytest.raises(ValueError, match="footprint_penalty_weight"):
        RegistrationOptions(footprint_penalty_weight=weight)
