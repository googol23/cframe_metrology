import json

import numpy as np
import pytest
import yaml

from detector_alignment.math3d import normal_from_angles, rotation_align_normal_to_z, rz
from detector_alignment.pipeline import AlignmentPipeline
from detector_alignment.plane import RobustPlaneFitter
from detector_alignment.reference import MeasuredReferenceProcessor
from detector_alignment.uncertainty import (
    final_transform_from_q,
    propagate_final_transform_covariance,
)


@pytest.mark.parametrize("direction,sign", [("+z", 1), ("-z", -1)])
@pytest.mark.parametrize("normal", [[0, 0, 1], [0, 0, -1], [0.2, -0.3, 1], [1, 0, 0]])
def test_selected_normal_uses_a_proper_rotation(direction, sign, normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    rotation = rotation_align_normal_to_z(normal, direction)
    np.testing.assert_allclose(rotation @ normal, [0, 0, sign], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_minus_z_leveling_is_continuous_near_scanner_plus_z():
    left = rotation_align_normal_to_z(np.array([1e-5, 0, 1]), "-z")
    right = rotation_align_normal_to_z(np.array([-1e-5, 0, 1]), "-z")
    assert np.linalg.norm(left - right) < 1e-4


@pytest.mark.parametrize("value", ["+Z", "z", "-x", "", None, True, 0, [], {}])
def test_invalid_cframe_normal_is_rejected_before_processing(value):
    with pytest.raises(ValueError, match="cframe.normal.*must be"):
        AlignmentPipeline({"cframe": {"normal": value}})


@pytest.mark.parametrize(
    "value",
    [None, True, False, "0.75", float("nan"), float("inf"), -float("inf"), [], {}],
)
def test_invalid_facet_offset_is_rejected_before_processing(value):
    with pytest.raises(ValueError, match="cframe.facet_offset.*finite number"):
        AlignmentPipeline({"cframe": {"facet_offset": value}})
    q = np.array([0, 0, -10, 1, 2, 1, 7], dtype=float)
    with pytest.raises(ValueError, match="cframe.facet_offset.*finite number"):
        final_transform_from_q(q, np.array([[0, 0], [0, 5]]), facet_offset=value)


@pytest.mark.parametrize("direction", ["+z", "-z"])
def test_fixed_facet_offset_changes_only_z_translation_not_covariance(direction):
    q = np.array([0.3, 0.7, -10, 1, 2, 1, 7], dtype=float)
    covariance = np.diag([1e-6, 2e-6, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
    nominal = np.array([[0, 0], [0, 5]], dtype=float)
    original = propagate_final_transform_covariance(
        q, covariance, nominal, normal_direction=direction
    )
    shifted = propagate_final_transform_covariance(
        q,
        covariance,
        nominal,
        normal_direction=direction,
        facet_offset=0.75,
    )
    np.testing.assert_allclose(shifted[0], original[0], atol=1e-12)
    np.testing.assert_allclose(shifted[1] - original[1], [0, 0, 0.75], atol=1e-12)
    assert shifted[2] == pytest.approx(original[2])
    np.testing.assert_allclose(shifted[3], original[3], rtol=1e-8, atol=1e-10)


def test_target_orientation_does_not_change_fitted_plane_or_inliers():
    rng = np.random.default_rng(9)
    xy = rng.uniform(-10, 10, (100, 2))
    points = np.column_stack((xy, 10 - 0.2 * xy[:, 0] + 0.3 * xy[:, 1]))
    points[:, 2] += rng.normal(0, 0.002, len(points))
    plus = RobustPlaneFitter(ransac_iterations=20, random_seed=1).fit(points)
    minus = RobustPlaneFitter(
        ransac_iterations=20, random_seed=1, normal_direction="-z"
    ).fit(points)
    np.testing.assert_allclose(minus.normal, plus.normal, atol=1e-12)
    np.testing.assert_allclose(minus.params, plus.params, atol=1e-12)
    np.testing.assert_allclose(
        minus.covariance_params, plus.covariance_params, atol=1e-12
    )
    np.testing.assert_array_equal(minus.inlier_mask, plus.inlier_mask)
    np.testing.assert_allclose(
        minus.rotation_to_xy @ minus.normal, [0, 0, -1], atol=1e-12
    )
    np.testing.assert_allclose(plus.rotation_to_xy @ plus.normal, [0, 0, 1], atol=1e-12)


def test_reference_projection_covariance_and_jacobian_follow_orientation():
    arguments = dict(
        point_xyz=np.array([3.0, 4.0, 5.0]),
        point_covariance_xyz=np.array(
            [[0.01, 0.002, 0.001], [0.002, 0.04, 0.003], [0.001, 0.003, 0.09]]
        ),
        plane_params=np.array([0.3, -0.4, -5.0]),
        nominal_center_xy=np.array([10.0, 20.0]),
        source="reference.xyz",
    )
    plus = MeasuredReferenceProcessor().process(**arguments)
    minus = MeasuredReferenceProcessor(normal_direction="-z").process(**arguments)
    flip = np.diag([1, -1])
    np.testing.assert_allclose(minus.center_xy, flip @ plus.center_xy, atol=1e-12)
    np.testing.assert_allclose(
        minus.covariance_center_conditional,
        flip @ plus.covariance_center_conditional @ flip,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        minus.jacobian_center_wrt_plane,
        flip @ plus.jacobian_center_wrt_plane,
        atol=1e-8,
    )


@pytest.mark.parametrize("direction,sign", [("+z", 1), ("-z", -1)])
@pytest.mark.parametrize("facet_offset", [0.0, 0.75, -0.75])
def test_final_transform_levels_tilted_plane_and_registers_reference_centers(
    direction, sign, facet_offset
):
    plane_params = np.array([0.35, 0.7, -10.0])
    normal = normal_from_angles(*plane_params[:2])
    xy = np.array([[1, 2], [4, 7]])
    z = (-plane_params[2] - xy @ normal[:2]) / normal[2]
    references = np.column_stack((xy, z))
    level_rotation = rotation_align_normal_to_z(normal, direction)
    leveled_centers = (references @ level_rotation.T)[:, :2]
    expected_yaw = 0.2
    nominal = leveled_centers @ rz(expected_yaw)[:2, :2].T + [7, -3]
    q = np.r_[plane_params, leveled_centers.ravel()]
    rotation, translation, yaw = final_transform_from_q(
        q, nominal, normal_direction=direction, facet_offset=facet_offset
    )
    np.testing.assert_allclose(rotation, rz(expected_yaw) @ level_rotation, atol=1e-12)
    np.testing.assert_allclose(
        translation, [7, -3, sign * plane_params[2] + facet_offset], atol=1e-12
    )
    np.testing.assert_allclose(rotation @ normal, [0, 0, sign], atol=1e-12)
    transformed = references @ rotation.T + translation
    np.testing.assert_allclose(transformed[:, :2], nominal, atol=1e-12)
    np.testing.assert_allclose(transformed[:, 2], facet_offset, atol=1e-12)
    assert yaw == pytest.approx(expected_yaw)


def test_minus_z_transform_covariance_does_not_differentiate_a_two_pi_jump():
    q = np.array([0, 0, -10, 1, -2, 1, -7], dtype=float)
    covariance = np.diag([1e-6, 1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    nominal = np.array([[0, 0], [0, 5]], dtype=float)
    rotation, translation, _, propagated = propagate_final_transform_covariance(
        q,
        covariance,
        nominal,
        normal_direction="-z",
    )
    np.testing.assert_allclose(rotation, np.diag([-1, 1, -1]), atol=1e-12)
    np.testing.assert_allclose(translation, [1, -2, 10], atol=1e-12)
    np.testing.assert_allclose(propagated, propagated.T, atol=1e-12)
    assert np.all(np.isfinite(propagated))
    assert np.linalg.eigvalsh(propagated).min() >= -1e-12
    # A principal-rotvec discontinuity would create order-1e8 variances here.
    assert np.diag(propagated).max() < 0.01
    assert np.trace(propagated[:3, :3]) > 1e-7
    assert propagated[5, 5] == pytest.approx(covariance[2, 2], rel=1e-8)


@pytest.mark.parametrize("direction", [None, "+z", "-z"])
@pytest.mark.parametrize("facet_offset", [None, 0.75, -0.75])
def test_cframe_normal_yaml_end_to_end(tmp_path, direction, facet_offset):
    points = np.array(
        [[x, y, 10] for x in np.linspace(-5, 5, 7) for y in np.linspace(-4, 4, 6)]
    )
    off_plane = np.array([[2, 3, 11], [4, 5, 9]], dtype=float)
    np.savetxt(tmp_path / "facet.xyz", points)
    np.savetxt(tmp_path / "ref1.xyz", [[1, 2, 10]])
    np.savetxt(tmp_path / "ref2.xyz", [[1, 7, 10]])
    np.savetxt(tmp_path / "off_plane.xyz", off_plane)
    config = {
        "output_dir": "output",
        "uncertainty": {"mode": "constant", "sigma_xyz_mm": [0.01, 0.02, 0.03]},
        "cframe": {"file": "facet.xyz"},
        "plane_fit": {"ransac_iterations": 20, "random_seed": 1},
        "references": [
            {"file": "ref1.xyz", "nominal_center": [0, 0]},
            {"file": "ref2.xyz", "nominal_center": [0, 5]},
        ],
        "fiducials": {"file": "off_plane.xyz", "fiducial_min_distance": 0.5},
        "ladder_files": ["off_plane.xyz"],
        "visualization": {
            "dpi": 40,
            "sample_size": 100,
            "draw": {
                "cframe": direction == "-z",
                "sensors": False,
                "full_ladders": False,
                "all_ladders_and_cframe": False,
            },
        },
    }
    if direction is not None:
        config["cframe"]["normal"] = direction
    if facet_offset is not None:
        config["cframe"]["facet_offset"] = facet_offset
    target_z = facet_offset if facet_offset is not None else 0.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    pipeline = AlignmentPipeline.from_yaml(path)
    result = pipeline.run()
    sign = -1 if direction == "-z" else 1
    expected_rotation = np.diag([sign, 1, sign])
    expected_translation = np.array([-sign, -2, -10 * sign + target_z])
    np.testing.assert_allclose(result.rotation, expected_rotation, atol=1e-8)
    np.testing.assert_allclose(result.translation, expected_translation, atol=1e-8)
    np.testing.assert_allclose(
        result.rotation @ result.plane.normal, [0, 0, sign], atol=1e-8
    )
    np.testing.assert_allclose(
        result.plane.rotation_to_xy @ result.plane.normal, [0, 0, sign], atol=1e-8
    )
    np.testing.assert_allclose(
        (points @ result.rotation.T + result.translation)[:, 2], target_z, atol=1e-8
    )
    assert np.linalg.det(result.rotation) == pytest.approx(1.0)
    assert pipeline.sensor_plane_fitter.normal_direction == "+z"
    payload = json.loads((tmp_path / "output/results.json").read_text())
    assert payload["plane"]["normal_direction"] == (direction or "+z")
    assert payload["plane"]["facet_offset_mm"] == target_z
    np.testing.assert_allclose(
        payload["plane"]["normal_detector"], [0, 0, sign], atol=1e-8
    )
    expected_points = off_plane @ result.rotation.T + result.translation
    exported = np.loadtxt(tmp_path / "output/point_clouds/off_plane_transformed.xyz")
    np.testing.assert_allclose(exported, expected_points, atol=1e-6)
    np.testing.assert_allclose(
        exported[:, 2], sign * (off_plane[:, 2] - 10) + target_z, atol=1e-6
    )
    assert len(payload["fiducials"]["markers"]) == 2
    for marker in payload["fiducials"]["markers"]:
        index = marker["measurement_indices"][0]
        np.testing.assert_allclose(
            marker["detector"]["mean_xyz_mm"], expected_points[index], atol=1e-8
        )
    if direction == "-z":
        assert (tmp_path / "output/images/cframe_final_alignment.png").is_file()
        assert np.max(np.diag(result.covariance_transform_params)) < 1.0
