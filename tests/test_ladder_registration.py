import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from detector_alignment.ladder_registration import LadderRegistrar, RegistrationOptions
from detector_alignment.nominal_geometry import NominalLadder, NominalSensor
from detector_alignment.registration_viz import RegistrationVisualizer


def box(
    center=(0, 0, 0), dimensions=(20, 30, 0.3), rotation=None, name="/ladder/sensor"
):
    return NominalSensor(
        name,
        np.array(center),
        np.array(dimensions),
        np.eye(3) if rotation is None else rotation,
    )


def surface_points(sensor):
    points = np.array(
        [
            [x, y, sensor.dimensions_mm[2] / 2]
            for x in np.linspace(-0.4, 0.4, 5) * sensor.dimensions_mm[0]
            for y in np.linspace(-0.4, 0.4, 5) * sensor.dimensions_mm[1]
        ]
    )
    return points @ sensor.rotation.T + sensor.center_mm


def geometry_payload(*, node=False):
    payload = {
        "schema_version": 2 if node else 1,
        "units": "cm",
        "coordinate_frame": "node" if node else "global",
        "node_path": "/top/Unit01L_1/LadderType01_1",
        "sensor_count": 1,
        "sensors": [
            {
                "node_path": "/top/Unit01L_1/LadderType01_1/sensor",
                "center": [1, 2, 3],
                "dimensions": [6.2, 2.2, 0.03],
                "rotation": np.eye(3).tolist(),
            }
        ],
    }
    if node:
        payload.update(
            reference_node="/top/Unit01L_1",
            reference_to_global={
                "translation": [50, 60, 70],
                "rotation": Rotation.from_euler("z", 90, degrees=True)
                .as_matrix()
                .tolist(),
            },
        )
    return payload


@pytest.mark.parametrize("node", [False, True])
def test_geometry_export_units_and_explicit_frame(tmp_path, node):
    payload = geometry_payload(node=node)
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))
    frame = payload.get("reference_node", "global")
    geometry = NominalLadder.from_json(path, expected_frame=frame)
    np.testing.assert_allclose(geometry.sensors[0].center_mm, [10, 20, 30])
    np.testing.assert_allclose(geometry.sensors[0].dimensions_mm, [62, 22, 0.3])
    with pytest.raises(ValueError, match="does not match"):
        NominalLadder.from_json(path, expected_frame="wrong_frame")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"units": "m"}, "units"),
        ({"sensor_count": 2}, "sensor_count"),
        ({"schema_version": 3}, "global v1"),
        ({"sensors": []}, "non-empty"),
    ],
)
def test_invalid_geometry_exports(tmp_path, change, match):
    payload = geometry_payload()
    payload.update(change)
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        NominalLadder.from_json(path, expected_frame="global")


@pytest.mark.parametrize("rotation", [np.diag([1, 1, -1]), np.eye(3) * 2])
def test_geometry_rejects_nonrigid_sensor_orientation(rotation):
    with pytest.raises(ValueError, match="proper orthonormal"):
        box(rotation=rotation)


def test_rigid_recovery_with_observable_surfaces():
    sensors = (
        box(center=(-25, 0, 0), name="/ladder/a"),
        box(
            center=(25, 0, 0),
            rotation=Rotation.from_euler("y", 55, degrees=True).as_matrix(),
            name="/ladder/b",
        ),
        box(
            center=(0, 40, 0),
            rotation=Rotation.from_euler("x", 65, degrees=True).as_matrix(),
            name="/ladder/c",
        ),
    )
    geometry = NominalLadder("/ladder", "global", sensors)
    rotation = Rotation.from_rotvec([0.008, -0.012, 0.006]).as_matrix()
    translation = np.array([0.25, -0.18, 0.32])
    points = (
        np.concatenate([surface_points(sensor) for sensor in sensors]) @ rotation.T
        + translation
    )
    result = LadderRegistrar().fit(geometry, points)
    assert result.accepted, result.optimizer_message
    np.testing.assert_allclose(result.rotation, rotation, atol=2e-6)
    np.testing.assert_allclose(result.translation_mm, translation, atol=2e-5)
    assert result.final.rms_mm < 1e-6
    assert result.surface_jacobian_rank == 6
    assert np.all(result.final.containment_margin_mm >= -1e-6)
    assert len(result.final.residuals_mm) == len(points)
    assert set(result.final.sensor_indices) == {0, 1, 2}
    summary = result.summary(geometry)
    assert summary["convention"].startswith("p_as_built = R @ p_nominal")
    json.dumps(summary, allow_nan=False)


def test_partial_plane_reports_unobservable_directions():
    sensor = box()
    geometry = NominalLadder("/ladder", "global", (sensor,))
    points = surface_points(sensor) + [0, 0, 0.5]
    result = LadderRegistrar().fit(geometry, points)
    assert result.accepted
    assert result.final.rms_mm < 1e-7
    assert result.surface_jacobian_rank == 3
    assert result.surface_nullspace_scaled.shape == (3, 6)
    assert any("only 3/6" in warning for warning in result.warnings)
    np.testing.assert_allclose(result.translation_mm[2], 0.5, atol=1e-7)


def test_surface_distance_not_volume_distance():
    sensor = box(dimensions=(20, 30, 2))
    geometry = NominalLadder("/ladder", "global", (sensor,))
    evaluation = LadderRegistrar().evaluate(
        geometry, np.zeros((3, 3)), np.eye(3), np.zeros(3)
    )
    np.testing.assert_allclose(evaluation.residuals_mm, -1)
    np.testing.assert_allclose(evaluation.containment_margin_mm, 1)


def test_containment_uses_oriented_boxes_and_projected_anisotropic_covariance():
    axes = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    sensor = box(dimensions=(2, 4, 0.3), rotation=axes)
    geometry = NominalLadder("/ladder", "global", (sensor,))
    points = np.array([[1.2, 0, 0], [0, 2.2, 0], [0, 0, 0.4]]) @ axes.T
    local_covariance = np.diag([0.1**2, 0.02**2, 0.03**2])
    covariance = np.repeat((axes @ local_covariance @ axes.T)[None], 3, axis=0)
    registrar = LadderRegistrar(RegistrationOptions(uncertainty_sigma=3))
    evaluation = registrar.evaluate(
        geometry, points, np.eye(3), np.zeros(3), covariance
    )
    # The first point is outside the true box but inside its projected 3-sigma envelope.
    np.testing.assert_allclose(
        evaluation.containment_margin_mm, [0.1, -0.14, -0.16], atol=1e-10
    )
    without = registrar.evaluate(geometry, points, np.eye(3), np.zeros(3))
    assert without.containment_margin_mm[0] < 0
    # An AABB of the rotated sensor would incorrectly accept this point.
    aabb_point = np.array([[2, 0, 0]])
    assert np.all(aabb_point >= sensor.corners_mm.min(axis=0))
    assert np.all(aabb_point <= sensor.corners_mm.max(axis=0))
    assert (
        registrar.evaluate(
            geometry, aabb_point, np.eye(3), np.zeros(3)
        ).containment_margin_mm[0]
        < 0
    )


def test_covariance_is_projected_after_ladder_rotation():
    geometry = NominalLadder("/ladder", "global", (box(dimensions=(2, 4, 0.3)),))
    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    points = np.repeat(np.array([[0, 0, 0.4]]) @ rotation.T, 3, axis=0)
    covariance = np.repeat(np.diag([0.1**2, 0.001**2, 0.001**2])[None], 3, axis=0)
    evaluation = LadderRegistrar().evaluate(
        geometry, points, rotation, np.zeros(3), covariance
    )
    np.testing.assert_allclose(evaluation.containment_margin_mm, 0.05, atol=1e-10)


def test_infeasible_point_is_not_discarded_or_reported_as_success():
    geometry = NominalLadder("/ladder", "global", (box(),))
    points = np.vstack((surface_points(geometry.sensors[0]), [100, 100, 100]))
    registrar = LadderRegistrar(
        RegistrationOptions(
            max_translation_mm=0.1, max_rotation_deg=0.1, max_iterations=30
        )
    )
    result = registrar.fit(geometry, points)
    assert not result.accepted
    assert not result.feasible
    assert len(result.final.residuals_mm) == len(points)
    assert result.summary(geometry)["final"]["outside_points"] >= 1
    assert any("containment failed" in warning for warning in result.warnings)


def test_initial_transform_and_pivot_convention():
    sensors = (
        box(center=(120, -200, 50), name="/ladder/a"),
        box(
            center=(160, -200, 50),
            name="/ladder/b",
            rotation=Rotation.from_euler("y", 55, degrees=True).as_matrix(),
        ),
        box(
            center=(120, -160, 50),
            name="/ladder/c",
            rotation=Rotation.from_euler("x", 65, degrees=True).as_matrix(),
        ),
    )
    geometry = NominalLadder("/ladder", "global", sensors)
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    initial[:3, 3] = [40, -20, 80]
    points = np.concatenate([surface_points(sensor) for sensor in sensors])
    points = points @ initial[:3, :3].T + initial[:3, 3] + [0, 0, 0.2]
    result = LadderRegistrar().fit(geometry, points, initial_transform=initial)
    assert result.accepted
    np.testing.assert_allclose(result.translation_mm, [40, -20, 80.2], atol=1e-6)
    np.testing.assert_allclose(
        result.pivot_displacement_mm,
        result.rotation @ result.pivot_mm + result.translation_mm - result.pivot_mm,
    )


@pytest.mark.parametrize(
    "covariance,match",
    [
        (np.ones((1, 3, 3)), "shape"),
        (np.repeat(np.diag([1, 1, -1])[None], 3, axis=0), "positive semidefinite"),
        (
            np.repeat(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])[None], 3, axis=0),
            "symmetric",
        ),
    ],
)
def test_registration_validates_covariances(covariance, match):
    geometry = NominalLadder("/ladder", "global", (box(),))
    with pytest.raises(ValueError, match=match):
        LadderRegistrar().fit(geometry, np.zeros((3, 3)), covariance)


@pytest.mark.parametrize(
    "options",
    [
        {"uncertainty_sigma": 0},
        {"containment_tolerance_mm": -1},
        {"max_translation_mm": float("nan")},
        {"max_rotation_deg": True},
        {"max_iterations": 1.5},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        RegistrationOptions(**options)


def test_three_view_plot(tmp_path, monkeypatch):
    import matplotlib.figure

    sensor = box()
    geometry = NominalLadder("/ladder", "global", (sensor,))
    points = surface_points(sensor) + [0, 0, 0.2]
    result = LadderRegistrar().fit(geometry, points)
    captured = []
    original_savefig = matplotlib.figure.Figure.savefig

    def inspect_figure(figure, *args, **kwargs):
        captured.append(
            (
                [ax.get_title() for ax in figure.axes[:3]],
                [collection.get_alpha() for collection in figure.axes[2].collections],
            )
        )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", inspect_figure)
    path = tmp_path / "registration.png"
    RegistrationVisualizer(dpi=50, sample_size=5).plot(
        geometry, points, result, path, ladder_name="L1"
    )
    assert path.stat().st_size > 1000
    assert captured[0][0] == ["XY", "YZ", "Perspective"]
    assert any(alpha is not None and 0 < alpha < 1 for alpha in captured[0][1])
