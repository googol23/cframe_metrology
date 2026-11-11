import numpy as np
from scipy.spatial.transform import Rotation

from detector_alignment.labeled_registration import LabeledFaceRegistrar
from detector_alignment.nominal_geometry import NominalLadder, NominalSensor


def test_single_rigid_fit_with_outlier_and_fixed_correspondences():
    sensors = tuple(
        NominalSensor(f"/ladder/s{i}", np.array([0., y, 2.]),
                      np.array([20., 18., 0.3]), np.eye(3))
        for i, y in enumerate((-30., 0., 30.))
    )
    geometry = NominalLadder("/ladder", "/cframe", sensors)
    points, labels = [], []
    rotation = Rotation.from_rotvec([0.002, -0.004, 0.0]).as_matrix()
    displacement = np.array([0., 0., 0.2])
    for i, sensor in enumerate(sensors):
        cloud = np.array([[x, y, 2.15] for x in (-5., 0., 5.)
                          for y in (sensor.center_mm[1] - 5., sensor.center_mm[1],
                                    sensor.center_mm[1] + 5.)])
        points.append(cloud @ rotation.T + displacement)
        labels.extend([i] * len(cloud))
    points = np.vstack(points)
    points[2, 2] += 4  # A point from the wrong surface is excluded before fitting.
    covariance = np.repeat(np.eye(3)[None] * 0.01**2, len(points), axis=0)
    result = LabeledFaceRegistrar(facet_z_mm=0).fit(
        geometry, points, np.asarray(labels), covariance
    )
    assert result.accepted, result.optimizer_message
    assert result.summary(geometry)["outlier_points"] == 1
    assert result.final.rms_mm < 1e-6
    np.testing.assert_allclose(result.rotation[:3, 2], rotation[:3, 2], atol=1e-5)
    assert result.final.sensor_indices[2] == 0
    assert not result.final.inlier_mask[2]


def test_requires_complete_explicit_labels():
    sensor = NominalSensor("/a", np.zeros(3), np.ones(3), np.eye(3))
    geometry = NominalLadder("/a", "global", (sensor,))
    try:
        LabeledFaceRegistrar().fit(geometry, np.zeros((3, 3)), np.array([0, 1, 0]))
    except ValueError as exc:
        assert "outside nominal sensor" in str(exc)
    else:
        raise AssertionError("Invalid labels were accepted")


def test_labeled_stage_fails_closed_without_optical_preparation(tmp_path):
    import logging
    import pytest
    from detector_alignment.registration_stage import LadderRegistrationStage

    stage = LadderRegistrationStage(None, lambda path: tmp_path / path,
                                    logging.getLogger("optical-check"))
    ladder = {"registration": {"enabled": True, "mode": "labeled_faces",
                 "geometry_file": "unused.json", "measurement_frame": "/cframe"}}
    with pytest.raises(ValueError, match="requires optical geometry preparation"):
        stage.run([("L1", ladder)], None, tmp_path, None)


def test_labeled_stage_receives_optically_corrected_geometry(tmp_path, monkeypatch):
    import logging
    from types import SimpleNamespace
    import pytest
    import detector_alignment.registration_stage as registration_stage

    original = NominalSensor("/ladder/s0", np.array([0., 0., 2.]),
                             np.array([20., 20., 0.3]), np.eye(3))
    # Optical survey correction is local to the nominal sensor box.
    corrected = NominalSensor(original.node_path, original.center_mm + [0., 0., 0.35],
                              original.dimensions_mm, original.rotation)
    geometry = NominalLadder("/ladder", "/cframe", (corrected,))

    class Preparation:
        def prepare(self):
            pass

        def load(self, ladder_config, config, *, ladder_name):
            return geometry, {"root_geometry": {"managed": True},
                              "optical_metrology": {"applied": True,
                                                    "corrected_sensors": [{"node_path": corrected.node_path}]}}

    class CaptureFit:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, observed_geometry, *args, **kwargs):
            assert observed_geometry is geometry
            np.testing.assert_allclose(observed_geometry.sensors[0].center_mm,
                                       [0., 0., 2.35])
            raise RuntimeError("corrected geometry reached optimizer")

    stage = registration_stage.LadderRegistrationStage(
        None, lambda path: tmp_path / path, logging.getLogger("corrected-geometry"),
        geometry_preparation=Preparation())
    stage.load_inputs = lambda *args, **kwargs: (
        np.zeros((3, 3)), None, ["sensor.xyz"], np.zeros(3, dtype=int))
    monkeypatch.setattr(registration_stage, "LabeledFaceRegistrar", CaptureFit)
    ladder = {"registration": {"enabled": True, "mode": "labeled_faces",
                 "geometry_file": "unused.json", "measurement_frame": "/cframe"}}
    with pytest.raises(RuntimeError, match="corrected geometry reached optimizer"):
        stage.run([("L1", ladder)], None, tmp_path, None)


def test_labeled_stage_rejects_uncorrected_preparation(tmp_path):
    import logging
    import pytest
    from detector_alignment.registration_stage import LadderRegistrationStage

    class Preparation:
        def prepare(self):
            pass

        def load(self, ladder_config, config, *, ladder_name):
            return None, {"root_geometry": {"managed": True},
                          "optical_metrology": {"applied": False,
                                                "reason": "optical_metrology_disabled"}}

    stage = LadderRegistrationStage(None, lambda path: tmp_path / path,
                                    logging.getLogger("no-optical"),
                                    geometry_preparation=Preparation())
    ladder = {"registration": {"enabled": True, "mode": "labeled_faces",
                 "geometry_file": "unused.json", "measurement_frame": "/cframe"}}
    with pytest.raises(ValueError, match="requires optically corrected nominal sensor"):
        stage.run([("L1", ladder)], None, tmp_path, None)


def test_compressed_objective_matches_dense_point_residuals():
    from detector_alignment.labeled_registration import (
        _compressed_residual, _sensor_statistics,
    )

    rng = np.random.default_rng(14)
    points = rng.normal(size=(1203, 3)) * [4, 9, 0.02] + [200, -180, 8]
    ids = np.repeat(np.arange(3), 401)
    mask = rng.uniform(size=len(points)) > 0.07
    statistics = _sensor_statistics(points, ids, mask, 3)
    for _ in range(8):
        normals = rng.normal(size=(3, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        faces = rng.normal(size=(3, 3)) + [200, -180, 8]
        dense = np.einsum('ij,ij->i', points[mask] - faces[ids[mask]], normals[ids[mask]])
        compact = _compressed_residual(statistics, normals, faces)
        np.testing.assert_allclose(np.dot(compact, compact), np.dot(dense, dense), rtol=1e-12, atol=1e-8)
    assert compact.shape == (12,)


def test_pca_eigendecomposition_rejects_single_remote_outlier():
    from detector_alignment.labeled_registration import _plane_inliers

    rng = np.random.default_rng(7)
    xy = rng.uniform(-10, 10, size=(1200, 2))
    cloud = np.column_stack((xy, 0.002 * xy[:, 0] - 0.003 * xy[:, 1] +
                                   rng.normal(0, 0.005, len(xy))))
    cloud = np.vstack((cloud, [1, 1, 4]))
    mask = _plane_inliers(cloud, 3.0, np.array([0, 0, 1.]))
    assert not mask[-1]
    assert mask[:-1].sum() > 1150
