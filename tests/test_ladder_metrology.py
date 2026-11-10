"""External producer contract and ROOT geometry composition; no optical fitter imports."""

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from detector_alignment.ladder_metrology import (
    SENSOR_FRAME,
    LadderGeometryPreparation,
    LadderMetrologyDriver,
    _digest,
)

LADDER = "L3DL100112"
MODULE = "M3DL1B0001120B2"


def optical_config(root):
    config = {
        "half_unit": {"name": "U3DL", "assembly_naming": "STS8"},
        "ladder_metrology": {
            "enabled": True,
            "source_dir": "source",
            "result_file": "optical/sensor_transforms.json",
            "sensor_frame": SENSOR_FRAME,
            "execution": {"python": "optical-python", "timeout_seconds": 12},
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return {**config, "_config_path": str(path), "_config_dir": str(root)}


def result_payload():
    deviation = np.eye(4)
    deviation[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
    deviation[:3, 3] = [0.2, -0.3, 0.1]
    sensors = {
        MODULE: {
            "status": "ok",
            "sensor_frame": SENSOR_FRAME,
            "nominal_to_measured": deviation.tolist(),
            "nominal_sensor_to_survey": np.eye(4).tolist(),
            "covariance": (np.eye(6) * 1e-8).tolist(),
            "dimensions_xy_mm": [20, 30],
            "covariance_parameters": [
                "tx",
                "ty",
                "tz",
                "rotvec_x",
                "rotvec_y",
                "rotvec_z",
            ],
            "covariance_units": ["mm", "mm", "mm", "rad", "rad", "rad"],
        }
    }
    ladders = {LADDER: {"module_order": [MODULE], "sensors": sensors}}
    return {
        "schema_version": 1,
        "producer": "sts_ladder_metrology",
        "half_unit": "U3DL",
        "assembly_naming": "STS8",
        "units": "mm",
        "direction": "nominal_to_measured",
        "sensor_frame": SENSOR_FRAME,
        "input_fingerprint": "current-inputs",
        "result_digest": _digest(ladders),
        "ladders": ladders,
    }


def make_driver(root, config=None):
    config = config or optical_config(root)
    return LadderMetrologyDriver(
        config, lambda path: root / path, logging.getLogger(__name__)
    )


def mock_producer(monkeypatch, root, payload, *, failure=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if isinstance(failure, Exception):
            raise failure
        if failure:
            return SimpleNamespace(
                returncode=2, stdout="", stderr="source input missing"
            )
        path = root / "optical/sensor_transforms.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload))
        receipt = {
            "result_file": str(path),
            "input_fingerprint": payload["input_fingerprint"],
            "result_digest": payload["result_digest"],
            "checked_only": "--check" in command,
        }
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt), stderr="producer log\n"
        )

    monkeypatch.setattr("detector_alignment.ladder_metrology.subprocess.run", run)
    return calls


@pytest.mark.parametrize("preview", [False, True])
def test_driver_invokes_external_module_and_collects_its_contract(
    tmp_path, monkeypatch, preview
):
    payload = result_payload()
    driver = make_driver(tmp_path)
    calls = mock_producer(monkeypatch, tmp_path, payload)
    assert driver.run(check_only=preview) == payload
    command, kwargs = calls[0]
    assert command[:3] == ["optical-python", "-m", "sts_ladder_metrology.main"]
    assert ("--check" in command) is preview
    assert kwargs["timeout"] == 12 and kwargs["cwd"] == tmp_path.resolve()
    assert kwargs.get("shell", False) is False
    assert (tmp_path / "optical/execution.log").exists() is (not preview)


@pytest.mark.parametrize(
    "failure,match",
    [
        (True, "exit 2"),
        (OSError("missing"), "Cannot execute"),
        (subprocess.TimeoutExpired("producer", 12), "exceeded"),
    ],
)
def test_driver_never_falls_back_after_execution_failure(
    tmp_path, monkeypatch, failure, match
):
    driver = make_driver(tmp_path)
    mock_producer(monkeypatch, tmp_path, result_payload(), failure=failure)
    with pytest.raises(RuntimeError, match=match):
        driver.run()


def test_explicit_working_directory_is_used_for_external_module(tmp_path, monkeypatch):
    producer_root = tmp_path / "producer-root"
    producer_root.mkdir()
    config = optical_config(tmp_path)
    config["ladder_metrology"]["execution"]["working_dir"] = "producer-root"
    Path(config["_config_path"]).write_text(
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")})
    )
    calls = mock_producer(monkeypatch, tmp_path, result_payload())
    make_driver(tmp_path, config).run()
    assert calls[0][1]["cwd"] == producer_root.resolve()


def test_legacy_output_directory_still_resolves_default_artifact(tmp_path):
    config = optical_config(tmp_path)
    options = config["ladder_metrology"]
    options["output_dir"] = "legacy-optical"
    del options["result_file"]
    Path(config["_config_path"]).write_text(
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")})
    )
    assert make_driver(tmp_path, config).result_path == (
        tmp_path / "legacy-optical/sensor_transforms.json"
    )


def test_disabled_stage_does_not_execute_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("Unexpected external execution")
    )
    driver = make_driver(tmp_path, {"ladder_metrology": {"enabled": False}})
    assert driver.run() is None


def test_changed_in_memory_configuration_is_not_sent_to_a_different_saved_config(
    tmp_path, monkeypatch
):
    config = optical_config(tmp_path)
    config["ladder_metrology"]["mode"] = "reuse"
    with pytest.raises(ValueError, match="changed in memory"):
        make_driver(tmp_path, config).run()


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", "measured_to_nominal"),
        ("units", "cm"),
        ("half_unit", "U3UL"),
        ("schema_version", 99),
    ],
)
def test_incompatible_producer_contract_is_rejected(tmp_path, field, value):
    payload = result_payload()
    payload[field] = value
    with pytest.raises(ValueError, match="incompatible"):
        make_driver(tmp_path)._validate_result(payload)


def test_result_integrity_and_failed_fits_are_rejected(tmp_path):
    driver = make_driver(tmp_path)
    payload = result_payload()
    payload["ladders"][LADDER]["sensors"][MODULE]["status"] = "failed"
    with pytest.raises(ValueError, match="modified"):
        driver._validate_result(payload)
    payload["result_digest"] = _digest(payload["ladders"])
    with pytest.raises(ValueError, match="Invalid optical fit"):
        driver._validate_result(payload)


def prepare_geometry(root, payload):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.4, 0.1, -0.6]).as_matrix()
    pose[:3, 3] = [10, 20, 30]
    geometry = {
        "schema_version": 1,
        "units": "cm",
        "coordinate_frame": "global",
        "node_path": "/ladder",
        "sensor_count": 1,
        "sensors": [
            {
                "node_path": "/ladder/sensor",
                "center": (pose[:3, 3] / 10).tolist(),
                "dimensions": [2, 3, 0.03],
                "rotation": pose[:3, :3].tolist(),
            }
        ],
    }
    (root / "geometry.json").write_text(json.dumps(geometry))
    driver = make_driver(root)
    preparation = LadderGeometryPreparation(driver, lambda path: root / path)
    preparation.result = payload
    bridge = np.eye(4)
    bridge[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    bridge[:3, 3] = [0.5, 0, 0]
    registration = {
        "geometry_file": "geometry.json",
        "measurement_frame": "global",
        "optical_sensor_map": {
            MODULE: {
                "node_path": "/ladder/sensor",
                "sensor_to_geometry": bridge.tolist(),
            }
        },
    }
    return preparation, registration, pose, bridge


@pytest.mark.parametrize("reflection", [False, True])
def test_sensor_local_transform_right_multiplies_rotated_root_pose_once(
    tmp_path, reflection
):
    payload = result_payload()
    preparation, registration, pose, bridge = prepare_geometry(tmp_path, payload)
    if reflection:
        bridge[:3, 0] *= -1
        registration["optical_sensor_map"][MODULE]["sensor_to_geometry"] = (
            bridge.tolist()
        )
    before = (tmp_path / "geometry.json").read_bytes()
    first, metadata = preparation.load({"assembly_ladder": LADDER}, registration)
    second, _ = preparation.load({"assembly_ladder": LADDER}, registration)
    deviation = np.array(
        payload["ladders"][LADDER]["sensors"][MODULE]["nominal_to_measured"]
    )
    expected = pose @ bridge @ deviation @ np.linalg.inv(bridge)
    np.testing.assert_allclose(first.sensors[0].center_mm, expected[:3, 3])
    np.testing.assert_allclose(first.sensors[0].rotation, expected[:3, :3])
    np.testing.assert_allclose(first.sensors[0].dimensions_mm, [20, 30, 0.3])
    np.testing.assert_allclose(second.corners_mm, first.corners_mm)
    optical = metadata["optical_metrology"]
    assert optical["applied"] and optical["result_digest"] == payload["result_digest"]
    assert metadata["root_geometry"] == {"managed": False}
    assert (tmp_path / "geometry.json").read_bytes() == before


@pytest.mark.parametrize(
    "bad_mapping",
    [
        {},
        {MODULE: {"node_path": "/wrong", "sensor_to_geometry": np.eye(4).tolist()}},
        {MODULE: {"node_path": "/ladder/sensor"}},
    ],
)
def test_missing_or_unverified_sensor_associations_are_errors(tmp_path, bad_mapping):
    preparation, registration, _, _ = prepare_geometry(tmp_path, result_payload())
    registration["optical_sensor_map"] = bad_mapping
    with pytest.raises(ValueError, match="map|requires"):
        preparation.load({"assembly_ladder": LADDER}, registration)
    with pytest.raises(ValueError, match="assembly_ladder"):
        preparation.load({}, registration)


def test_y_order_mapping_resolves_nodes_but_keeps_explicit_bridge(tmp_path):
    payload = result_payload()
    preparation, registration, pose, bridge = prepare_geometry(tmp_path, payload)
    registration["optical_sensor_map"] = {
        "mode": "y_order",
        "sensor_to_geometry": bridge.tolist(),
    }
    # One-sided synthetic geometry cannot be ordered by top/bottom; fail closed.
    with pytest.raises(ValueError, match="equal B/T"):
        preparation.load({"assembly_ladder": LADDER}, registration)


def test_y_order_mapping_applies_correction_to_selected_root_node(tmp_path):
    payload = result_payload()
    top = "M3DL1T0001120A2"
    payload["ladders"][LADDER]["sensors"][top] = json.loads(
        json.dumps(payload["ladders"][LADDER]["sensors"][MODULE])
    )
    payload["ladders"][LADDER]["sensors"][top]["nominal_to_measured"][0][3] = 1.5
    preparation, registration, pose, bridge = prepare_geometry(tmp_path, payload)
    geometry = json.loads((tmp_path / "geometry.json").read_text())
    geometry["sensors"][0]["center"][1] = -1
    top_node = json.loads(json.dumps(geometry["sensors"][0]))
    top_node["node_path"] = "/ladder/top"
    top_node["center"][1] = 1
    geometry["sensors"].append(top_node)
    geometry["sensor_count"] = 2
    (tmp_path / "geometry.json").write_text(json.dumps(geometry))
    registration["optical_sensor_map"] = {
        "mode": "y_order", "sensor_to_geometry": bridge.tolist()
    }
    corrected, metadata = preparation.load({"assembly_ladder": LADDER}, registration)
    assert metadata["optical_metrology"]["sensor_mapping"][top]["node_path"] == "/ladder/top"
    assert metadata["optical_metrology"]["sensor_mapping"][MODULE]["node_path"] == "/ladder/sensor"
    for sensor, module in zip(corrected.sensors, (MODULE, top)):
        original = next(row for row in geometry["sensors"] if row["node_path"] == sensor.node_path)
        deviation = np.asarray(payload["ladders"][LADDER]["sensors"][module]["nominal_to_measured"])
        displacement = bridge @ deviation @ np.linalg.inv(bridge)
        expected = np.asarray(original["center"]) * 10 + pose[:3, :3] @ displacement[:3, 3]
        np.testing.assert_allclose(sensor.center_mm, expected)


def test_ladder_axis_bridge_from_root_boxes_applies_about_box_centres(tmp_path):
    payload = result_payload()
    top = "M3DL1T0001120A2"
    payload["ladders"][LADDER]["sensors"][top] = json.loads(
        json.dumps(payload["ladders"][LADDER]["sensors"][MODULE])
    )
    preparation, registration, pose, _ = prepare_geometry(tmp_path, payload)
    geometry = json.loads((tmp_path / "geometry.json").read_text())
    geometry["sensors"][0]["center"][1] = -1
    geometry["sensors"][0]["rotation"] = (
        pose[:3, :3] @ np.diag([-1, -1, 1])
    ).tolist()
    top_node = json.loads(json.dumps(geometry["sensors"][0]))
    top_node["node_path"] = "/ladder/top"
    top_node["center"][1] = 1
    top_node["rotation"] = pose[:3, :3].tolist()
    geometry["sensors"].append(top_node)
    geometry["sensor_count"] = 2
    (tmp_path / "geometry.json").write_text(json.dumps(geometry))
    registration["optical_sensor_map"] = {
        "mode": "y_order", "sensor_to_geometry": "ladder_aligned_box_center"
    }
    corrected, metadata = preparation.load({}, registration, ladder_name=LADDER)
    mapping = metadata["optical_metrology"]["sensor_mapping"]
    assert metadata["optical_metrology"]["assembly_ladder"] == LADDER
    np.testing.assert_allclose(
        mapping[top]["sensor_to_geometry"], np.eye(4), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(mapping[MODULE]["sensor_to_geometry"])[:3, :3],
        np.diag([-1, -1, 1]),
        atol=1e-12,
    )
    for sensor, module in zip(corrected.sensors, (MODULE, top)):
        original = next(row for row in geometry["sensors"] if row["node_path"] == sensor.node_path)
        deviation = np.asarray(payload["ladders"][LADDER]["sensors"][module]["nominal_to_measured"])
        expected_center = np.asarray(original["center"]) * 10 + pose[:3, :3] @ deviation[:3, 3]
        expected_rotation = np.asarray(original["rotation"]) @ (
            np.asarray(mapping[module]["sensor_to_geometry"])[:3, :3]
            @ deviation[:3, :3]
            @ np.asarray(mapping[module]["sensor_to_geometry"])[:3, :3].T
        )
        np.testing.assert_allclose(sensor.center_mm, expected_center)
        np.testing.assert_allclose(sensor.rotation, expected_rotation)


def test_reuse_mode_checks_without_refitting(tmp_path, monkeypatch):
    config = optical_config(tmp_path)
    config["ladder_metrology"]["mode"] = "reuse"
    Path(config["_config_path"]).write_text(
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")})
    )
    calls = mock_producer(monkeypatch, tmp_path, result_payload())
    make_driver(tmp_path, config).run()
    assert "--check" in calls[0][0]
