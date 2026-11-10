"""External geo-inspector driver contract and staged nominal-geometry publication."""

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from detector_alignment.root_geometry import RootGeometryDriver


REFERENCE = "/sts_v26a/Unit04L_10"
LADDER = REFERENCE + "/Ladder_LadderType12_401"


def root_config(root):
    (root / "geometry.root").write_bytes(b"root-geometry")
    config = {
        "half_unit": {"name": "U3DL", "assembly_naming": "STS8"},
        "root_geometry": {
            "enabled": True,
            "source_file": "geometry.root",
            "node": "Unit04L_10",
            "frame": "node",
            "output_dir": "output/U3DL/root_geo",
            "execution": {
                "python": "geometry-python",
                "root": "configured-root",
                "working_dir": ".",
                "timeout_seconds": 15,
            },
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return {**config, "_config_path": str(path), "_config_dir": str(root)}


def make_driver(root, config=None):
    return RootGeometryDriver(
        config or root_config(root),
        lambda path: root / path,
        logging.getLogger(__name__),
    )


def geometry_payload(source):
    return {
        "schema_version": 2,
        "units": "cm",
        "coordinate_frame": "node",
        "reference_node": REFERENCE,
        "reference_to_global": {
            "translation": [0, 0, 1.425],
            "rotation": np.eye(3).tolist(),
        },
        "source_geometry": str(source.resolve()),
        "node_path": LADDER,
        "sensor_count": 1,
        "bounds": {"min": [-1, -1.5, -0.015], "max": [1, 1.5, 0.015]},
        "sensors": [
            {
                "node_path": LADDER + "/Module_0/Sensor01_0",
                "center": [0, 0, 0],
                "dimensions": [2, 3, 0.03],
                "rotation": np.eye(3).tolist(),
                "corners": [],
                "bounds": {"min": [-1, -1.5, -0.015], "max": [1, 1.5, 0.015]},
            }
        ],
    }


def mock_inspector(monkeypatch, root, *, failure=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if isinstance(failure, Exception):
            raise failure
        if failure:
            return SimpleNamespace(returncode=2, stdout="", stderr="ROOT failed")
        destination = Path(command[command.index("--output-dir") + 1])
        path = destination / "Ladder_LadderType12_401.json"
        path.write_text(json.dumps(geometry_payload(root / "geometry.root")))
        receipt = {
            "node_path": REFERENCE,
            "ladder_count": 1,
            "files": [str(path)],
            "coordinate_frame": "node",
            "reference_node": REFERENCE,
            "reference_to_global": {
                "translation": [0, 0, 1.425],
                "rotation": np.eye(3).tolist(),
            },
            "source_geometry": str((root / "geometry.root").resolve()),
        }
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt), stderr="ROOT extraction log\n"
        )

    monkeypatch.setattr("detector_alignment.root_geometry.subprocess.run", run)
    return calls


def test_driver_runs_geo_inspector_and_publishes_validated_batch(tmp_path, monkeypatch):
    driver = make_driver(tmp_path)
    calls = mock_inspector(monkeypatch, tmp_path)
    result = driver.run()
    command, kwargs = calls[0]
    assert command[:3] == ["geometry-python", "-m", "geo_inspector"]
    assert command[3:5] == ["ladder", str((tmp_path / "geometry.root").resolve())]
    assert command[command.index("--root") + 1] == "configured-root"
    assert command[command.index("--frame") + 1] == "node"
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["timeout"] == 15
    assert kwargs.get("shell", False) is False
    assert result["measurement_frame"] == REFERENCE
    assert result["files"][0]["node_path"] == LADDER
    assert driver.manifest_path.is_file()
    assert (driver.output_dir / "execution.log").is_file()
    metadata = driver.geometry_metadata(result["files"][0]["path"])
    assert metadata["managed"]
    assert metadata["result_digest"] == result["result_digest"]


def test_check_mode_reuses_and_integrity_checks_without_root(tmp_path, monkeypatch):
    driver = make_driver(tmp_path)
    mock_inspector(monkeypatch, tmp_path)
    result = driver.run()
    monkeypatch.setattr(
        "detector_alignment.root_geometry.subprocess.run",
        lambda *a, **k: pytest.fail("Freshness check must not execute ROOT"),
    )
    assert driver.run(check_only=True) == result
    geometry = Path(result["files"][0]["path"])
    geometry.write_text(geometry.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        driver.run(check_only=True)


@pytest.mark.parametrize(
    "failure,match",
    [
        (True, "exit 2"),
        (OSError("missing"), "Cannot execute"),
        (subprocess.TimeoutExpired("geo_inspector", 15), "exceeded"),
    ],
)
def test_failed_extraction_never_replaces_last_valid_batch(
    tmp_path, monkeypatch, failure, match
):
    driver = make_driver(tmp_path)
    mock_inspector(monkeypatch, tmp_path)
    original = driver.run()
    manifest_before = driver.manifest_path.read_bytes()
    mock_inspector(monkeypatch, tmp_path, failure=failure)
    with pytest.raises(RuntimeError, match=match):
        driver.run()
    assert driver.manifest_path.read_bytes() == manifest_before
    assert driver.run(check_only=True) == original


def test_only_manifested_geometry_can_feed_preparation(tmp_path, monkeypatch):
    driver = make_driver(tmp_path)
    mock_inspector(monkeypatch, tmp_path)
    driver.run()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="produced by"):
        driver.geometry_metadata(outside)


def test_changed_in_memory_root_configuration_is_rejected(tmp_path):
    config = root_config(tmp_path)
    config["root_geometry"]["frame"] = "global"
    with pytest.raises(ValueError, match="changed in memory"):
        make_driver(tmp_path, config).run()
