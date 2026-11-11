"""Stage-oriented YAML and legacy schema must resolve to equivalent inputs."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from detector_alignment.pipeline import AlignmentPipeline


def legacy_case(tmp_path):
    points = np.array([[x, y, 10] for x in (-5, 0, 5) for y in (-4, 0, 4)])
    np.savetxt(tmp_path / "facet.xyz", points)
    np.savetxt(tmp_path / "ref1.xyz", [[1, 2, 10]])
    np.savetxt(tmp_path / "ref2.xyz", [[1, 7, 10]])
    np.savetxt(tmp_path / "sensor.xyz", [[0, 0, 11], [1, 0, 11], [0, 1, 11]])
    return {
        "output_dir": "old_output",
        "output_precision": 3,
        "io": {"delimiter": "whitespace"},
        "uncertainty": {"mode": "constant", "sigma_xyz_mm": [0.01] * 3},
        "cframe": {"file": "facet.xyz", "facet_offset": 0.75},
        "plane_fit": {"ransac_iterations": 20, "random_seed": 1},
        "references": [
            {"file": "ref1.xyz", "nominal_center": [0, 0]},
            {"file": "ref2.xyz", "nominal_center": [0, 5]},
        ],
        "visualization": {"dpi": 40, "sample_size": 10,
                          "draw": {"cframe": False, "sensors": False,
                                   "full_ladders": False, "all_ladders_and_cframe": False}},
        "ladder_analysis": {
            "enabled": True, "analyze_individual_sensors": False,
            "L1": {"segmentation": {"mode": "segmented_files"},
                   "sensors": [{"name": "S1", "file": "sensor.xyz"}]},
        },
    }


def stage_case(old):
    new = deepcopy(old)
    new["io"].update(output_dir="new_output", output_precision=new.pop("output_precision"))
    new.pop("output_dir")
    new["half_unit"] = {
        "name": "test", "assembly_naming": "STS8",
        "uncertainty": new.pop("uncertainty"),
        "cframe": {
            **new.pop("cframe"),
            "plane_fit": new.pop("plane_fit"),
            "references": new.pop("references"),
        },
        "ladders": {"L1": new["ladder_analysis"].pop("L1")},
    }
    return new


def test_stage_yaml_preserves_alignment_uncertainty_paths_and_exports(tmp_path):
    old = legacy_case(tmp_path)
    new = stage_case(old)
    old_path, new_path = tmp_path / "old.yaml", tmp_path / "new.yaml"
    old_path.write_text(yaml.safe_dump(old))
    new_path.write_text(yaml.safe_dump(new))
    original = deepcopy(new)
    old_pipeline, new_pipeline = (AlignmentPipeline.from_yaml(p) for p in (old_path, new_path))
    assert new == original
    assert new_pipeline._resolve(new["half_unit"]["cframe"]["file"]) == tmp_path / "facet.xyz"
    assert new_pipeline._ladder_configs()[0][1]["sensors"][0]["file"] == "sensor.xyz"
    old_result, new_result = old_pipeline.run(), new_pipeline.run()
    np.testing.assert_allclose(new_result.homogeneous, old_result.homogeneous)
    np.testing.assert_allclose(new_result.covariance_joint, old_result.covariance_joint)
    exported = (tmp_path / "new_output/point_clouds/L1_sensor_transformed.xyz").read_text()
    assert exported == (tmp_path / "old_output/point_clouds/L1_sensor_transformed.xyz").read_text()
    assert exported.splitlines()[0].endswith("1.750")
    assert "L1" not in json.loads((tmp_path / "new_output/results.json").read_text()).get("ladder_registrations", {})


def test_enabled_registration_runs_with_ladder_analysis_disabled(tmp_path, monkeypatch):
    new = stage_case(legacy_case(tmp_path))
    new["ladder_analysis"]["enabled"] = False
    new["registration"] = {
        "enabled": True, "measurement_frame": "global", "draw": False,
        "ladders": {"L1": {"geometry_file": "geometry.json"}},
    }
    geometry = {
        "schema_version": 1, "units": "mm", "coordinate_frame": "global",
        "node_path": "/ladder", "sensor_count": 1,
        "sensors": [{"node_path": "/ladder/sensor", "center": [-1, -2, 1.6],
                     "dimensions": [20, 20, 0.3], "rotation": np.eye(3).tolist()}],
    }
    (tmp_path / "geometry.json").write_text(json.dumps(geometry))
    from detector_alignment.ladder_metrology import SENSOR_FRAME, LadderMetrologyDriver
    new["half_unit"].update(name="U3DL")
    new["half_unit"]["ladders"]["L1"]["assembly_ladder"] = "L3DL100112"
    new["ladder_metrology"] = {
        "enabled": True, "source_dir": "source", "output_dir": "optical",
        "sensor_frame": SENSOR_FRAME,
    }
    new["registration"]["ladders"]["L1"]["optical_sensor_map"] = {
        "M3DL1B0001120B2": {"node_path": "/ladder/sensor",
                            "sensor_to_geometry": np.eye(4).tolist()}
    }
    deviation = np.eye(4)
    deviation[2, 3] = 0.1
    monkeypatch.setattr(LadderMetrologyDriver, "run", lambda self, **kwargs: {
        "input_fingerprint": "current", "result_digest": "result",
        "ladders": {"L3DL100112": {"sensors": {
            "M3DL1B0001120B2": {"nominal_to_measured": deviation.tolist()}
        }}},
    })
    path = tmp_path / "stages.yaml"
    path.write_text(yaml.safe_dump(new))
    pipeline = AlignmentPipeline.from_yaml(path)
    pipeline.run()
    assert "L1" in pipeline.registration_stage.results
    assert not pipeline.ladder_processors


def test_registration_is_independent_of_ladder_analysis_and_keeps_preview_inputs(tmp_path):
    new = stage_case(legacy_case(tmp_path))
    new["ladder_analysis"]["enabled"] = False
    new["registration"] = {
        "enabled": False, "measurement_frame": "global",
        "ladders": {"L1": {"geometry_file": "geometry.json"}},
    }
    path = tmp_path / "stages.yaml"
    path.write_text(yaml.safe_dump(new))
    pipeline = AlignmentPipeline.from_yaml(path)
    name, ladder = pipeline._ladder_configs()[0]
    assert name == "L1"
    assert ladder["registration"]["geometry_file"] == "geometry.json"
    assert ladder["registration"]["enabled"] is False
    pipeline.run()
    assert pipeline.registration_stage.results == {}
    assert (tmp_path / "new_output/point_clouds/L1_sensor_transformed.xyz").exists()
    assert not pipeline.ladder_processors


@pytest.mark.parametrize("alter,match", [
    (lambda c: c.update(output_dir="other"), "io.output_dir"),
    (lambda c: c.update(uncertainty={"mode": "none"}), "half_unit.uncertainty"),
    (lambda c: c.update(cframe={"file": "other"}), "half_unit.cframe"),
    (lambda c: c["ladder_analysis"].update(L1={}), "mix legacy"),
    (lambda c: c["half_unit"]["ladders"]["L1"].update(registration={}), "registration.ladders"),
    (lambda c: c.update(registration={"ladders": {"missing": {}}}), "unknown half-unit"),
])
def test_mixed_or_unknown_stage_keys_fail_before_processing(tmp_path, alter, match):
    config = stage_case(legacy_case(tmp_path))
    alter(config)
    with pytest.raises(ValueError, match=match):
        AlignmentPipeline(config)


def test_dataset_config_uses_stage_sections_and_resolves_paths():
    path = Path(__file__).resolve().parents[1] / "input/u3dl/config.yaml"
    config = yaml.safe_load(path.read_text())
    pipeline = AlignmentPipeline.from_yaml(path)
    assert len(pipeline._ladder_configs()) == 3
    assert len(config["half_unit"]["ladders"]["L3DL100112"]["sensors"]) == 10
    assert config["half_unit"]["uncertainty"]["sigma_xyz_mm"] == [0.05] * 3
    assert pipeline._resolve(config["io"]["output_dir"]) == (path.parent / "../../../output/U3DL")
    registration = pipeline._ladder_configs()[0][1]["registration"]
    assert "mode" not in registration
    assert registration["optical_sensor_map"]["mode"] == "y_order"
