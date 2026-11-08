import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from detector_alignment.ladder_registration import LadderRegistrar
from detector_alignment.pipeline import AlignmentPipeline
from detector_alignment.registration_preview import RegistrationPreview
from detector_alignment.registration_stage import LadderRegistrationStage


def pipeline_config(root: Path, *, segmented=False):
    np.savetxt(root / "cframe.xyz", [[x, y, 10] for x in range(5) for y in range(5)])
    np.savetxt(root / "ref1.xyz", [[1, 2, 10]])
    np.savetxt(root / "ref2.xyz", [[1, 7, 10]])
    points = np.array([[x, y, 0.35] for x in [-8, 0, 8] for y in [-10, 0, 10]])
    np.savetxt(root / "ladder.xyz", points + [1, 2, 10])
    geometry = {
        "schema_version": 1,
        "units": "cm",
        "coordinate_frame": "global",
        "node_path": "/ladder",
        "sensor_count": 1,
        "sensors": [
            {
                "node_path": "/ladder/sensor",
                "center": [0, 0, 0],
                "dimensions": [2, 3, 0.03],
                "rotation": np.eye(3).tolist(),
            }
        ],
    }
    (root / "geometry.json").write_text(json.dumps(geometry))
    ladder = {
        "segmentation": {
            "mode": "automatic",
            "config": {"min_points_per_sensor": 100000},
        },
        "file": "ladder.xyz",
        "registration": {
            "geometry_file": "geometry.json",
            "measurement_frame": "global",
        },
    }
    if segmented:
        np.savetxt(root / "sensor_a.xyz", points[:5] + [1, 2, 10])
        np.savetxt(root / "sensor_b.xyz", points[5:] + [1, 2, 10])
        ladder = {
            **ladder,
            "segmentation": {"mode": "segmented_files"},
            "sensors": [
                {"name": "measured_a", "file": "sensor_a.xyz"},
                {"name": "measured_b", "file": "sensor_b.xyz"},
            ],
        }
    return {
        "_config_dir": str(root),
        "output_dir": "output",
        "uncertainty": {"mode": "constant", "sigma_xyz_mm": [0.01, 0.02, 0.03]},
        "cframe": {"file": "cframe.xyz"},
        "plane_fit": {
            "ransac_iterations": 20,
            "ransac_threshold_mm": 0.01,
            "random_seed": 1,
        },
        "references": [
            {"file": "ref1.xyz", "nominal_center": [0, 0]},
            {"file": "ref2.xyz", "nominal_center": [0, 5]},
        ],
        "visualization": {
            "dpi": 50,
            "sample_size": 5,
            "draw": {
                "cframe": False,
                "sensors": False,
                "full_ladders": False,
                "all_ladders_and_cframe": False,
            },
        },
        "ladder_analysis": {
            "enabled": True,
            "analyze_individual_sensors": False,
            "L1": ladder,
        },
    }, points


@pytest.mark.parametrize("segmented", [False, True])
def test_registration_runs_without_sensor_analysis_and_exports_every_point(
    tmp_path, monkeypatch, segmented
):
    config, points = pipeline_config(tmp_path, segmented=segmented)
    pipeline = AlignmentPipeline(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("Individual sensor-plane analysis must remain disabled")

    monkeypatch.setattr(pipeline.sensor_plane_fitter, "fit", forbidden)
    monkeypatch.setattr("detector_alignment.pipeline.segment_ladder", forbidden)
    pipeline.run()
    root = tmp_path / "output"
    payload = json.loads((root / "results.json").read_text())
    assert "ladders" not in payload
    summary = payload["ladder_registrations"]["L1"]
    assert summary["accepted"]
    assert summary["feasible"]
    assert summary["total_points"] == 9
    assert summary["final"]["outside_points"] == 0
    assert summary["final"]["surface_rms_mm"] < 1e-6
    assert summary["uncertainty_sigma"] == 3
    assert (root / "images" / "L1_ladder_registration.png").is_file()
    matrix = np.load(root / "matrices" / "L1_nominal_to_as_built.npy")
    np.testing.assert_allclose(matrix, summary["homogeneous_matrix"])
    residuals = np.loadtxt(
        root / "registrations" / "L1_point_residuals.csv", delimiter=",", skiprows=1
    )
    assert residuals.shape == (9, 7)
    np.testing.assert_allclose(residuals[:, 1:4], points, atol=1e-9)
    np.testing.assert_array_equal(residuals[:, 0], np.arange(9))
    np.testing.assert_allclose(
        residuals[:, 5], pipeline.registration_stage.results["L1"].final.residuals_mm
    )
    assert len(summary["input_files"]) == (2 if segmented else 1)
    assert "task=register_ladder" in (root / "logs" / "pipeline.log").read_text()


def test_registration_drawing_can_be_disabled_without_skipping_fit(tmp_path):
    config, _ = pipeline_config(tmp_path)
    registration = config["ladder_analysis"]["L1"]["registration"]
    registration["draw"] = False
    stale = tmp_path / "output" / "images" / "L1_ladder_registration.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    pipeline = AlignmentPipeline(config)
    pipeline.run()
    assert not stale.exists()
    assert pipeline.registration_stage.results["L1"].accepted


def test_disabled_registration_does_not_load_geometry_or_clouds(tmp_path, monkeypatch):
    config, _ = pipeline_config(tmp_path)
    config["ladder_analysis"]["L1"]["registration"] = {"enabled": False}
    pipeline = AlignmentPipeline(config)
    original_load = pipeline.loader.load

    def load(path, uncertainty):
        assert Path(path).name != "ladder.xyz"
        return original_load(path, uncertainty)

    monkeypatch.setattr(pipeline.loader, "load", load)
    stale_paths = [
        tmp_path / "output" / "images" / "L1_ladder_registration.png",
        tmp_path / "output" / "matrices" / "L1_nominal_to_as_built.npy",
        tmp_path / "output" / "registrations" / "L1_point_residuals.csv",
    ]
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")
    pipeline.run()
    payload = json.loads((tmp_path / "output" / "results.json").read_text())
    assert "ladder_registrations" not in payload
    assert all(not path.exists() for path in stale_paths)


def test_infeasible_registration_is_exported_as_diagnostic_not_success(tmp_path):
    config, points = pipeline_config(tmp_path)
    points = np.vstack((points, [100, 100, 100]))
    np.savetxt(tmp_path / "ladder.xyz", points + [1, 2, 10])
    config["ladder_analysis"]["L1"]["registration"].update(
        draw=False,
        max_translation_mm=0.1,
        max_rotation_deg=0.1,
        max_iterations=20,
    )
    AlignmentPipeline(config).run()
    payload = json.loads((tmp_path / "output" / "results.json").read_text())
    summary = payload["ladder_registrations"]["L1"]
    assert not summary["accepted"]
    assert not summary["feasible"]
    assert summary["total_points"] == 10
    assert summary["final"]["outside_points"] >= 1


@pytest.mark.parametrize(
    "config,match",
    [
        ([], "mapping"),
        ({"enabled": "true"}, "true or false"),
        ({"draw": 0}, "true or false"),
        ({"typo": 1}, "Unknown"),
        ({"measurement_frame": "global"}, "geometry_file"),
        ({"geometry_file": "g.json"}, "measurement_frame"),
        (
            {
                "geometry_file": "g.json",
                "measurement_frame": "global",
                "initial_transform": np.zeros((4, 4)).tolist(),
            },
            "rigid",
        ),
    ],
)
def test_registration_configuration_validation(config, match):
    with pytest.raises(ValueError, match=match):
        LadderRegistrationStage.validate_config(config)


@pytest.mark.parametrize(
    "defaults,overrides,expected",
    [
        (None, None, None),
        ({"enabled": False}, None, {"enabled": False}),
        (
            {"measurement_frame": "global", "uncertainty_sigma": 2.0},
            {"geometry_file": "g.json", "uncertainty_sigma": 4.0},
            {
                "measurement_frame": "global",
                "geometry_file": "g.json",
                "uncertainty_sigma": 4.0,
            },
        ),
        (
            {"geometry_file": "g.json", "measurement_frame": "global", "draw": True},
            {"draw": False},
            {"geometry_file": "g.json", "measurement_frame": "global", "draw": False},
        ),
        (
            {"enabled": False, "measurement_frame": "global"},
            {"enabled": True, "geometry_file": "g.json"},
            {"enabled": True, "measurement_frame": "global", "geometry_file": "g.json"},
        ),
        (
            {"enabled": True, "measurement_frame": "global"},
            {"enabled": False},
            {"enabled": False, "measurement_frame": "global"},
        ),
    ],
)
def test_registration_merges_fields_without_mutating_configuration(
    defaults, overrides, expected
):
    original_defaults, original_overrides = deepcopy(defaults), deepcopy(overrides)
    assert LadderRegistrationStage.resolve_config(defaults, overrides) == expected
    assert defaults == original_defaults
    assert overrides == original_overrides


def test_initial_transform_override_replaces_the_whole_matrix():
    initial = np.eye(4).tolist()
    replacement = np.eye(4)
    replacement[:3, 3] = [1, 2, 3]
    effective = LadderRegistrationStage.resolve_config(
        {
            "geometry_file": "g.json",
            "measurement_frame": "global",
            "initial_transform": initial,
        },
        {"initial_transform": replacement.tolist()},
    )
    assert effective["initial_transform"] == replacement.tolist()
    assert initial == np.eye(4).tolist()


@pytest.mark.parametrize(
    "defaults,overrides,match",
    [
        ([], None, "mapping"),
        ({"draw": "false"}, {"enabled": False}, "true or false"),
        ({"typo": 1}, {"enabled": False}, "Unknown"),
        ({"measurement_frame": "global"}, {}, "geometry_file"),
        ({"geometry_file": "g.json"}, {}, "measurement_frame"),
        (
            {"geometry_file": "g.json", "measurement_frame": "global"},
            {"geometry_file": None},
            "geometry_file",
        ),
    ],
)
def test_invalid_global_or_merged_registration(defaults, overrides, match):
    with pytest.raises(ValueError, match=match):
        LadderRegistrationStage.resolve_config(defaults, overrides)


def test_shared_registration_defaults_and_independent_ladder_overrides(tmp_path):
    config, _ = pipeline_config(tmp_path)
    ladders = config["ladder_analysis"]
    defaults = {
        "enabled": True,
        "geometry_file": "geometry.json",
        "measurement_frame": "global",
        "uncertainty_sigma": 2.0,
        "max_rotation_deg": 1.5,
        "draw": False,
    }
    ladders["registration"] = defaults
    ladders["L1"]["registration"] = {"uncertainty_sigma": 4.0, "draw": True}
    ladders["L2"] = deepcopy(ladders["L1"])
    ladders["L2"].pop("registration")
    ladders["L3"] = deepcopy(ladders["L2"])
    ladders["L3"]["registration"] = {"enabled": False}
    original = deepcopy(config)
    pipeline = AlignmentPipeline(config)
    assert [name for name, _ in pipeline._ladder_configs()] == ["L1", "L2", "L3"]
    pipeline.run()
    assert config == original
    root = tmp_path / "output"
    summaries = json.loads((root / "results.json").read_text())["ladder_registrations"]
    assert set(summaries) == {"L1", "L2"}
    assert summaries["L1"]["accepted"] and summaries["L2"]["accepted"]
    assert summaries["L1"]["options"]["uncertainty_sigma"] == 4.0
    assert summaries["L2"]["options"]["uncertainty_sigma"] == 2.0
    assert summaries["L1"]["options"]["max_rotation_deg"] == 1.5
    assert summaries["L2"]["options"]["max_rotation_deg"] == 1.5
    assert (root / "images" / "L1_ladder_registration.png").is_file()
    assert not (root / "images" / "L2_ladder_registration.png").exists()
    assert summaries["L2"]["image_file"] is None
    assert summaries["L1"]["measurement_frame"] == "global"
    assert summaries["L2"]["geometry_file"].endswith("geometry.json")


def test_local_enable_overrides_globally_disabled_registration(tmp_path):
    config, _ = pipeline_config(tmp_path)
    ladders = config["ladder_analysis"]
    ladders["registration"] = {
        "enabled": False,
        "measurement_frame": "global",
        "draw": False,
    }
    ladders["L1"]["registration"] = {"enabled": True, "geometry_file": "geometry.json"}
    pipeline = AlignmentPipeline(config)
    pipeline.run()
    assert pipeline.registration_stage.results["L1"].accepted
    assert not (tmp_path / "output/images/L1_ladder_registration.png").exists()


def test_global_registration_disable_is_inherited(tmp_path):
    config, _ = pipeline_config(tmp_path)
    config["ladder_analysis"]["registration"] = {"enabled": False}
    pipeline = AlignmentPipeline(config)
    pipeline.run()
    assert not pipeline.registration_stage.results


@pytest.mark.parametrize("segmented", [False, True])
def test_initial_preview_never_fits_and_preserves_existing_products(
    tmp_path, monkeypatch, segmented
):
    config, points = pipeline_config(tmp_path, segmented=segmented)
    registration = config["ladder_analysis"]["L1"]["registration"]
    initial = np.eye(4)
    initial[:3, 3] = [3, -4, 2]
    registration.update(enabled=False, draw=False, initial_transform=initial.tolist())
    pipeline = AlignmentPipeline(config)
    original_config = deepcopy(config)
    matrices = tmp_path / "output/matrices"
    matrices.mkdir(parents=True)
    scanner_to_detector = np.eye(4)
    scanner_to_detector[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    scanner_to_detector[:3, 3] = [-2, 1, -10]
    np.save(matrices / "transform_matrix.npy", scanner_to_detector)
    protected = [
        tmp_path / "output/results.json",
        matrices / "L1_nominal_to_as_built.npy",
        tmp_path / "output/images/L1_ladder_registration.png",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing fit product: do not modify")
    protected.append(matrices / "transform_matrix.npy")
    before = {path: path.read_bytes() for path in protected}

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Preview must not invoke alignment, plane fitting or registration"
        )

    monkeypatch.setattr(AlignmentPipeline, "run", forbidden)
    monkeypatch.setattr(pipeline.plane_fitter, "fit", forbidden)
    monkeypatch.setattr(pipeline.sensor_plane_fitter, "fit", forbidden)
    monkeypatch.setattr(LadderRegistrar, "fit", forbidden)
    report = RegistrationPreview(pipeline).run(dpi=50, sample_size=3)
    assert report["registration_performed"] is False
    assert config == original_config
    summary = report["ladders"]["L1"]
    assert "accepted" not in summary
    assert summary["total_points"] == 9
    assert summary["displayed_points"] == 3
    expected = (points + [1, 2, 10]) @ scanner_to_detector[
        :3, :3
    ].T + scanner_to_detector[:3, 3]
    np.testing.assert_allclose(summary["measured_bounds_mm"]["min"], expected.min(0))
    np.testing.assert_allclose(summary["measured_bounds_mm"]["max"], expected.max(0))
    np.testing.assert_allclose(
        summary["initial_geometry_bounds_mm"]["min"], [-7, -19, 1.85]
    )
    np.testing.assert_allclose(
        summary["initial_geometry_bounds_mm"]["max"], [13, 11, 2.15]
    )
    assert summary["initial_nominal_to_as_built_matrix"] == initial.tolist()
    assert len(summary["input_files"]) == (2 if segmented else 1)
    assert Path(summary["image_file"]).stat().st_size > 1000
    saved_report = json.loads(
        (
            tmp_path / "output/images/registration_initial/preview_summary.json"
        ).read_text()
    )
    assert saved_report == report
    assert all(path.read_bytes() == content for path, content in before.items())


def test_initial_preview_requires_a_saved_alignment(tmp_path, monkeypatch):
    config, _ = pipeline_config(tmp_path)
    pipeline = AlignmentPipeline(config)
    monkeypatch.setattr(
        LadderRegistrar, "fit", lambda *args, **kwargs: pytest.fail("Unexpected fit")
    )
    with pytest.raises(
        FileNotFoundError, match="Saved scanner-to-detector matrix not found"
    ):
        RegistrationPreview(pipeline).run()
    assert not (tmp_path / "output").exists()


def test_initial_preview_rejects_nonrigid_cached_alignment(tmp_path):
    config, _ = pipeline_config(tmp_path)
    matrix = np.eye(4)
    matrix[0, 0] = 2
    np.save(tmp_path / "bad.npy", matrix)
    with pytest.raises(ValueError, match="proper orthonormal"):
        RegistrationPreview(AlignmentPipeline(config)).run(alignment_matrix="bad.npy")
    assert not (tmp_path / "output").exists()


def test_initial_preview_ladder_selection_and_path_overrides(tmp_path):
    config, _ = pipeline_config(tmp_path)
    config["ladder_analysis"]["L2"] = deepcopy(config["ladder_analysis"]["L1"])
    np.save(tmp_path / "detector.npy", np.eye(4))
    preview = RegistrationPreview(AlignmentPipeline(config))
    report = preview.run(
        alignment_matrix="detector.npy",
        output_dir="previews",
        ladder_names=["L2"],
        dpi=50,
        sample_size=2,
    )
    assert set(report["ladders"]) == {"L2"}
    assert (tmp_path / "previews/L2_initial_registration.png").exists()
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="Unknown preview ladder"):
        preview.run(alignment_matrix="detector.npy", ladder_names=["unknown"])


def test_registration_refuses_mixed_uncertainty_availability(tmp_path):
    config, _ = pipeline_config(tmp_path, segmented=True)
    config["ladder_analysis"]["L1"]["sensors"][0]["uncertainty"] = {"mode": "none"}
    with pytest.raises(ValueError, match="covariance for all input files or none"):
        AlignmentPipeline(config).run()
