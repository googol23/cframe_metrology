import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from detector_alignment.pipeline import AlignmentPipeline


def _write_synthetic_case(
    root: Path,
    *,
    obsolete_radius: bool = False,
    fiducial_draw: bool | None = None,
) -> Path:
    data = root / "data"
    data.mkdir()

    # Exact z=10 plane with a non-degenerate XY footprint.
    pts = []
    for x in np.linspace(-5.0, 5.0, 7):
        for y in np.linspace(-4.0, 4.0, 6):
            pts.append((x, y, 10.0))
    np.savetxt(data / "cframe.xyz", np.asarray(pts))
    np.savetxt(data / "ref1.xyz", np.asarray([[1.0, 2.0, 10.0]]))
    np.savetxt(data / "ref2.xyz", np.asarray([[1.0, 7.0, 10.0]]))

    if fiducial_draw is not None:
        (data / "cframe_fiducials.xyz").write_text(
            "1,2,10\n1,2,10\n3\t4\t10\n",
            encoding="utf-8",
        )

    refs = [
        {"file": "data/ref1.xyz", "nominal_center": [0.0, 0.0]},
        {"file": "data/ref2.xyz", "nominal_center": [0.0, 5.0]},
    ]
    if obsolete_radius:
        refs[0]["radius_mm"] = 1.5

    cfg = {
        "output_dir": "output",
        "uncertainty": {"mode": "constant", "sigma_xyz_mm": [0.01, 0.01, 0.01]},
        "cframe": {"file": "data/cframe.xyz"},
        "plane_fit": {
            "ransac_iterations": 30,
            "ransac_threshold_mm": 0.01,
            "random_seed": 1,
        },
        "references": refs,
        "visualization": {"dpi": 72, "sample_size": 1000, "random_seed": 1},
    }
    if fiducial_draw is not None:
        cfg["fiducials"] = {
            "file": "data/cframe_fiducials.xyz",
            "draw": fiducial_draw,
            "marker": "x",
        }

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pipeline_direct_reference_centers_end_to_end(tmp_path: Path):
    config = _write_synthetic_case(tmp_path)
    result = AlignmentPipeline.from_yaml(config).run()

    np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(result.translation, [-1.0, -2.0, -10.0], atol=1e-8)
    assert len(result.references) == 2

    payload = json.loads((tmp_path / "output" / "results.json").read_text(encoding="utf-8"))
    assert len(payload["references"]) == 2
    assert "radius_mm" not in payload["references"][0]
    assert (
        tmp_path / "output" / "point_clouds" / "ref1_transformed.xyz"
    ).exists()


@pytest.mark.parametrize("draw", [True, False])
def test_pipeline_loads_transforms_and_optionally_draws_fiducials(
    tmp_path: Path,
    draw: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _write_synthetic_case(tmp_path, fiducial_draw=draw)
    pipeline = AlignmentPipeline.from_yaml(config)
    plot_calls = []
    original_plot = pipeline.visualizer.plot_plane_alignment

    def record_plot(*args, **kwargs):
        plot_calls.append(kwargs)
        if draw:
            original_plot(*args, **kwargs)

    monkeypatch.setattr(pipeline.visualizer, "plot_plane_alignment", record_plot)
    pipeline.run()

    assert len(plot_calls) == 2
    for call in plot_calls:
        assert call["fiducial_marker"] == "x"
        if draw:
            np.testing.assert_allclose(
                call["fiducial_points"],
                [[1, 2, 10], [1, 2, 10], [3, 4, 10]],
            )
        else:
            assert call["fiducial_points"] is None

    transformed = np.loadtxt(
        tmp_path / "output" / "point_clouds" / "cframe_fiducials_transformed.xyz"
    )
    np.testing.assert_allclose(
        transformed,
        [[0, 0, 0], [0, 0, 0], [2, 2, 0]],
        atol=1e-8,
    )


def test_pipeline_processes_named_ladders_with_mixed_segmentation_modes(
    tmp_path: Path,
):
    config_path = _write_synthetic_case(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = tmp_path / "data"

    detector_bottom = np.asarray(
        [
            [-2.0, -3.0, 1.0],
            [0.0, -3.0, 1.0],
            [-2.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    detector_top = np.asarray(
        [
            [-2.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [-2.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
        ]
    )
    scanner_offset = np.asarray([1.0, 2.0, 10.0])
    np.savetxt(
        data_dir / "L3DL0.xyz",
        np.vstack([detector_bottom, detector_top]) + scanner_offset,
    )
    np.savetxt(
        data_dir / "L1T0.xyz",
        detector_bottom + scanner_offset,
    )

    config["ladder_analysis"] = {
        "enabled": True,
        "L3DL0": {
            "segmentation": {
                "mode": "automatic",
                "config": {
                    "z_gap_mm": 0.3,
                    "min_point_per_sensor": 3,
                },
            },
            "file": "data/L3DL0.xyz",
        },
        "L3DL1": {
            "segmentation": {"mode": "segmented_files"},
            "sensors": [
                {"name": "L1T0", "file": "data/L1T0.xyz"},
            ],
        },
        "visualization": {"dpi": 72, "sample_size": 1000},
        "write_individual_plots": True,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    pipeline = AlignmentPipeline.from_yaml(config_path)
    pipeline.run()

    images_dir = tmp_path / "output" / "images"
    assert (images_dir / "L3DL0_ladder_sensor_plane_residuals.png").exists()
    assert (images_dir / "L3DL0_sensor_S0_plane_residuals.png").exists()
    assert (images_dir / "L3DL0_sensor_S1_plane_residuals.png").exists()
    assert (images_dir / "L3DL1_ladder_sensor_plane_residuals.png").exists()
    assert (images_dir / "L3DL1_sensor_L1T0_plane_residuals.png").exists()
    assert not (images_dir / "ladder_sensor_plane_residuals.png").exists()

    payload = json.loads(
        (tmp_path / "output" / "results.json").read_text(encoding="utf-8")
    )
    assert list(payload["ladders"]) == ["L3DL0", "L3DL1"]
    assert payload["ladders"]["L3DL0"]["segmentation_mode"] == "automatic"
    assert payload["ladders"]["L3DL0"]["sensor_count"] == 2
    assert (
        payload["ladders"]["L3DL1"]["segmentation_mode"]
        == "segmented_files"
    )
    assert payload["ladders"]["L3DL1"]["sensor_count"] == 1
    assert list(pipeline.ladder_processors) == ["L3DL0", "L3DL1"]

    point_clouds_dir = tmp_path / "output" / "point_clouds"
    assert (point_clouds_dir / "L3DL0_transformed.xyz").exists()
    assert (point_clouds_dir / "L3DL1_L1T0_transformed.xyz").exists()


def test_legacy_single_ladder_configuration_is_rejected():
    with pytest.raises(ValueError, match="named ladder mappings"):
        AlignmentPipeline(
            {
                "ladder_analysis": {
                    "enabled": True,
                    "segmentation": {"mode": "automatic"},
                    "file": "data/L3DL0.xyz",
                }
            }
        )


def test_obsolete_hole_radius_is_rejected(tmp_path: Path):
    config = _write_synthetic_case(tmp_path, obsolete_radius=True)
    with pytest.raises(ValueError, match="obsolete 'radius_mm'"):
        AlignmentPipeline.from_yaml(config).run()
