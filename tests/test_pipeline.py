import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from matplotlib.axes import Axes

import detector_alignment.pipeline as pipeline_module
from detector_alignment.ladder import SensorPlaneResult
from detector_alignment.ladder_viz import LadderVisualizer
from detector_alignment.models import PlaneFitResult
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
            "1,2,10\n1.02,1.98,10.01\n3\t4\t10\n",
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
            "fiducial_min_distance": 0.5,
            "draw": fiducial_draw,
            "marker": "x",
        }

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pipeline_direct_reference_centers_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
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

    console_log = capsys.readouterr().err
    expected_events = [
        "Pipeline | status=started",
        "Pipeline | task=prepare_outputs",
        "Pipeline | task=fit_cframe_plane",
        "Pipeline | task=process_references | count=2",
        "Pipeline | task=write_alignment_diagnostic | stage=plane",
        "Pipeline | task=compute_detector_transform | references=2",
        "Pipeline | task=write_alignment_diagnostic | stage=final",
        (
            "Pipeline | task=analyze_ladders | action=skip | "
            "reason=not_configured"
        ),
        "Pipeline | task=write_numerical_results",
        "Pipeline | task=write_transformed_clouds",
        (
            "Pipeline | status=completed | references=2 | ladders=0 | "
            "sensors=0"
        ),
    ]
    positions = [console_log.index(event) for event in expected_events]
    assert positions == sorted(positions)
    assert "Output root:" not in console_log

    file_log = (
        tmp_path / "output" / "logs" / "pipeline.log"
    ).read_text(encoding="utf-8")
    for event in expected_events:
        assert event in file_log
    assert "DEBUG | Output root:" in file_log


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
    assert [call["residual_y_legend_location"] for call in plot_calls] == [
        "lower left",
        "upper left",
    ]
    for call in plot_calls:
        assert call["fiducial_marker"] == "x"
        if draw:
            np.testing.assert_allclose(
                call["fiducial_points"],
                [[1.01, 1.99, 10.005], [3, 4, 10]],
            )
        else:
            assert call["fiducial_points"] is None

    transformed = np.loadtxt(
        tmp_path / "output" / "point_clouds" / "cframe_fiducials_transformed.xyz"
    )
    np.testing.assert_allclose(
        transformed,
        [[0, 0, 0], [0.02, -0.02, 0.01], [2, 2, 0]],
        atol=1e-8,
    )

    averaged = np.loadtxt(
        tmp_path
        / "output"
        / "point_clouds"
        / "cframe_fiducials_averaged_transformed.xyz"
    )
    np.testing.assert_allclose(
        averaged,
        [[0.01, -0.01, 0.005], [2, 2, 0]],
        atol=1e-8,
    )

    payload = json.loads(
        (tmp_path / "output" / "results.json").read_text(encoding="utf-8")
    )
    assert payload["fiducials"]["marker_count"] == 2
    file_log = (
        tmp_path / "output" / "logs" / "pipeline.log"
    ).read_text(encoding="utf-8")
    assert "Pipeline | task=process_fiducials" in file_log
    assert "min_distance_mm=0.5" in file_log
    first_marker = payload["fiducials"]["markers"][0]
    assert first_marker["measurement_indices"] == [0, 1]
    np.testing.assert_allclose(
        first_marker["scanner"]["mean_xyz_mm"],
        [1.01, 1.99, 10.005],
    )
    np.testing.assert_allclose(
        first_marker["scanner"]["mean_uncertainty_stddev_xyz_mm"],
        np.full(3, 0.01 / np.sqrt(2.0)),
    )
    np.testing.assert_allclose(
        first_marker["detector"]["mean_xyz_mm"],
        [0.01, -0.01, 0.005],
        atol=1e-8,
    )


def test_cframe_diagnostics_use_fixed_legend_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _write_synthetic_case(tmp_path)
    pipeline = AlignmentPipeline.from_yaml(config)
    legend_locations = []
    original_legend = Axes.legend

    def record_legend(axis, *args, **kwargs):
        legend_locations.append(kwargs.get("loc"))
        return original_legend(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", record_legend)
    pipeline.run()

    assert legend_locations == [
        "upper right",
        "center right",
        "center right",
        "upper right",
        "lower left",
        "upper right",
        "center right",
        "center right",
        "upper right",
        "upper left",
    ]


def test_fiducial_min_distance_is_required():
    pipeline = AlignmentPipeline(
        {"fiducials": {"file": "data/cframe_fiducials.xyz"}}
    )
    with pytest.raises(ValueError, match="fiducial_min_distance is required"):
        pipeline._fiducials_config()


@pytest.mark.parametrize(
    ("draw_config", "error_match"),
    [
        (False, "visualization.draw.*mapping"),
        ({"cframe": "yes"}, "visualization.draw.cframe.*true or false"),
        ({"unknown": True}, "Unknown visualization.draw stage"),
    ],
)
def test_visualization_draw_configuration_is_validated(
    draw_config,
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        AlignmentPipeline({"visualization": {"draw": draw_config}})


@pytest.mark.parametrize(
    ("configured_workers", "available_cpus", "expected_workers"),
    [
        (None, 2, 2),
        (8, 2, 2),
        (1, 8, 1),
        (4, None, 1),
    ],
)
def test_individual_plot_workers_are_capped_by_available_cpus(
    monkeypatch: pytest.MonkeyPatch,
    configured_workers: int | None,
    available_cpus: int | None,
    expected_workers: int,
):
    monkeypatch.setattr(
        pipeline_module.os,
        "cpu_count",
        lambda: available_cpus,
    )
    ladder_visualization = {}
    if configured_workers is not None:
        ladder_visualization["individual_plot_workers"] = configured_workers

    pipeline = AlignmentPipeline(
        {"ladder_analysis": {"visualization": ladder_visualization}}
    )

    assert pipeline.individual_plot_workers == expected_workers


@pytest.mark.parametrize("invalid_workers", [True, 0, -1, 1.5, "4"])
def test_individual_plot_workers_must_be_a_positive_integer(invalid_workers):
    with pytest.raises(ValueError, match="individual_plot_workers.*positive integer"):
        AlignmentPipeline(
            {
                "ladder_analysis": {
                    "visualization": {
                        "individual_plot_workers": invalid_workers,
                    }
                }
            }
        )


def test_parallel_sensor_plots_match_sequential_pngs(tmp_path: Path):
    x = np.linspace(-2.0, 2.0, 8)
    y = np.linspace(-1.0, 1.0, 6)
    xy = np.asarray([(x_value, y_value) for x_value in x for y_value in y])

    def sensor_result(name: str, y_offset: float) -> SensorPlaneResult:
        residuals = np.linspace(-0.02, 0.02, len(xy))
        points = np.column_stack(
            [xy[:, 0], xy[:, 1] + y_offset, residuals]
        )
        plane = PlaneFitResult(
            normal=np.asarray([0.0, 0.0, 1.0]),
            d=0.0,
            centroid=np.mean(points, axis=0),
            inlier_mask=np.ones(len(points), dtype=bool),
            rms_mm=float(np.sqrt(np.mean(residuals**2))),
            params=np.zeros(3),
            covariance_params=np.eye(3),
            rotation_to_xy=np.eye(3),
        )
        return SensorPlaneResult(name=name, points=points, plane=plane)

    results = [sensor_result("S0", 0.0), sensor_result("S1", 3.0)]
    sequential_dir = tmp_path / "sequential"
    parallel_dir = tmp_path / "parallel"

    sequential_visualizer = LadderVisualizer(
        dpi=72,
        sample_size=20,
        random_seed=23,
    )
    sequential_paths = sequential_visualizer.plot_individual_residuals(
        results,
        sequential_dir,
        filename_prefix="L0_",
    )

    parallel_visualizer = LadderVisualizer(
        dpi=72,
        sample_size=20,
        random_seed=23,
    )
    prepared = parallel_visualizer.prepare_individual_residuals(
        results,
        parallel_dir,
        filename_prefix="L0_",
    )
    pipeline = AlignmentPipeline(
        {
            "ladder_analysis": {
                "visualization": {"individual_plot_workers": 2}
            }
        }
    )
    pipeline.ladder_visualizer = parallel_visualizer
    parallel_paths = pipeline._render_prepared_individual_plots(
        prepared,
        point_size=2.0,
    )

    assert [path.name for path in sequential_paths] == [
        path.name for path in parallel_paths
    ]
    sequential_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sequential_paths
    ]
    parallel_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in parallel_paths
    ]
    assert sequential_hashes == parallel_hashes


@pytest.mark.parametrize(
    ("draw_sensors", "legacy_write_individual", "expected"),
    [
        (None, False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_sensor_draw_stage_supports_legacy_configuration(
    draw_sensors: bool | None,
    legacy_write_individual: bool,
    expected: bool,
):
    visualization = {}
    if draw_sensors is not None:
        visualization["draw"] = {"sensors": draw_sensors}

    pipeline = AlignmentPipeline(
        {
            "visualization": visualization,
            "ladder_analysis": {
                "write_individual_plots": legacy_write_individual,
            },
        }
    )

    assert pipeline.visualization_draw["sensors"] is expected


def test_pipeline_processes_named_ladders_with_mixed_segmentation_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        "visualization": {
            "dpi": 72,
            "sample_size": 1000,
            "individual_plot_workers": 2,
        },
        "write_individual_plots": True,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    pipeline = AlignmentPipeline.from_yaml(config_path)
    overview_calls = []
    overview_text_calls = []
    original_axes_text = Axes.text
    original_overview_plot = (
        pipeline.ladder_visualizer.plot_cframe_and_ladder_residuals
    )

    def record_axes_text(ax, *args, **kwargs):
        overview_text_calls.append((args, kwargs))
        return original_axes_text(ax, *args, **kwargs)

    def record_overview_plot(
        cframe_points,
        cframe_residuals_mm,
        cframe_inlier_mask,
        ladder_results,
        output_path,
        **kwargs,
    ):
        overview_calls.append(
            (
                cframe_points.copy(),
                cframe_residuals_mm.copy(),
                cframe_inlier_mask.copy(),
                ladder_results,
                output_path,
            )
        )
        with monkeypatch.context() as overview_monkeypatch:
            overview_monkeypatch.setattr(Axes, "text", record_axes_text)
            return original_overview_plot(
                cframe_points,
                cframe_residuals_mm,
                cframe_inlier_mask,
                ladder_results,
                output_path,
                **kwargs,
            )

    monkeypatch.setattr(
        pipeline.ladder_visualizer,
        "plot_cframe_and_ladder_residuals",
        record_overview_plot,
    )
    alignment = pipeline.run()

    assert len(overview_calls) == 1
    assert overview_text_calls == []
    (
        plotted_cframe,
        plotted_cframe_residuals,
        plotted_cframe_inliers,
        plotted_ladders,
        overview_path,
    ) = overview_calls[0]
    scanner_cframe = np.loadtxt(data_dir / "cframe.xyz")
    expected_cframe = scanner_cframe - scanner_offset
    expected_cframe_residuals = (
        scanner_cframe @ alignment.plane.normal + alignment.plane.d
    )
    np.testing.assert_allclose(plotted_cframe, expected_cframe, atol=1e-8)
    np.testing.assert_allclose(
        plotted_cframe_residuals,
        expected_cframe_residuals,
        atol=1e-8,
    )
    np.testing.assert_array_equal(
        plotted_cframe_inliers,
        alignment.plane.inlier_mask,
    )
    assert list(plotted_ladders) == ["L3DL0", "L3DL1"]

    images_dir = tmp_path / "output" / "images"
    assert overview_path == (
        images_dir / "cframe_all_ladders_sensor_plane_residuals.png"
    )
    assert overview_path.exists()
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

    file_log = (
        tmp_path / "output" / "logs" / "pipeline.log"
    ).read_text(encoding="utf-8")
    assert "Pipeline | task=analyze_ladders | count=2" in file_log
    assert (
        "Pipeline | task=analyze_ladder | ladder=L3DL0 | mode=automatic"
        in file_log
    )
    assert (
        "Pipeline | task=analyze_ladder | ladder=L3DL1 | "
        "mode=segmented_files"
        in file_log
    )
    assert "Pipeline | task=write_ladder_overview | ladders=2" in file_log
    assert "DEBUG | Sensor S0 plane fit:" in file_log

    point_clouds_dir = tmp_path / "output" / "point_clouds"
    assert (point_clouds_dir / "L3DL0_transformed.xyz").exists()
    assert (point_clouds_dir / "L3DL1_L1T0_transformed.xyz").exists()


@pytest.mark.parametrize(
    ("disabled_stage", "expected_calls", "stale_images"),
    [
        (
            "cframe",
            {"cframe": 0, "sensors": 1, "full_ladders": 1, "overview": 1},
            [
                "cframe_plane_alignment.png",
                "cframe_plane_alignment_top_view.png",
                "cframe_final_alignment.png",
                "cframe_final_alignment_top_view.png",
            ],
        ),
        (
            "sensors",
            {"cframe": 2, "sensors": 0, "full_ladders": 1, "overview": 1},
            ["L3DL0_sensor_S0_plane_residuals.png"],
        ),
        (
            "full_ladders",
            {"cframe": 2, "sensors": 1, "full_ladders": 0, "overview": 1},
            ["L3DL0_ladder_sensor_plane_residuals.png"],
        ),
        (
            "all_ladders_and_cframe",
            {"cframe": 2, "sensors": 1, "full_ladders": 1, "overview": 0},
            ["cframe_all_ladders_sensor_plane_residuals.png"],
        ),
    ],
)
def test_pipeline_can_disable_each_visualization_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disabled_stage: str,
    expected_calls: dict[str, int],
    stale_images: list[str],
):
    config_path = _write_synthetic_case(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = tmp_path / "data"
    scanner_offset = np.asarray([1.0, 2.0, 10.0])
    detector_sensor = np.asarray(
        [
            [-2.0, -3.0, 1.0],
            [0.0, -3.0, 1.0],
            [-2.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    np.savetxt(data_dir / "S0.xyz", detector_sensor + scanner_offset)

    draw = {
        "cframe": True,
        "sensors": True,
        "full_ladders": True,
        "all_ladders_and_cframe": True,
    }
    draw[disabled_stage] = False
    config["visualization"]["draw"] = draw
    config["ladder_analysis"] = {
        "enabled": True,
        "L3DL0": {
            "segmentation": {"mode": "segmented_files"},
            "sensors": [{"name": "S0", "file": "data/S0.xyz"}],
        },
        "visualization": {
            "dpi": 72,
            "sample_size": 1000,
            "individual_plot_workers": 1,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    images_dir = tmp_path / "output" / "images"
    images_dir.mkdir(parents=True)
    for filename in stale_images:
        (images_dir / filename).write_text("stale", encoding="utf-8")

    pipeline = AlignmentPipeline.from_yaml(config_path)
    calls = {stage: 0 for stage in expected_calls}

    def record(stage):
        def recorder(*args, **kwargs):
            calls[stage] += 1

        return recorder

    monkeypatch.setattr(
        pipeline.visualizer,
        "plot_plane_alignment",
        record("cframe"),
    )
    monkeypatch.setattr(
        pipeline.ladder_visualizer,
        "plot_individual_residuals",
        record("sensors"),
    )
    monkeypatch.setattr(
        pipeline.ladder_visualizer,
        "plot_combined_residuals",
        record("full_ladders"),
    )
    monkeypatch.setattr(
        pipeline.ladder_visualizer,
        "plot_cframe_and_ladder_residuals",
        record("overview"),
    )

    pipeline.run()

    assert calls == expected_calls
    assert all(not (images_dir / filename).exists() for filename in stale_images)
    payload = json.loads(
        (tmp_path / "output" / "results.json").read_text(encoding="utf-8")
    )
    assert payload["ladders"]["L3DL0"]["sensor_count"] == 1


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
    pipeline = AlignmentPipeline.from_yaml(config)

    with pytest.raises(ValueError, match="obsolete 'radius_mm'"):
        pipeline.run()

    assert pipeline._log_handlers == []
    assert (tmp_path / "output" / "logs" / "pipeline.log").exists()
