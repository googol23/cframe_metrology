from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .fiducials import (
    FiducialMarkerResult,
    FiducialMeasurementResult,
    FiducialProcessor,
)
from .io import AsciiPointCloudLoader
from .ladder import LadderProcessor, SensorPlaneResult
from .ladder_viz import LadderVisualizer
from .math3d import homogeneous
from .models import AlignmentResult
from .plane import RobustPlaneFitter
from .reference import MeasuredReferenceProcessor, MeasuredReferenceResult
from .segmentation import segment_ladder
from .uncertainty import (
    build_joint_covariance,
    propagate_final_transform_covariance,
)
from .viz import AlignmentVisualizer


class AlignmentPipeline:
    """
    End-to-end detector alignment from a C-frame plane and measured centers.

    The C-frame analysis first determines the final scanner -> detector rigid
    transformation. That final transformation is then passed to LadderProcessor.

    LadderProcessor owns application of that transformation: every ladder/sensor
    file is transformed immediately after loading and before it is cached,
    segmented, fitted, or plotted.

    Output structure:

        output/
        ├── results.json
        ├── images/
        ├── logs/
        ├── matrices/
        └── point_clouds/

    Diagnostic images, numerical matrix products, and transformed point clouds
    are kept separate. The machine-readable results.json remains directly in
    the output root.
    """

    _LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    _VISUALIZATION_DRAW_STAGES = frozenset(
        {
            "cframe",
            "sensors",
            "full_ladders",
            "all_ladders_and_cframe",
        }
    )

    _LADDER_GLOBAL_KEYS = frozenset(
        {
            "enabled",
            "sensor_plane_fit",
            "visualization",
            "write_individual_plots",
        }
    )
    _LEGACY_SINGLE_LADDER_KEYS = frozenset(
        {
            "segmentation",
            "sensors",
            "file",
            "uncertainty",
        }
    )

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("config must be a mapping.")

        self.config = config
        self._logger = logging.getLogger(
            f"{__name__}.AlignmentPipeline.{id(self)}"
        )
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._log_handlers: list[logging.Handler] = []

        io_cfg = config.get("io", {})
        comment_prefixes = io_cfg.get("comment_prefixes", ["#", "//", "$"])
        self.loader = AsciiPointCloudLoader(
            comment_prefixes=comment_prefixes,
            delimiter=io_cfg.get("delimiter", None),
        )
        self.fiducial_loader = AsciiPointCloudLoader(
            comment_prefixes=comment_prefixes,
            delimiter="auto",
        )
        self.fiducial_processor = FiducialProcessor()

        p_cfg = config.get("plane_fit", {})
        self.plane_fitter = RobustPlaneFitter(
            ransac_iterations=p_cfg.get("ransac_iterations", 600),
            ransac_threshold_mm=p_cfg.get("ransac_threshold_mm", 0.05),
            random_seed=p_cfg.get("random_seed", 42),
        )

        r_cfg = config.get("reference_points", {})
        self.reference_processor = MeasuredReferenceProcessor(
            derivative_step_rad=r_cfg.get("derivative_step_rad", 1.0e-7),
        )

        v_cfg = config.get("visualization", {})
        if not isinstance(v_cfg, dict):
            raise ValueError("'visualization' must be a YAML mapping.")
        draw_cfg = v_cfg.get("draw", {})
        if not isinstance(draw_cfg, dict):
            raise ValueError("'visualization.draw' must be a YAML mapping.")
        unknown_draw_stages = sorted(
            set(draw_cfg).difference(self._VISUALIZATION_DRAW_STAGES)
        )
        if unknown_draw_stages:
            stages = ", ".join(repr(stage) for stage in unknown_draw_stages)
            raise ValueError(f"Unknown visualization.draw stage(s): {stages}.")
        for stage, enabled in draw_cfg.items():
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"'visualization.draw.{stage}' must be true or false."
                )
        self.visualization_draw = {
            stage: draw_cfg.get(stage, True)
            for stage in self._VISUALIZATION_DRAW_STAGES
        }

        self.visualizer = AlignmentVisualizer(
            dpi=v_cfg.get("dpi", 600),
            sample_size=v_cfg.get("sample_size", 50_000),
            random_seed=v_cfg.get("random_seed", 42),
            point_alpha=v_cfg.get("point_alpha", 0.25),
            plane_alpha=v_cfg.get("plane_alpha", 0.18),
            sigma_band_alpha=v_cfg.get("sigma_band_alpha", 0.22),
        )

        ladder_cfg = config.get("ladder_analysis", {})
        if ladder_cfg is None:
            ladder_cfg = {}
        if not isinstance(ladder_cfg, dict):
            raise ValueError("'ladder_analysis' must be a YAML mapping.")

        write_individual_plots = ladder_cfg.get("write_individual_plots", True)
        if not isinstance(write_individual_plots, bool):
            raise ValueError(
                "'ladder_analysis.write_individual_plots' must be true or false."
            )
        if "sensors" not in draw_cfg:
            self.visualization_draw["sensors"] = write_individual_plots

        sensor_fit_cfg = ladder_cfg.get("sensor_plane_fit", {})
        if not isinstance(sensor_fit_cfg, dict):
            raise ValueError("'ladder_analysis.sensor_plane_fit' must be a mapping.")
        self.sensor_plane_fitter = RobustPlaneFitter(
            ransac_iterations=sensor_fit_cfg.get(
                "ransac_iterations",
                p_cfg.get("ransac_iterations", 600),
            ),
            ransac_threshold_mm=sensor_fit_cfg.get(
                "ransac_threshold_mm",
                p_cfg.get("ransac_threshold_mm", 0.05),
            ),
            random_seed=sensor_fit_cfg.get(
                "random_seed",
                p_cfg.get("random_seed", 42),
            ),
        )

        # LadderProcessor instances are intentionally NOT constructed here. The
        # final C-frame transformation does not exist until run() reaches step 3.
        self.ladder_processors: dict[str, LadderProcessor] = {}

        ladder_viz_cfg = ladder_cfg.get("visualization", {})
        if not isinstance(ladder_viz_cfg, dict):
            raise ValueError("'ladder_analysis.visualization' must be a mapping.")
        self.ladder_visualizer = LadderVisualizer(
            dpi=ladder_viz_cfg.get("dpi", v_cfg.get("dpi", 600)),
            sample_size=ladder_viz_cfg.get(
                "sample_size",
                v_cfg.get("sample_size", 50_000),
            ),
            random_seed=ladder_viz_cfg.get(
                "random_seed",
                v_cfg.get("random_seed", 42),
            ),
        )

        self._ladder_configs()

    def _configure_logging(self, outdir: Path) -> Path:
        """Configure isolated console and file logging for one pipeline run."""
        self._close_logging()

        logs_dir = outdir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "pipeline.log"
        formatter = logging.Formatter(
            self._LOG_FORMAT,
            datefmt=self._LOG_DATE_FORMAT,
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            log_path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self._log_handlers = [console_handler, file_handler]
        for handler in self._log_handlers:
            self._logger.addHandler(handler)

        return log_path

    def _close_logging(self) -> None:
        """Flush, detach, and close handlers owned by this pipeline instance."""
        for handler in self._log_handlers:
            handler.flush()
            self._logger.removeHandler(handler)
            handler.close()
        self._log_handlers.clear()

    @staticmethod
    def _log_value(value: object) -> str:
        """Keep user-controlled values on one physical log line."""
        return str(value).replace("\r", "\\r").replace("\n", "\\n")

    def _log_task(self, task: str, **details: object) -> None:
        context = "".join(
            f" | {key}={self._log_value(value)}"
            for key, value in details.items()
        )
        self._logger.info("Pipeline | task=%s%s", task, context)

    def _remove_generated_image(self, path: Path) -> None:
        if path.exists():
            path.unlink()
            self._logger.debug("Removed disabled visualization output: %s", path)

    def _remove_cframe_diagnostics(self, images_dir: Path) -> None:
        for stem in ("cframe_plane_alignment", "cframe_final_alignment"):
            for suffix in ("", "_top_view", "_residual_histogram"):
                self._remove_generated_image(images_dir / f"{stem}{suffix}.png")

    def _remove_disabled_ladder_diagnostics(self, images_dir: Path) -> None:
        if not self.visualization_draw["full_ladders"]:
            for path in images_dir.glob("*_ladder_sensor_plane_residuals.png"):
                self._remove_generated_image(path)

        if not self.visualization_draw["sensors"]:
            for path in images_dir.glob("*_sensor_*_plane_residuals.png"):
                if not path.name.endswith("_ladder_sensor_plane_residuals.png"):
                    self._remove_generated_image(path)

    def _ladder_configs(self) -> list[tuple[str, dict]]:
        """Return validated ``(ladder_name, config)`` pairs in YAML order."""
        cfg = self.config.get("ladder_analysis")
        if cfg is None:
            return []
        if not isinstance(cfg, dict):
            raise ValueError("'ladder_analysis' must be a YAML mapping.")

        enabled = cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("'ladder_analysis.enabled' must be true or false.")
        if not enabled:
            return []

        legacy_keys = sorted(self._LEGACY_SINGLE_LADDER_KEYS.intersection(cfg))
        if legacy_keys:
            keys = ", ".join(repr(key) for key in legacy_keys)
            raise ValueError(
                "ladder_analysis must contain named ladder mappings. Move the "
                f"legacy top-level key(s) {keys} under a ladder name."
            )

        ladder_configs: list[tuple[str, dict]] = []
        output_prefixes: dict[str, str] = {}

        for raw_name, ladder_cfg in cfg.items():
            if raw_name in self._LADDER_GLOBAL_KEYS:
                continue
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Every ladder name must be a non-empty string.")
            if raw_name != raw_name.strip():
                raise ValueError(
                    f"Ladder name {raw_name!r} must not have leading or trailing "
                    "whitespace."
                )
            if not isinstance(ladder_cfg, dict):
                raise ValueError(
                    f"ladder_analysis.{raw_name} must be a YAML mapping."
                )

            segmentation = ladder_cfg.get("segmentation")
            if not isinstance(segmentation, dict):
                raise ValueError(
                    f"ladder_analysis.{raw_name}.segmentation must be a mapping."
                )
            mode = segmentation.get("mode")
            if mode not in {"segmented_files", "automatic"}:
                raise ValueError(
                    f"Unknown segmentation mode {mode!r} for ladder {raw_name!r}. "
                    "Expected 'segmented_files' or 'automatic'."
                )
            if mode == "segmented_files":
                sensors = ladder_cfg.get("sensors")
                if not isinstance(sensors, list) or not sensors:
                    raise ValueError(
                        f"ladder_analysis.{raw_name}.sensors must be a non-empty "
                        "list in segmented_files mode."
                    )
            else:
                if not ladder_cfg.get("file"):
                    raise ValueError(
                        f"ladder_analysis.{raw_name}.file is required in automatic mode."
                    )
                segmentation_options = segmentation.get("config", {})
                if not isinstance(segmentation_options, dict):
                    raise ValueError(
                        f"ladder_analysis.{raw_name}.segmentation.config must be "
                        "a mapping."
                    )

            safe_prefix = LadderVisualizer._safe_name(raw_name)
            collision_key = safe_prefix.casefold()
            previous_name = output_prefixes.get(collision_key)
            if previous_name is not None:
                raise ValueError(
                    f"Ladder names {previous_name!r} and {raw_name!r} produce "
                    f"colliding output prefix {safe_prefix!r}."
                )
            output_prefixes[collision_key] = raw_name
            ladder_configs.append((raw_name, ladder_cfg))

        if not ladder_configs:
            raise ValueError(
                "Enabled ladder_analysis requires at least one named ladder mapping."
            )

        return ladder_configs

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlignmentPipeline":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError("Configuration root must be a YAML mapping.")

        cfg["_config_dir"] = str(path.parent.resolve())
        cfg["_config_path"] = str(path.resolve())
        return cls(cfg)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(self.config.get("_config_dir", ".")) / path

    def _fiducials_config(self) -> dict | None:
        cfg = self.config.get("fiducials")
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            raise ValueError("fiducials must be a mapping.")
        if "file" not in cfg:
            raise ValueError("fiducials.file is required when fiducials are configured.")
        if "fiducial_min_distance" not in cfg:
            raise ValueError(
                "fiducials.fiducial_min_distance is required when fiducials "
                "are configured."
            )
        distance = cfg["fiducial_min_distance"]
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not np.isfinite(distance)
            or distance <= 0.0
        ):
            raise ValueError(
                "fiducials.fiducial_min_distance must be a finite positive number."
            )
        if "uncertainty" in cfg and not isinstance(cfg["uncertainty"], dict):
            raise ValueError("fiducials.uncertainty must be a mapping.")
        if "draw" in cfg and not isinstance(cfg["draw"], bool):
            raise ValueError("fiducials.draw must be true or false.")
        marker = cfg.get("marker", "x")
        if not isinstance(marker, str) or not marker:
            raise ValueError("fiducials.marker must be a non-empty marker string.")
        return cfg

    def _reference_config(self) -> list[dict]:
        has_references = "references" in self.config
        has_reference_holes = "reference_holes" in self.config

        if has_references and has_reference_holes:
            raise ValueError("Use either 'references' or 'reference_holes', not both.")

        refs = self.config.get(
            "references",
            self.config.get("reference_holes", []),
        )
        if not isinstance(refs, list) or len(refs) < 2:
            raise ValueError("At least two measured references are required.")
        if not all(isinstance(ref, dict) for ref in refs):
            raise ValueError("Every reference entry must be a mapping.")

        return refs

    def _process_reference(
        self,
        cfg: dict,
        plane,
        uncertainty_default: dict,
        index: int,
    ) -> MeasuredReferenceResult:
        if "file" not in cfg or "nominal_center" not in cfg:
            raise ValueError(
                f"Reference {index} requires both 'file' and 'nominal_center'."
            )

        obsolete = {"radius_mm", "type"}.intersection(cfg)
        if "radius_mm" in obsolete:
            raise ValueError(
                f"Reference {index} contains obsolete 'radius_mm'. "
                "Hole-opening fitting has been removed; provide a file "
                "containing one measured XYZ center."
            )
        if (
            "type" in obsolete
            and str(cfg["type"]).lower().replace("-", "_")
            not in {"measured_point", "direct_point", "point"}
        ):
            raise ValueError(
                f"Reference {index} has unsupported type {cfg['type']!r}; "
                "only directly measured center points are supported."
            )

        nominal = np.asarray(cfg["nominal_center"], dtype=float)
        if nominal.shape != (2,) or not np.all(np.isfinite(nominal)):
            raise ValueError(
                f"Reference {index} nominal_center must contain two finite "
                "values [x, y]."
            )

        path = self._resolve(cfg["file"])
        data = self.loader.load(
            path,
            cfg.get("uncertainty", uncertainty_default),
        )

        if len(data.points) != 1:
            raise ValueError(
                f"Reference {index} file {data.source} must contain exactly "
                f"one measured XYZ center point; found {len(data.points)} numeric rows."
            )

        point_covariance = (
            None if data.covariances is None else data.covariances[0]
        )

        return self.reference_processor.process(
            point_xyz=data.points[0],
            point_covariance_xyz=point_covariance,
            plane_params=plane.params,
            nominal_center_xy=nominal,
            source=data.source,
        )

    @staticmethod
    def _validate_reference_geometry(
        measured: np.ndarray,
        nominal: np.ndarray,
    ) -> None:
        if (
            measured.shape != nominal.shape
            or measured.ndim != 2
            or measured.shape[1] != 2
        ):
            raise ValueError(
                "Measured and nominal references must both have shape (N, 2)."
            )
        if len(measured) < 2:
            raise ValueError("At least two measured references are required.")
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(nominal)):
            raise ValueError("Reference coordinates must be finite.")
        if np.max(np.linalg.norm(measured - measured[0], axis=1)) <= 1.0e-12:
            raise ValueError(
                "Measured reference centers are degenerate (not distinct)."
            )
        if np.max(np.linalg.norm(nominal - nominal[0], axis=1)) <= 1.0e-12:
            raise ValueError(
                "Nominal reference centers are degenerate (not distinct)."
            )

    def run(self) -> AlignmentResult:
        outdir = self._resolve(self.config.get("output_dir", "output"))
        log_path = self._configure_logging(outdir)
        config_source = self.config.get("_config_path", "<in-memory>")
        self._logger.info(
            "Pipeline | status=started | config=%s | output_dir=%s | log_file=%s",
            self._log_value(config_source),
            self._log_value(outdir),
            self._log_value(log_path),
        )
        try:
            return self._run(outdir)
        finally:
            self._close_logging()

    def _run(self, outdir: Path) -> AlignmentResult:
        # ------------------------------------------------------------------
        # Output structure
        #
        # output/
        # ├── results.json
        # ├── images/
        # ├── logs/
        # ├── matrices/
        # └── point_clouds/
        # ------------------------------------------------------------------
        self._log_task("prepare_outputs", output_dir=outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        images_dir = outdir / "images"
        matrices_dir = outdir / "matrices"
        point_clouds_dir = outdir / "point_clouds"

        images_dir.mkdir(parents=True, exist_ok=True)
        matrices_dir.mkdir(parents=True, exist_ok=True)
        point_clouds_dir.mkdir(parents=True, exist_ok=True)

        if not self.visualization_draw["cframe"]:
            self._remove_cframe_diagnostics(images_dir)
        self._remove_disabled_ladder_diagnostics(images_dir)
        if not self.visualization_draw["all_ladders_and_cframe"]:
            self._remove_generated_image(
                images_dir / "cframe_all_ladders_sensor_plane_residuals.png"
            )

        self._logger.debug("Output root: %s", outdir)
        self._logger.debug("Image output: %s", images_dir)
        self._logger.debug("Matrix output: %s", matrices_dir)
        self._logger.debug("Point-cloud output: %s", point_clouds_dir)

        uncertainty_default = self.config.get(
            "uncertainty",
            {"mode": "none"},
        )

        if "hole_fit" in self.config:
            raise ValueError(
                "The 'hole_fit' configuration block is obsolete. "
                "Reference files must contain directly measured XYZ centers."
            )

        cframe_cfg = self.config.get("cframe")
        if not isinstance(cframe_cfg, dict) or "file" not in cframe_cfg:
            raise ValueError("Configuration requires cframe.file.")

        # C-frame plane
        cframe_path = self._resolve(cframe_cfg["file"])
        self._log_task("fit_cframe_plane", source=cframe_path)
        cframe = self.loader.load(
            cframe_path,
            cframe_cfg.get("uncertainty", uncertainty_default),
        )
        plane = self.plane_fitter.fit(
            cframe.points,
            cframe.covariances,
        )

        fiducials_cfg = self._fiducials_config()
        fiducials = None
        transformed_fiducials = None
        fiducial_points = None
        fiducial_marker = "x"
        if fiducials_cfg is not None:
            fiducials_path = self._resolve(fiducials_cfg["file"])
            self._log_task(
                "process_fiducials",
                source=fiducials_path,
                min_distance_mm=fiducials_cfg["fiducial_min_distance"],
            )
            fiducial_data = self.fiducial_loader.load(
                fiducials_path,
                fiducials_cfg.get("uncertainty", uncertainty_default),
            )
            fiducials = self.fiducial_processor.process(
                fiducial_data,
                fiducials_cfg["fiducial_min_distance"],
            )
            fiducial_marker = fiducials_cfg.get("marker", "x")
            if fiducials_cfg.get("draw", True):
                fiducial_points = fiducials.means

        # Measured references
        reference_cfg = self._reference_config()
        self._log_task("process_references", count=len(reference_cfg))
        references = [
            self._process_reference(
                cfg,
                plane,
                uncertainty_default,
                i,
            )
            for i, cfg in enumerate(reference_cfg, start=1)
        ]

        scanner_reference_points = np.vstack(
            [ref.measured_xyz for ref in references]
        )

        if self.visualization_draw["cframe"]:
            cframe_plane_plot = images_dir / "cframe_plane_alignment.png"
            self._log_task(
                "write_alignment_diagnostic",
                stage="plane",
                destination=cframe_plane_plot,
            )
            self.visualizer.plot_plane_alignment(
                cframe.points,
                plane.rotation_to_xy,
                plane,
                cframe_plane_plot,
                reference_points=scanner_reference_points,
                fiducial_points=fiducial_points,
                fiducial_marker=fiducial_marker,
            )
        else:
            self._logger.debug("C-frame plane diagnostic disabled; skipping.")

        # Final scanner -> detector transformation
        self._log_task(
            "compute_detector_transform",
            references=len(references),
        )
        nominal = np.vstack([ref.nominal_center_xy for ref in references])
        measured = np.vstack([ref.center_xy for ref in references])
        self._validate_reference_geometry(measured, nominal)

        q0 = np.r_[plane.params, measured.reshape(-1)]
        Cq = build_joint_covariance(
            plane.covariance_params,
            references,
        )
        R, t, yaw, Ct = propagate_final_transform_covariance(
            q0,
            Cq,
            nominal,
        )

        result = AlignmentResult(
            rotation=R,
            translation=t,
            homogeneous=homogeneous(R, t),
            covariance_transform_params=Ct,
            covariance_joint=Cq,
            plane=plane,
            references=references,
            yaw_rad=yaw,
        )
        if fiducials is not None:
            transformed_fiducials = fiducials.transformed(R, t)

        if self.visualization_draw["cframe"]:
            final_plot = images_dir / "cframe_final_alignment.png"
            self._log_task(
                "write_alignment_diagnostic",
                stage="final",
                destination=final_plot,
            )
            self.visualizer.plot_plane_alignment(
                cframe.points,
                R,
                plane,
                final_plot,
                translation=t,
                reference_points=scanner_reference_points,
                fiducial_points=fiducial_points,
                fiducial_marker=fiducial_marker,
                title="C-frame final alignment",
            )
        else:
            self._logger.debug("C-frame final diagnostic disabled; skipping.")

        # Ladder/sensor analysis. The final C-frame result is passed into
        # LadderProcessor; the pipeline does not transform ladder clouds itself.
        ladder_results = self._process_ladders(
            alignment=result,
            images_dir=images_dir,
            uncertainty_default=uncertainty_default,
        )
        if (
            ladder_results
            and self.visualization_draw["all_ladders_and_cframe"]
        ):
            self._plot_cframe_and_ladder_residuals(
                cframe.points,
                alignment=result,
                ladder_results=ladder_results,
                images_dir=images_dir,
            )
        elif ladder_results:
            self._logger.debug(
                "All-ladders and C-frame diagnostic disabled; skipping."
            )

        # results.json -> output/
        # *.npy        -> output/matrices/
        self._log_task(
            "write_numerical_results",
            destination=outdir / "results.json",
        )
        self._save_results(
            result,
            outdir,
            matrices_dir,
            ladder_results=ladder_results,
            fiducials=fiducials,
            transformed_fiducials=transformed_fiducials,
        )

        # transformed point clouds -> output/point_clouds/
        self._log_task(
            "write_transformed_clouds",
            destination=point_clouds_dir,
        )
        self._transform_outputs(
            result,
            point_clouds_dir,
            transformed_fiducials=transformed_fiducials,
        )

        sensor_count = sum(len(sensors) for sensors in ladder_results.values())
        self._logger.info(
            "Pipeline | status=completed | references=%d | ladders=%d | "
            "sensors=%d | output_dir=%s",
            len(references),
            len(ladder_results),
            sensor_count,
            self._log_value(outdir),
        )
        return result

    def _process_ladders(
        self,
        alignment: AlignmentResult,
        images_dir: Path,
        uncertainty_default: dict,
    ) -> dict[str, list[SensorPlaneResult]]:
        """Process every configured ladder independently."""
        ladder_configs = self._ladder_configs()
        if not ladder_configs:
            self._log_task(
                "analyze_ladders",
                action="skip",
                reason="not_configured",
            )
            return {}

        self._log_task("analyze_ladders", count=len(ladder_configs))
        cfg = self.config["ladder_analysis"]
        viz_cfg = cfg.get("visualization", {})
        residual_percentile = float(
            viz_cfg.get("residual_percentile", 99.0)
        )
        show_outliers = bool(viz_cfg.get("show_outliers", False))
        point_size = float(viz_cfg.get("point_size", 2.0))
        self.ladder_processors.clear()
        ladder_results: dict[str, list[SensorPlaneResult]] = {}

        for ladder_name, ladder_cfg in ladder_configs:
            mode = ladder_cfg["segmentation"]["mode"]
            self._log_task(
                "analyze_ladder",
                ladder=ladder_name,
                mode=mode,
            )

            processor = LadderProcessor(
                loader=self.loader,
                plane_fitter=self.sensor_plane_fitter,
                resolve_path=self._resolve,
                automatic_segmenter=segment_ladder,
                transformation=alignment,
            )
            self.ladder_processors[ladder_name] = processor

            # Clouds are transformed to detector coordinates in both modes. In
            # automatic mode this occurs before the segmenter sees the cloud.
            sensors = processor.get_sensor_clouds(
                ladder_cfg,
                default_uncertainty=uncertainty_default,
            )
            self._logger.debug(
                "Ladder %s segmentation/loading complete: %d sensor cloud(s).",
                ladder_name,
                len(sensors),
            )
            for sensor in sensors:
                self._logger.debug(
                    "Sensor %s: %d points%s",
                    sensor.name,
                    len(sensor.points),
                    (
                        f", source={sensor.source}"
                        if sensor.source is not None
                        else ""
                    ),
                )

            self._logger.debug(
                "Fitting one plane per sensor for ladder %s in detector coordinates.",
                ladder_name,
            )
            results = processor.fit_sensor_planes(sensors)
            ladder_results[ladder_name] = results

            for sensor_result in results:
                residuals = sensor_result.residuals_mm
                self._logger.debug(
                    "Sensor %s plane fit: RMS=%.6f mm (%.3f µm), "
                    "inliers=%d/%d, max|residual|=%.6f mm",
                    sensor_result.name,
                    sensor_result.plane.rms_mm,
                    sensor_result.plane.rms_mm * 1000.0,
                    int(np.count_nonzero(sensor_result.plane.inlier_mask)),
                    len(sensor_result.points),
                    float(np.max(np.abs(residuals))),
                )

            output_prefix = LadderVisualizer._safe_name(ladder_name)
            combined_plot = (
                images_dir
                / f"{output_prefix}_ladder_sensor_plane_residuals.png"
            )
            if self.visualization_draw["full_ladders"]:
                self.ladder_visualizer.plot_combined_residuals(
                    results,
                    combined_plot,
                    residual_percentile=residual_percentile,
                    use_inliers_for_scale=bool(
                        viz_cfg.get("use_inliers_for_scale", True)
                    ),
                    show_outliers=show_outliers,
                    point_size=point_size,
                )
            else:
                self._remove_generated_image(combined_plot)
                self._logger.debug(
                    "Full-ladder diagnostic disabled for %s; skipping.",
                    ladder_name,
                )

            if self.visualization_draw["sensors"]:
                self.ladder_visualizer.plot_individual_residuals(
                    results,
                    images_dir,
                    filename_prefix=f"{output_prefix}_",
                    residual_percentile=residual_percentile,
                    show_outliers=show_outliers,
                    point_size=point_size,
                )
            else:
                for sensor_result in results:
                    sensor_name = LadderVisualizer._safe_name(sensor_result.name)
                    self._remove_generated_image(
                        images_dir
                        / f"{output_prefix}_sensor_{sensor_name}_plane_residuals.png"
                    )
                self._logger.debug(
                    "Individual sensor diagnostics disabled for %s; skipping.",
                    ladder_name,
                )

            self._logger.debug(
                "Ladder %s sensor analysis completed.",
                ladder_name,
            )

        return ladder_results

    def _plot_cframe_and_ladder_residuals(
        self,
        cframe_points: np.ndarray,
        alignment: AlignmentResult,
        ladder_results: dict[str, list[SensorPlaneResult]],
        images_dir: Path,
    ) -> None:
        """Write a detector-wide residual map with the C-frame as context."""
        viz_cfg = self.config["ladder_analysis"].get("visualization", {})
        cframe_points = np.asarray(cframe_points, dtype=float)
        transformed_cframe = (
            cframe_points @ alignment.rotation.T + alignment.translation
        )
        cframe_residuals_mm = (
            cframe_points @ alignment.plane.normal + alignment.plane.d
        )
        output_path = (
            images_dir / "cframe_all_ladders_sensor_plane_residuals.png"
        )
        self._log_task(
            "write_ladder_overview",
            ladders=len(ladder_results),
            destination=output_path,
        )
        self.ladder_visualizer.plot_cframe_and_ladder_residuals(
            transformed_cframe,
            cframe_residuals_mm,
            alignment.plane.inlier_mask,
            ladder_results,
            output_path,
            residual_percentile=float(
                viz_cfg.get("residual_percentile", 99.0)
            ),
            use_inliers_for_scale=bool(
                viz_cfg.get("use_inliers_for_scale", True)
            ),
            show_outliers=bool(viz_cfg.get("show_outliers", False)),
            point_size=float(viz_cfg.get("point_size", 2.0)),
        )

    def _transform_outputs(
        self,
        result: AlignmentResult,
        point_clouds_dir: Path,
        transformed_fiducials: FiducialMeasurementResult | None = None,
    ) -> None:
        """
        Write transformed copies of configured input point-cloud files.

        All transformed point clouds are written to point_clouds_dir.

        This output-writing operation is independent of LadderProcessor's
        in-memory transformed cache.
        """
        files: list[tuple[str | Path, str | None]] = [
            (self.config["cframe"]["file"], None)
        ]
        files.extend((ref["file"], None) for ref in self._reference_config())

        fiducials_cfg = self._fiducials_config()
        fiducials_path = None
        if fiducials_cfg is not None:
            files.append((fiducials_cfg["file"], None))
            fiducials_path = self._resolve(fiducials_cfg["file"]).resolve()

        files.extend(
            (
                item["file"] if isinstance(item, dict) else item,
                None,
            )
            for item in self.config.get("ladder_files", [])
        )

        for ladder_name, ladder_cfg in self._ladder_configs():
            output_prefix = LadderVisualizer._safe_name(ladder_name)
            mode = ladder_cfg["segmentation"]["mode"]
            if mode == "segmented_files":
                for sensor_cfg in ladder_cfg.get("sensors", []):
                    if isinstance(sensor_cfg, dict) and sensor_cfg.get("file"):
                        files.append((sensor_cfg["file"], output_prefix))
            elif ladder_cfg.get("file"):
                files.append((ladder_cfg["file"], output_prefix))

        seen: set[tuple[str, str | None]] = set()
        unique_files: list[tuple[Path, str | None]] = []

        for value, output_prefix in files:
            src = self._resolve(value)
            key = (str(src.resolve()), output_prefix)
            if key in seen:
                continue
            seen.add(key)
            unique_files.append((src, output_prefix))

        for src, output_prefix in unique_files:
            suffix = src.suffix or ".xyz"
            output_stem = src.stem
            if output_prefix is not None:
                normalized_stem = output_stem.casefold()
                normalized_prefix = output_prefix.casefold()
                if normalized_stem != normalized_prefix and not normalized_stem.startswith(
                    f"{normalized_prefix}_"
                ):
                    output_stem = f"{output_prefix}_{output_stem}"
            dst = point_clouds_dir / f"{output_stem}_transformed{suffix}"

            transformer = self.loader
            if fiducials_path is not None and src.resolve() == fiducials_path:
                transformer = self.fiducial_loader

            transformer.transform_file_streaming(
                src,
                dst,
                result.rotation,
                result.translation,
                precision=self.config.get("output_precision", 6),
                preserve_header=True,
            )

        if transformed_fiducials is not None:
            source = transformed_fiducials.source
            suffix = source.suffix or ".xyz"
            averaged_path = (
                point_clouds_dir
                / f"{source.stem}_averaged_transformed{suffix}"
            )
            precision = int(self.config.get("output_precision", 6))
            np.savetxt(
                averaged_path,
                transformed_fiducials.means,
                fmt=f"%.{precision}f",
            )
            self._logger.debug(
                "Averaged transformed fiducials written to %s",
                averaged_path,
            )

    @staticmethod
    def _serialize_reference(
        ref: MeasuredReferenceResult,
    ) -> dict:
        return {
            "file": str(ref.source),
            "measured_xyz_mm": ref.measured_xyz.tolist(),
            "leveled_center_xy_mm": ref.center_xy.tolist(),
            "nominal_center_xy_mm": ref.nominal_center_xy.tolist(),
            "covariance_point_xyz": ref.covariance_point_xyz.tolist(),
            "covariance_center_conditional": (
                ref.covariance_center_conditional.tolist()
            ),
            "jacobian_center_wrt_plane": (
                ref.jacobian_center_wrt_plane.tolist()
            ),
        }

    @staticmethod
    def _serialize_fiducial_frame(
        marker: FiducialMarkerResult,
    ) -> dict:
        measurement_stddev = marker.measurement_stddev
        return {
            "measurements_xyz_mm": marker.measurement_points.tolist(),
            "mean_xyz_mm": marker.mean.tolist(),
            "measurement_stddev_xyz_mm": (
                None
                if measurement_stddev is None
                else measurement_stddev.tolist()
            ),
            "measurement_covariance_xyz_mm2": (
                None
                if marker.measurement_covariance is None
                else marker.measurement_covariance.tolist()
            ),
            "mean_uncertainty_stddev_xyz_mm": (
                marker.mean_uncertainty_stddev.tolist()
            ),
            "mean_covariance_xyz_mm2": marker.mean_covariance.tolist(),
        }

    @classmethod
    def _serialize_fiducials(
        cls,
        scanner: FiducialMeasurementResult,
        detector: FiducialMeasurementResult,
    ) -> dict:
        markers = []
        for scanner_marker, detector_marker in zip(
            scanner.markers,
            detector.markers,
            strict=True,
        ):
            if scanner_marker.marker_id != detector_marker.marker_id:
                raise ValueError("Scanner and detector fiducial IDs do not match.")
            markers.append(
                {
                    "id": scanner_marker.marker_id,
                    "measurement_indices": (
                        scanner_marker.measurement_indices.tolist()
                    ),
                    "measurement_count": len(
                        scanner_marker.measurement_indices
                    ),
                    "scanner": cls._serialize_fiducial_frame(scanner_marker),
                    "detector": cls._serialize_fiducial_frame(detector_marker),
                }
            )
        return {
            "source_file": str(scanner.source),
            "fiducial_min_distance_mm": scanner.fiducial_min_distance,
            "marker_count": len(markers),
            "markers": markers,
        }

    def _save_results(
        self,
        result: AlignmentResult,
        outdir: Path,
        matrices_dir: Path,
        ladder_results: dict[str, list[SensorPlaneResult]] | None = None,
        fiducials: FiducialMeasurementResult | None = None,
        transformed_fiducials: FiducialMeasurementResult | None = None,
    ) -> None:
        """
        Save numerical alignment products.

        results.json is written directly to outdir.

        Binary NumPy matrix and covariance products are written to
        matrices_dir.
        """

        # ------------------------------------------------------------------
        # Matrix / covariance products
        # ------------------------------------------------------------------
        np.save(
            matrices_dir / "rotation_matrix.npy",
            result.rotation,
        )
        np.save(
            matrices_dir / "translation_vector.npy",
            result.translation,
        )
        np.save(
            matrices_dir / "transform_matrix.npy",
            result.homogeneous,
        )
        np.save(
            matrices_dir / "covariance_transform_params.npy",
            result.covariance_transform_params,
        )
        np.save(
            matrices_dir / "covariance_joint.npy",
            result.covariance_joint,
        )
        np.save(
            matrices_dir / "covariance_plane_params.npy",
            result.plane.covariance_params,
        )

        for i, ref in enumerate(result.references, start=1):
            np.save(
                matrices_dir
                / f"covariance_reference_{i:02d}_conditional.npy",
                ref.covariance_center_conditional,
            )

        # ------------------------------------------------------------------
        # JSON summary
        # ------------------------------------------------------------------
        rotvec = Rotation.from_matrix(result.rotation).as_rotvec()
        sigma = np.sqrt(
            np.maximum(
                np.diag(result.covariance_transform_params),
                0.0,
            )
        )

        spacing = None
        if len(result.references) == 2:
            measured_xy = np.vstack(
                [r.center_xy for r in result.references]
            )
            nominal_xy = np.vstack(
                [r.nominal_center_xy for r in result.references]
            )
            measured_spacing = float(
                np.linalg.norm(measured_xy[1] - measured_xy[0])
            )
            nominal_spacing = float(
                np.linalg.norm(nominal_xy[1] - nominal_xy[0])
            )
            spacing = {
                "measured_mm": measured_spacing,
                "nominal_mm": nominal_spacing,
                "difference_mm": measured_spacing - nominal_spacing,
            }

        payload = {
            "convention": (
                "p_aligned = R @ p + T; column-vector convention"
            ),
            "units": {
                "length": "mm",
                "rotation_vector": "rad",
            },
            "rotation_matrix": result.rotation.tolist(),
            "translation_vector_mm": result.translation.tolist(),
            "homogeneous_matrix": result.homogeneous.tolist(),
            "rotation_vector_rad": rotvec.tolist(),
            "yaw_about_leveled_z_rad": float(result.yaw_rad),
            "transform_parameter_order": [
                "rx",
                "ry",
                "rz",
                "tx",
                "ty",
                "tz",
            ],
            "transform_parameter_1sigma": sigma.tolist(),
            "covariance_transform_params": (
                result.covariance_transform_params.tolist()
            ),
            "plane": {
                "normal": result.plane.normal.tolist(),
                "d_mm": float(result.plane.d),
                "params_theta_phi_d": result.plane.params.tolist(),
                "rms_mm": float(result.plane.rms_mm),
                "inliers": int(result.plane.inlier_mask.sum()),
                "total_points": int(len(result.plane.inlier_mask)),
                "covariance_params": (
                    result.plane.covariance_params.tolist()
                ),
            },
            "references": [
                self._serialize_reference(ref)
                for ref in result.references
            ],
            "reference_spacing_check": spacing,
        }

        if fiducials is not None:
            if transformed_fiducials is None:
                raise ValueError(
                    "Transformed fiducial results are required for serialization."
                )
            payload["fiducials"] = self._serialize_fiducials(
                fiducials,
                transformed_fiducials,
            )

        if ladder_results:
            ladder_configs = dict(self._ladder_configs())
            payload["ladders"] = {
                ladder_name: {
                    "segmentation_mode": ladder_configs[ladder_name][
                        "segmentation"
                    ]["mode"],
                    "sensor_count": len(sensor_results),
                    "sensors": LadderProcessor.results_summary(sensor_results),
                }
                for ladder_name, sensor_results in ladder_results.items()
            }

        # results.json intentionally remains directly in the output root.
        results_path = outdir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self._logger.debug("Numerical results written to %s", results_path)