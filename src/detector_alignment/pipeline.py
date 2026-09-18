from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .segmentation import segment_ladder
from .io import AsciiPointCloudLoader
from .ladder import LadderProcessor, SensorPlaneResult
from .ladder_viz import LadderVisualizer
from .math3d import homogeneous
from .models import AlignmentResult
from .plane import RobustPlaneFitter
from .reference import MeasuredReferenceProcessor, MeasuredReferenceResult
from .uncertainty import build_joint_covariance, propagate_final_transform_covariance
from .viz import AlignmentVisualizer


logger = logging.getLogger(__name__)


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
        ├── matrices/
        └── point_clouds/

    Diagnostic images, numerical matrix products, and transformed point clouds
    are kept separate. The machine-readable results.json remains directly in
    the output root.
    """

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("config must be a mapping.")

        self.config = config
        logger.debug("Initialising AlignmentPipeline.")

        io_cfg = config.get("io", {})
        self.loader = AsciiPointCloudLoader(
            comment_prefixes=io_cfg.get("comment_prefixes", ["#", "//", "$"]),
            delimiter=io_cfg.get("delimiter", None),
        )

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
        self.visualizer = AlignmentVisualizer(
            dpi=v_cfg.get("dpi", 600),
            sample_size=v_cfg.get("sample_size", 50_000),
            random_seed=v_cfg.get("random_seed", 42),
            point_alpha=v_cfg.get("point_alpha", 0.25),
            plane_alpha=v_cfg.get("plane_alpha", 0.18),
            sigma_band_alpha=v_cfg.get("sigma_band_alpha", 0.22),
        )

        ladder_cfg = config.get("ladder_analysis", {})
        sensor_fit_cfg = ladder_cfg.get("sensor_plane_fit", {})
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

        # LadderProcessor is intentionally NOT constructed here. The final
        # C-frame transformation does not exist until run() reaches step 3.
        self.ladder_processor: LadderProcessor | None = None

        ladder_viz_cfg = ladder_cfg.get("visualization", {})
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

        if ladder_cfg.get("enabled", False):
            mode = ladder_cfg.get("segmentation", {}).get("mode")
            logger.info(
                "Ladder analysis enabled with segmentation mode '%s'.",
                mode,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlignmentPipeline":
        path = Path(path)
        logger.info("Loading configuration from %s", path)

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError("Configuration root must be a YAML mapping.")

        cfg["_config_dir"] = str(path.parent.resolve())
        return cls(cfg)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(self.config.get("_config_dir", ".")) / path

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
        logger.info("Starting detector-alignment pipeline.")

        # ------------------------------------------------------------------
        # Output structure
        #
        # output/
        # ├── results.json
        # ├── images/
        # ├── matrices/
        # └── point_clouds/
        # ------------------------------------------------------------------
        outdir = self._resolve(self.config.get("output_dir", "output"))
        outdir.mkdir(parents=True, exist_ok=True)

        images_dir = outdir / "images"
        matrices_dir = outdir / "matrices"
        point_clouds_dir = outdir / "point_clouds"

        images_dir.mkdir(parents=True, exist_ok=True)
        matrices_dir.mkdir(parents=True, exist_ok=True)
        point_clouds_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Output root: %s", outdir)
        logger.debug("Image output: %s", images_dir)
        logger.debug("Matrix output: %s", matrices_dir)
        logger.debug("Point-cloud output: %s", point_clouds_dir)

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

        # 1. C-frame plane
        cframe_path = self._resolve(cframe_cfg["file"])
        logger.info("Step 1/5: loading C-frame point cloud: %s", cframe_path)
        cframe = self.loader.load(
            cframe_path,
            cframe_cfg.get("uncertainty", uncertainty_default),
        )
        plane = self.plane_fitter.fit(
            cframe.points,
            cframe.covariances,
        )

        # 2. Measured references
        reference_cfg = self._reference_config()
        logger.info(
            "Step 2/5: processing %d reference point(s).",
            len(reference_cfg),
        )
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

        cframe_plane_plot = images_dir / "cframe_plane_alignment.png"
        self.visualizer.plot_plane_alignment(
            cframe.points,
            plane.rotation_to_xy,
            plane,
            cframe_plane_plot,
            reference_points=scanner_reference_points,
        )

        # 3. Final scanner -> detector transformation
        logger.info(
            "Step 3/5: calculating final scanner-to-detector transformation."
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

        final_plot = images_dir / "cframe_final_alignment.png"
        self.visualizer.plot_plane_alignment(
            cframe.points,
            R,
            plane,
            final_plot,
            translation=t,
            reference_points=scanner_reference_points,
            title="C-frame final alignment",
        )

        # 4. Ladder/sensor analysis. The final C-frame result is passed into
        # LadderProcessor; the pipeline does not transform ladder clouds itself.
        logger.info("Step 4/5: checking ladder-analysis configuration.")
        sensor_results = self._process_ladder(
            alignment=result,
            images_dir=images_dir,
            uncertainty_default=uncertainty_default,
        )

        # 5. Outputs
        logger.info("Step 5/5: writing numerical and transformed outputs.")

        # results.json -> output/
        # *.npy        -> output/matrices/
        self._save_results(
            result,
            outdir,
            matrices_dir,
            sensor_results=sensor_results,
        )

        # transformed point clouds -> output/point_clouds/
        self._transform_outputs(
            result,
            point_clouds_dir,
        )

        logger.info("Detector-alignment pipeline completed successfully.")
        return result

    def _process_ladder(
        self,
        alignment: AlignmentResult,
        images_dir: Path,
        uncertainty_default: dict,
    ) -> list[SensorPlaneResult]:
        """
        Process optional ladder sensor data.

        The final C-frame AlignmentResult is injected into LadderProcessor here.
        Therefore all ladder/sensor input files are transformed during loading,
        before caching and, in automatic mode, before segmentation.

        All ladder/sensor diagnostic figures are written to images_dir.
        """
        cfg = self.config.get("ladder_analysis")

        if cfg is None:
            logger.info(
                "No ladder_analysis block found; skipping ladder analysis."
            )
            return []
        if not isinstance(cfg, dict):
            raise ValueError("'ladder_analysis' must be a YAML mapping.")
        if not cfg.get("enabled", False):
            logger.info("Ladder analysis disabled; skipping.")
            return []

        segmentation_cfg = cfg.get("segmentation", {})
        mode = segmentation_cfg.get("mode")
        logger.info(
            "Starting ladder analysis using segmentation mode '%s'.",
            mode,
        )

        # Construct the processor only after the final C-frame transform exists.
        self.ladder_processor = LadderProcessor(
            loader=self.loader,
            plane_fitter=self.sensor_plane_fitter,
            resolve_path=self._resolve,
            automatic_segmenter=segment_ladder,
            transformation=alignment,
        )

        # get_sensor_clouds() returns detector-frame clouds in BOTH modes.
        # In automatic mode the full ladder cloud is transformed before the
        # automatic segmenter sees it.
        sensors = self.ladder_processor.get_sensor_clouds(
            cfg,
            default_uncertainty=uncertainty_default,
        )

        logger.info(
            "Sensor segmentation/loading complete: %d sensor cloud(s).",
            len(sensors),
        )
        for sensor in sensors:
            logger.info(
                "  Sensor %s: %d points%s",
                sensor.name,
                len(sensor.points),
                (
                    f", source={sensor.source}"
                    if sensor.source is not None
                    else ""
                ),
            )

        # No transform_sensors() call here. Applying the transform again would
        # double-transform the clouds.
        logger.info("Fitting one plane per sensor in detector coordinates.")
        results = self.ladder_processor.fit_sensor_planes(sensors)

        for sensor_result in results:
            residuals = sensor_result.residuals_mm
            logger.info(
                "  Sensor %s plane fit: RMS=%.6f mm (%.3f µm), "
                "inliers=%d/%d, max|residual|=%.6f mm",
                sensor_result.name,
                sensor_result.plane.rms_mm,
                sensor_result.plane.rms_mm * 1000.0,
                int(np.count_nonzero(sensor_result.plane.inlier_mask)),
                len(sensor_result.points),
                float(np.max(np.abs(residuals))),
            )

        viz_cfg = cfg.get("visualization", {})
        residual_percentile = float(
            viz_cfg.get("residual_percentile", 99.0)
        )
        show_outliers = bool(viz_cfg.get("show_outliers", False))
        point_size = float(viz_cfg.get("point_size", 2.0))

        combined_plot = images_dir / "ladder_sensor_plane_residuals.png"
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

        if cfg.get("write_individual_plots", True):
            self.ladder_visualizer.plot_individual_residuals(
                results,
                images_dir,
                residual_percentile=residual_percentile,
                show_outliers=show_outliers,
                point_size=point_size,
            )

        logger.info("Ladder sensor analysis completed.")
        return results

    def _transform_outputs(
        self,
        result: AlignmentResult,
        point_clouds_dir: Path,
    ) -> None:
        """
        Write transformed copies of configured input point-cloud files.

        All transformed point clouds are written to point_clouds_dir.

        This output-writing operation is independent of LadderProcessor's
        in-memory transformed cache.
        """
        files: list[str | Path] = [self.config["cframe"]["file"]]
        files.extend(ref["file"] for ref in self._reference_config())

        files.extend(
            item["file"] if isinstance(item, dict) else item
            for item in self.config.get("ladder_files", [])
        )

        ladder_cfg = self.config.get("ladder_analysis")
        if isinstance(ladder_cfg, dict) and ladder_cfg.get("enabled", False):
            segmentation_cfg = ladder_cfg.get("segmentation", {})
            mode = segmentation_cfg.get("mode")

            if mode == "segmented_files":
                for sensor_cfg in ladder_cfg.get("sensors", []):
                    if isinstance(sensor_cfg, dict) and sensor_cfg.get("file"):
                        files.append(sensor_cfg["file"])
            elif mode == "automatic":
                if ladder_cfg.get("file"):
                    files.append(ladder_cfg["file"])

        seen: set[str] = set()
        unique_files: list[Path] = []

        for value in files:
            src = self._resolve(value)
            key = str(src.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique_files.append(src)

        for src in unique_files:
            suffix = src.suffix or ".xyz"
            dst = point_clouds_dir / f"{src.stem}_transformed{suffix}"

            self.loader.transform_file_streaming(
                src,
                dst,
                result.rotation,
                result.translation,
                precision=self.config.get("output_precision", 6),
                preserve_header=True,
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

    def _save_results(
        self,
        result: AlignmentResult,
        outdir: Path,
        matrices_dir: Path,
        sensor_results: list[SensorPlaneResult] | None = None,
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

        if sensor_results:
            ladder_cfg = self.config.get("ladder_analysis", {})
            segmentation_cfg = ladder_cfg.get("segmentation", {})
            payload["ladder"] = {
                "segmentation_mode": segmentation_cfg.get("mode"),
                "sensor_count": len(sensor_results),
                "sensors": LadderProcessor.results_summary(sensor_results),
            }

        # results.json intentionally remains directly in the output root.
        results_path = outdir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Numerical results written to %s", results_path)