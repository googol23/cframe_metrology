from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .io import AsciiPointCloudLoader
from .math3d import homogeneous
from .models import AlignmentResult
from .plane import RobustPlaneFitter
from .reference import MeasuredReferenceProcessor, MeasuredReferenceResult
from .uncertainty import build_joint_covariance, propagate_final_transform_covariance
from .viz import AlignmentVisualizer


class AlignmentPipeline:
    """End-to-end detector alignment from a C-frame plane and measured centers.

    Every configured reference file must contain exactly one directly measured XYZ
    center point.  Its paired ``nominal_center`` in the YAML config defines the
    corresponding nominal detector-plane XY coordinate.
    """

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("config must be a mapping.")
        self.config = config

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlignmentPipeline":
        path = Path(path)
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
        # ``reference_holes`` is retained as a compatibility alias because these
        # points are still the measured centers of physical reference holes.
        has_references = "references" in self.config
        has_reference_holes = "reference_holes" in self.config
        if has_references and has_reference_holes:
            raise ValueError("Use either 'references' or 'reference_holes', not both.")

        refs = self.config.get("references", self.config.get("reference_holes", []))
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
                f"Reference {index} contains obsolete 'radius_mm'. Hole-opening fitting "
                "has been removed; provide a file containing one measured XYZ center."
            )
        if "type" in obsolete and str(cfg["type"]).lower().replace("-", "_") not in {
            "measured_point",
            "direct_point",
            "point",
        }:
            raise ValueError(
                f"Reference {index} has unsupported type {cfg['type']!r}; only directly "
                "measured center points are supported."
            )

        nominal = np.asarray(cfg["nominal_center"], dtype=float)
        if nominal.shape != (2,) or not np.all(np.isfinite(nominal)):
            raise ValueError(
                f"Reference {index} nominal_center must contain two finite values [x, y]."
            )

        data = self.loader.load(
            self._resolve(cfg["file"]),
            cfg.get("uncertainty", uncertainty_default),
        )
        if len(data.points) != 1:
            raise ValueError(
                f"Reference {index} file {data.source} must contain exactly one measured "
                f"XYZ center point; found {len(data.points)} numeric rows."
            )

        point_covariance = None if data.covariances is None else data.covariances[0]
        return self.reference_processor.process(
            point_xyz=data.points[0],
            point_covariance_xyz=point_covariance,
            plane_params=plane.params,
            nominal_center_xy=nominal,
            source=data.source,
        )

    @staticmethod
    def _validate_reference_geometry(measured: np.ndarray, nominal: np.ndarray) -> None:
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

        # A rigid in-plane transform needs at least two distinct points in each set.
        if np.max(np.linalg.norm(measured - measured[0], axis=1)) <= 1.0e-12:
            raise ValueError(
                "Measured reference centers are degenerate (not distinct)."
            )
        if np.max(np.linalg.norm(nominal - nominal[0], axis=1)) <= 1.0e-12:
            raise ValueError("Nominal reference centers are degenerate (not distinct).")

    def run(self) -> AlignmentResult:
        outdir = self._resolve(self.config.get("output_dir", "output"))
        outdir.mkdir(parents=True, exist_ok=True)
        uncertainty_default = self.config.get("uncertainty", {"mode": "none"})

        if "hole_fit" in self.config:
            raise ValueError(
                "The 'hole_fit' configuration block is obsolete. Reference files must "
                "contain directly measured XYZ centers."
            )

        cframe_cfg = self.config.get("cframe")
        if not isinstance(cframe_cfg, dict) or "file" not in cframe_cfg:
            raise ValueError("Configuration requires cframe.file.")

        cframe = self.loader.load(
            self._resolve(cframe_cfg["file"]),
            cframe_cfg.get("uncertainty", uncertainty_default),
        )
        plane = self.plane_fitter.fit(cframe.points, cframe.covariances)

        reference_cfg = self._reference_config()
        references = [
            self._process_reference(cfg, plane, uncertainty_default, i)
            for i, cfg in enumerate(reference_cfg, start=1)
        ]
        scanner_reference_points = np.vstack([ref.measured_xyz for ref in references])

        self.visualizer.plot_plane_alignment(
            cframe.points,
            plane.rotation_to_xy,
            plane,
            outdir / "cframe_plane_alignment.png",
            reference_points=scanner_reference_points,
        )

        nominal = np.vstack([ref.nominal_center_xy for ref in references])
        measured = np.vstack([ref.center_xy for ref in references])
        self._validate_reference_geometry(measured, nominal)

        q0 = np.r_[plane.params, measured.reshape(-1)]
        Cq = build_joint_covariance(plane.covariance_params, references)
        R, t, yaw, Ct = propagate_final_transform_covariance(q0, Cq, nominal)

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

        self.visualizer.plot_plane_alignment(
            cframe.points,
            R,
            plane,
            outdir / "cframe_final_alignment.png",
            translation=t,
            reference_points=scanner_reference_points,
            title="C-frame final alignment",
        )

        self._save_results(result, outdir)
        self._transform_outputs(result, outdir)
        return result

    def _transform_outputs(self, result: AlignmentResult, outdir: Path) -> None:
        files: list[str | Path] = [self.config["cframe"]["file"]]
        files.extend(ref["file"] for ref in self._reference_config())
        files.extend(
            item["file"] if isinstance(item, dict) else item
            for item in self.config.get("ladder_files", [])
        )

        seen: set[str] = set()
        for value in files:
            src = self._resolve(value)
            key = str(src.resolve())
            if key in seen:
                continue
            seen.add(key)
            suffix = src.suffix or ".xyz"
            dst = outdir / f"{src.stem}_transformed{suffix}"
            self.loader.transform_file_streaming(
                src,
                dst,
                result.rotation,
                result.translation,
                precision=self.config.get("output_precision", 6),
                preserve_header=True,
            )

    @staticmethod
    def _serialize_reference(ref: MeasuredReferenceResult) -> dict:
        return {
            "file": str(ref.source),
            "measured_xyz_mm": ref.measured_xyz.tolist(),
            "leveled_center_xy_mm": ref.center_xy.tolist(),
            "nominal_center_xy_mm": ref.nominal_center_xy.tolist(),
            "covariance_point_xyz": ref.covariance_point_xyz.tolist(),
            "covariance_center_conditional": ref.covariance_center_conditional.tolist(),
            "jacobian_center_wrt_plane": ref.jacobian_center_wrt_plane.tolist(),
        }

    def _save_results(self, result: AlignmentResult, outdir: Path) -> None:
        np.save(outdir / "rotation_matrix.npy", result.rotation)
        np.save(outdir / "translation_vector.npy", result.translation)
        np.save(outdir / "transform_matrix.npy", result.homogeneous)
        np.save(
            outdir / "covariance_transform_params.npy",
            result.covariance_transform_params,
        )
        np.save(outdir / "covariance_joint.npy", result.covariance_joint)
        np.save(outdir / "covariance_plane_params.npy", result.plane.covariance_params)
        for i, ref in enumerate(result.references, start=1):
            np.save(
                outdir / f"covariance_reference_{i:02d}_conditional.npy",
                ref.covariance_center_conditional,
            )

        rotvec = Rotation.from_matrix(result.rotation).as_rotvec()
        sigma = np.sqrt(np.maximum(np.diag(result.covariance_transform_params), 0.0))

        spacing = None
        if len(result.references) == 2:
            measured_xy = np.vstack([r.center_xy for r in result.references])
            nominal_xy = np.vstack([r.nominal_center_xy for r in result.references])
            measured_spacing = float(np.linalg.norm(measured_xy[1] - measured_xy[0]))
            nominal_spacing = float(np.linalg.norm(nominal_xy[1] - nominal_xy[0]))
            spacing = {
                "measured_mm": measured_spacing,
                "nominal_mm": nominal_spacing,
                "difference_mm": measured_spacing - nominal_spacing,
            }

        payload = {
            "convention": "p_aligned = R @ p + T; column-vector convention",
            "units": {"length": "mm", "rotation_vector": "rad"},
            "rotation_matrix": result.rotation.tolist(),
            "translation_vector_mm": result.translation.tolist(),
            "homogeneous_matrix": result.homogeneous.tolist(),
            "rotation_vector_rad": rotvec.tolist(),
            "yaw_about_leveled_z_rad": float(result.yaw_rad),
            "transform_parameter_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
            "transform_parameter_1sigma": sigma.tolist(),
            "covariance_transform_params": result.covariance_transform_params.tolist(),
            "plane": {
                "normal": result.plane.normal.tolist(),
                "d_mm": float(result.plane.d),
                "params_theta_phi_d": result.plane.params.tolist(),
                "rms_mm": float(result.plane.rms_mm),
                "inliers": int(result.plane.inlier_mask.sum()),
                "total_points": int(len(result.plane.inlier_mask)),
                "covariance_params": result.plane.covariance_params.tolist(),
            },
            "references": [self._serialize_reference(ref) for ref in result.references],
            "reference_spacing_check": spacing,
        }
        with (outdir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
