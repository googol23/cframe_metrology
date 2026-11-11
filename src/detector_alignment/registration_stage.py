"""Opt-in pipeline adapter for nominal ladder registration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .ladder_registration import RegistrationOptions
from .registration_diagnostics import (
    InstrumentedNearestBoxRegistrar, registration_heartbeat,
)
from .labeled_registration import LabeledFaceRegistrar
from .ladder_viz import LadderVisualizer
from .nominal_geometry import NominalLadder, rigid_matrix
from .registration_viz import RegistrationVisualizer


class LadderRegistrationStage:
    _OPTION_KEYS = frozenset(RegistrationOptions.__dataclass_fields__)
    _CONFIG_KEYS = _OPTION_KEYS | {
        "enabled",
        "geometry_file",
        "measurement_frame",
        "initial_transform",
        "draw",
        "optical_sensor_map",
        "mode",
        "facet_z_mm",
        "outlier_sigma",
        "pairing_tolerance_mm",
        "diagnostic_interval_s",
    }

    def __init__(
        self,
        loader,
        resolve_path,
        logger,
        *,
        geometry_preparation=None,
        dpi=200,
        sample_size=50_000,
        random_seed=42,
    ):
        self.loader = loader
        self.resolve_path = resolve_path
        self.logger = logger
        self.visualizer = RegistrationVisualizer(dpi, sample_size, random_seed)
        self.results = {}
        self.geometry_preparation = geometry_preparation

    @classmethod
    def validate_config(cls, config: dict | None, *, partial: bool = False) -> bool:
        if config is None:
            return False
        if not isinstance(config, dict):
            raise ValueError("Ladder registration must be a mapping.")
        unknown = set(config) - cls._CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown ladder registration options: {sorted(unknown)}")
        for key in ("enabled", "draw"):
            if key in config and not isinstance(config[key], bool):
                raise ValueError(f"registration.{key} must be true or false.")
        if not config.get("enabled", True):
            return False
        for key in ("geometry_file", "measurement_frame"):
            if partial and key not in config:
                continue
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"registration.{key} must be a non-empty string.")
        if config.get("mode", "nearest_boxes") not in ("nearest_boxes", "labeled_faces"):
            raise ValueError("registration.mode must be nearest_boxes or labeled_faces")
        for key in ("facet_z_mm", "outlier_sigma", "pairing_tolerance_mm", "diagnostic_interval_s"):
            if key in config:
                value = config[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"registration.{key} must be finite and numeric")
                if key != "facet_z_mm" and value <= 0:
                    raise ValueError(f"registration.{key} must be positive")
        RegistrationOptions(
            **{key: config[key] for key in cls._OPTION_KEYS if key in config}
        )
        if "initial_transform" in config:
            rigid_matrix(config["initial_transform"], "registration.initial_transform")
        return True

    @classmethod
    def resolve_config(
        cls,
        defaults: dict | None,
        overrides: dict | None,
    ) -> dict | None:
        """Merge whole fields without changing either source mapping.

        Partial mappings may omit required geometry/frame fields; the effective
        mapping must provide them unless registration is explicitly disabled.
        """
        cls.validate_config(defaults, partial=True)
        cls.validate_config(overrides, partial=True)
        if defaults is None and overrides is None:
            return None
        effective = {**(defaults or {}), **(overrides or {})}
        cls.validate_config(effective)
        return effective

    def load_inputs(self, ladder_config: dict, alignment, default_uncertainty, *,
                    geometry=None, pairing_tolerance_mm=15.0, initial_transform=None):
        """Load all source points and covariances in the fixed detector frame."""
        if ladder_config["segmentation"]["mode"] == "automatic":
            inputs = [
                (
                    ladder_config["file"],
                    ladder_config.get("uncertainty", default_uncertainty),
                )
            ]
        else:
            inputs = [
                (sensor["file"], sensor.get("uncertainty", default_uncertainty))
                for sensor in ladder_config["sensors"]
            ]
        clouds = [
            self.loader.load(self.resolve_path(filename), uncertainty)
            for filename, uncertainty in inputs
        ]
        points = np.concatenate([cloud.points for cloud in clouds])
        points = points @ alignment.rotation.T + alignment.translation
        present = [cloud.covariances is not None for cloud in clouds]
        if any(present) and not all(present):
            raise ValueError(
                "Ladder registration requires covariance for all input files or none; "
                "mixed uncertainty availability is ambiguous."
            )
        covariances = None
        if all(present):
            covariances = np.einsum(
                "ij,njk,lk->nil",
                alignment.rotation,
                np.concatenate([cloud.covariances for cloud in clouds]),
                alignment.rotation,
            )
        input_files = [str(cloud.source) for cloud in clouds]
        if geometry is None:
            return points, covariances, input_files
        if ladder_config["segmentation"]["mode"] != "segmented_files":
            raise ValueError("labeled_faces requires one input file per sensor")
        if len(clouds) != len(geometry.sensors):
            raise ValueError("labeled_faces requires exactly one cloud per ROOT sensor")
        transformed = [cloud.points @ alignment.rotation.T + alignment.translation
                       for cloud in clouds]
        measured_y = np.array([np.median(cloud[:, 1]) for cloud in transformed])
        nominal = np.stack([sensor.center_mm for sensor in geometry.sensors])
        start = rigid_matrix(np.eye(4) if initial_transform is None else initial_transform,
                             "registration.initial_transform")
        expected_y = (nominal @ start[:3, :3].T + start[:3, 3])[:, 1]
        measured_order = np.argsort(measured_y, kind="stable")
        nominal_order = np.argsort(expected_y, kind="stable")
        if (np.any(np.diff(np.sort(measured_y)) <= 0) or
                np.any(np.diff(np.sort(expected_y)) <= 0)):
            raise ValueError("Ambiguous overlapping sensor Y centers")
        if np.any(np.abs(measured_y[measured_order] -
                         expected_y[nominal_order]) > pairing_tolerance_mm):
            raise ValueError("Sensor Y correspondence exceeds pairing tolerance; verify ROOT/C-frame mapping")
        ids_by_file = np.empty(len(clouds), dtype=int)
        ids_by_file[measured_order] = nominal_order
        labels = np.concatenate([np.full(len(cloud.points), ids_by_file[i], dtype=int)
                                 for i, cloud in enumerate(clouds)])
        return points, covariances, input_files, labels

    def run(self, ladder_configs, alignment, outdir: Path, default_uncertainty) -> dict:
        self.results.clear()
        if self.geometry_preparation is not None:
            self.geometry_preparation.prepare()
        summaries = {}
        for ladder_name, ladder_config in ladder_configs:
            config = ladder_config.get("registration")
            prefix = LadderVisualizer._safe_name(ladder_name)
            image_path = (
                outdir / "images" / "registration" / f"{prefix}_ladder_registration.png"
            )
            matrix_path = outdir / "registration" / f"{prefix}_nominal_to_as_built.npy"
            residual_path = outdir / "registration" / f"{prefix}_point_residuals.csv"
            legacy_paths = (
                outdir / "images" / f"{prefix}_ladder_registration.png",
                outdir / "matrices" / f"{prefix}_nominal_to_as_built.npy",
                outdir / "registrations" / f"{prefix}_point_residuals.csv",
            )
            # Remove only the known products of disabled, still-configured ladders.
            if not self.validate_config(config):
                for path in (image_path, matrix_path, residual_path, *legacy_paths):
                    path.unlink(missing_ok=True)
                continue
            stage_started = perf_counter()
            self.logger.info("Pipeline | task=register_ladder | ladder=%s | mode=%s", ladder_name, config.get("mode", "nearest_boxes"))
            geometry_path = self.resolve_path(config["geometry_file"]).resolve()
            preparation = {
                "root_geometry": {"managed": False},
                "optical_metrology": {"applied": False},
            }
            labeled = config.get("mode", "nearest_boxes") == "labeled_faces"
            if labeled and self.geometry_preparation is None:
                raise ValueError(
                    "labeled_faces requires optical geometry preparation; "
                    "uncorrected ROOT sensor boxes cannot be used for fitting"
                )
            if self.geometry_preparation is None:
                geometry = NominalLadder.from_json(
                    geometry_path, expected_frame=config["measurement_frame"]
                )
            else:
                geometry, preparation = self.geometry_preparation.load(
                    ladder_config, config, ladder_name=ladder_name
                )
            if labeled and preparation.get("optical_metrology", {}).get("applied") is not True:
                raise ValueError(
                    "labeled_faces requires optically corrected nominal sensor "
                    "geometry; enable optical metrology and provide an explicit "
                    "optical_sensor_map for every sensor"
                )
            # The corrected geometry returned by LadderGeometryPreparation.load is
            # used for Y correspondence, outward-face selection, fitting and plots.
            load_started = perf_counter()
            loaded = self.load_inputs(
                ladder_config,
                alignment,
                default_uncertainty,
                geometry=geometry if labeled else None,
                pairing_tolerance_mm=config.get("pairing_tolerance_mm", 15.0),
                initial_transform=config.get("initial_transform"),
            )
            points, covariances, input_files = loaded[:3]
            load_seconds = perf_counter() - load_started
            self.logger.info(
                "Ladder %s registration | phase=load | points=%d | sensors=%d | files=%d | covariance=%s | elapsed_s=%.3f | optical_corrected=%s",
                ladder_name, len(points), len(geometry.sensors), len(input_files),
                covariances is not None, load_seconds,
                preparation.get("optical_metrology", {}).get("applied") is True,
            )
            interval = float(config.get("diagnostic_interval_s", 15.0))
            if not labeled:
                self.logger.warning(
                    "Ladder %s registration | mode=nearest_boxes | "
                    "optimized_labeled_faces=False | points=%d | sensors=%d | "
                    "hint=set registration.mode=labeled_faces to use compressed fitting",
                    ladder_name, len(points), len(geometry.sensors),
                )
            options = RegistrationOptions(
                **{key: config[key] for key in self._OPTION_KEYS if key in config}
            )
            if labeled:
                with registration_heartbeat(self.logger, ladder_name,
                                            "labeled_faces_fit", interval):
                    result = LabeledFaceRegistrar(
                        options,
                        facet_z_mm=config.get("facet_z_mm", 0.0),
                        outlier_sigma=config.get("outlier_sigma", 3.0),
                        logger=self.logger,
                    ).fit(geometry, points, loaded[3], covariances,
                          config.get("initial_transform"))
            else:
                registrar = InstrumentedNearestBoxRegistrar(options)
                with registration_heartbeat(self.logger, ladder_name,
                                            "nearest_boxes_fit", interval, registrar):
                    result = registrar.fit(
                        geometry, points, covariances, config.get("initial_transform")
                    )
                self.logger.info(
                    "Ladder %s registration | phase=nearest_boxes_fit | "
                    "evaluate_calls=%d | evaluate_seconds=%.3f",
                    ladder_name, registrar.evaluate_calls, registrar.evaluate_seconds,
                )
            self.results[ladder_name] = result
            summary = result.summary(geometry)
            summary.update(
                {
                    "geometry_file": str(geometry_path),
                    "root_geometry": preparation["root_geometry"],
                    "optical_metrology": preparation["optical_metrology"],
                    "registration_reference_geometry": (
                        "optically_corrected_nominal_sensors"
                        if labeled else "prepared_nominal_sensors"
                    ),
                    "input_files": input_files,
                    "point_order": "input_files order, preserving numeric row order in each file",
                    "options": asdict(options),
                    "input_loading_seconds": load_seconds,
                    "registration_mode": config.get("mode", "nearest_boxes"),
                    "facet_z_mm": config.get("facet_z_mm", 0.0) if labeled else None,
                    "outlier_sigma": config.get("outlier_sigma", 3.0) if labeled else None,
                    "initial_transform": config.get(
                        "initial_transform", np.eye(4).tolist()
                    ),
                    "point_residuals_file": str(residual_path.relative_to(outdir)),
                    "transform_file": str(matrix_path.relative_to(outdir)),
                    "image_file": str(image_path.relative_to(outdir))
                    if config.get("draw", True)
                    else None,
                    "uncertainty_scope": "point measurement covariance conditional on the fixed C-frame alignment; excludes common frame uncertainty",
                }
            )
            summaries[ladder_name] = summary
            for warning in result.warnings:
                self.logger.warning("Ladder %s registration: %s", ladder_name, warning)
            self.logger.info(
                "Ladder %s registration | accepted=%s | feasible=%s | surface_rms_mm=%.6g | rank=%d/6",
                ladder_name,
                result.accepted,
                result.feasible,
                result.final.rms_mm,
                result.surface_jacobian_rank,
            )
            export_started = perf_counter()
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(matrix_path, result.homogeneous)
            residual_path.parent.mkdir(parents=True, exist_ok=True)
            inlier_mask = (result.final.inlier_mask if labeled else
                           np.ones(len(points), dtype=bool))
            np.savetxt(
                residual_path,
                np.column_stack(
                    (
                        np.arange(len(points)),
                        points,
                        result.final.sensor_indices,
                        result.final.residuals_mm,
                        result.final.containment_margin_mm,
                        *((inlier_mask.astype(int),) if labeled else ()),
                    )
                ),
                delimiter=",",
                fmt=["%d", "%.12g", "%.12g", "%.12g", "%d", "%.12g", "%.12g"] +
                    (["%d"] if labeled else []),
                header=("point_index,x_mm,y_mm,z_mm,assigned_sensor_index,signed_surface_distance_mm,containment_margin_mm,used_for_fit"
                        if labeled else
                        "point_index,x_mm,y_mm,z_mm,nearest_surface_sensor_index,signed_surface_distance_mm,containment_margin_mm"),
                comments="",
            )
            export_seconds = perf_counter() - export_started
            self.logger.info(
                "Ladder %s registration | phase=export | csv_rows=%d | elapsed_s=%.3f",
                ladder_name, len(points), export_seconds,
            )
            plot_started = perf_counter()
            if config.get("draw", True):
                self.visualizer.plot(
                    geometry, points, result, image_path, ladder_name=ladder_name
                )
            else:
                image_path.unlink(missing_ok=True)
            plot_seconds = perf_counter() - plot_started
            if config.get("draw", True):
                self.logger.info(
                    "Ladder %s registration | phase=plot | elapsed_s=%.3f",
                    ladder_name, plot_seconds,
                )
            for path in legacy_paths:
                path.unlink(missing_ok=True)
            self.logger.info(
                "Ladder %s registration | phase=complete | total_stage_s=%.3f | fit_s=%s | export_s=%.3f | plot_s=%.3f | plot_written=%s | csv_rows=%d",
                ladder_name, perf_counter() - stage_started,
                (f"{result.performance['total_seconds']:.3f}" if labeled else "not_instrumented"),
                export_seconds, plot_seconds, bool(config.get("draw", True)), len(points),
            )
        return summaries
