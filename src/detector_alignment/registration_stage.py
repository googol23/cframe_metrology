"""Opt-in pipeline adapter for nominal ladder registration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ladder_registration import LadderRegistrar, RegistrationOptions
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

    def load_inputs(self, ladder_config: dict, alignment, default_uncertainty):
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
        return points, covariances, [str(cloud.source) for cloud in clouds]

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
            self.logger.info("Pipeline | task=register_ladder | ladder=%s", ladder_name)
            geometry_path = self.resolve_path(config["geometry_file"]).resolve()
            preparation = {
                "root_geometry": {"managed": False},
                "optical_metrology": {"applied": False},
            }
            if self.geometry_preparation is None:
                geometry = NominalLadder.from_json(
                    geometry_path, expected_frame=config["measurement_frame"]
                )
            else:
                geometry, preparation = self.geometry_preparation.load(
                    ladder_config, config, ladder_name=ladder_name
                )
            points, covariances, input_files = self.load_inputs(
                ladder_config,
                alignment,
                default_uncertainty,
            )
            options = RegistrationOptions(
                **{key: config[key] for key in self._OPTION_KEYS if key in config}
            )
            result = LadderRegistrar(options).fit(
                geometry,
                points,
                covariances,
                config.get("initial_transform"),
            )
            self.results[ladder_name] = result
            summary = result.summary(geometry)
            summary.update(
                {
                    "geometry_file": str(geometry_path),
                    "root_geometry": preparation["root_geometry"],
                    "optical_metrology": preparation["optical_metrology"],
                    "input_files": input_files,
                    "point_order": "input_files order, preserving numeric row order in each file",
                    "options": asdict(options),
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
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(matrix_path, result.homogeneous)
            residual_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                residual_path,
                np.column_stack(
                    (
                        np.arange(len(points)),
                        points,
                        result.final.sensor_indices,
                        result.final.residuals_mm,
                        result.final.containment_margin_mm,
                    )
                ),
                delimiter=",",
                fmt=["%d", "%.12g", "%.12g", "%.12g", "%d", "%.12g", "%.12g"],
                header="point_index,x_mm,y_mm,z_mm,nearest_surface_sensor_index,signed_surface_distance_mm,containment_margin_mm",
                comments="",
            )
            if config.get("draw", True):
                self.visualizer.plot(
                    geometry, points, result, image_path, ladder_name=ladder_name
                )
            else:
                image_path.unlink(missing_ok=True)
            for path in legacy_paths:
                path.unlink(missing_ok=True)
        return summaries
