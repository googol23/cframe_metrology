"""Preview registration inputs without running any alignment or ladder fit.

Run: python -m detector_alignment.registration_preview CONFIG.yaml
The saved scanner-to-detector matrix defaults to output_dir/matrices/transform_matrix.npy.
All optional path overrides are relative to the configuration directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .ladder_registration import LadderRegistrar, RegistrationOptions
from .nominal_geometry import NominalLadder, rigid_matrix
from .pipeline import AlignmentPipeline
from .registration_stage import LadderRegistrationStage
from .registration_viz import RegistrationVisualizer


class RegistrationPreview:
    """Read original measurements once per ladder and preserve existing fit products."""

    def __init__(self, pipeline: AlignmentPipeline):
        self.pipeline = pipeline

    def run(
        self,
        *,
        alignment_matrix: str | Path | None = None,
        output_dir: str | Path | None = None,
        dpi: int = 150,
        sample_size: int = 20_000,
        ladder_names: list[str] | None = None,
    ) -> dict:
        pipeline = self.pipeline
        output_root = pipeline._resolve(pipeline.config.get("output_dir", "output"))
        alignment_path = (
            pipeline._resolve(alignment_matrix)
            if alignment_matrix is not None
            else output_root / "matrices" / "transform_matrix.npy"
        ).resolve()
        if not alignment_path.is_file():
            raise FileNotFoundError(
                f"Saved scanner-to-detector matrix not found: {alignment_path}. "
                "Supply --alignment-matrix or first compute the C-frame alignment "
                "with ladder registration disabled. Preview never runs that fit."
            )
        matrix = rigid_matrix(
            np.load(alignment_path, allow_pickle=False), "saved alignment"
        )
        alignment = SimpleNamespace(rotation=matrix[:3, :3], translation=matrix[:3, 3])
        configs = dict(pipeline._ladder_configs())
        if ladder_names:
            unknown = set(ladder_names) - set(configs)
            if unknown:
                raise ValueError(f"Unknown preview ladder(s): {sorted(unknown)}")
            configs = {name: configs[name] for name in dict.fromkeys(ladder_names)}
        selected = {}
        for name, config in configs.items():
            registration = config.get("registration")
            if registration is None:
                continue
            # Preview is explicit and independent of enabled/draw switches. Validate
            # the required geometry fields without changing the saved configuration.
            registration = {**registration, "enabled": True}
            LadderRegistrationStage.validate_config(registration)
            selected[name] = (config, registration)
        if not selected:
            raise ValueError(
                "No ladders with configured geometry are available for preview."
            )
        destination = (
            pipeline._resolve(output_dir)
            if output_dir is not None
            else output_root / "images" / "registration_initial"
        ).resolve()
        visualizer = RegistrationVisualizer(
            dpi=dpi,
            sample_size=sample_size,
            random_seed=pipeline.config.get("visualization", {}).get("random_seed", 42),
        )
        report = {
            "kind": "registration_initial_preview",
            "registration_performed": False,
            "configuration_file": pipeline.config.get("_config_path"),
            "alignment_matrix_file": str(alignment_path),
            "scanner_to_detector_matrix": matrix.tolist(),
            "display_sample_size": sample_size,
            "dpi": dpi,
            "warnings": [
                "Saved C-frame transform reused, not recomputed: verify it matches the current measurement inputs/configuration.",
                "Geometry-frame identity and measured detector-frame equivalence are not physically verified by this preview.",
            ],
            "ladders": {},
        }
        print(
            f"Preview only: reusing saved scanner-to-detector matrix {alignment_path}",
            flush=True,
        )
        for name, (ladder_config, registration) in selected.items():
            print(
                f"Preview {name}: loading original measurements and nominal boxes (no optimization)",
                flush=True,
            )
            geometry_path = pipeline._resolve(registration["geometry_file"]).resolve()
            geometry = NominalLadder.from_json(
                geometry_path,
                expected_frame=registration["measurement_frame"],
            )
            points, covariances, input_files = pipeline.registration_stage.load_inputs(
                ladder_config,
                alignment,
                pipeline.config.get("uncertainty"),
            )
            initial = rigid_matrix(
                registration.get("initial_transform", np.eye(4)),
                "preview initial_transform",
            )
            options = RegistrationOptions(
                **{
                    key: registration[key]
                    for key in RegistrationOptions.__dataclass_fields__
                    if key in registration
                }
            )
            evaluation = LadderRegistrar(options).evaluate_initial(
                geometry,
                points,
                covariances,
                initial,
            )
            corners = geometry.corners_mm @ initial[:3, :3].T + initial[:3, 3]
            image_path = (
                destination
                / f"{pipeline.ladder_visualizer._safe_name(name)}_initial_registration.png"
            )
            visualizer.plot_initial(
                geometry,
                points,
                evaluation,
                initial,
                image_path,
                ladder_name=name,
            )
            report["ladders"][name] = {
                "geometry_file": str(geometry_path),
                "geometry_node": geometry.node_path,
                "geometry_frame": geometry.frame,
                "measured_frame": "detector frame defined by the saved scanner-to-detector matrix",
                "input_files": input_files,
                "initial_nominal_to_as_built_matrix": initial.tolist(),
                "total_points": len(points),
                "displayed_points": min(len(points), sample_size),
                "measured_bounds_mm": {
                    "min": points.min(axis=0).tolist(),
                    "max": points.max(axis=0).tolist(),
                },
                "initial_geometry_bounds_mm": {
                    "min": corners.min(axis=0).tolist(),
                    "max": corners.max(axis=0).tolist(),
                },
                "surface_rms_mm": evaluation.rms_mm,
                "surface_max_abs_mm": float(np.max(np.abs(evaluation.residuals_mm))),
                "outside_points": int(
                    np.count_nonzero(
                        evaluation.containment_margin_mm
                        < -options.containment_tolerance_mm
                    )
                ),
                "uncertainty_sigma": options.uncertainty_sigma,
                "containment_tolerance_mm": options.containment_tolerance_mm,
                "image_file": str(image_path),
            }
            print(f"Preview {name}: wrote {image_path}", flush=True)
        summary_path = destination / "preview_summary.json"
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"Preview summary: {summary_path}", flush=True)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Pipeline YAML configuration")
    parser.add_argument(
        "--alignment-matrix", help="Saved scanner-to-detector .npy matrix"
    )
    parser.add_argument("--output-dir", help="Preview-only output directory")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument(
        "--ladder",
        action="append",
        dest="ladder_names",
        help="Preview only this ladder; repeat to select more",
    )
    args = parser.parse_args(argv)
    RegistrationPreview(AlignmentPipeline.from_yaml(args.config)).run(
        alignment_matrix=args.alignment_matrix,
        output_dir=args.output_dir,
        dpi=args.dpi,
        sample_size=args.sample_size,
        ladder_names=args.ladder_names,
    )


if __name__ == "__main__":
    main()
