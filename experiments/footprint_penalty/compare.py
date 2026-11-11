"""Isolated, reproducible U3DL objective comparison; never runs external producers.

From PointCloudMetrology:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py prepare
  OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py baseline
  OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py 100

Preparation freezes exact fit arrays, corrected geometry and provenance. Runs read
only those snapshots. Existing snapshots/reports are never overwritten.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import yaml

from detector_alignment.labeled_registration import LabeledFaceRegistrar
from detector_alignment.ladder_registration import RegistrationOptions
from detector_alignment.nominal_geometry import NominalLadder, NominalSensor
from detector_alignment.pipeline import AlignmentPipeline

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
NAMES = ("L3DL100112", "L3DL300112")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare():
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("Snapshot already exists; use it rather than changing the comparison inputs")
    config_path = PROJECT / "input/u3dl/config.yaml"
    cfg = yaml.safe_load((ROOT / "baseline/config.original.yaml").read_text())
    original_tolerance = cfg["registration"]["containment_tolerance_mm"]
    # Explicit experiment-only type correction, not a production YAML modification.
    cfg["registration"]["containment_tolerance_mm"] = float(original_tolerance)
    cfg["_config_dir"] = str(config_path.parent)
    # The producer's read-only --check resolves its paths from the saved YAML.
    # Verify that it still matches our frozen copy before allowing that check.
    assert digest(config_path) == digest(ROOT / "baseline/config.original.yaml")
    cfg["_config_path"] = str(config_path)
    pipeline = AlignmentPipeline(cfg)
    pipeline.geometry_preparation.prepare(preview=True)
    assert digest(config_path) == digest(ROOT / "baseline/config.original.yaml")
    save_json(ROOT / "config.numeric.json", cfg)
    alignment_path = pipeline._resolve(pipeline.config["output_dir"]) / "matrices/transform_matrix.npy"
    matrix = np.load(alignment_path, allow_pickle=False)
    alignment = SimpleNamespace(rotation=matrix[:3, :3], translation=matrix[:3, 3])
    configs = dict(pipeline._ladder_configs())
    manifest = {
        "scope": "Current cached C-frame alignment and validated cached optical/ROOT geometry; not a reconstruction of the historical logged run",
        "configuration_adjustment": {"original_tolerance": original_tolerance,
            "original_type": type(original_tolerance).__name__, "numeric_tolerance_mm": float(original_tolerance)},
        "alignment": {"file": str(alignment_path), "sha256": digest(alignment_path), "matrix": matrix.tolist()},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "ladders": {},
    }
    for name in NAMES:
        ladder = configs[name]
        config = ladder["registration"]
        geometry, provenance = pipeline.geometry_preparation.load(ladder, config, ladder_name=name)
        assert provenance["optical_metrology"]["applied"]
        points, covariance, files, ids = pipeline.registration_stage.load_inputs(
            ladder, alignment, pipeline.config.get("uncertainty"), geometry=geometry,
            pairing_tolerance_mm=config.get("pairing_tolerance_mm", 15.),
            initial_transform=config.get("initial_transform"))
        path = ROOT / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, points=points, covariance=covariance, ids=ids,
                centers=np.stack([s.center_mm for s in geometry.sensors]),
                dimensions=np.stack([s.dimensions_mm for s in geometry.sensors]),
                axes=np.stack([s.rotation for s in geometry.sensors]),
                initial=np.asarray(config.get("initial_transform", np.eye(4))))
        manifest["ladders"][name] = {
            "snapshot": path.name, "sha256": digest(path), "config": config,
            "node_path": geometry.node_path, "frame": geometry.frame,
            "sensor_nodes": [s.node_path for s in geometry.sensors], "provenance": provenance,
            "inputs": [{"path": f, "sha256": digest(f)} for f in files],
        }
        print(name, len(points), "points frozen", flush=True)
    manifest["baseline_sources"] = {name: digest(ROOT / "baseline" / name)
        for name in ("config.original.yaml", "labeled_registration.py", "ladder_registration.py")}
    save_json(ROOT / "manifest.json", manifest)


def footprint_metrics(geometry, points, ids, covariance, matrix, options, mask):
    axes = np.einsum("ij,njk->nik", matrix[:3, :3], np.stack([s.rotation for s in geometry.sensors]))[ids]
    centers = np.stack([s.center_mm for s in geometry.sensors]) @ matrix[:3, :3].T + matrix[:3, 3]
    local = np.einsum("ni,nij->nj", points - centers[ids], axes)
    padding = options.uncertainty_sigma * np.sqrt(np.maximum(0,
        np.einsum("nia,nij,nja->na", axes, covariance, axes)))
    half = np.stack([s.dimensions_mm / 2 for s in geometry.sensors])[ids]
    excess = np.maximum(0, np.max(np.abs(local[:, :2]) - half[:, :2] - padding[:, :2], axis=1)
                        - options.containment_tolerance_mm)
    def metrics(selected):
        values = excess[selected]
        return {"outside_footprint_points": int(np.count_nonzero(values)),
                "max_footprint_excess_mm": float(values.max()),
                "sum_squared_footprint_excess_mm2": float(values @ values)}
    return {"all": metrics(mask), "sensors": {s.node_path: metrics(mask & (ids == i))
            for i, s in enumerate(geometry.sensors)}}


def compare(mode):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if mode == "baseline":
        path = ROOT / "baseline/labeled_registration.py"
        assert digest(path) == manifest["baseline_sources"][path.name]
        spec = importlib.util.spec_from_file_location("detector_alignment._frozen_baseline", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        registrar = module.LabeledFaceRegistrar
        weight = 0.
    else:
        registrar, weight = LabeledFaceRegistrar, float(mode)
    for name, record in manifest["ladders"].items():
        report_path = ROOT / f"{name}.weight_{mode}.json"
        if report_path.exists():
            raise FileExistsError(report_path)
        path = ROOT / record["snapshot"]
        assert digest(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as data:
            geometry = NominalLadder(record["node_path"], record["frame"], tuple(
                NominalSensor(node, c, d, a) for node, c, d, a in zip(record["sensor_nodes"],
                    data["centers"], data["dimensions"], data["axes"])))
            config = record["config"]
            options = RegistrationOptions(**{k: config[k] for k in RegistrationOptions.__dataclass_fields__
                if k in config and k != "footprint_penalty_weight"}, footprint_penalty_weight=weight)
            result = registrar(options, facet_z_mm=config.get("facet_z_mm", 0.),
                outlier_sigma=config.get("outlier_sigma", 3.)).fit(
                    geometry, data["points"], data["ids"], data["covariance"], data["initial"])
            report = result.summary(geometry)
            report["comparison_weight"] = weight
            report["snapshot_sha256"] = record["sha256"]
            report["inlier_mask_sha256"] = hashlib.sha256(result.final.inlier_mask.tobytes()).hexdigest()
            report["implementation_sha256"] = digest(ROOT / "baseline/labeled_registration.py" if mode == "baseline"
                else PROJECT / "src/detector_alignment/labeled_registration.py")
            report["xy_footprints"] = {label: footprint_metrics(geometry, data["points"], data["ids"],
                data["covariance"], pose, options, result.final.inlier_mask)
                for label, pose in (("initial", data["initial"]), ("final", result.homogeneous))}
            report["rotation_vector_deg"] = np.rad2deg(Rotation.from_matrix(result.rotation).as_rotvec()).tolist()
            save_json(report_path, report)
            print(name, mode, "accepted", result.accepted, "RMS", result.final.rms_mm,
                  "outside", report["final"]["outside_points"], "XY", report["xy_footprints"]["final"]["all"],
                  "rot_deg", report["rotation_vector_deg"], "pivot", result.pivot_displacement_mm.tolist(),
                  "seconds", result.performance["total_seconds"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Pass prepare, baseline, or a nonnegative penalty weight")
    prepare() if sys.argv[1] == "prepare" else compare(sys.argv[1])
