"""External geo-inspector execution and validated nominal-geometry publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from .nominal_geometry import NominalLadder


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class RootGeometryDriver:
    """Produce/check a complete geo-inspector half-unit ladder export."""

    def __init__(self, config, resolve_path, logger):
        self.config = config
        self.resolve_path = resolve_path
        self.logger = logger
        self.options = config.get("root_geometry", {})
        if not isinstance(self.options, dict):
            raise ValueError("root_geometry must be a mapping")
        self.enabled = self.options.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("root_geometry.enabled must be true or false")
        self.result = None
        if not self.enabled:
            return
        unknown = set(self.options) - {
            "enabled",
            "source_file",
            "node",
            "frame",
            "output_dir",
            "mode",
            "execution",
        }
        if unknown:
            raise ValueError(f"Unknown root_geometry options: {sorted(unknown)}")
        half = config.get("half_unit", {})
        self.half_unit = half.get("name") if isinstance(half, dict) else None
        if not isinstance(self.half_unit, str) or not self.half_unit:
            raise ValueError("root_geometry requires half_unit.name")
        for key in ("source_file", "node", "output_dir"):
            value = self.options.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"root_geometry.{key} must be a non-empty path/string")
        self.source_path = self.resolve_path(self.options["source_file"]).resolve()
        self.output_dir = self.resolve_path(self.options["output_dir"]).resolve()
        self.manifest_path = self.output_dir / "manifest.json"
        self.frame = self.options.get("frame", "node")
        if self.frame not in ("node", "global"):
            raise ValueError("root_geometry.frame must be node or global")
        self.mode = self.options.get("mode", "run")
        if self.mode not in ("run", "reuse"):
            raise ValueError("root_geometry.mode must be run or reuse")
        execution = self.options.get("execution", {})
        if not isinstance(execution, dict) or set(execution) - {
            "python",
            "root",
            "working_dir",
            "timeout_seconds",
        }:
            raise ValueError(
                "root_geometry.execution accepts python, root, working_dir and timeout_seconds"
            )
        self.python = execution.get("python", sys.executable)
        self.root = execution.get("root", "root")
        working_dir = execution.get("working_dir", ".")
        for key, value in (
            ("python", self.python),
            ("root", self.root),
            ("working_dir", working_dir),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"root_geometry.execution.{key} must be a non-empty path/command"
                )
        self.working_dir = self.resolve_path(working_dir).resolve()
        if not self.working_dir.is_dir():
            raise ValueError(
                f"root_geometry.execution.working_dir is not a directory: {self.working_dir}"
            )
        self.timeout = execution.get("timeout_seconds", 120)
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not np.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError(
                "root_geometry.execution.timeout_seconds must be finite and positive"
            )

    def run(self, *, check_only=False):
        if not self.enabled:
            return None
        if not self.source_path.is_file():
            raise FileNotFoundError(f"ROOT geometry file does not exist: {self.source_path}")
        self._check_saved_configuration()
        source = _file_record(self.source_path)
        fingerprint = _digest(
            {
                "source_geometry": source,
                "node": self.options["node"],
                "frame": self.frame,
                "half_unit": self.half_unit,
            }
        )
        checking = check_only or self.mode == "reuse"
        if checking:
            result = self._load_manifest(fingerprint)
        else:
            result = self._extract(source, fingerprint)
        self.result = result
        return result

    def _check_saved_configuration(self) -> None:
        config_path = self.config.get("_config_path")
        if not config_path or not Path(config_path).is_file():
            raise ValueError(
                "External ROOT inspection requires a saved configuration; "
                "use AlignmentPipeline.from_yaml(config.yaml)"
            )
        saved = yaml.safe_load(Path(config_path).read_text())
        for key in ("half_unit", "root_geometry"):
            if saved.get(key) != self.config.get(key):
                raise ValueError(
                    "ROOT geometry configuration changed in memory; "
                    "save/reload the configuration before execution"
                )

    def _extract(self, source: dict[str, str], fingerprint: str) -> dict:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}-staging-", dir=self.output_dir.parent
            )
        )
        command = [
            self.python,
            "-m",
            "geo_inspector",
            "ladder",
            str(self.source_path),
            "--node",
            self.options["node"],
            "--frame",
            self.frame,
            "--output-dir",
            str(stage),
            "--root",
            self.root,
        ]
        self.logger.info(
            "ROOT geometry | action=extract | half_unit=%s | node=%s",
            self.half_unit,
            self.options["node"],
        )
        try:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"ROOT geometry inspection exceeded {self.timeout} seconds"
                ) from error
            except OSError as error:
                raise RuntimeError(
                    f"Cannot execute geo_inspector interpreter {self.python}: {error}"
                ) from error
            if completed.returncode:
                raise RuntimeError(
                    "geo_inspector failed "
                    f"(exit {completed.returncode}): {completed.stderr.strip()}\n"
                    f"{completed.stdout.strip()}"
                )
            try:
                receipt = json.loads(completed.stdout)
            except ValueError as error:
                raise ValueError(
                    "geo_inspector did not return a readable JSON receipt"
                ) from error
            manifest = self._manifest_from_receipt(receipt, stage, source, fingerprint)
            (stage / "execution.log").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self._publish(stage)
            stage = None
            return self._load_manifest(fingerprint)
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)

    def _manifest_from_receipt(self, receipt, stage, source, fingerprint):
        if not isinstance(receipt, dict):
            raise ValueError("geo_inspector receipt must contain an object")
        expected_frame = "node" if self.frame == "node" else "global"
        if (
            Path(receipt.get("source_geometry", "")).resolve() != self.source_path
            or receipt.get("coordinate_frame") != expected_frame
            or not isinstance(receipt.get("node_path"), str)
        ):
            raise ValueError(
                "geo_inspector receipt has incompatible source, node or coordinate frame"
            )
        measurement_frame = (
            receipt.get("reference_node") if self.frame == "node" else "global"
        )
        if not isinstance(measurement_frame, str) or not measurement_frame:
            raise ValueError("Node-frame geo_inspector receipt requires reference_node")
        files = receipt.get("files")
        if (
            not isinstance(files, list)
            or not files
            or receipt.get("ladder_count") != len(files)
        ):
            raise ValueError("geo_inspector receipt has invalid ladder file coverage")
        records = []
        seen_paths = set()
        seen_nodes = set()
        for raw_path in files:
            path = Path(raw_path).resolve()
            if not path.is_relative_to(stage.resolve()) or not path.is_file():
                raise ValueError(
                    f"geo_inspector result is outside its staging directory: {path}"
                )
            relative = path.relative_to(stage.resolve())
            destination = (self.output_dir / relative).resolve()
            if destination in seen_paths:
                raise ValueError("geo_inspector returned duplicate output files")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if Path(payload.get("source_geometry", "")).resolve() != self.source_path:
                raise ValueError("Nominal geometry source does not match configured ROOT file")
            geometry = NominalLadder.from_json(path, expected_frame=measurement_frame)
            if geometry.node_path in seen_nodes:
                raise ValueError("geo_inspector returned duplicate ladder nodes")
            seen_paths.add(destination)
            seen_nodes.add(geometry.node_path)
            records.append(
                {
                    "path": str(destination),
                    "relative_path": relative.as_posix(),
                    "node_path": geometry.node_path,
                    "sensor_count": len(geometry.sensors),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        records.sort(key=lambda item: item["node_path"])
        return {
            "schema_version": 1,
            "producer": "geo_inspector",
            "half_unit": self.half_unit,
            "source_geometry": source,
            "node_selector": self.options["node"],
            "node_path": receipt["node_path"],
            "coordinate_frame": expected_frame,
            "measurement_frame": measurement_frame,
            "input_fingerprint": fingerprint,
            "files": records,
            "result_digest": _digest(records),
        }

    def _load_manifest(self, fingerprint: str) -> dict:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepare ROOT nominal geometry first: {self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("ROOT geometry manifest is not readable JSON") from error
        expected = {
            "schema_version": 1,
            "producer": "geo_inspector",
            "half_unit": self.half_unit,
            "node_selector": self.options["node"],
            "coordinate_frame": "node" if self.frame == "node" else "global",
            "input_fingerprint": fingerprint,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "ROOT geometry result is stale or incompatible with the configuration"
            )
        source = manifest.get("source_geometry")
        if source != _file_record(self.source_path):
            raise ValueError("ROOT geometry source changed after inspection")
        records = manifest.get("files")
        if (
            not isinstance(records, list)
            or not records
            or manifest.get("result_digest") != _digest(records)
        ):
            raise ValueError("ROOT geometry manifest has missing or modified file data")
        measurement_frame = manifest.get("measurement_frame")
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("ROOT geometry manifest contains an invalid file record")
            relative = Path(record.get("relative_path", ""))
            path = (self.output_dir / relative).resolve()
            if (
                not relative.parts
                or path.parent == path
                or not path.is_relative_to(self.output_dir)
                or str(path) != record.get("path")
                or path in seen
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != record.get("sha256")
            ):
                raise ValueError("ROOT geometry output is missing, moved or modified")
            geometry = NominalLadder.from_json(path, expected_frame=measurement_frame)
            if (
                geometry.node_path != record.get("node_path")
                or len(geometry.sensors) != record.get("sensor_count")
            ):
                raise ValueError("ROOT geometry output does not match its manifest")
            seen.add(path)
        return manifest

    def _publish(self, stage: Path) -> None:
        if not self.output_dir.exists():
            stage.replace(self.output_dir)
            return
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}-previous-", dir=self.output_dir.parent
            )
        )
        backup.rmdir()
        self.output_dir.replace(backup)
        try:
            stage.replace(self.output_dir)
        except Exception:
            backup.replace(self.output_dir)
            raise
        shutil.rmtree(backup)

    def geometry_metadata(self, path) -> dict:
        if not self.enabled:
            return {"managed": False}
        if self.result is None:
            raise RuntimeError("ROOT nominal geometry was not prepared")
        resolved = Path(path).resolve()
        records = {
            Path(record["path"]).resolve(): record for record in self.result["files"]
        }
        if resolved not in records:
            raise ValueError(
                "registration.geometry_file must reference a file produced by the configured root_geometry batch"
            )
        return {
            "managed": True,
            "manifest_file": str(self.manifest_path),
            "source_geometry": self.result["source_geometry"],
            "node_selector": self.result["node_selector"],
            "node_path": self.result["node_path"],
            "measurement_frame": self.result["measurement_frame"],
            "input_fingerprint": self.result["input_fingerprint"],
            "result_digest": self.result["result_digest"],
            "geometry_file": records[resolved],
        }
