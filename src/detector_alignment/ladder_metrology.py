"""External optical-metrology execution and geometry composition, never optical fitting."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .nominal_geometry import NominalLadder, NominalSensor, rigid_matrix
from .sensor_node_mapping import (
    ladder_aligned_box_center_mapping,
    map_modules_by_y,
)

SENSOR_FRAME = "nominal_sensor_local_v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _sensor_basis(value):
    """A coordinate-basis bridge may reflect axes; an as-built pose may not."""
    bridge = np.asarray(value, dtype=float)
    probe = bridge.copy()
    if probe.shape == (4, 4) and np.linalg.det(probe[:3, :3]) < 0:
        probe[:3, 0] *= -1
    rigid_matrix(probe, "orthogonal optical sensor_to_geometry basis")
    return bridge.copy()


class LadderMetrologyDriver:
    """Run/check the independent producer through a versioned JSON boundary."""

    def __init__(self, config, resolve_path, logger):
        self.config = config
        self.resolve_path = resolve_path
        self.logger = logger
        self.options = config.get("ladder_metrology", {})
        if not isinstance(self.options, dict):
            raise ValueError(
                "ladder_metrology must be a mapping; migrate the old filename setting"
            )
        self.enabled = self.options.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("ladder_metrology.enabled must be true or false")
        if not self.enabled:
            return
        unknown = set(self.options) - {
            "enabled",
            "source_dir",
            "result_file",
            "output_dir",
            "sensor_frame",
            "execution",
            "mode",
            "ladders",
            "selection",
        }
        if unknown:
            raise ValueError(f"Unknown ladder_metrology options: {sorted(unknown)}")
        half = config.get("half_unit", {})
        if (
            not isinstance(half, dict)
            or half.get("assembly_naming") != "STS8"
            or not re.fullmatch(r"U[0-7][UD][LR]", str(half.get("name", "")))
        ):
            raise ValueError(
                "Optical metrology requires half_unit.name and assembly_naming: STS8"
            )
        self.half_unit = half["name"]
        if self.options.get("sensor_frame") != SENSOR_FRAME:
            raise ValueError(f"ladder_metrology.sensor_frame must be {SENSOR_FRAME}")
        if (
            not isinstance(self.options.get("source_dir"), str)
            or not self.options["source_dir"].strip()
        ):
            raise ValueError("ladder_metrology.source_dir must be a non-empty path")
        result_file = self.options.get("result_file")
        output_dir = self.options.get("output_dir")
        if result_file is not None and output_dir is not None:
            raise ValueError(
                "ladder_metrology must use result_file or legacy output_dir, not both"
            )
        if result_file is not None:
            if not isinstance(result_file, str) or not result_file.strip():
                raise ValueError("ladder_metrology.result_file must be a non-empty path")
            self.result_path = self.resolve_path(result_file).resolve()
        elif isinstance(output_dir, str) and output_dir.strip():
            self.result_path = (
                self.resolve_path(output_dir) / "sensor_transforms.json"
            ).resolve()
        else:
            raise ValueError(
                "ladder_metrology.result_file must name the producer JSON artifact"
            )
        self.mode = self.options.get("mode", "run")
        if self.mode not in ("run", "reuse"):
            raise ValueError("ladder_metrology.mode must be run or reuse")
        execution = self.options.get("execution", {})
        if not isinstance(execution, dict) or set(execution) - {
            "python",
            "timeout_seconds",
            "working_dir",
        }:
            raise ValueError(
                "ladder_metrology.execution accepts python, timeout_seconds and working_dir"
            )
        self.python = execution.get("python", sys.executable)
        if not isinstance(self.python, str) or not self.python.strip():
            raise ValueError(
                "ladder_metrology.execution.python must be a non-empty executable"
            )
        working_dir = execution.get("working_dir", ".")
        if not isinstance(working_dir, str) or not working_dir.strip():
            raise ValueError(
                "ladder_metrology.execution.working_dir must be a non-empty path"
            )
        self.working_dir = self.resolve_path(working_dir).resolve()
        if not self.working_dir.is_dir():
            raise ValueError(
                f"ladder_metrology.execution.working_dir is not a directory: {self.working_dir}"
            )
        self.timeout = execution.get("timeout_seconds", 120)
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not np.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError(
                "ladder_metrology.execution.timeout_seconds must be finite and positive"
            )
        self.selection = self.options.get("selection")
        if self.selection is not None:
            if (
                not isinstance(self.selection, list)
                or not self.selection
                or any(
                    not isinstance(name, str)
                    or not re.fullmatch("L" + self.half_unit[1:] + r"\d{6}", name)
                    for name in self.selection
                )
                or len(set(self.selection)) != len(self.selection)
            ):
                raise ValueError(
                    "ladder_metrology.selection must list unique STS8 ladders belonging to the half-unit"
                )

    def run(self, *, check_only=False):
        if not self.enabled:
            return None
        config_path = self.config.get("_config_path")
        if not config_path or not Path(config_path).is_file():
            raise ValueError(
                "External optical analysis requires a saved configuration; use AlignmentPipeline.from_yaml(config.yaml)"
            )
        config_path = Path(config_path).resolve()
        # Never let the child analyze a different configuration from the in-memory consumer.
        import yaml

        saved = yaml.safe_load(config_path.read_text())
        for key in ("half_unit", "ladder_metrology"):
            if saved.get(key) != self.config.get(key):
                raise ValueError(
                    "Optical configuration changed in memory; save/reload the configuration before execution"
                )
        checking = check_only or self.mode == "reuse"
        command = [
            self.python,
            "-m",
            "sts_ladder_metrology.main",
            "--config",
            str(config_path),
        ]
        if checking:
            command.append("--check")
        for name in self.selection or []:
            command.extend(("--ladder", name))
        self.logger.info(
            "Optical metrology | action=%s | half_unit=%s",
            "check" if checking else "run",
            self.half_unit,
        )
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
                f"Optical metrology exceeded {self.timeout} seconds"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Cannot execute optical metrology interpreter {self.python}: {error}"
            ) from error
        result_path = self.result_path
        if not checking:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            (result_path.parent / "execution.log").write_text(
                completed.stdout + completed.stderr
            )
        if completed.returncode:
            raise RuntimeError(
                f"External optical metrology failed (exit {completed.returncode}): {completed.stderr.strip()}\n{completed.stdout.strip()}"
            )
        try:
            receipt = json.loads(completed.stdout)
            payload = json.loads(result_path.read_text())
        except (ValueError, OSError) as error:
            raise ValueError(
                "Optical producer did not return a readable JSON receipt/result"
            ) from error
        if (
            Path(receipt.get("result_file", "")).resolve() != result_path.resolve()
            or receipt.get("checked_only") is not checking
            or receipt.get("input_fingerprint") != payload.get("input_fingerprint")
            or receipt.get("result_digest") != payload.get("result_digest")
        ):
            raise ValueError(
                "Optical producer receipt does not match its result artifact"
            )
        self._validate_result(payload)
        return payload

    def _validate_result(self, payload):
        expected = {
            "schema_version": 1,
            "producer": "sts_ladder_metrology",
            "half_unit": self.half_unit,
            "assembly_naming": "STS8",
            "units": "mm",
            "direction": "nominal_to_measured",
            "sensor_frame": SENSOR_FRAME,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Optical result has incompatible schema, identity, units, direction or sensor frame"
            )
        ladders = payload.get("ladders")
        if (
            not isinstance(ladders, dict)
            or not ladders
            or payload.get("result_digest") != _digest(ladders)
        ):
            raise ValueError("Optical result has missing or modified ladder data")
        if self.selection is not None and set(ladders) != set(self.selection):
            raise ValueError("Optical result does not cover the selected ladders")
        for name, ladder in ladders.items():
            if not re.fullmatch("L" + self.half_unit[1:] + r"\d{6}", name):
                raise ValueError(
                    "Optical result contains a ladder from another half-unit"
                )
            sensors = ladder.get("sensors")
            order = ladder.get("module_order")
            if (
                not isinstance(sensors, dict)
                or not sensors
                or not isinstance(order, list)
                or len(order) != len(set(order))
                or set(order) != set(sensors)
            ):
                raise ValueError(
                    "Optical result has missing/duplicate module identities"
                )
            for module, sensor in sensors.items():
                if not re.fullmatch(
                    "M" + name[1:5] + r"[TB][0-4]\d{6}[ABCD]\d", module
                ):
                    raise ValueError(
                        f"Optical module {module} does not belong to ladder {name}"
                    )
                if sensor.get("covariance_parameters") != [
                    "tx",
                    "ty",
                    "tz",
                    "rotvec_x",
                    "rotvec_y",
                    "rotvec_z",
                ] or sensor.get("covariance_units") != [
                    "mm",
                    "mm",
                    "mm",
                    "rad",
                    "rad",
                    "rad",
                ]:
                    raise ValueError(
                        f"Unknown optical covariance convention for {module}"
                    )
                dimensions = np.asarray(sensor.get("dimensions_xy_mm"), dtype=float)
                if (
                    dimensions.shape != (2,)
                    or not np.isfinite(dimensions).all()
                    or np.any(dimensions <= 0)
                ):
                    raise ValueError(f"Invalid optical sensor dimensions for {module}")
                if (
                    sensor.get("status") != "ok"
                    or sensor.get("sensor_frame") != SENSOR_FRAME
                ):
                    raise ValueError(f"Invalid optical fit for {module}")
                rigid_matrix(sensor.get("nominal_to_measured"), module)
                rigid_matrix(sensor.get("nominal_sensor_to_survey"), module)
                covariance = np.asarray(sensor.get("covariance"), dtype=float)
                if (
                    covariance.shape != (6, 6)
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
                    or np.linalg.eigvalsh(covariance).min() < -1e-12
                ):
                    raise ValueError(f"Invalid optical covariance for {module}")


class LadderGeometryPreparation:
    """Apply producer transforms to original ROOT boxes, identically for fit/preview."""

    def __init__(self, driver, resolve_path, root_geometry_driver=None):
        self.driver = driver
        self.resolve_path = resolve_path
        self.root_geometry_driver = root_geometry_driver
        self.result = None

    def prepare(self, *, preview=False):
        if self.root_geometry_driver is not None:
            self.root_geometry_driver.run(check_only=preview)
        self.result = self.driver.run(check_only=preview)

    def load(self, ladder_config, registration, *, apply_optical=True, ladder_name=None):
        path = self.resolve_path(registration["geometry_file"]).resolve()
        root_geometry = (
            {"managed": False}
            if self.root_geometry_driver is None
            else self.root_geometry_driver.geometry_metadata(path)
        )
        nominal = NominalLadder.from_json(
            path, expected_frame=registration["measurement_frame"]
        )
        if not self.driver.enabled:
            return nominal, {
                "root_geometry": root_geometry,
                "optical_metrology": {
                    "applied": False,
                    "reason": "optical_metrology_disabled",
                },
            }
        if self.result is None:
            raise RuntimeError("Optical geometry was not prepared")
        if not apply_optical:
            return nominal, {
                "root_geometry": root_geometry,
                "optical_metrology": {
                    "applied": False,
                    "reason": "no_explicit_ladder_sensor_mapping",
                    "input_fingerprint": self.result["input_fingerprint"],
                    "result_digest": self.result["result_digest"],
                    "result_file": str(self.driver.result_path),
                    "sensor_frame": SENSOR_FRAME,
                },
            }
        assembly = ladder_config.get("assembly_ladder", ladder_name)
        if assembly not in self.result["ladders"]:
            raise ValueError(
                "Each corrected ladder requires an explicit assembly_ladder in the optical half-unit result"
            )
        sensors = self.result["ladders"][assembly]["sensors"]
        mapping = registration.get("optical_sensor_map")
        if isinstance(mapping, dict) and mapping.get("mode") == "y_order":
            if set(mapping) != {"mode", "sensor_to_geometry"}:
                raise ValueError("Y-order optical_sensor_map requires mode and sensor_to_geometry")
            if mapping["sensor_to_geometry"] == "ladder_aligned_box_center":
                mapping = ladder_aligned_box_center_mapping(sensors, nominal)
            else:
                bridge = _sensor_basis(mapping["sensor_to_geometry"])
                node_paths = map_modules_by_y(sensors, nominal)
                mapping = {
                    module: {
                        "node_path": node_path,
                        "sensor_to_geometry": bridge.tolist(),
                    }
                    for module, node_path in node_paths.items()
                }
        if not isinstance(mapping, dict) or set(mapping) != set(sensors):
            raise ValueError(
                f"{assembly}: optical_sensor_map must explicitly map every assembly module to a ROOT sensor"
            )
        by_node = {}
        for module, entry in mapping.items():
            if not isinstance(entry, dict) or set(entry) != {
                "node_path",
                "sensor_to_geometry",
            }:
                raise ValueError(
                    "Each optical_sensor_map entry requires node_path and sensor_to_geometry"
                )
            node = entry["node_path"]
            if not isinstance(node, str) or node in by_node:
                raise ValueError(
                    "Optical sensor mapping contains invalid/duplicate ROOT nodes"
                )
            bridge = _sensor_basis(entry["sensor_to_geometry"])
            deviation = rigid_matrix(sensors[module]["nominal_to_measured"], module)
            by_node[node] = (bridge @ deviation @ np.linalg.inv(bridge), module)
        if set(by_node) != {sensor.node_path for sensor in nominal.sensors}:
            raise ValueError(
                "Optical sensor mapping must cover exactly the ROOT geometry sensors"
            )
        corrected = []
        for sensor in nominal.sensors:
            deviation, module = by_node[sensor.node_path]
            # The bridge is optical-local -> ROOT box-local, in mm. Local corrections
            # right-multiply the original box pose; never modify cached nominal input.
            corrected.append(
                NominalSensor(
                    sensor.node_path,
                    sensor.center_mm + sensor.rotation @ deviation[:3, 3],
                    sensor.dimensions_mm,
                    sensor.rotation @ deviation[:3, :3],
                )
            )
        optical = {
            "applied": True,
            "assembly_ladder": assembly,
            "input_fingerprint": self.result["input_fingerprint"],
            "result_digest": self.result["result_digest"],
            "result_file": str(self.driver.result_path),
            "sensor_frame": SENSOR_FRAME,
            "sensor_mapping": mapping,
            "uncertainty_scope": "optical corrections treated as fixed geometry; optical covariance is not propagated into registration",
            "corrected_sensors": [
                {
                    "node_path": sensor.node_path,
                    "center_mm": sensor.center_mm.tolist(),
                    "rotation": sensor.rotation.tolist(),
                }
                for sensor in corrected
            ],
        }
        return NominalLadder(
            nominal.node_path, nominal.frame, tuple(corrected)
        ), {
            "root_geometry": root_geometry,
            "optical_metrology": optical,
        }
