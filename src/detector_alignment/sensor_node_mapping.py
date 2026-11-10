"""Associate assembly module identities with ROOT sensor boxes by nominal Y order.

This resolves identities only. It does not determine the optical-to-ROOT local
coordinate bridge or establish physical equivalence of the two reference frames.
"""

from __future__ import annotations

import re

import numpy as np

from .nominal_geometry import NominalLadder


_POSITION = re.compile(r"^M\d[A-Z]{2}\d+([BT])(\d)")


def map_modules_by_y(modules, geometry: NominalLadder) -> dict[str, str]:
    """Match B(outside→centre), T(centre→outside) to ascending ROOT centre Y."""
    if not isinstance(modules, (list, tuple, set, dict)) or not modules:
        raise ValueError("Y-order mapping requires a non-empty module collection")
    positions = {}
    for module in modules:
        if not isinstance(module, str) or not (match := _POSITION.match(module)):
            raise ValueError(f"Invalid assembly module position: {module!r}")
        side, index = match.group(1), int(match.group(2))
        if (side, index) in positions:
            raise ValueError(f"Duplicate assembly module position: {side}{index}")
        positions[side, index] = module

    counts = [sum(side == label for side, _ in positions) for label in ("B", "T")]
    if counts[0] != counts[1] or len(positions) != len(geometry.sensors):
        raise ValueError("Y-order mapping requires equal B/T counts matching ROOT sensors")
    for side, count in zip(("B", "T"), counts):
        if {index for label, index in positions if label == side} != set(range(count)):
            raise ValueError(f"Y-order mapping requires consecutive {side} indices from 0")

    ordered_sensors = sorted(geometry.sensors, key=lambda sensor: sensor.center_mm[1])
    ys = [sensor.center_mm[1] for sensor in ordered_sensors]
    if any(left >= right for left, right in zip(ys, ys[1:])):
        raise ValueError("ROOT sensor centre Y values must be distinct")
    if any(y >= 0 for y in ys[: counts[0]]) or any(y <= 0 for y in ys[counts[0] :]):
        raise ValueError("ROOT bottom/top sensor centres must have negative/positive Y")

    ordered_modules = (
        [positions["B", index] for index in range(counts[0] - 1, -1, -1)]
        + [positions["T", index] for index in range(counts[1])]
    )
    return {module: sensor.node_path for module, sensor in zip(ordered_modules, ordered_sensors)}


def ladder_aligned_box_center_mapping(modules, geometry: NominalLadder) -> dict:
    """Resolve Y-order identities and optical ladder axes at each ROOT box center.

    Top ROOT boxes define the ladder axes; bottom boxes must differ by a half
    turn about Z. No marker-surface displacement enters this centre-origin bridge.
    """
    paths = map_modules_by_y(modules, geometry)
    sensors = {sensor.node_path: sensor for sensor in geometry.sensors}
    top = [sensors[path] for module, path in paths.items() if _POSITION.match(module).group(1) == "T"]
    bottom = [sensors[path] for module, path in paths.items() if _POSITION.match(module).group(1) == "B"]
    ladder_rotation = top[0].rotation
    half_turn = np.diag([-1.0, -1.0, 1.0])
    if any(not np.allclose(sensor.rotation, ladder_rotation, atol=1e-7, rtol=0) for sensor in top) or any(
        not np.allclose(sensor.rotation, ladder_rotation @ half_turn, atol=1e-7, rtol=0)
        for sensor in bottom
    ):
        raise ValueError("ROOT sensor axes are not ladder-aligned top / half-turn bottom")
    mapping = {}
    for module, path in paths.items():
        bridge = np.eye(4)
        bridge[:3, :3] = sensors[path].rotation.T @ ladder_rotation
        mapping[module] = {
            "node_path": path,
            "sensor_to_geometry": bridge.tolist(),
        }
    return mapping

