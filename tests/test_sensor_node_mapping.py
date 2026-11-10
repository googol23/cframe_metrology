"""Position-only association of assembly modules with ROOT sensor nodes."""

import numpy as np
import pytest

from detector_alignment.nominal_geometry import NominalLadder, NominalSensor
from detector_alignment.sensor_node_mapping import (
    ladder_aligned_box_center_mapping,
    map_modules_by_y,
)


def ladder(ys):
    sensors = tuple(
        NominalSensor(f"/ladder/node{index}", [0, y, 0], [1, 1, 0.32], np.eye(3))
        for index, y in enumerate(reversed(ys))
    )
    return NominalLadder("/ladder", "global", sensors)


def test_y_order_ignores_root_export_order_and_nonposition_name_digits():
    geometry = ladder([-3, -2, -1, 1, 2, 3])
    modules = [
        "M3DL5T2001142A2", "M3DL5B0101140B2", "M3DL5B2001142B2",
        "M3DL5T1001141A2", "M3DL5B1001141B2", "M3DL5T0001140A2",
    ]
    mapping = map_modules_by_y(modules, geometry)
    assert mapping["M3DL5B2001142B2"] == "/ladder/node5"
    assert mapping["M3DL5B0101140B2"] == "/ladder/node3"
    assert mapping["M3DL5T0001140A2"] == "/ladder/node2"
    assert mapping["M3DL5T2001142A2"] == "/ladder/node0"
    ten_modules = [
        f"M3DL3{side}{index}001121{side}2"
        for side in ("B", "T") for index in range(5)
    ]
    ten_modules[-1] = "M3DL3T4101124A2"
    assert map_modules_by_y(ten_modules, ladder([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]))[
        "M3DL3T4101124A2"
    ] == "/ladder/node0"


def test_ladder_aligned_box_center_uses_side_dependent_root_rotations():
    geometry = ladder([-1, 1])
    bottom, top = geometry.sensors[1], geometry.sensors[0]
    from detector_alignment.nominal_geometry import NominalSensor

    bottom = NominalSensor(
        bottom.node_path, bottom.center_mm, bottom.dimensions_mm,
        np.diag([-1, -1, 1]),
    )
    geometry = NominalLadder("/ladder", "global", (top, bottom))
    modules = ["M3DL1T0001120A2", "M3DL1B0001120B2"]
    result = ladder_aligned_box_center_mapping(modules, geometry)
    np.testing.assert_allclose(result[modules[0]]["sensor_to_geometry"], np.eye(4))
    np.testing.assert_allclose(
        result[modules[1]]["sensor_to_geometry"][:3],
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]]),
    )
    with pytest.raises(ValueError, match="not ladder-aligned"):
        ladder_aligned_box_center_mapping(modules, ladder([-1, 1]))


@pytest.mark.parametrize(
    "modules,ys,reason",
    [
        (["M3DL5B0001140B2", "M3DL5B0001140B3", "M3DL5T0001140A2"], [-1, 1, 2], "Duplicate"),
        (["M3DL5B1001141B2", "M3DL5T0001140A2"], [-1, 1], "consecutive"),
        (["M3DL5B0001140B2", "M3DL5T0001140A2"], [-1, -1], "distinct"),
        (["M3DL5B0001140B2", "M3DL5T0001140A2"], [1, 2], "negative/positive"),
        (["M3DL5B0001140B2", "M3DL5T0001140A2", "M3DL5T1001141A2"], [-2, 1, 2], "equal B/T"),
        (["unknown", "M3DL5T0001140A2"], [-1, 1], "Invalid"),
    ],
)
def test_rejects_ambiguous_or_inconsistent_associations(modules, ys, reason):
    with pytest.raises(ValueError, match=reason):
        map_modules_by_y(modules, ladder(ys))
