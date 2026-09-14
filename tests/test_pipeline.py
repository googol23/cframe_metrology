import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from detector_alignment.pipeline import AlignmentPipeline


def _write_synthetic_case(root: Path, *, obsolete_radius: bool = False) -> Path:
    data = root / "data"
    data.mkdir()

    # Exact z=10 plane with a non-degenerate XY footprint.
    pts = []
    for x in np.linspace(-5.0, 5.0, 7):
        for y in np.linspace(-4.0, 4.0, 6):
            pts.append((x, y, 10.0))
    np.savetxt(data / "cframe.xyz", np.asarray(pts))
    np.savetxt(data / "ref1.xyz", np.asarray([[1.0, 2.0, 10.0]]))
    np.savetxt(data / "ref2.xyz", np.asarray([[1.0, 7.0, 10.0]]))

    refs = [
        {"file": "data/ref1.xyz", "nominal_center": [0.0, 0.0]},
        {"file": "data/ref2.xyz", "nominal_center": [0.0, 5.0]},
    ]
    if obsolete_radius:
        refs[0]["radius_mm"] = 1.5

    cfg = {
        "output_dir": "output",
        "uncertainty": {"mode": "constant", "sigma_xyz_mm": [0.01, 0.01, 0.01]},
        "cframe": {"file": "data/cframe.xyz"},
        "plane_fit": {
            "ransac_iterations": 30,
            "ransac_threshold_mm": 0.01,
            "random_seed": 1,
        },
        "references": refs,
        "visualization": {"dpi": 72, "sample_size": 1000, "random_seed": 1},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pipeline_direct_reference_centers_end_to_end(tmp_path: Path):
    config = _write_synthetic_case(tmp_path)
    result = AlignmentPipeline.from_yaml(config).run()

    np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(result.translation, [-1.0, -2.0, -10.0], atol=1e-8)
    assert len(result.references) == 2

    payload = json.loads((tmp_path / "output" / "results.json").read_text(encoding="utf-8"))
    assert len(payload["references"]) == 2
    assert "radius_mm" not in payload["references"][0]
    assert (tmp_path / "output" / "ref1_transformed.xyz").exists()


def test_obsolete_hole_radius_is_rejected(tmp_path: Path):
    config = _write_synthetic_case(tmp_path, obsolete_radius=True)
    with pytest.raises(ValueError, match="obsolete 'radius_mm'"):
        AlignmentPipeline.from_yaml(config).run()
