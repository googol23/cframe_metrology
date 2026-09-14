from pathlib import Path

import numpy as np

from detector_alignment.io import AsciiPointCloudLoader


def test_loader_constant_covariance_and_headers(tmp_path: Path):
    src = tmp_path / "points.xyz"
    src.write_text("# header\n1 2 3\n4 5 6\n", encoding="utf-8")
    data = AsciiPointCloudLoader().load(
        src,
        {"mode": "constant", "sigma_xyz_mm": [0.1, 0.2, 0.3]},
    )
    np.testing.assert_allclose(data.points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(data.covariances[0], np.diag([0.01, 0.04, 0.09]))
    assert data.header_lines == ["# header"]


def test_streaming_transform_preserves_extra_columns(tmp_path: Path):
    src = tmp_path / "input.xyz"
    dst = tmp_path / "output.xyz"
    src.write_text("// h\n1 2 3 99\n", encoding="utf-8")
    loader = AsciiPointCloudLoader()
    loader.transform_file_streaming(src, dst, np.eye(3), np.array([1.0, -1.0, 2.0]))
    assert dst.read_text(encoding="utf-8") == "// h\n2.000000 1.000000 5.000000 99\n"
