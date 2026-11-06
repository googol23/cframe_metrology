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


def test_streaming_transform_batches_rows_and_preserves_non_numeric_lines(
    tmp_path: Path,
):
    src = tmp_path / "input.xyz"
    dst = tmp_path / "output.xyz"
    rows = ["# header", "1,2,3,extra", "not numeric", "4 5 6 99", ""]
    src.write_text("\n".join(rows), encoding="utf-8")

    AsciiPointCloudLoader(delimiter="auto").transform_file_streaming(
        src,
        dst,
        np.eye(3),
        np.array([1.0, -1.0, 2.0]),
    )

    assert dst.read_text(encoding="utf-8") == (
        "# header\n2.000000 1.000000 5.000000 extra\n"
        "not numeric\n5.000000 4.000000 8.000000 99\n"
    )


def test_streaming_transform_handles_rows_across_batch_boundary(tmp_path: Path):
    src = tmp_path / "input.xyz"
    dst = tmp_path / "output.xyz"
    src.write_text("".join(f"{i} 0 0\n" for i in range(8193)), encoding="utf-8")

    AsciiPointCloudLoader().transform_file_streaming(
        src,
        dst,
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )

    output_lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 8193
    assert output_lines[8191] == "8192.000000 0.000000 0.000000"
    assert output_lines[8192] == "8193.000000 0.000000 0.000000"


def test_auto_delimiter_accepts_comma_whitespace_and_repeated_points(tmp_path: Path):
    src = tmp_path / "fiducials.xyz"
    src.write_text(
        "1,2,3\n1,2,3\n4 5 6\n7\t8\t9\n",
        encoding="utf-8",
    )

    data = AsciiPointCloudLoader(delimiter="auto").load(src)

    np.testing.assert_allclose(
        data.points,
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )
