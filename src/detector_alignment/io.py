from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import PointCloudData


class AsciiPointCloudLoader:
    """ASCII XYZ loader with configurable comments, delimiters and covariance input."""

    def __init__(self, comment_prefixes=("#", "//", "$"), delimiter=None):
        self.comment_prefixes = tuple(comment_prefixes)
        self.delimiter = delimiter

    def _is_comment(self, stripped: str) -> bool:
        return not stripped or any(stripped.startswith(p) for p in self.comment_prefixes)

    def _split(self, s: str) -> list[str]:
        if self.delimiter in (None, "whitespace"):
            return s.split()
        return s.split(self.delimiter)

    def load(self, path: str | Path, uncertainty: dict | None = None) -> PointCloudData:
        path = Path(path)
        rows: list[list[float]] = []
        header: list[str] = []
        ncols: int | None = None
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if self._is_comment(s):
                    header.append(line.rstrip("\n"))
                    continue
                parts = self._split(s)
                if len(parts) < 3:
                    header.append(line.rstrip("\n"))
                    continue
                try:
                    row = [float(v) for v in parts]
                except ValueError:
                    header.append(line.rstrip("\n"))
                    continue
                if ncols is None:
                    ncols = len(row)
                elif len(row) != ncols:
                    raise ValueError(
                        f"Inconsistent numeric column count in {path}: line {line_no} "
                        f"has {len(row)} columns; expected {ncols}."
                    )
                rows.append(row)

        if not rows:
            raise ValueError(f"No numeric points found in {path}")
        arr = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite numeric values found in {path}")
        points = arr[:, :3]
        covs = self._build_covariances(arr, uncertainty, path)
        return PointCloudData(points=points, covariances=covs, header_lines=header, source=path)

    @staticmethod
    def _validate_columns(arr: np.ndarray, cols, needed: int, label: str) -> np.ndarray:
        cols = np.asarray(cols, dtype=int)
        if cols.shape != (needed,) or np.any(cols < 0) or np.any(cols >= arr.shape[1]):
            raise ValueError(
                f"{label} columns must contain {needed} valid zero-based column indices "
                f"for an input with {arr.shape[1]} columns."
            )
        return cols

    @staticmethod
    def _symmetrize_and_validate_covariances(C: np.ndarray, label: str) -> np.ndarray:
        C = 0.5 * (C + np.swapaxes(C, 1, 2))
        if not np.all(np.isfinite(C)):
            raise ValueError(f"{label} contains non-finite covariance values.")
        if np.any(np.diagonal(C, axis1=1, axis2=2) < 0.0):
            raise ValueError(f"{label} contains negative covariance diagonal entries.")
        return C

    def _build_covariances(self, arr: np.ndarray, uncertainty: dict | None, point_path: Path):
        if not uncertainty or uncertainty.get("mode", "none") == "none":
            return None
        mode = uncertainty["mode"]
        n = len(arr)
        C = np.zeros((n, 3, 3), dtype=float)

        if mode == "constant":
            sig = np.asarray(uncertainty["sigma_xyz_mm"], float)
            if sig.shape != (3,) or not np.all(np.isfinite(sig)) or np.any(sig < 0.0):
                raise ValueError("sigma_xyz_mm must be three finite non-negative values.")
            C[:] = np.diag(sig**2)
            return C

        if mode == "columns_sigma":
            cols = self._validate_columns(arr, uncertainty.get("columns", [3, 4, 5]), 3, mode)
            sig = arr[:, cols]
            if np.any(sig < 0.0):
                raise ValueError("Sigma columns must be non-negative.")
            C[:, 0, 0] = sig[:, 0] ** 2
            C[:, 1, 1] = sig[:, 1] ** 2
            C[:, 2, 2] = sig[:, 2] ** 2
            return C

        if mode == "columns_covariance":
            cols = self._validate_columns(arr, uncertainty.get("columns", [3, 4, 5, 6, 7, 8]), 6, mode)
            v = arr[:, cols]
            C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
            C[:, 0, 1] = C[:, 1, 0] = v[:, 3]
            C[:, 0, 2] = C[:, 2, 0] = v[:, 4]
            C[:, 1, 2] = C[:, 2, 1] = v[:, 5]
            return self._symmetrize_and_validate_covariances(C, mode)

        if mode in {"file_sigma", "file_covariance"}:
            upath = Path(uncertainty["path"])
            if not upath.is_absolute():
                upath = point_path.parent / upath
            vals: list[list[float]] = []
            with upath.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if self._is_comment(s):
                        continue
                    vals.append([float(x) for x in self._split(s)])
            u = np.asarray(vals, float)
            expected_cols = 3 if mode == "file_sigma" else 6
            if u.ndim != 2 or u.shape != (n, expected_cols):
                raise ValueError(
                    f"Uncertainty file {upath} must have shape ({n}, {expected_cols}); "
                    f"got {u.shape}."
                )
            if not np.all(np.isfinite(u)):
                raise ValueError(f"Uncertainty file {upath} contains non-finite values.")
            if mode == "file_sigma":
                if np.any(u < 0.0):
                    raise ValueError(f"Uncertainty file {upath} contains negative sigma values.")
                C[:, 0, 0] = u[:, 0] ** 2
                C[:, 1, 1] = u[:, 1] ** 2
                C[:, 2, 2] = u[:, 2] ** 2
                return C
            C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = u[:, 0], u[:, 1], u[:, 2]
            C[:, 0, 1] = C[:, 1, 0] = u[:, 3]
            C[:, 0, 2] = C[:, 2, 0] = u[:, 4]
            C[:, 1, 2] = C[:, 2, 1] = u[:, 5]
            return self._symmetrize_and_validate_covariances(C, mode)

        raise ValueError(f"Unknown uncertainty mode: {mode}")

    def transform_file_streaming(
        self,
        src: str | Path,
        dst: str | Path,
        R: np.ndarray,
        t: np.ndarray,
        precision: int = 6,
        preserve_header: bool = True,
    ):
        src, dst = Path(src), Path(dst)
        R = np.asarray(R, float)
        t = np.asarray(t, float)
        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError("R and t must have shapes (3, 3) and (3,).")
        dst.parent.mkdir(parents=True, exist_ok=True)
        fmt = f"{{:.{int(precision)}f}}"
        with src.open("r", encoding="utf-8", errors="replace") as fi, dst.open("w", encoding="utf-8") as fo:
            for line in fi:
                s = line.strip()
                if self._is_comment(s):
                    if preserve_header:
                        fo.write(line if line.endswith("\n") else line + "\n")
                    continue
                parts = self._split(s)
                if len(parts) < 3:
                    if preserve_header:
                        fo.write(line if line.endswith("\n") else line + "\n")
                    continue
                try:
                    xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    if preserve_header:
                        fo.write(line if line.endswith("\n") else line + "\n")
                    continue
                out = R @ xyz + t
                parts[:3] = [fmt.format(v) for v in out]
                sep = " " if self.delimiter in (None, "whitespace") else self.delimiter
                fo.write(sep.join(parts) + "\n")
