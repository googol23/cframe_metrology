from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .ladder import SensorPlaneResult


class LadderVisualizer:
    """Plots for ladder sensor plane fits."""

    def __init__(
        self,
        dpi: int = 300,
        sample_size: int = 50_000,
        random_seed: int = 42,
    ) -> None:
        self.dpi = int(dpi)
        self.sample_size = int(sample_size)
        self.rng = np.random.default_rng(random_seed)

        if self.dpi <= 0:
            raise ValueError("dpi must be greater than zero.")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be greater than zero.")

    def _sample_indices(self, n: int) -> np.ndarray:
        if n <= self.sample_size:
            return np.arange(n)

        return np.sort(
            self.rng.choice(
                n,
                size=self.sample_size,
                replace=False,
            )
        )

    @staticmethod
    def _residual_limit_um(
        results: list[SensorPlaneResult],
        percentile: float,
        use_inliers: bool,
        additional_residuals_mm: Iterable[np.ndarray] = (),
    ) -> float:
        values = [
            np.abs(np.asarray(residuals, dtype=float)) * 1000.0
            for residuals in additional_residuals_mm
            if len(residuals)
        ]

        for result in results:
            residuals = result.residuals_mm

            if use_inliers:
                residuals = residuals[result.plane.inlier_mask]

            if len(residuals):
                values.append(np.abs(residuals) * 1000.0)

        if not values:
            return 1.0

        combined = np.concatenate(values)
        limit = float(np.percentile(combined, percentile))

        if not np.isfinite(limit) or limit <= 0.0:
            limit = float(np.max(combined)) if len(combined) else 1.0

        return max(limit, 1e-9)

    def plot_combined_residuals(
        self,
        results: Iterable[SensorPlaneResult],
        output_path: str | Path,
        *,
        residual_percentile: float = 99.0,
        use_inliers_for_scale: bool = True,
        show_outliers: bool = False,
        point_size: float = 2.0,
    ) -> Path:
        """
        Combined XY projection of all sensors.

        Colour represents signed point-to-plane distance for each point
        relative to the plane fitted to its own sensor.
        """
        results = list(results)

        if not results:
            raise ValueError("No sensor-plane results supplied.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        limit_um = self._residual_limit_um(
            results,
            percentile=residual_percentile,
            use_inliers=use_inliers_for_scale,
        )

        norm = TwoSlopeNorm(
            vmin=-limit_um,
            vcenter=0.0,
            vmax=limit_um,
        )

        fig, ax = plt.subplots(figsize=(9, 11))
        scatter_artist = None

        for result in results:
            idx = self._sample_indices(len(result.points))

            if not show_outliers:
                idx = idx[result.plane.inlier_mask[idx]]

            if len(idx) == 0:
                continue

            points = result.points[idx]
            residuals_um = result.residuals_mm[idx] * 1000.0

            scatter_artist = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=residuals_um,
                s=point_size,
                norm=norm,
                cmap="coolwarm",
                linewidths=0,
                rasterized=True,
            )

            centroid = result.plane.centroid
            ax.text(
                centroid[0],
                centroid[1],
                result.name,
                ha="center",
                va="center",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )

        if scatter_artist is None:
            plt.close(fig)
            raise RuntimeError("No sensor points remained for plotting.")

        colorbar = fig.colorbar(scatter_artist, ax=ax)
        colorbar.set_label("Distance to fitted sensor plane [µm]")

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_title("Ladder sensor plane residuals")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def plot_cframe_and_ladder_residuals(
        self,
        cframe_points: np.ndarray,
        cframe_residuals_mm: np.ndarray,
        cframe_inlier_mask: np.ndarray,
        ladder_results: Mapping[str, Iterable[SensorPlaneResult]],
        output_path: str | Path,
        *,
        residual_percentile: float = 99.0,
        use_inliers_for_scale: bool = True,
        show_outliers: bool = False,
        point_size: float = 2.0,
    ) -> Path:
        """Plot the detector-frame C-frame facet and every ladder sensor residual."""
        cframe_points = np.asarray(cframe_points, dtype=float)
        if (
            cframe_points.ndim != 2
            or cframe_points.shape[1] != 3
            or len(cframe_points) == 0
        ):
            raise ValueError("cframe_points must have shape (N, 3) with N > 0.")
        if not np.all(np.isfinite(cframe_points)):
            raise ValueError("cframe_points contain NaN or infinite values.")

        cframe_residuals_mm = np.asarray(cframe_residuals_mm, dtype=float)
        cframe_inlier_mask = np.asarray(cframe_inlier_mask, dtype=bool)
        if cframe_residuals_mm.shape != (len(cframe_points),):
            raise ValueError(
                "cframe_residuals_mm must have one value per C-frame point."
            )
        if cframe_inlier_mask.shape != (len(cframe_points),):
            raise ValueError(
                "cframe_inlier_mask must have one value per C-frame point."
            )
        if not np.all(np.isfinite(cframe_residuals_mm)):
            raise ValueError(
                "cframe_residuals_mm contain NaN or infinite values."
            )

        results = [
            result
            for ladder_sensor_results in ladder_results.values()
            for result in ladder_sensor_results
        ]
        if not results:
            raise ValueError("No ladder sensor-plane results supplied.")

        cframe_scale_residuals = cframe_residuals_mm
        if use_inliers_for_scale:
            cframe_scale_residuals = cframe_residuals_mm[cframe_inlier_mask]

        limit_um = self._residual_limit_um(
            results,
            percentile=residual_percentile,
            use_inliers=use_inliers_for_scale,
            additional_residuals_mm=(cframe_scale_residuals,),
        )
        norm = TwoSlopeNorm(
            vmin=-limit_um,
            vcenter=0.0,
            vmax=limit_um,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 11))
        cframe_idx = self._sample_indices(len(cframe_points))
        if not show_outliers:
            cframe_idx = cframe_idx[cframe_inlier_mask[cframe_idx]]
        sampled_cframe = cframe_points[cframe_idx]
        ax.scatter(
            sampled_cframe[:, 0],
            sampled_cframe[:, 1],
            c=cframe_residuals_mm[cframe_idx] * 1000.0,
            s=max(point_size, 1.0),
            norm=norm,
            cmap="coolwarm",
            marker="s",
            alpha=0.65,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        scatter_artist = None
        for result in results:
            idx = self._sample_indices(len(result.points))
            if not show_outliers:
                idx = idx[result.plane.inlier_mask[idx]]
            if len(idx) == 0:
                continue

            points = result.points[idx]
            residuals_um = result.residuals_mm[idx] * 1000.0
            scatter_artist = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=residuals_um,
                s=point_size,
                norm=norm,
                cmap="coolwarm",
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        if scatter_artist is None:
            plt.close(fig)
            raise RuntimeError("No ladder sensor points remained for plotting.")

        colorbar = fig.colorbar(scatter_artist, ax=ax)
        colorbar.set_label("Distance to fitted plane [µm]")

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_title("C-frame facet and all ladder sensor plane residuals")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        return output_path

    def plot_individual_residuals(
        self,
        results: Iterable[SensorPlaneResult],
        output_dir: str | Path,
        *,
        filename_prefix: str = "",
        residual_percentile: float = 99.0,
        show_outliers: bool = False,
        point_size: float = 2.0,
    ) -> list[Path]:
        """Write one XY residual map per sensor."""
        results = list(results)

        if not results:
            raise ValueError("No sensor-plane results supplied.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []

        for result in results:
            residuals = result.residuals_mm
            scale_values = residuals[result.plane.inlier_mask]

            if len(scale_values) == 0:
                scale_values = residuals

            limit_um = float(
                np.percentile(
                    np.abs(scale_values) * 1000.0,
                    residual_percentile,
                )
            )
            limit_um = max(limit_um, 1e-9)

            norm = TwoSlopeNorm(
                vmin=-limit_um,
                vcenter=0.0,
                vmax=limit_um,
            )

            idx = self._sample_indices(len(result.points))
            if not show_outliers:
                idx = idx[result.plane.inlier_mask[idx]]

            points = result.points[idx]
            residuals_um = residuals[idx] * 1000.0

            fig, ax = plt.subplots(figsize=(8, 7))

            scatter_artist = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=residuals_um,
                s=point_size,
                norm=norm,
                cmap="coolwarm",
                linewidths=0,
                rasterized=True,
            )

            colorbar = fig.colorbar(scatter_artist, ax=ax)
            colorbar.set_label("Distance to fitted plane [µm]")

            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_title(
                f"{result.name} plane residuals\n"
                f"RMS = {result.plane.rms_mm * 1000.0:.2f} µm"
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

            fig.tight_layout()

            path = output_dir / (
                f"{filename_prefix}sensor_{self._safe_name(result.name)}"
                "_plane_residuals.png"
            )
            fig.savefig(
                path,
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close(fig)

            written.append(path)

        return written

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
