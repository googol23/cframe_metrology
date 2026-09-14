from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.spatial import ConvexHull


class AlignmentVisualizer:
    """
    Visualization utilities for detector alignment processing.

    Notes
    -----
    - Plotting uses a random subsample of the input point cloud to keep
      visualization responsive for large data sets.
    - All saved figures use the configured DPI.
    - Plane-alignment visualization displays the transformed / leveled
      C-frame point cloud and reference centers when supplied.
    - Point-cloud markers are drawn with transparency.
    - The fitted plane is shown explicitly.
    - A residual diagnostic panel shows signed point-to-plane distance
      as a function of X.
    - The residual panel includes a transparent yellow ±1 sigma band,
      where sigma is estimated from fitted-plane inlier residuals.
    - A separate histogram of fitted-plane inlier residuals is saved
      alongside the main diagnostic plot.
    """

    def __init__(
        self,
        dpi: int = 600,
        sample_size: int = 50_000,
        random_seed: int = 42,
        point_alpha: float = 0.25,
        plane_alpha: float = 0.18,
        sigma_band_alpha: float = 0.22,
    ):
        self.dpi = int(dpi)
        self.sample_size = int(sample_size)
        self.rng = np.random.default_rng(random_seed)

        self.point_alpha = float(point_alpha)
        self.plane_alpha = float(plane_alpha)
        self.sigma_band_alpha = float(sigma_band_alpha)

        if not 0.0 <= self.point_alpha <= 1.0:
            raise ValueError("point_alpha must be between 0 and 1.")

        if not 0.0 <= self.plane_alpha <= 1.0:
            raise ValueError("plane_alpha must be between 0 and 1.")

        if not 0.0 <= self.sigma_band_alpha <= 1.0:
            raise ValueError("sigma_band_alpha must be between 0 and 1.")

    # ==================================================================
    # Sampling utilities
    # ==================================================================

    def _sample(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Return a random sample of the supplied points.

        If the number of points is smaller than ``sample_size``,
        all points are returned.
        """
        points = np.asarray(points)

        if len(points) <= self.sample_size:
            return points

        indices = self.rng.choice(
            len(points),
            self.sample_size,
            replace=False,
        )

        return points[indices]

    def _sample_indices(
        self,
        n_points: int,
    ) -> np.ndarray:
        """
        Return indices for synchronized visualization sampling.

        This is used when several arrays must remain aligned, for
        example XYZ coordinates, residuals and the inlier mask.
        """
        if n_points <= self.sample_size:
            return np.arange(n_points)

        return self.rng.choice(
            n_points,
            self.sample_size,
            replace=False,
        )

    # ==================================================================
    # Axis utilities
    # ==================================================================

    @staticmethod
    def _set_equal_2d_limits(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        padding_fraction: float = 0.03,
    ) -> None:
        """
        Set equal geometric scaling for a 2D point-cloud view.
        """
        xmin = float(np.min(x))
        xmax = float(np.max(x))

        ymin = float(np.min(y))
        ymax = float(np.max(y))

        x_span = xmax - xmin
        y_span = ymax - ymin

        span = max(
            x_span,
            y_span,
        )

        if span <= 0.0:
            span = 1.0

        padding = padding_fraction * span

        half_span = 0.5 * span + padding

        x_center = 0.5 * (xmin + xmax)

        y_center = 0.5 * (ymin + ymax)

        ax.set_xlim(
            x_center - half_span,
            x_center + half_span,
        )

        ax.set_ylim(
            y_center - half_span,
            y_center + half_span,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    # ==================================================================
    # Plane footprint utility
    # ==================================================================

    @staticmethod
    def _plane_footprint(
        xy: np.ndarray,
    ) -> np.ndarray | None:
        """
        Compute the 2D convex-hull footprint of fitted-plane inliers.

        Parameters
        ----------
        xy
            Inlier XY coordinates after leveling.

        Returns
        -------
        np.ndarray | None
            Ordered hull coordinates with shape (M, 2), or None if a
            valid hull cannot be constructed.
        """
        xy = np.asarray(
            xy,
            dtype=float,
        )

        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
            return None

        xy_unique = np.unique(
            xy,
            axis=0,
        )

        if len(xy_unique) < 3:
            return None

        try:
            hull = ConvexHull(xy_unique)
        except Exception:
            return None

        return xy_unique[hull.vertices]

    # ==================================================================
    # Histogram utilities
    # ==================================================================

    @staticmethod
    def _histogram_bin_count(
        values: np.ndarray,
        minimum_bins: int = 30,
        maximum_bins: int = 250,
    ) -> int:
        """
        Determine histogram bin count using the Freedman-Diaconis rule.

        The final number of bins is limited to a practical range to
        avoid excessively fine histograms for million-point datasets.
        """
        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if len(values) < 2:
            return minimum_bins

        q25, q75 = np.percentile(
            values,
            [25.0, 75.0],
        )

        iqr = float(q75 - q25)

        data_min = float(np.min(values))

        data_max = float(np.max(values))

        data_range = data_max - data_min

        if iqr <= 0.0 or data_range <= 0.0:
            return 50

        bin_width = 2.0 * iqr / np.cbrt(len(values))

        if not np.isfinite(bin_width) or bin_width <= 0.0:
            return 50

        n_bins = int(np.ceil(data_range / bin_width))

        return int(
            np.clip(
                n_bins,
                minimum_bins,
                maximum_bins,
            )
        )

    # ==================================================================
    # C-frame plane visualization
    # ==================================================================

    def plot_plane_alignment(
        self,
        points: np.ndarray,
        R: np.ndarray,
        result,
        path: str | Path,
        translation: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
        title: str = "C-frame plane leveling",
    ) -> None:
        """
        Plot the rotated C-frame point cloud using three diagnostic views.

        Output figures
        --------------
        TOP VIEW:
            Standalone leveled X-Y view. Points are colored by their
            signed distance to the fitted plane. Saved as:

                <output stem>_top_view.png

        DIAGNOSTICS:
            The requested output path contains:

            - Lateral X-Z view with fitted plane.
            - Lateral Y-Z view with fitted plane.
            - Signed point-to-plane residual versus X.
            - Plane-fit statistics.

            The X-Z and residual frames have no vertical gap.

        HISTOGRAM:
            A third PNG is automatically created containing the residual
            histogram:

                <output stem>_residual_histogram.png


        Residual definition
        -------------------
        The fitted plane is:

            n . p + d = 0

        After applying a rigid transform p' = R @ p + t, where R maps
        n onto +Z, the plane height is transformed consistently. The
        signed perpendicular residual remains:

            residual = n . p + d

        because rigid transformations preserve distances.
        """

        points = np.asarray(
            points,
            dtype=float,
        )

        R = np.asarray(
            R,
            dtype=float,
        )

        if translation is None:
            translation = np.zeros(3, dtype=float)
        else:
            translation = np.asarray(
                translation,
                dtype=float,
            )

        if reference_points is not None:
            reference_points = np.asarray(
                reference_points,
                dtype=float,
            )

        # --------------------------------------------------------------
        # Validate input
        # --------------------------------------------------------------

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")

        if R.shape != (3, 3):
            raise ValueError("R must have shape (3, 3).")

        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,).")

        if reference_points is not None and (
            reference_points.ndim != 2 or reference_points.shape[1] != 3
        ):
            raise ValueError("reference_points must have shape (N, 3).")

        inlier_mask = np.asarray(
            result.inlier_mask,
            dtype=bool,
        )

        if len(inlier_mask) != len(points):
            raise ValueError("result.inlier_mask must have the same length as points.")

        # ==============================================================
        # Transform complete cloud
        # ==============================================================

        leveled_points = points @ R.T + translation

        transformed_references = None
        if reference_points is not None:
            transformed_references = reference_points @ R.T + translation

        leveled_inliers = leveled_points[inlier_mask]

        if len(leveled_inliers) < 3:
            raise ValueError("Plane fit contains fewer than 3 inlier points.")

        # ==============================================================
        # Plane parameters
        # ==============================================================

        theta, phi, d = result.params

        normal = np.asarray(
            result.normal,
            dtype=float,
        )

        transformed_normal = R @ normal
        transformed_d = float(d) - float(transformed_normal @ translation)
        plane_z = -transformed_d / float(transformed_normal[2])

        # ==============================================================
        # Signed residuals
        # ==============================================================

        all_residuals = points @ normal + float(d)

        inlier_residuals = all_residuals[inlier_mask]

        finite_inlier_residuals = inlier_residuals[np.isfinite(inlier_residuals)]

        if len(finite_inlier_residuals) < 1:
            raise ValueError("No finite plane residuals available.")

        # ==============================================================
        # Residual statistics
        # ==============================================================
        residual_mean = float(np.mean(finite_inlier_residuals))
        residual_median = float(np.median(finite_inlier_residuals))

        if len(finite_inlier_residuals) > 1:
            residual_sigma = float(
                np.std(
                    finite_inlier_residuals,
                    ddof=1,
                )
            )
        else:
            residual_sigma = 0.0

        residual_mad = float(
            np.median(np.abs(finite_inlier_residuals - residual_median))
        )

        rms = float(result.rms_mm)

        # ==============================================================
        # Visualization sampling
        # ==============================================================

        sample_indices = self._sample_indices(len(leveled_points))

        p = leveled_points[sample_indices]

        sampled_residuals = all_residuals[sample_indices]

        # Use a robust symmetric color range centered on the fitted
        # plane. Limiting the range to the 98th percentile prevents a
        # few extreme values from reducing the contrast of most points.
        absolute_inlier_residuals = np.abs(finite_inlier_residuals)

        residual_color_limit = float(
            np.percentile(
                absolute_inlier_residuals,
                98.0,
            )
        )

        # TwoSlopeNorm requires a finite, nonzero range around zero.
        if not np.isfinite(residual_color_limit) or residual_color_limit <= 0.0:
            residual_color_limit = float(np.finfo(float).eps)

        residual_colormap = LinearSegmentedColormap.from_list(
            "plane_residual",
            [
                "#0057ff",  # Negative residual
                "#00b83f",  # On the fitted plane
                "#e60026",  # Positive residual
            ],
        )

        residual_color_norm = TwoSlopeNorm(
            vmin=-residual_color_limit,
            vcenter=0.0,
            vmax=residual_color_limit,
        )

        sampled_inlier_mask = inlier_mask[sample_indices]

        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        ref_x = np.empty(0, dtype=float)
        ref_y = np.empty(0, dtype=float)
        ref_z = np.empty(0, dtype=float)

        if transformed_references is not None:
            ref_x = transformed_references[:, 0]
            ref_y = transformed_references[:, 1]
            ref_z = transformed_references[:, 2]

        # ==============================================================
        # Fitted-plane footprint
        # ==============================================================

        p_inliers = self._sample(leveled_inliers)

        footprint = self._plane_footprint(p_inliers[:, :2])

        # ==============================================================
        # Output paths
        # ==============================================================

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        top_view_path = output_path.parent / (f"{output_path.stem}_top_view.png")

        # ==============================================================
        # TOP-VIEW FIGURE
        # ==============================================================

        fig_top, ax_top = plt.subplots(figsize=(8.5, 8.0))

        if footprint is not None:
            ax_top.fill(
                footprint[:, 0],
                footprint[:, 1],
                alpha=self.plane_alpha,
                label="Fitted plane footprint",
                zorder=1,
            )

        top_scatter = ax_top.scatter(
            x,
            y,
            c=sampled_residuals,
            cmap=residual_colormap,
            norm=residual_color_norm,
            s=0.8,
            alpha=max(
                self.point_alpha,
                0.75,
            ),
            rasterized=True,
            label="Point cloud",
            zorder=2,
        )

        residual_colorbar = fig_top.colorbar(
            top_scatter,
            ax=ax_top,
            fraction=0.046,
            pad=0.04,
            extend="both",
        )

        residual_colorbar.set_label("Signed distance to fitted plane [mm]")

        if transformed_references is not None:
            ax_top.scatter(
                ref_x,
                ref_y,
                marker="*",
                s=90,
                facecolor="yellow",
                edgecolor="black",
                linewidth=0.8,
                label="References",
                zorder=4,
            )
            for i, (ref_point_x, ref_point_y) in enumerate(
                zip(ref_x, ref_y),
                start=1,
            ):
                ax_top.annotate(
                    f"ref_{i}",
                    (ref_point_x, ref_point_y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=5,
                )

        ax_top.set_title("C-frame facet — top view")

        ax_top.set_xlabel("X leveled [mm]")

        ax_top.set_ylabel("Y leveled [mm]")

        limit_x = x if transformed_references is None else np.r_[x, ref_x]
        limit_y = y if transformed_references is None else np.r_[y, ref_y]

        self._set_equal_2d_limits(
            ax_top,
            limit_x,
            limit_y,
        )

        ax_top.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
        )

        ax_top.legend(
            loc="best",
            fontsize=8,
        )

        # ==============================================================
        # Main plane-fit statistics
        # ==============================================================

        n_inliers = int(np.sum(inlier_mask))

        n_total = len(inlier_mask)

        inlier_fraction = n_inliers / n_total if n_total > 0 else np.nan

        plane_fit_text = (
            "Robust plane fit\n"
            "\n"
            f"normal = ["
            f"{normal[0]:.8f}, "
            f"{normal[1]:.8f}, "
            f"{normal[2]:.8f}]\n"
            f"theta = "
            f"{np.degrees(theta):.6f} deg\n"
            f"phi   = "
            f"{np.degrees(phi):.6f} deg\n"
            f"d     = "
            f"{d:.6f} mm\n"
            f"plane Z = "
            f"{plane_z:.6f} mm\n"
            f"RMS   = "
            f"{rms:.6f} mm\n"
            f"sigma = "
            f"{residual_sigma:.6f} mm\n"
            f"inliers = "
            f"{n_inliers}/{n_total} "
            f"({100.0 * inlier_fraction:.2f} %)\n"
            f"plot sample = "
            f"{len(p)} points"
        )

        fig_top.text(
            0.50,
            0.02,
            plane_fit_text,
            ha="center",
            va="bottom",
            fontsize=8,
            family="monospace",
        )

        fig_top.suptitle(
            f"{title} — top view",
            fontsize=12,
        )

        fig_top.subplots_adjust(
            left=0.10,
            right=0.88,
            top=0.92,
            bottom=0.25,
        )

        fig_top.savefig(
            top_view_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close(fig_top)

        # ==============================================================
        # DIAGNOSTICS FIGURE
        # ==============================================================

        fig_diagnostics = plt.figure(figsize=(14, 7.5))

        diagnostics_grid = fig_diagnostics.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[
                1.0,
                1.0,
            ],
            height_ratios=[
                3.0,
                1.15,
            ],
            hspace=0.0,
            wspace=0.18,
        )

        # Each lateral view shares its horizontal coordinate with the
        # residual view directly below it. The zero vertical spacing
        # makes the corresponding axes frames touch.
        ax_side_x = fig_diagnostics.add_subplot(diagnostics_grid[0, 0])

        ax_residual_x = fig_diagnostics.add_subplot(
            diagnostics_grid[1, 0],
            sharex=ax_side_x,
        )

        ax_side_y = fig_diagnostics.add_subplot(diagnostics_grid[0, 1])

        ax_residual_y = fig_diagnostics.add_subplot(
            diagnostics_grid[1, 1],
            sharex=ax_side_y,
        )

        # ==============================================================
        # X-Z LATERAL VIEW
        # ==============================================================

        ax_side_x.scatter(
            x,
            z,
            s=0.35,
            alpha=self.point_alpha,
            rasterized=True,
            label="Point cloud",
            zorder=2,
        )

        if transformed_references is not None:
            ax_side_x.scatter(
                ref_x,
                ref_z,
                marker="*",
                s=70,
                facecolor="yellow",
                edgecolor="black",
                linewidth=0.8,
                label="References",
                zorder=4,
            )
            for i, (ref_point_x, ref_point_z) in enumerate(
                zip(ref_x, ref_z),
                start=1,
            ):
                ax_side_x.annotate(
                    f"ref_{i}",
                    (ref_point_x, ref_point_z),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=5,
                )

        ax_side_x.axhline(
            y=plane_z,
            linewidth=1.5,
            alpha=0.95,
            label=(f"Fitted plane (Z = {plane_z:.4f} mm)"),
            zorder=3,
        )

        # Draw the widest band first. Overlapping transparent yellow
        # regions become progressively darker toward the fitted plane.
        if (
            np.isfinite(residual_sigma)
            and residual_sigma > 0.0
        ):
            for sigma_multiple in (
                3,
                2,
                1,
            ):
                sigma_limit = (
                    sigma_multiple
                    * residual_sigma
                )

                ax_side_x.axhspan(
                    plane_z - sigma_limit,
                    plane_z + sigma_limit,
                    facecolor="yellow",
                    alpha=self.sigma_band_alpha,
                    label=(
                        f"±{sigma_multiple}σ "
                        f"({sigma_limit:.4f} mm)"
                    ),
                    zorder=1,
                )


        ax_side_x.set_title("C-frame facet — X-Z lateral view")
        ax_side_x.set_ylabel("Z leveled [mm]")
        ax_side_x.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
        )

        ax_side_x.legend(
            loc="best",
            fontsize=8,
        )

        # The residual panel immediately below supplies the X labels.
        ax_side_x.tick_params(
            axis="x",
            labelbottom=False,
        )

        # ==============================================================
        # Y-Z LATERAL VIEW
        # ==============================================================

        ax_side_y.scatter(
            y,
            z,
            s=0.35,
            alpha=self.point_alpha,
            rasterized=True,
            label="Point cloud",
            zorder=2,
        )

        if transformed_references is not None:
            ax_side_y.scatter(
                ref_y,
                ref_z,
                marker="*",
                s=70,
                facecolor="yellow",
                edgecolor="black",
                linewidth=0.8,
                label="References",
                zorder=4,
            )
            for i, (ref_point_y, ref_point_z) in enumerate(
                zip(ref_y, ref_z),
                start=1,
            ):
                ax_side_y.annotate(
                    f"ref_{i}",
                    (ref_point_y, ref_point_z),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=5,
                )

        ax_side_y.axhline(
            y=plane_z,
            linewidth=1.5,
            alpha=0.95,
            label=(f"Fitted plane (Z = {plane_z:.4f} mm)"),
            zorder=3,
        )

        # Use the same sigma bands and drawing order as the X-Z view.
        if (
            np.isfinite(residual_sigma)
            and residual_sigma > 0.0
        ):
            for sigma_multiple in (
                3,
                2,
                1,
            ):
                sigma_limit = (
                    sigma_multiple
                    * residual_sigma
                )

                ax_side_y.axhspan(
                    plane_z - sigma_limit,
                    plane_z + sigma_limit,
                    facecolor="yellow",
                    alpha=self.sigma_band_alpha,
                    label=(
                        f"±{sigma_multiple}σ "
                        f"({sigma_limit:.4f} mm)"
                    ),
                    zorder=1,
                )


        ax_side_y.set_title("C-frame facet — Y-Z lateral view")

        ax_side_y.set_ylabel("Z leveled [mm]")

        ax_side_y.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
        )

        ax_side_y.legend(
            loc="best",
            fontsize=8,
        )

        # The residual panel immediately below supplies the Y labels.
        ax_side_y.tick_params(
            axis="x",
            labelbottom=False,
        )

        # ==============================================================
        # RESIDUAL SIGMA BANDS
        # ==============================================================

        if np.isfinite(residual_sigma) and residual_sigma > 0.0:
            # Draw the widest band first. The overlapping transparent
            # yellow regions become progressively darker toward zero:
            #
            #     ±3 sigma: one layer
            #     ±2 sigma: two layers
            #     ±1 sigma: three layers
            for sigma_multiple in (
                3,
                2,
                1,
            ):
                sigma_limit = sigma_multiple * residual_sigma

                for residual_axis in (
                    ax_residual_x,
                    ax_residual_y,
                ):
                    residual_axis.axhspan(
                        -sigma_limit,
                        sigma_limit,
                        facecolor="yellow",
                        alpha=self.sigma_band_alpha,
                        label=(f"±{sigma_multiple}σ ({sigma_limit:.4f} mm)"),
                        zorder=1,
                    )

        # ==============================================================
        # RESIDUALS VERSUS X
        # ==============================================================

        ax_residual_x.scatter(
            x,
            sampled_residuals,
            s=0.35,
            alpha=self.point_alpha,
            rasterized=True,
            label="Point-to-plane residual",
            zorder=2,
        )

        ax_residual_x.axhline(
            y=0.0,
            linewidth=1.2,
            alpha=0.95,
            label="Zero residual",
            zorder=3,
        )

        ax_residual_x.set_xlabel("X leveled [mm]")

        ax_residual_x.set_ylabel("Residual [mm]")

        ax_residual_x.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
        )

        ax_residual_x.legend(
            loc="best",
            fontsize=7,
        )

        # ==============================================================
        # RESIDUALS VERSUS Y
        # ==============================================================

        ax_residual_y.scatter(
            y,
            sampled_residuals,
            s=0.35,
            alpha=self.point_alpha,
            rasterized=True,
            label="Point-to-plane residual",
            zorder=2,
        )

        ax_residual_y.axhline(
            y=0.0,
            linewidth=1.2,
            alpha=0.95,
            label="Zero residual",
            zorder=3,
        )

        ax_residual_y.set_xlabel("Y leveled [mm]")

        ax_residual_y.set_ylabel("Residual [mm]")

        ax_residual_y.grid(
            True,
            linewidth=0.4,
            alpha=0.4,
        )

        ax_residual_y.legend(
            loc="best",
            fontsize=7,
        )

        # Use identical residual limits so the X and Y diagnostics can
        # be compared visually without rescaling.
        finite_sampled_residuals = sampled_residuals[np.isfinite(sampled_residuals)]

        if len(finite_sampled_residuals) > 0:
            residual_plot_limit = float(np.max(np.abs(finite_sampled_residuals)))

            if np.isfinite(residual_sigma) and residual_sigma > 0.0:
                residual_plot_limit = max(
                    residual_plot_limit,
                    3.0 * residual_sigma,
                )

            if residual_plot_limit > 0.0:
                residual_margin = 1.05 * residual_plot_limit

                ax_residual_x.set_ylim(
                    -residual_margin,
                    residual_margin,
                )

                ax_residual_y.set_ylim(
                    -residual_margin,
                    residual_margin,
                )

        fig_diagnostics.suptitle(
            f"{title} — lateral diagnostics",
            fontsize=12,
        )

        fig_diagnostics.subplots_adjust(
            left=0.07,
            right=0.97,
            top=0.91,
            bottom=0.10,
        )

        # ==============================================================
        # Save diagnostics figure
        # ==============================================================

        fig_diagnostics.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close(fig_diagnostics)
