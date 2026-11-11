"""Three-view overlays for nominal-to-as-built ladder registration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from .ladder_registration import RegistrationEvaluation
from .labeled_registration import LabeledRegistrationResult
from .nominal_geometry import NominalLadder, rigid_matrix


class RegistrationVisualizer:
    _FACES = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )

    def __init__(
        self, dpi: int = 200, sample_size: int = 50_000, random_seed: int = 42
    ):
        if dpi <= 0 or sample_size <= 0:
            raise ValueError("Registration plot dpi and sample_size must be positive.")
        self.dpi = int(dpi)
        self.sample_size = int(sample_size)
        self.random_seed = random_seed

    def plot(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        result: LabeledRegistrationResult,
        path: str | Path,
        *,
        ladder_name: str,
        covariances: np.ndarray | None = None,
        facet_z_mm: float = 0.0,
    ) -> Path:
        expanded = self._expanded_facets(
            geometry, result, covariances, facet_z_mm
        )
        status = (
            "accepted local fit"
            if result.accepted
            else "NOT ACCEPTED — diagnostic candidate"
        )
        title = (
            f"{ladder_name}: {status}\n"
            f"Surface RMS {result.initial.rms_mm * 1000:.2f} → {result.final.rms_mm * 1000:.2f} µm; "
            f"surface rank {result.surface_jacobian_rank}/6; frame: {geometry.frame}"
        )
        return self._plot(
            geometry,
            points,
            result.rotation,
            result.translation_mm,
            result.final.residuals_mm,
            path,
            title=title,
            pose_label="Fitted sensor volumes",
            residual_scale=1000.0,
            residual_unit="µm",
            sensor_indices=result.final.sensor_indices,
            expanded_corners=expanded,
        )

    @staticmethod
    def _expanded_facets(geometry, result, covariances, facet_z_mm):
        """Plot-only outer envelope of the assigned-face containment bands.

        Per-point uncertainty varies; max inlier padding per sensor makes this
        a visual outer envelope, never an alternative acceptance test.
        """
        labels = result.final.sensor_indices
        inliers = result.final.inlier_mask
        expanded = []
        for index, sensor in enumerate(geometry.sensors):
            axes = result.rotation @ sensor.rotation
            padding = np.zeros(3)
            if covariances is not None:
                selected = covariances[(labels == index) & inliers]
                variance = np.einsum("ia,nij,ja->na", axes, selected, axes)
                padding = result.options.uncertainty_sigma * np.sqrt(
                    np.maximum(variance, 0).max(axis=0)
                )
            allowance = padding + result.options.containment_tolerance_mm
            halfsize = sensor.dimensions_mm / 2
            local = (sensor.corners_mm - sensor.center_mm) @ sensor.rotation
            local[:, :2] *= (halfsize[:2] + allowance[:2]) / halfsize[:2]
            outward = np.sign((sensor.center_mm[2] - facet_z_mm) * sensor.rotation[2, 2])
            local[:, 2] = outward * halfsize[2] + np.sign(local[:, 2]) * allowance[2]
            expanded.append(local @ axes.T + sensor.center_mm @ result.rotation.T
                            + result.translation_mm)
        return expanded

    def plot_initial(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        evaluation: RegistrationEvaluation,
        initial_transform: np.ndarray,
        path: str | Path,
        *,
        ladder_name: str,
        original_geometry: NominalLadder | None = None,
    ) -> Path:
        transform = rigid_matrix(initial_transform, "preview initial_transform")
        title = (
            f"{ladder_name}: INITIAL POSE — no registration performed\n"
            f"Surface RMS {evaluation.rms_mm:.2f} mm; points {len(points):,}; "
            f"geometry frame: {geometry.frame}\n"
            "Measured points use the saved detector transform; physical frame equivalence is not verified"
        )
        return self._plot(
            geometry,
            points,
            transform[:3, :3],
            transform[:3, 3],
            evaluation.residuals_mm,
            path,
            title=title,
            pose_label=(
                "Optically corrected sensor volumes"
                if original_geometry is not None
                else "Initial sensor volumes"
            ),
            residual_scale=1.0,
            residual_unit="mm",
            original_geometry=original_geometry,
        )

    def _plot(
        self,
        geometry: NominalLadder,
        points: np.ndarray,
        rotation: np.ndarray,
        translation_mm: np.ndarray,
        residuals_mm: np.ndarray,
        path: str | Path,
        *,
        title: str,
        pose_label: str,
        residual_scale: float,
        residual_unit: str,
        original_geometry: NominalLadder | None = None,
        sensor_indices: np.ndarray | None = None,
        expanded_corners: list[np.ndarray] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        indices = np.arange(len(points))
        if len(points) > self.sample_size:
            indices = np.sort(
                np.random.default_rng(self.random_seed).choice(
                    len(points), self.sample_size, replace=False
                )
            )
        shown = points[indices]
        residual_values = np.abs(residuals_mm[indices]) * residual_scale
        color_limit = max(float(np.max(np.abs(residuals_mm))) * residual_scale, 1.0)
        sensor_colors = None
        point_colors = residual_values
        if sensor_indices is not None:
            labels = np.asarray(sensor_indices)
            if (labels.shape != (len(points),)
                    or not np.issubdtype(labels.dtype, np.integer)
                    or np.any(labels < 0) or np.any(labels >= len(geometry.sensors))):
                raise ValueError("Registration plot requires a valid sensor label per point")
            palette = np.asarray(plt.get_cmap("tab20").colors)
            sensor_colors = palette[np.arange(len(geometry.sensors)) % len(palette)]
            point_colors = sensor_colors[labels[indices]]
        if original_geometry is not None:
            original_by_node = {s.node_path: s for s in original_geometry.sensors}
            if set(original_by_node) != {s.node_path for s in geometry.sensors}:
                raise ValueError("Original ROOT geometry must contain the corrected sensor nodes")
            nominal = [original_by_node[s.node_path].corners_mm for s in geometry.sensors]
        else:
            nominal = [sensor.corners_mm for sensor in geometry.sensors]
        fitted = [s.corners_mm @ rotation.T + translation_mm for s in geometry.sensors]
        # Limits and colour scale use the full cloud, not only displayed samples.
        extent = np.vstack(
            (
                *nominal,
                *fitted,
                *(expanded_corners or []),
                np.min(points, axis=0),
                np.max(points, axis=0),
            )
        )
        low, high = np.min(extent, axis=0), np.max(extent, axis=0)
        spans = np.maximum(high - low, 0.1)
        middle = (low + high) / 2
        fig = plt.figure(figsize=(16, 6))
        try:
            grid = fig.add_gridspec(
                1,
                4,
                width_ratios=(1, 1, 1.2, 0.045),
                left=0.05,
                right=0.94,
                bottom=0.16,
                top=0.80,
                wspace=0.5,
            )
            axes = [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[0, 2], projection="3d"),
            ]
            colorbar_axis = fig.add_subplot(grid[0, 3])
            for ax, dimensions, view_title in zip(
                axes[:2], ((0, 1), (1, 2)), ("XY", "YZ")
            ):
                for sensor_index, (corners, moved) in enumerate(zip(nominal, fitted)):
                    volume_color = ("tab:blue" if sensor_colors is None
                                    else sensor_colors[sensor_index])
                    for box, is_nominal in ((corners, True), (moved, False)):
                        projection = box[:, dimensions]
                        polygon = projection[ConvexHull(projection).vertices]
                        ax.add_collection(
                            PolyCollection(
                                [polygon],
                                facecolors="none" if is_nominal else volume_color,
                                edgecolors=("0.4" if sensor_colors is None and is_nominal
                                            else volume_color),
                                alpha=0.65 if is_nominal else 0.22,
                                linestyles="--" if is_nominal else "-",
                                linewidths=0.8,
                            )
                        )
                    if expanded_corners is not None:
                        outline = expanded_corners[sensor_index][:, dimensions]
                        polygon = outline[ConvexHull(outline).vertices]
                        ax.add_collection(PolyCollection(
                            [polygon], facecolors="none", edgecolors=volume_color,
                            linestyles=":", linewidths=1.2,
                        ))
                ax.scatter(
                    shown[:, dimensions[0]],
                    shown[:, dimensions[1]],
                    c=point_colors,
                    **({"cmap": "magma", "vmin": 0, "vmax": color_limit}
                       if sensor_colors is None else {}),
                    s=3,
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_xlabel(f"{'XYZ'[dimensions[0]]} [mm]")
                ax.set_ylabel(f"{'XYZ'[dimensions[1]]} [mm]")
                ax.set_title(view_title)
                ax.set_xlim(
                    middle[dimensions[0]] - spans[dimensions[0]] * 0.55,
                    middle[dimensions[0]] + spans[dimensions[0]] * 0.55,
                )
                ax.set_ylim(
                    middle[dimensions[1]] - spans[dimensions[1]] * 0.55,
                    middle[dimensions[1]] + spans[dimensions[1]] * 0.55,
                )
                if view_title == "XY":
                    ax.set_aspect("equal", adjustable="box")
                else:
                    ax.text(
                        0.02,
                        0.98,
                        "Independent Y/Z display scales",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
                ax.grid(alpha=0.2)
            ax3d = axes[2]
            for sensor_index, (corners, moved) in enumerate(zip(nominal, fitted)):
                volume_color = ("tab:blue" if sensor_colors is None
                                else sensor_colors[sensor_index])
                ax3d.add_collection3d(
                    Poly3DCollection(
                        [moved[list(face)] for face in self._FACES],
                        facecolors=volume_color,
                        edgecolors=volume_color,
                        alpha=0.18,
                        linewidths=0.4,
                    )
                )
                for face in self._FACES:
                    line = corners[list(face) + [face[0]]]
                    ax3d.plot(
                        *line.T,
                        color="0.4" if sensor_colors is None else volume_color,
                        linestyle="--", linewidth=0.5, alpha=0.65
                    )
                if expanded_corners is not None:
                    corners = expanded_corners[sensor_index]
                    for vertex in range(8):
                        for bit in (1, 2, 4):
                            if not vertex & bit:
                                ax3d.plot(*corners[[vertex, vertex | bit]].T,
                                          color=volume_color, linestyle=":",
                                          linewidth=0.8)
            scatter = ax3d.scatter(
                *shown.T,
                c=point_colors,
                **({"cmap": "magma", "vmin": 0, "vmax": color_limit}
                   if sensor_colors is None else {}),
                s=3,
                linewidths=0,
                rasterized=True,
            )

            for dimension, setter in enumerate(
                (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim)
            ):
                setter(
                    middle[dimension] - spans[dimension] * 0.55,
                    middle[dimension] + spans[dimension] * 0.55,
                )
            display_spans = spans.copy()
            display_spans[2] = max(spans[2], float(np.max(spans[:2])) * 0.15)
            ax3d.set_box_aspect(display_spans)
            ax3d.set_xticks(np.linspace(low[0], high[0], 3))
            ax3d.set_yticks(np.linspace(low[1], high[1], 5))
            ax3d.set_zticks(np.linspace(low[2], high[2], 3))
            ax3d.tick_params(labelsize=8, pad=0)
            if display_spans[2] > spans[2]:
                ax3d.text2D(
                    0.5,
                    0.98,
                    "Z display scale expanded",
                    transform=ax3d.transAxes,
                    ha="center",
                    fontsize=8,
                )
            ax3d.set_xlabel("X [mm]")
            ax3d.set_ylabel("Y [mm]")
            ax3d.set_zlabel("Z [mm]")
            ax3d.set_title("Perspective")
            if sensor_colors is None:
                fig.colorbar(
                    scatter,
                    cax=colorbar_axis,
                    label=f"Distance to sensor surface [{residual_unit}]",
                )
            else:
                colorbar_axis.axis("off")
            fig.suptitle(title)
            fig.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color="0.4" if sensor_colors is None else sensor_colors[0],
                        linestyle="--",
                        label=("Corrected nominal sensor outlines"
                               if sensor_colors is not None else
                               "Original ROOT geometry" if original_geometry is not None
                               else "Nominal geometry"),
                    ),
                    *([Line2D(
                        [], [], color=sensor_colors[0], linestyle=":",
                        label="Expanded assigned facet (max inlier uncertainty; visual only)",
                    )] if expanded_corners is not None else []),
                    Patch(
                        facecolor="tab:blue" if sensor_colors is None else sensor_colors[0],
                        alpha=0.22,
                        label=pose_label,
                    ),
                    Line2D(
                        [],
                        [],
                        marker=".",
                        linestyle="none",
                        color="black" if sensor_colors is None else sensor_colors[0],
                        label=("Measured surface points" if sensor_colors is None
                               else "Points coloured by assigned sensor"),
                    ),
                ],
                loc="lower center",
                ncol=3,
            )

            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
        return path
