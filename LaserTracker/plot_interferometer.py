#!/usr/bin/env python3
"""Animate a plane-mirror interferometer as the reflector moves by ±λ/4.

The model shows monochromatic incident and reflected electric fields,
their longitudinal standing-wave intensity, and the brightness measured
at a fixed sensor plane.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MOVE_LIMIT_LAMBDA = 0.5
DEFAULT_FRAMES = 100
DEFAULT_FPS = 10
FIELD_X_MIN_LAMBDA = -3.0
FIELD_X_MAX_LAMBDA = +3.0
DEFAULT_SENSOR_POSITION_LAMBDA = -1.0
PATTERN_DISPLAY_ROWS = 50
RENDER_DPI = 110


@dataclass(frozen=True)
class RenderConfig:
    frames: int
    fps: int
    reflectivity: float
    reflection_phase: float
    sensor_position_lambda: float


def normalized_interference_intensity(
    phase_difference: np.ndarray,
    reflectivity: float,
) -> np.ndarray:
    """Return normalized two-beam intensity for a given phase difference."""
    intensity = (
        1.0 + reflectivity**2
        + 2.0 * reflectivity * np.cos(phase_difference)
    ) / (1.0 + reflectivity) ** 2
    return np.clip(intensity, 0.0, 1.0)


def fields_and_intensity(
    x_lambda: np.ndarray,
    mirror_lambda: float,
    reflectivity: float,
    reflection_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return incident E, reflected E, and normalized time-averaged intensity.

    x_lambda and mirror_lambda are positions in units of the optical wavelength.
    The incident field has unit amplitude. The reflected field amplitude is
    reflectivity. reflection_phase is the phase shift at the mirror in radians.
    """
    kx = 2.0 * np.pi * x_lambda
    incident = np.cos(kx)
    reflected = reflectivity * np.cos(
        2.0 * np.pi * (2.0 * mirror_lambda - x_lambda) + reflection_phase
    )

    phase_difference = (
        4.0 * np.pi * (mirror_lambda - x_lambda) + reflection_phase
    )
    intensity = normalized_interference_intensity(
        phase_difference, reflectivity
    )
    return incident, reflected, intensity



def create_figure(config: RenderConfig):
    """Create one worker-local figure and its frame update function."""
    x_lambda = np.linspace(FIELD_X_MIN_LAMBDA, FIELD_X_MAX_LAMBDA, 1800)
    frames = np.arange(config.frames)
    mirror_positions = MOVE_LIMIT_LAMBDA * np.sin(
        2.0 * np.pi * frames / (config.frames - 1)
    )

    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(3.0, 1.2),
        hspace=0.32,
        wspace=0.28,
    )
    ax_fields = fig.add_subplot(grid[:, :2])
    ax_intensity = fig.add_subplot(grid[0, 2])
    ax_sensor = fig.add_subplot(grid[1, 2])
    fig.suptitle(
        "Interference from a moving plane reflector",
        fontsize=15,
    )

    mirror_color = "tab:orange"
    incident_color = "tab:blue"
    reflected_color = "tab:red"

    # Panel 1: snapshot of the two traveling electric fields.
    incident_line, = ax_fields.plot(
        [], [], color=incident_color, lw=1.8, label="Emitted wave"
    )
    reflected_line, = ax_fields.plot(
        [], [], color=reflected_color, lw=1.8, ls="--", label="Reflected wave"
    )
    # Requested plane-travel limits: dotted and the same color as the mirror.
    ax_fields.axvline(
        -MOVE_LIMIT_LAMBDA,
        color=mirror_color,
        ls=":",
        lw=1.8,
        label="Plane travel limits (±λ/4)",
    )
    ax_fields.axvline(
        MOVE_LIMIT_LAMBDA,
        color=mirror_color,
        ls=":",
        lw=1.8,
    )
    mirror_line = ax_fields.axvline(
        0.0, color=mirror_color, lw=3.0, label="Reflective plane"
    )
    ax_fields.set_xlim(x_lambda.min(), x_lambda.max())
    ax_fields.set_ylim(-1.25, 1.25)
    ax_fields.set_xlabel("Position from mean plane, x / λ")
    ax_fields.set_ylabel("Electric-field amplitude (normalized)")
    ax_fields.set_title("Incident and reflected fields")
    ax_fields.grid(alpha=0.2)
    ax_fields.legend(loc="upper left", fontsize=8)
    field_note = ax_fields.text(
        0.02, 0.02, "",
        transform=ax_fields.transAxes,
        va="bottom",
        fontsize=9,
    )

    # Panel 2: longitudinal standing-wave intensity. Each vertical slice is a
    # plane across the beam; the red line marks the fixed sensor plane.
    pattern_image = ax_intensity.imshow(
        np.zeros((PATTERN_DISPLAY_ROWS, x_lambda.size)),
        extent=(x_lambda.min(), x_lambda.max(), -0.5, 0.5),
        origin="lower",
        aspect="auto",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax_intensity.set_xlim(x_lambda.min(), x_lambda.max())
    ax_intensity.set_ylim(-0.5, 0.5)
    ax_intensity.set_yticks([])
    ax_intensity.set_xlabel("Position along optical axis, x / λ")
    ax_intensity.set_ylabel("Beam cross-section (schematic)")
    ax_intensity.set_title("Longitudinal interference intensity")
    colorbar = fig.colorbar(pattern_image, ax=ax_intensity, pad=0.02)
    colorbar.set_label("Time-averaged intensity (normalized)")
    intensity_note = ax_intensity.text(
        0.02, 0.02, "",
        transform=ax_intensity.transAxes,
        va="bottom",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
    )
    ax_intensity.axvline(
        config.sensor_position_lambda,
        color="red",
        linewidth=3.0,
        zorder=3,
    )
    ax_intensity.text(
        config.sensor_position_lambda,
        0.43,
        "Sensor plane",
        ha="center",
        va="center",
        color="red",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        zorder=4,
    )
    mirror_line_intensity = ax_intensity.axvline(
        0.0,
        color=mirror_color,
        linewidth=3.0,
        zorder=3,
    )
    mirror_label_intensity = ax_intensity.text(
        0.0,
        0.43,
        "Reflector",
        ha="center",
        va="center",
        color=mirror_color,
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        zorder=4,
    )
    blocked_region_note = ax_intensity.text(
        FIELD_X_MAX_LAMBDA / 2.0,
        0.0,
        "No field behind\nreflector",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        zorder=4,
    )

    sensor_display_left = 0.38
    sensor_display_width = 0.20
    sensor_readout_image = ax_sensor.imshow(
        np.zeros((PATTERN_DISPLAY_ROWS, PATTERN_DISPLAY_ROWS)),
        extent=(
            sensor_display_left,
            sensor_display_left + sensor_display_width,
            0.0,
            1.0,
        ),
        origin="lower",
        aspect="auto",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_sensor.set_xlim(0.0, 1.0)
    ax_sensor.set_ylim(0.0, 1.0)
    ax_sensor.set_xticks([])
    ax_sensor.set_yticks([])
    ax_sensor.set_title(
        f"Brightness measured at x = {config.sensor_position_lambda:g} λ"
    )
    for spine in ax_sensor.spines.values():
        spine.set_visible(False)
    ax_sensor.add_patch(
        Rectangle(
            (sensor_display_left, 0.0),
            sensor_display_width,
            1.0,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
            zorder=3,
        )
    )
    sensor_readout_note = ax_sensor.text(
        0.64,
        0.5,
        "",
        transform=ax_sensor.transAxes,
        ha="left",
        va="center",
        color="black",
        fontsize=13,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.90)

    def update(frame: int):
        mirror_lambda = float(mirror_positions[frame])
        incident, reflected, intensity = fields_and_intensity(
            x_lambda,
            mirror_lambda,
            config.reflectivity,
            config.reflection_phase,
        )

        # The light exists in the region between the source side and plane.
        in_front_of_plane = x_lambda <= mirror_lambda
        x_visible = x_lambda[in_front_of_plane]
        incident_line.set_data(x_visible, incident[in_front_of_plane])
        reflected_line.set_data(x_visible, reflected[in_front_of_plane])

        visible_intensity = np.where(in_front_of_plane, intensity, 0.0)
        pattern_image.set_data(
            np.tile(visible_intensity, (PATTERN_DISPLAY_ROWS, 1))
        )
        sensor_phase = (
            4.0
            * np.pi
            * (mirror_lambda - config.sensor_position_lambda)
            + config.reflection_phase
        )
        sensor_brightness = float(
            normalized_interference_intensity(
                np.array([sensor_phase]), config.reflectivity
            )[0]
        )
        sensor_readout_image.set_data(
            np.full(
                (PATTERN_DISPLAY_ROWS, PATTERN_DISPLAY_ROWS),
                sensor_brightness,
            )
        )
        sensor_readout_note.set_text(
            f"Sensor signal:\n{sensor_brightness:.3f}"
        )

        mirror_line.set_xdata([mirror_lambda, mirror_lambda])
        mirror_line_intensity.set_xdata([mirror_lambda, mirror_lambda])
        mirror_label_intensity.set_position((mirror_lambda, 0.43))
        blocked_region_note.set_position(
            ((mirror_lambda + FIELD_X_MAX_LAMBDA) / 2.0, 0.0)
        )

        displacement = mirror_lambda
        field_note.set_text(
            f"Reflector displacement: {displacement:+.3f} λ"
        )
        intensity_note.set_text(
            "Standing-wave spacing: λ/2\n"
            f"Mirror phase change: {4.0 * np.pi * displacement:+.2f} rad"
        )
        return (
            incident_line,
            reflected_line,
            pattern_image,
            sensor_readout_image,
            mirror_line,
            mirror_line_intensity,
            mirror_label_intensity,
            blocked_region_note,
            field_note,
            intensity_note,
            sensor_readout_note,
        )

    return fig, update


def partition_frames(frame_count: int, worker_count: int) -> list[tuple[int, ...]]:
    """Split ordered frame indices into balanced contiguous worker chunks."""
    chunk_size, remainder = divmod(frame_count, worker_count)
    chunks: list[tuple[int, ...]] = []
    start = 0
    for worker_index in range(worker_count):
        stop = start + chunk_size + (worker_index < remainder)
        chunks.append(tuple(range(start, stop)))
        start = stop
    return chunks


def render_frame_chunk(
    config: RenderConfig,
    frame_indices: tuple[int, ...],
    temporary_directory: str,
) -> int:
    """Render one frame chunk in a worker process using one reusable figure."""
    fig, update = create_figure(config)
    try:
        for frame_index in frame_indices:
            update(frame_index)
            frame_path = Path(temporary_directory) / f"frame_{frame_index:06d}.png"
            fig.savefig(frame_path, dpi=RENDER_DPI)
    finally:
        plt.close(fig)
    return len(frame_indices)


def assemble_gif(
    frame_directory: Path,
    output_path: Path,
    frame_count: int,
    fps: int,
) -> None:
    """Assemble ordered PNG frames into the final animated GIF."""
    images: list[Image.Image] = []
    try:
        for frame_index in range(frame_count):
            frame_path = frame_directory / f"frame_{frame_index:06d}.png"
            with Image.open(frame_path) as image:
                images.append(image.convert("RGB"))

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / fps)),
            loop=0,
        )
    finally:
        for image in images:
            image.close()


def render_animation(
    config: RenderConfig,
    output_path: Path,
    requested_workers: int,
) -> int:
    """Render frames concurrently and return the worker count used."""
    available_cpus = os.cpu_count() or 1
    automatic_workers = min(available_cpus, 4)
    worker_count = min(
        config.frames,
        requested_workers if requested_workers > 0 else automatic_workers,
    )
    frame_chunks = partition_frames(config.frames, worker_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Rendering {config.frames} frames with {worker_count} worker "
        f"process{'es' if worker_count != 1 else ''}...",
        flush=True,
    )
    with TemporaryDirectory(
        prefix="interferometer_frames_", dir=output_path.parent
    ) as temporary_directory:
        if worker_count == 1:
            render_frame_chunk(config, frame_chunks[0], temporary_directory)
            print(f"Rendered {config.frames}/{config.frames} frames", flush=True)
        else:
            completed_frames = 0
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        render_frame_chunk,
                        config,
                        frame_chunk,
                        temporary_directory,
                    ): len(frame_chunk)
                    for frame_chunk in frame_chunks
                }
                for future in as_completed(futures):
                    completed_frames += future.result()
                    print(
                        f"Rendered {completed_frames}/{config.frames} frames",
                        flush=True,
                    )

        assemble_gif(
            Path(temporary_directory),
            output_path,
            config.frames,
            config.fps,
        )
    return worker_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate interference from a plane reflector moving ±λ/4."
    )
    parser.add_argument(
        "--output",
        default="moving_plane_interference.gif",
        help="Output GIF path (default: moving_plane_interference.gif).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of animation frames (default: {DEFAULT_FRAMES}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"GIF frames per second (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (default: 0 selects up to 4 automatically; 1 disables concurrency).",
    )
    parser.add_argument(
        "--reflectivity",
        type=float,
        default=1.0,
        help="Reflected electric-field amplitude relative to incident field (0–1).",
    )
    parser.add_argument(
        "--reflection-phase-deg",
        type=float,
        default=180.0,
        help="Reflection phase shift in degrees (default: 180 for an ideal conductor).",
    )
    parser.add_argument(
        "--sensor-position-lambda",
        type=float,
        default=DEFAULT_SENSOR_POSITION_LAMBDA,
        help=(
            "Sensor-plane position along the optical axis in wavelengths "
            f"(default: {DEFAULT_SENSOR_POSITION_LAMBDA:g})."
        ),
    )
    args = parser.parse_args()

    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.workers < 0:
        parser.error("--workers must be zero or greater")
    if not 0.0 <= args.reflectivity <= 1.0:
        parser.error("--reflectivity must be between 0 and 1")
    if not (
        FIELD_X_MIN_LAMBDA
        <= args.sensor_position_lambda
        <= -MOVE_LIMIT_LAMBDA
    ):
        parser.error(
            "--sensor-position-lambda must stay between the plotted source side "
            f"({FIELD_X_MIN_LAMBDA:g} λ) and the nearest reflector position "
            f"({-MOVE_LIMIT_LAMBDA:g} λ)"
        )

    config = RenderConfig(
        frames=args.frames,
        fps=args.fps,
        reflectivity=args.reflectivity,
        reflection_phase=np.deg2rad(args.reflection_phase_deg),
        sensor_position_lambda=args.sensor_position_lambda,
    )
    output_path = Path(args.output).expanduser()
    worker_count = render_animation(config, output_path, args.workers)

    print(f"Saved animation: {output_path.resolve()}")
    print(f"Worker processes used: {worker_count}")
    print("Reflector travel: -0.25 λ to +0.25 λ")
    print("Total travel is λ/2, producing one full intensity fringe at a fixed point.")


if __name__ == "__main__":
    main()

