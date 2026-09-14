# Detector Alignment

`detector_alignment` computes a rigid transform from scanner coordinates into a nominal detector coordinate system using:

1. a measured C-frame facet to determine plane tilt and the detector `Z=0` plane;
2. two or more **directly measured reference-hole center points** to determine in-plane yaw and XY translation;
3. measurement covariance propagated into the final six-parameter rigid-transform covariance.

The transform convention is:

```text
p_aligned = R @ p + T
```

All lengths are millimetres. Rotation-vector covariance uses radians.

## Installation

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Development/test dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
`detector-align process examples/config.yaml
````

or:

```bash
python -m detector_alignment.cli process examples/config.yaml
```

## Inputs

### C-frame point cloud

`cframe.file` is an ASCII point cloud. Numeric rows require at least `X Y Z`.
Comment/header prefixes default to:

```text
#
//
$
```

Extra numeric columns are preserved by the streaming transform writer and may also carry uncertainty values.

### Direct reference centers

Each entry in `references` pairs one measurement file with one nominal detector-plane coordinate:

```yaml
references:
  - file: data/reference_01.xyz
    nominal_center: [0.0, 0.0]
  - file: data/reference_02.xyz
    nominal_center: [0.0, 500.0]
```

Each reference file must contain **exactly one numeric XYZ row**: the scanner-measured center of that reference hole. The package does not fit hole openings, radii, circles, or boundary point clouds.

`reference_holes` is accepted as a compatibility alias for `references`, but new configurations should use `references`.

At least two distinct measured references and two distinct nominal references are required. Their list order establishes correspondence; there is no artificial requirement that reference 2 have larger nominal Y than reference 1.

## Measurement uncertainty

Supported uncertainty modes are:

- `none`
- `constant` with `sigma_xyz_mm: [sx, sy, sz]`
- `columns_sigma` with three zero-based sigma-column indices
- `columns_covariance` with six zero-based columns `cxx cyy czz cxy cxz cyz`
- `file_sigma` with a row-aligned three-column sigma file
- `file_covariance` with a row-aligned six-column covariance file

The global `uncertainty` block is the default for the C-frame and references. A `cframe` or individual reference entry may override it.

If a reference uses `mode: none`, its conditional point covariance is treated as zero; the shared plane uncertainty is still propagated into its leveled XY coordinates.

## Plane fit

The C-frame facet is estimated using:

1. robust random plane hypotheses;
2. adaptive residual classification;
3. orthogonal SVD refinement;
4. robust nonlinear least squares;
5. covariance from the refined Jacobian.

The plane parameterization is `[theta, phi, d]`:

```text
n = [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]
n dot p + d = 0
```

The minimal rotation maps `n` to `+Z`. After that rotation the fitted plane is at `z = -d`, so the final transform uses `tz = d` to place it at `z = 0`.

## Reference processing and registration

For each scanner-measured XYZ center, the fitted plane rotation is applied and the leveled XY coordinate is retained. The original point covariance is projected into this leveled XY system. A numerical derivative captures sensitivity of each reference XY position to the shared fitted-plane orientation.

The leveled measured centers are then registered to their paired nominal XY coordinates with a 2D rigid least-squares fit. The final 3D rotation is:

```text
R = Rz(yaw) @ R_plane
```

and the final translation is:

```text
T = [tx, ty, d]
```

## Covariance propagation

The joint state is:

```text
q = [theta, phi, d, ref1_x, ref1_y, ref2_x, ref2_y, ...]
```

The joint covariance includes:

- plane-parameter covariance;
- each reference's conditional measurement covariance;
- plane/reference cross-covariance;
- reference/reference cross-covariance induced by the common plane fit.

It is propagated numerically into final transform parameters ordered as:

```text
[rx, ry, rz, tx, ty, tz]
```

where `[rx, ry, rz]` is a rotation vector in radians and translation is in millimetres.

## Outputs

The output directory contains:

```text
results.json
rotation_matrix.npy
translation_vector.npy
transform_matrix.npy
covariance_transform_params.npy
covariance_joint.npy
covariance_plane_params.npy
covariance_reference_XX_conditional.npy
cframe_plane_alignment.png
cframe_plane_alignment_top_view.png
cframe_plane_alignment_residual_histogram.png
cframe_final_alignment.png
cframe_final_alignment_top_view.png
cframe_final_alignment_residual_histogram.png
*_transformed.xyz
```

The configured C-frame, reference files, and optional ladder point clouds are transformed with the same final rigid transform. Visualization samples are bounded for large clouds; numerical fitting uses the full loaded C-frame except for bounded internal RANSAC hypothesis/scoring pools.

## Scope

Ladder-specific analysis is not implemented. `ladder_files` are transformed only.

Hole-opening/circle fitting has been removed from the project because reference centers are supplied directly by the measurement system.
