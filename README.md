# Detector Alignment and Ladder Metrology

`detector_alignment` processes laser-tracker / laser-scanner metrology data to determine the rigid transformation from scanner coordinates into the nominal detector coordinate system.

The current workflow uses:

1. a measured C-frame facet to determine the detector plane orientation and `Z = 0`;
2. two or more directly measured reference-hole centers to determine the in-plane translation and rotation;
3. covariance propagation to estimate the uncertainty of the final six-parameter rigid transformation;
4. optional ladder point-cloud processing, where individual sensor surfaces are fitted independently to planes and their surface deviations are visualized.

The global transformation convention is

```text
p_detector = R @ p_scanner + T
```

using the column-vector convention.

All lengths are expressed in **millimetres** unless explicitly stated otherwise. Rotation-vector covariance uses **radians**.

---

## Overview

The processing chain is:

```text
C-frame point cloud
        │
        ▼
robust C-frame plane fit
        │
        ▼
plane normal aligned with detector +Z
        │
        ▼
measured reference centers
        │
        ▼
2D rigid registration to nominal reference positions
        │
        ▼
scanner → detector rigid transformation
        │
        ├──────────────► transformed point clouds
        │
        ▼
optional ladder analysis
        │
        ▼
sensor point clouds
        │
        ▼
independent sensor plane fits
        │
        ▼
surface residual maps and numerical results
```

The C-frame establishes the detector coordinate system. Ladder and sensor measurements are then transformed into that same coordinate system before sensor-specific analysis is performed.

---

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```text
.venv\Scripts\activate
```

Install the package:

```bash
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

---

## Running the pipeline

Using the installed command:

```bash
detector-align process examples/config.yaml
```

or directly through Python:

```bash
python -m detector_alignment.cli process examples/config.yaml
```

Paths in the YAML configuration are resolved relative to the directory containing the configuration file.

---

# Input data

## C-frame point cloud

The C-frame measurement is configured with:

```yaml
cframe:
  file: data/c_frame_facet.txt
```

The file must be an ASCII point cloud containing at least:

```text
X Y Z
```

Additional numeric columns are allowed.

Comment or header lines may be identified using configurable prefixes:

```yaml
io:
  comment_prefixes: ["#", "//", "$"]
  delimiter: whitespace
```

The default parser accepts whitespace-separated numerical data.

---

## Reference points

Each reference measurement file must contain exactly **one measured XYZ point**, representing the measured center of a detector reference feature.

Example:

```yaml
references:
  - file: data/hole_1.xyz
    nominal_center: [0.0, 0.0]

  - file: data/hole_2.xyz
    nominal_center: [0.0, 500.0]
```

`nominal_center` contains the corresponding detector-coordinate position:

```text
[X, Y]
```

At least two distinct measured and nominal reference positions are required.

Their list order defines their correspondence.

For backward compatibility, `reference_holes` is accepted as an alias for `references`.

The software does **not** fit hole radii, circles, or hole-opening point clouds. The reference centers must already be supplied by the measurement system.

---

# Measurement uncertainty

The default measurement uncertainty can be defined globally:

```yaml
uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]
```

Individual C-frame, reference, or ladder measurements may override this configuration.

Supported modes are:

### No measurement uncertainty

```yaml
uncertainty:
  mode: none
```

### Constant XYZ uncertainty

```yaml
uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]
```

### Sigma values stored in point-cloud columns

```yaml
uncertainty:
  mode: columns_sigma
  columns: [3, 4, 5]
```

Column indices are zero-based.

### Covariance stored in point-cloud columns

```yaml
uncertainty:
  mode: columns_covariance
  columns: [3, 4, 5, 6, 7, 8]
```

The expected order is:

```text
Cxx Cyy Czz Cxy Cxz Cyz
```

### Separate sigma file

```yaml
uncertainty:
  mode: file_sigma
  path: uncertainty.xyz
```

### Separate covariance file

```yaml
uncertainty:
  mode: file_covariance
  path: covariance.xyz
```

The uncertainty file must contain the same number of rows as the corresponding point cloud.

---

# C-frame plane fit

The C-frame facet is estimated using a robust plane-fitting procedure consisting of:

1. random plane hypotheses;
2. adaptive residual classification;
3. orthogonal SVD plane refinement;
4. robust nonlinear least-squares refinement;
5. covariance estimation from the refined Jacobian.

Configuration:

```yaml
plane_fit:
  ransac_iterations: 600
  ransac_threshold_mm: 0.05
  random_seed: 23
```

The plane is parameterized by:

```text
[theta, phi, d]
```

with

```text
n = [
    sin(theta) cos(phi),
    sin(theta) sin(phi),
    cos(theta)
]

n · p + d = 0
```

The fitted normal is rotated onto detector `+Z`.

After this rotation, the fitted C-frame plane is translated to:

```text
Z = 0
```

---

# Reference registration

After the C-frame orientation is determined, each measured reference point is rotated into the leveled coordinate system.

Only its XY coordinates are then used for the in-plane registration.

The measured reference coordinates are registered to their nominal detector positions using a 2D rigid least-squares transformation.

The final rotation is:

```text
R = Rz(yaw) @ R_plane
```

and the final translation is:

```text
T = [tx, ty, d]
```

The complete scanner-to-detector transformation is therefore:

```text
p_detector = R @ p_scanner + T
```

---

# Transformation covariance

The joint state used for uncertainty propagation is:

```text
q = [
    theta,
    phi,
    d,
    ref1_x,
    ref1_y,
    ref2_x,
    ref2_y,
    ...
]
```

The joint covariance includes contributions from:

* C-frame plane uncertainty;
* individual reference measurement uncertainty;
* plane/reference correlations;
* correlations between reference coordinates introduced by the common plane fit.

The covariance is propagated into the final transformation parameters ordered as:

```text
[rx, ry, rz, tx, ty, tz]
```

where:

```text
[rx, ry, rz]
```

is a rotation vector in radians and:

```text
[tx, ty, tz]
```

is expressed in millimetres.

---

# Ladder and sensor analysis

Ladder processing is optional.

It is enabled through:

```yaml
ladder_analysis:
  enabled: true
```

The ladder processing pipeline is:

```text
sensor measurement data
        │
        ▼
sensor segmentation / loading
        │
        ▼
scanner → detector transformation
        │
        ▼
independent plane fit for each sensor
        │
        ▼
point-to-plane residual calculation
        │
        ▼
combined and individual XY residual maps
```

All sensors are processed through the same plane-fitting and visualization code regardless of how segmentation is obtained.

---

## Pre-segmented CloudCompare mode

This is the currently operational ladder-analysis mode.

The complete ladder point cloud can be manually segmented using **CloudCompare**, with one ASCII point-cloud file exported per sensor.

CloudCompare should only be used for point selection / segmentation.

The exported sensor files should remain in the original scanner coordinate system. The detector transformation is applied by this package.

Example:

```yaml
ladder_analysis:
  enabled: true

  segmentation:
    mode: segmented_files

  sensors:
    - name: S0
      file: data/L1_000000.txt

    - name: S1
      file: data/L1_000001.txt

    - name: S2
      file: data/L1_000002.txt

    - name: S3
      file: data/L1_000003.txt
```

There is no requirement on the original CloudCompare filenames because the YAML configuration explicitly associates each file with a sensor name.

Sensor names must be unique.

---

## Automatic segmentation mode

The software architecture also defines an automatic segmentation mode:

```yaml
ladder_analysis:
  enabled: true

  segmentation:
    mode: automatic

  file: data/L1.txt
```

The intended processing path is:

```text
complete ladder point cloud
        │
        ▼
automatic sensor segmentation
        │
        ▼
SensorPointCloud objects
        │
        ▼
shared sensor processing pipeline
```

The adapter boundary is implemented in:

```text
src/detector_alignment/cbm_sts_segmenter.py
```

and is intended to connect to the ladder segmentation tools from:

```text
cbm_sts_tools/metrology
```

**Automatic segmentation is not wired to `cbm_sts_tools` yet.**

Selecting:

```yaml
segmentation:
  mode: automatic
```

currently raises a runtime error from the adapter.

The rest of the ladder processing pipeline is already independent of the segmentation method, so only the adapter needs to be implemented when the exact `cbm_sts_tools` segmentation API is integrated.

---

# Sensor plane fitting

Each segmented sensor cloud is transformed into detector coordinates and then fitted independently.

A separate plane-fit configuration can optionally be provided:

```yaml
ladder_analysis:
  sensor_plane_fit:
    ransac_iterations: 600
    ransac_threshold_mm: 0.03
    random_seed: 23
```

If this block is omitted, the global `plane_fit` configuration is reused.

For every sensor, the fitted plane is:

```text
n · p + d = 0
```

and the signed orthogonal residual for every measured point is:

```text
r = n · p + d
```

The plane normal returned by the fitter is normalized, so `r` directly represents the signed point-to-plane distance in millimetres.

---

# Sensor residual visualization

For each sensor, the point-cloud XY projection is displayed with colour representing its signed deviation from the fitted sensor plane.

Internally:

```text
residual [mm] = n · p + d
```

For visualization the residual is converted to micrometres:

```text
residual [µm] = 1000 × residual [mm]
```

## Combined plot

The combined ladder plot is written as:

```text
ladder_sensor_plane_residuals.png
```

All sensors retain their detector-coordinate XY positions.

Each point is coloured by its deviation from the plane fitted to its own sensor.

A common symmetric colour scale is used across all sensors, which allows residual magnitudes to be compared directly between sensors.

The default colour range is derived from the 99th percentile of the absolute residual distribution.

---

## Individual sensor plots

By default, one additional plot is written per sensor:

```text
sensor_S0_plane_residuals.png
sensor_S1_plane_residuals.png
sensor_S2_plane_residuals.png
...
```

These plots include the sensor plane-fit RMS in the title.

They can be disabled with:

```yaml
ladder_analysis:
  write_individual_plots: false
```

---

## Ladder visualization configuration

Example:

```yaml
ladder_analysis:

  visualization:
    dpi: 600
    sample_size: 50000
    random_seed: 23
    residual_percentile: 99.0
    use_inliers_for_scale: true
    show_outliers: false
    point_size: 2.0
```

`sample_size` affects visualization only.

Numerical sensor plane fitting uses the full loaded sensor point cloud.

---

# Example configuration

A configuration using pre-segmented ladder measurements may look like:

```yaml
# All lengths are millimetres.

output_dir: output
output_precision: 6

io:
  comment_prefixes: ["#", "//", "$"]
  delimiter: whitespace

uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]

cframe:
  file: data/c_frame_facet.txt

plane_fit:
  ransac_iterations: 600
  ransac_threshold_mm: 0.05
  random_seed: 23

references:
  - file: data/hole_1.xyz
    nominal_center: [0.0, 0.0]

  - file: data/hole_2.xyz
    nominal_center: [0.0, 500.0]

reference_points:
  derivative_step_rad: 1.0e-7

visualization:
  dpi: 600
  sample_size: 50000
  random_seed: 23

ladder_analysis:
  enabled: true

  segmentation:
    mode: segmented_files

  sensors:
    - name: S0
      file: data/L1_000000.txt

    - name: S1
      file: data/L1_000001.txt

    - name: S2
      file: data/L1_000002.txt

    - name: S3
      file: data/L1_000003.txt

  sensor_plane_fit:
    ransac_iterations: 600
    ransac_threshold_mm: 0.03
    random_seed: 23

  visualization:
    residual_percentile: 99.0
    use_inliers_for_scale: true
    show_outliers: false
    point_size: 2.0

  write_individual_plots: true
```

---

# Outputs

The output directory contains the numerical detector alignment results:

```text
results.json
rotation_matrix.npy
translation_vector.npy
transform_matrix.npy
covariance_transform_params.npy
covariance_joint.npy
covariance_plane_params.npy
covariance_reference_XX_conditional.npy
```

C-frame diagnostic plots include:

```text
cframe_plane_alignment.png
cframe_plane_alignment_top_view.png
cframe_plane_alignment_residual_histogram.png

cframe_final_alignment.png
cframe_final_alignment_top_view.png
cframe_final_alignment_residual_histogram.png
```

When ladder analysis is enabled, the additional outputs include:

```text
ladder_sensor_plane_residuals.png

sensor_<name>_plane_residuals.png
```

and transformed copies of the corresponding point-cloud inputs:

```text
*_transformed.xyz
```

or the equivalent original file extension.

For example:

```text
L1_000000_transformed.txt
L1_000001_transformed.txt
...
```

---

# `results.json`

The JSON output contains the complete scanner-to-detector transformation, its covariance, the C-frame plane result, and the reference measurements.

When ladder analysis is enabled it also contains a `ladder` section.

Conceptually:

```json
{
  "ladder": {
    "segmentation_mode": "segmented_files",
    "sensor_count": 4,
    "sensors": [
      {
        "name": "S0",
        "source_file": "data/L1_000000.txt",
        "total_points": 100000,
        "inliers": 99500,
        "plane": {
          "normal": [0.0, 0.0, 1.0],
          "d_mm": 0.0,
          "centroid_mm": [0.0, 0.0, 0.0],
          "rms_mm": 0.01
        },
        "residuals": {
          "mean_mm": 0.0,
          "std_mm": 0.01,
          "max_abs_mm": 0.05,
          "inlier_mean_mm": 0.0,
          "inlier_std_mm": 0.008,
          "inlier_max_abs_mm": 0.03
        }
      }
    ]
  }
}
```

Values above are illustrative only.

---

# Transformed point-cloud files

The following inputs are written in detector coordinates using the final rigid transformation:

* C-frame point cloud;
* measured reference files;
* legacy `ladder_files`;
* pre-segmented sensor point clouds;
* the complete ladder cloud when automatic mode is selected.

The writer preserves non-coordinate numeric columns and header/comment lines where possible.

Only the first three numeric columns are transformed.

---

# Current scope

Implemented:

* robust C-frame plane fitting;
* scanner-to-detector plane alignment;
* direct measured-reference registration;
* propagation of measurement uncertainty;
* final rigid-transform covariance;
* transformed point-cloud export;
* manually/pre-segmented ladder sensor input;
* independent sensor plane fitting;
* combined ladder sensor residual visualization;
* individual sensor residual maps;
* ladder plane-fit summaries in `results.json`.

Not yet implemented:

* the concrete `cbm_sts_tools` automatic sensor-segmentation adapter;
* nominal 2D sensor-shape registration;
* sensor XY position determination from nominal geometry;
* sensor rotation `Rz` determination from nominal geometry;
* complete sensor six-degree-of-freedom alignment.

The next sensor-alignment stage is intended to use each fitted sensor plane together with the nominal 2D sensor geometry to constrain its in-plane position and rotation.

---

# Legacy ladder input

The older configuration:

```yaml
ladder_files:
  - data/ladder.xyz
```

is still supported for transformation-only workflows.

Files listed there are transformed using the final scanner-to-detector registration but are not segmented or fitted as sensors.

New ladder metrology configurations should use:

```yaml
ladder_analysis:
```

instead.

---

# Related project

Automatic ladder sensor segmentation is intended to reuse the metrology tools from:

```text
https://github.com/googol23/cbm_sts_tools
```

In particular, the integration boundary has been isolated so that `cbm_sts_tools`-specific API calls do not propagate into the rest of the detector-alignment package.
