# C-Frame Metrology and Detector Alignment

Metrology-based geometrical alignment tools for registering laser-tracker / laser-scanner measurements of detector structures to a nominal detector coordinate system.

The package is currently developed for the **CBM Silicon Tracking System (STS)** metrology workflow. It establishes a detector reference frame from measured C-frame geometry and reference points, propagates measurement uncertainties into the resulting rigid transformation, and applies that transformation to ladder and sensor point clouds.

The current implementation also supports **automatic geometrical segmentation of ladder point clouds into individual sensor clouds**, followed by independent robust plane fitting and surface-residual analysis.

The package is installed as:

```text
detector-alignment
```

and provides the command-line program:

```text
detector-align
```

---

## 1. Purpose

The detector geometry used for reconstruction represents an idealized or nominal detector. The physically assembled detector is not expected to reproduce this geometry exactly.

Manufacturing tolerances, assembly tolerances, installation, support deformation, and service routing introduce differences between the **as-designed** and **as-built** detector.

Laser scanning adds another complication: the measured point cloud is not a complete copy of the CAD geometry. It contains only visible surfaces and can additionally contain cables, electronics, supports, temporary objects, and other structures that should not constrain detector alignment.

This project therefore does **not** attempt to globally force an entire measured point cloud onto an entire CAD model.

Instead, the processing is hierarchical:

1. mechanically meaningful C-frame geometry establishes the detector reference frame;
2. measured reference centers determine the in-plane position and orientation;
3. the resulting scanner-to-detector transformation is applied to ladder measurements;
4. ladder clouds are separated into sensor clouds;
5. every sensor surface is fitted independently;
6. residuals describe local measured surface deviations rather than being absorbed into a global registration.

This separation between **reference-frame determination** and **as-built deviation measurement** is central to the project.

---

## 2. Current scope

The repository currently implements two connected processing stages.

### C-frame alignment

The C-frame measurement establishes the transformation

```text
scanner coordinates -> detector coordinates
```

using:

* a robust fit of a measured C-frame reference plane;
* direct measurements of two or more reference centers;
* a two-dimensional rigid registration of those centers to their nominal detector positions;
* propagation of measurement and fit uncertainties into the final rigid transformation.

### Ladder and sensor metrology

Once the C-frame transformation is known, ladder point clouds are transformed into detector coordinates and individual sensor surfaces are analysed.

Two sensor-input modes are supported:

```text
segmented_files
automatic
```

`segmented_files` reads one previously segmented point-cloud file per sensor.

`automatic` reads one complete ladder cloud and performs the current geometrical sensor segmentation automatically.

Both modes feed the same downstream sensor-plane fitting and visualization pipeline.

---

## 3. Coordinate and transformation convention

The global rigid transformation is

```text
p_detector = R @ p_scanner + T
```

where:

* `p_scanner` is a point in the original measurement coordinate system;
* `R` is a `3 x 3` rotation matrix;
* `T` is a three-component translation vector;
* `p_detector` is the transformed point in detector coordinates.

The implementation uses the column-vector convention conceptually, although arrays of points are represented internally as `N x 3` NumPy arrays.

The equivalent homogeneous transformation is

```text
        [ R  T ]
H   =   [      ]
        [ 0  1 ]
```

All lengths are expressed in **millimetres** unless stated otherwise.

Rotation-vector uncertainties are expressed in **radians**.

---

## 4. Processing overview

The current processing chain is:

```text
C-frame point cloud
        |
        v
robust C-frame plane fit
        |
        v
align fitted plane normal with detector +Z
        |
        v
place reference plane at detector Z = 0
        |
        v
measured reference centers
        |
        v
2D registration to nominal reference positions
        |
        v
scanner -> detector rigid transformation
        |
        +------------------------------+
        |                              |
        v                              v
C-frame diagnostics              ladder point cloud
                                       |
                                       v
                         scanner -> detector transform
                                       |
                                       v
                         sensor segmentation/loading
                                       |
                                       v
                         independent sensor plane fits
                                       |
                                       v
                         residual maps and diagnostics
```

A deliberate implementation rule is that ladder data are transformed into detector coordinates **before automatic segmentation**.

The automatic segmenter therefore works with detector-frame quantities such as detector `Y` and `Z`, rather than with arbitrary scanner coordinates.

---

# Installation

## 5. Requirements

Python **3.11 or newer** is required.

The principal runtime dependencies are:

```text
NumPy >= 1.26
SciPy >= 1.11
Matplotlib >= 3.8
PyYAML >= 6.0
```

For development and testing:

```text
pytest >= 8.0
```

---

## 6. Create an environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```text
.venv\Scripts\activate
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

---

# Running the pipeline

## 7. Command line

Using the installed command:

```bash
detector-align process examples/config.yaml
```

or directly through Python:

```bash
python -m detector_alignment.cli process examples/config.yaml
```

Configuration paths are resolved relative to the directory containing the YAML configuration file.

The command prints the final rotation matrix, translation vector, and propagated one-standard-deviation uncertainties:

```text
[rx ry rz tx ty tz]
```

with rotations in radians and translations in millimetres.

---

# C-frame alignment

## 8. C-frame point cloud

The reference C-frame measurement is configured with:

```yaml
cframe:
  file: data/c_frame_facet.txt
```

The input is an ASCII point cloud containing at least three numerical columns:

```text
X Y Z
```

Additional numerical columns are allowed and can, for example, contain measurement uncertainty information.

Input parsing is configured with:

```yaml
io:
  comment_prefixes: ["#", "//", "$"]
  delimiter: whitespace
```

The default parser therefore accepts whitespace-separated ASCII measurements while ignoring configured header/comment lines.

---

## 9. C-frame reference plane

The C-frame plane is fitted using the robust plane-fitting implementation in:

```text
src/detector_alignment/plane.py
```

A typical configuration is:

```yaml
plane_fit:
  ransac_iterations: 600
  ransac_threshold_mm: 0.05
  random_seed: 23
```

The fitting procedure combines robust hypothesis generation with subsequent plane refinement and uncertainty estimation.

The plane is represented as:

```text
n . p + d = 0
```

with a unit normal

```text
n = [
    sin(theta) cos(phi),
    sin(theta) sin(phi),
    cos(theta)
]
```

and parameter vector

```text
[theta, phi, d]
```

The fitted C-frame normal defines the detector-plane orientation.

The fitted normal is aligned with detector `+Z`, and the corresponding reference plane is placed at:

```text
Z = 0
```

---

## 10. Reference points

After the plane orientation has been established, measured reference centers determine the remaining in-plane translation and rotation.

Each reference file must contain exactly **one measured XYZ point**.

Example:

```yaml
references:
  - file: data/hole_1.xyz
    nominal_center: [0.0, 0.0]

  - file: data/hole_2.xyz
    nominal_center: [0.0, 500.0]
```

`nominal_center` contains the corresponding nominal detector coordinate:

```text
[X, Y]
```

At least two distinct references are required.

The correspondence between measured and nominal references is defined by the YAML entries.

For backward compatibility:

```yaml
reference_holes:
```

is accepted as an alias for:

```yaml
references:
```

but both keys must not be supplied simultaneously.

### Important

The software does **not** determine the centers of physical reference holes from hole-opening point clouds.

Reference files must already contain the directly measured center coordinates supplied by the measurement system or preceding metrology procedure.

Legacy `radius_mm` hole-fitting configuration is rejected.

---

## 11. In-plane reference registration

Each measured reference point is first transformed by the C-frame plane rotation.

The resulting detector-plane coordinates are then registered against their nominal `[X, Y]` positions using a two-dimensional rigid least-squares transformation.

The final rotation has the form

```text
R = Rz(yaw) @ R_plane
```

and the final transformation obeys

```text
p_detector = R @ p_scanner + T
```

This transformation establishes the coordinate system in which subsequent ladder and sensor measurements are analysed.

---

# Measurement uncertainty

## 12. Uncertainty models

A default measurement uncertainty can be configured globally.

For example:

```yaml
uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]
```

Individual measurements can override the global uncertainty configuration.

Supported modes include the following.

### No measurement uncertainty

```yaml
uncertainty:
  mode: none
```

### Constant XYZ standard deviations

```yaml
uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]
```

### Standard deviations stored in point-cloud columns

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

The covariance-column order is:

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

A separate uncertainty file must contain the same number of rows as its corresponding point cloud.

---

## 13. Transformation covariance

The uncertainty propagation includes the fitted C-frame plane and measured reference coordinates.

Conceptually, the joint state is:

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

The propagated covariance accounts for:

* uncertainty of the C-frame plane;
* uncertainty of the measured reference points;
* correlations introduced by using the common plane fit;
* correlations between transformed reference coordinates.

The final transformation covariance uses the parameter order:

```text
[rx, ry, rz, tx, ty, tz]
```

where:

```text
rx, ry, rz
```

form a rotation vector in radians and:

```text
tx, ty, tz
```

are translations in millimetres.

---

# Ladder and sensor metrology

## 14. Enabling ladder analysis

Ladder processing is optional.

Enable it with:

```yaml
ladder_analysis:
  enabled: true
```

The C-frame alignment is always solved first.

The resulting scanner-to-detector transformation is then passed to the ladder processor.

This ordering is important:

```text
C-frame alignment
        |
        v
final scanner -> detector transform
        |
        v
load ladder cloud
        |
        v
transform ladder cloud
        |
        v
segment sensors
        |
        v
fit sensor planes
```

The transformation must not be applied again after segmentation.

---

## 15. Segmentation modes

Two modes are currently supported:

```text
segmented_files
automatic
```

Both ultimately produce `SensorPointCloud` objects and use the same sensor fitting code.

---

## 16. Automatic segmentation

Automatic segmentation is now implemented and is the mode used by the current example configuration.

Example:

```yaml
ladder_analysis:
  enabled: true

  segmentation:
    mode: automatic
    config:
      z_gap_mm: 0.3
      min_points_per_sensor: 50

  file: data/L1.txt
```

An optional detector-frame `Y` center can also be supplied:

```yaml
segmentation:
  mode: automatic
  config:
    z_gap_mm: 0.3
    y_center_mm: 0.0
    min_points_per_sensor: 50
```

If `y_center_mm` is omitted, the current implementation estimates it from the median detector-frame `Y` coordinate of the ladder cloud.

### Current segmentation algorithm

The current automatic segmenter is deliberately simple and geometrical.

The complete ladder point cloud is first transformed into detector coordinates.

The algorithm then:

1. sorts all points by detector `Z`;
2. identifies gaps in consecutive `Z` coordinates larger than `z_gap_mm`;
3. uses these gaps to divide the cloud into mechanically separated `Z` groups;
4. divides each `Z` group across the ladder `Y` center line;
5. rejects groups containing fewer than `min_points_per_sensor` points;
6. sorts the resulting sensor clouds geometrically by mean `Y`;
7. assigns stable names:

```text
S0
S1
S2
...
```

The relevant implementation is:

```text
src/detector_alignment/segmentation.py
```

### Important limitation

The current automatic segmentation is **not yet a general CAD-aware sensor segmentation algorithm**.

It assumes that the ladder geometry can be separated using distinct detector-`Z` layers and a detector-`Y` center line.

This is suitable for the current ladder geometry under development but should not be interpreted as a general solution for arbitrary STS point clouds.

Future segmentation work can replace this implementation without changing the downstream `SensorPointCloud` processing interface.

---

## 17. Pre-segmented sensor files

Sensor clouds can alternatively be supplied explicitly.

This is useful when sensor regions have been manually selected in software such as CloudCompare or generated by another segmentation procedure.

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

Sensor names must be unique.

The input files should remain in the **original scanner coordinate system**.

The package applies the final C-frame transformation while loading them.

Do not pre-transform the files and then pass them through the same alignment configuration, because that would apply the detector transformation twice.

---

## 18. Detector-frame invariant

`LadderProcessor` owns the scanner-to-detector transformation.

Every ladder or sensor cloud loaded by it is immediately transformed into detector coordinates.

This gives both segmentation modes the same invariant:

### `segmented_files`

```text
load sensor file
    ->
transform to detector coordinates
    ->
cache
    ->
fit
```

### `automatic`

```text
load complete ladder
    ->
transform to detector coordinates
    ->
cache
    ->
segment
    ->
fit
```

The internal cache therefore contains detector-frame clouds only.

Per-point covariance matrices, when available, are rotated consistently with the point-cloud transformation.

---

# Sensor surface analysis

## 19. Independent plane fitting

Every segmented sensor cloud is fitted independently.

A separate sensor plane-fit configuration can be supplied:

```yaml
ladder_analysis:
  sensor_plane_fit:
    ransac_iterations: 600
    ransac_threshold_mm: 0.03
    random_seed: 23
```

If values are omitted, the corresponding global C-frame `plane_fit` values are used.

Each sensor plane is represented by:

```text
n . p + d = 0
```

with normalized `n`.

The signed orthogonal residual of a measured point is therefore:

```text
r = n . p + d
```

and is directly expressed in millimetres.

---

## 20. What the current sensor fit determines

The present ladder analysis determines the measured **sensor surface plane**.

A plane strongly constrains:

* displacement normal to the sensor surface;
* the two rotations that tilt the sensor plane.

For a sensor whose nominal normal is approximately detector `Z`, these correspond approximately to:

```text
Tz
Rx
Ry
```

A plane alone does **not** fully determine the remaining in-plane degrees of freedom:

```text
Tx
Ty
Rz
```

Those require additional geometrical information such as:

* sensor edges;
* corners;
* fiducials;
* nominal sensor footprints;
* other reliable in-plane features.

Accordingly, the current per-sensor plane fitting should be understood as the surface/alignment stage of the larger six-degree-of-freedom sensor-alignment problem, not yet the complete sensor-to-ladder rigid transformation.

---

## 21. Sensor fit diagnostics

For every fitted sensor the processing records information including:

```text
sensor name
source file
total number of points
number of inliers
plane normal
plane offset
plane centroid
plane-fit RMS
residual mean
residual standard deviation
maximum absolute residual
inlier residual statistics
```

A low plane RMS by itself does not prove that the correct physical sensor has been identified.

Segmentation, sensor location, orientation, extent, and neighborhood should also be checked.

---

# Visualization

## 22. Sensor residual maps

Sensor residuals are stored internally in millimetres:

```text
residual_mm = n . p + d
```

For visualization they are converted to micrometres:

```text
residual_um = 1000 * residual_mm
```

The combined ladder visualization preserves the detector-coordinate positions of all sensor clouds while colouring each point according to its deviation from the plane fitted to its own sensor.

The principal output is:

```text
ladder_sensor_plane_residuals.png
```

Individual sensor plots can also be produced:

```text
sensor_S0_plane_residuals.png
sensor_S1_plane_residuals.png
sensor_S2_plane_residuals.png
...
```

---

## 23. Ladder visualization configuration

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

  write_individual_plots: true
```

`sample_size` controls visualization sampling only.

The numerical plane fit uses the complete loaded sensor cloud.

`residual_percentile` can be used to prevent a small number of extreme residuals from making the colour scale uninformative.

---

# Complete example

## 24. Automatic ladder segmentation

The following illustrates the current intended workflow:

```yaml
# All lengths are millimetres.

output_dir: output
output_precision: 6

io:
  comment_prefixes: ["#", "//", "$"]
  delimiter: whitespace

# Default measurement uncertainty.
uncertainty:
  mode: constant
  sigma_xyz_mm: [0.008, 0.008, 0.008]

# C-frame reference surface.
cframe:
  file: data/c_frame_facet.txt

plane_fit:
  ransac_iterations: 600
  ransac_threshold_mm: 0.05
  random_seed: 23

# Directly measured reference centers.
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
    mode: automatic
    config:
      z_gap_mm: 0.3
      # y_center_mm: 0.0
      min_points_per_sensor: 50

  file: data/L1.txt

  sensor_plane_fit:
    ransac_iterations: 600
    ransac_threshold_mm: 0.03
    random_seed: 23

  visualization:
    dpi: 600
    sample_size: 50000
    random_seed: 23
    residual_percentile: 99.0
    use_inliers_for_scale: true
    show_outliers: false
    point_size: 2.0

  write_individual_plots: true
```

The repository also contains:

```text
examples/config.yaml
```

which should be treated as the current executable configuration example.

---

# Outputs

## 25. Numerical alignment products

The output directory contains the scanner-to-detector alignment products, including files such as:

```text
results.json
rotation_matrix.npy
translation_vector.npy
transform_matrix.npy
covariance_transform_params.npy
covariance_joint.npy
covariance_plane_params.npy
```

Additional covariance products associated with the reference measurements can also be written by the pipeline.

---

## 26. C-frame diagnostics

C-frame diagnostic figures include the plane-alignment and final-alignment views generated by the visualization pipeline, for example:

```text
cframe_plane_alignment.png
cframe_final_alignment.png
```

Associated top-view and residual diagnostics may also be produced by the visualization stage.

---

## 27. Ladder diagnostics

When ladder analysis is enabled, the pipeline produces the combined sensor residual visualization:

```text
ladder_sensor_plane_residuals.png
```

and, when enabled:

```text
sensor_<name>_plane_residuals.png
```

for the individual sensors.

---

## 28. Transformed point clouds

Configured measurement inputs can be written back in transformed detector coordinates using names of the form:

```text
*_transformed.xyz
```

or with the corresponding original ASCII extension.

These files are useful for independent inspection of the final detector-frame geometry in point-cloud visualization software.

---

# `results.json`

## 29. Machine-readable results

`results.json` contains the numerical alignment result and associated metrology information.

The global section contains the scanner-to-detector transformation and uncertainty information.

When ladder analysis is enabled, sensor-plane results are additionally recorded.

A sensor result conceptually contains:

```json
{
  "name": "S0",
  "source_file": "data/L1.txt",
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
    "max_abs_mm": 0.04,
    "inlier_mean_mm": 0.0,
    "inlier_std_mm": 0.008,
    "inlier_max_abs_mm": 0.025
  }
}
```

The exact numerical values depend on the measurement.

---

# Repository structure

## 30. Main modules

```text
cframe_metrology/
|
|-- examples/
|   `-- config.yaml
|
|-- src/
|   `-- detector_alignment/
|       |-- __init__.py
|       |-- cli.py
|       |-- io.py
|       |-- ladder.py
|       |-- ladder_viz.py
|       |-- math3d.py
|       |-- models.py
|       |-- pipeline.py
|       |-- plane.py
|       |-- reference.py
|       |-- segmentation.py
|       |-- uncertainty.py
|       `-- viz.py
|
|-- tests/
|   |-- test_io.py
|   |-- test_pipeline.py
|   |-- test_reference.py
|   |-- test_registration.py
|   `-- test_uncertainty.py
|
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

The principal responsibilities are:

| Module            | Responsibility                                                                   |
| ----------------- | -------------------------------------------------------------------------------- |
| `pipeline.py`     | End-to-end alignment workflow and output generation                              |
| `io.py`           | ASCII point-cloud and uncertainty loading                                        |
| `plane.py`        | Robust plane estimation and plane uncertainty                                    |
| `reference.py`    | Processing of directly measured reference centers                                |
| `math3d.py`       | Rigid-transformation mathematics                                                 |
| `uncertainty.py`  | Joint covariance construction and transformation uncertainty propagation         |
| `ladder.py`       | Ladder/sensor cloud loading, detector transformation, caching, and plane fitting |
| `segmentation.py` | Current automatic ladder-to-sensor segmentation                                  |
| `ladder_viz.py`   | Sensor-plane residual visualization                                              |
| `viz.py`          | C-frame alignment diagnostics                                                    |
| `models.py`       | Core result/data structures                                                      |
| `cli.py`          | `detector-align` command-line entry point                                        |

---

# Development

## 31. Running tests

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Then run:

```bash
pytest
```

or:

```bash
python -m pytest
```

The current test suite covers core point-cloud loading, pipeline registration, reference handling, rigid-registration mathematics, and uncertainty propagation.

---

# Metrology interpretation

## 32. Reference geometry versus measured geometry

The nominal detector model should be treated as a **reference design**, not as exact point-by-point ground truth for the assembled detector.

For alignment it is useful to distinguish three classes of geometry:

### Reference geometry

Stable features deliberately used to establish a coordinate system or rigid pose.

Examples include suitable C-frame surfaces and accurately measured reference centers.

### Alignable geometry

Detector components whose deviation from nominal is itself a quantity to be measured.

Sensors and ladders belong to this category.

### Nuisance geometry

Visible structures that should not determine the alignment transformation.

Examples can include:

* cables;
* flexible services;
* electronics;
* temporary tooling;
* occluding structures;
* scan artifacts;
* geometry absent from or simplified in the nominal model.

A large number of nuisance points does not make them valid registration constraints.

---

## 33. Why global ICP is not the primary alignment strategy

A complete scan and a complete CAD model do not generally represent identical surfaces.

The scan is:

* partial;
* line-of-sight dependent;
* noisy;
* affected by incidence angle and occlusion;
* potentially populated by non-model objects.

The nominal geometry is:

* idealized;
* complete;
* not visibility limited;
* potentially different from the actual installed services and support geometry.

Consequently, minimizing a global cloud-to-model distance can partially absorb real as-built differences into the estimated detector transformation.

This project instead determines the coordinate system from selected reference geometry and analyses local detector components only after that reference system has been established.

---

# Relation to the CBM STS alignment hierarchy

## 34. Intended hierarchy

The larger STS alignment problem contains transformations at several mechanical levels:

```text
sensor -> ladder
ladder -> C-frame
sensor -> C-frame
```

For sensor `i` in ladder `L`:

```text
x_L = R_SiL @ x_Si + T_SiL
```

and for ladder `L` on C-frame `C`:

```text
x_C = R_LC @ x_L + T_LC
```

The composed sensor-to-C-frame transformation is:

```text
R_SiC = R_LC @ R_SiL

T_SiC = R_LC @ T_SiL + T_LC
```

The current repository establishes important pieces of this hierarchy:

* the C-frame reference coordinate transformation;
* detector-frame ladder point-cloud handling;
* sensor segmentation;
* independent sensor surface-plane measurements;
* uncertainty-aware alignment products.

Full sensor rigid-body alignment additionally requires reliable in-plane constraints.

---

## 35. Relation to track-based alignment

Metrology is intended to provide an independent **as-built geometrical initialization** for the detector reconstruction geometry.

It does not replace track-based alignment.

The two techniques constrain the detector differently:

```text
metrology
    -> external geometrical measurement
    -> physically interpretable detector reference
    -> useful initial geometry

track-based alignment
    -> particle trajectory constraints
    -> sensitive-element optimization
    -> potentially higher final tracking precision
```

Comparing the two is valuable because significant discrepancies can indicate:

* survey biases;
* coordinate-convention mistakes;
* installation changes;
* mechanical deformation;
* weak modes in track-based alignment.

---

# Current limitations and development direction

## 36. Current limitations

The present implementation should be understood as an active metrology-development package.

Important current limitations include:

* automatic segmentation is based on detector-frame `Z` gaps and a `Y` center line rather than a full nominal sensor geometry model;
* sensor plane fitting does not by itself determine `Tx`, `Ty`, and `Rz`;
* the current workflow relies on externally supplied measured reference centers rather than fitting physical reference features from raw scans;
* instrument-specific systematic uncertainty still needs to be represented by an appropriate measurement uncertainty model;
* the current sensor analysis is primarily surface based rather than a complete sensor-to-ladder six-degree-of-freedom alignment;
* automatic semantic rejection of cables, services, and other nuisance geometry is not yet a general capability.

These are analysis limitations rather than reasons to force the full point cloud onto the nominal geometry.

---

## 37. Near-term development

The next development steps are expected to concentrate on the sensor and ladder hierarchy:

```text
nominal sensor geometry
        |
        v
detector-frame ladder cloud
        |
        v
geometry-aware sensor candidate selection
        |
        v
robust independent sensor plane fits
        |
        v
sensor edge / footprint / fiducial constraints
        |
        v
complete sensor rigid-body transformations
        |
        v
sensor -> ladder -> C-frame composition
        |
        v
CBMROOT alignment constants
```

In particular, the current automatic segmentation can evolve from the present `Z`/`Y` geometrical split toward segmentation based explicitly on nominal STS sensor footprints while preserving the existing `SensorPointCloud` interface.

---

# Scientific context

This repository is part of a broader detector-metrology effort for the **Compressed Baryonic Matter (CBM) Silicon Tracking System at FAIR**.

The underlying problem is the registration of **as-built three-dimensional measurements** to an **as-designed detector geometry**, while avoiding the assumption that every measured surface must be an exact rigidly transformed copy of CAD.

Relevant topics include:

* detector geometrical survey;
* laser-tracker and laser-scanner metrology;
* point-cloud registration;
* Scan-vs-CAD / Scan-vs-BIM registration;
* robust geometrical estimation;
* uncertainty propagation;
* silicon-sensor alignment;
* hierarchical detector alignment;
* track-based alignment.

---

# Status

The current codebase implements:

```text
[implemented] C-frame point-cloud loading
[implemented] robust C-frame plane fitting
[implemented] direct measured-reference processing
[implemented] scanner -> detector rigid registration
[implemented] measurement uncertainty handling
[implemented] transformation covariance propagation
[implemented] C-frame diagnostic visualization
[implemented] transformed point-cloud output
[implemented] pre-segmented ladder sensor input
[implemented] automatic ladder sensor segmentation
[implemented] transform-before-segmentation detector-frame workflow
[implemented] per-sensor robust plane fitting
[implemented] sensor residual statistics
[implemented] combined ladder residual visualization
[implemented] individual sensor residual visualization

[in development] nominal-geometry-aware sensor segmentation
[in development] complete sensor in-plane alignment
[in development] complete sensor -> ladder transformations
[in development] ladder -> C-frame alignment products
[in development] CBMROOT alignment-constant integration and validation
```

The distinction between implemented and planned functionality should be preserved as the project evolves.

---

# License and collaboration

No license file is currently included in this repository.

Before distributing or reusing the package outside its intended collaboration context, add an explicit license and the appropriate CBM/STS collaboration and institutional attribution.
