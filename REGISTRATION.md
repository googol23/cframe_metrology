# Nominal-to-as-built ladder registration

This opt-in stage fits **one rigid correction per ladder**, holding the relative
sensor positions, orientations and dimensions fixed **after optional optical
geometry preparation**. It does not change the C-frame alignment or refit
individual sensor placements. It is independent of the existing sensor-plane
analysis.

## Inputs and coordinate contract

1. Export the selected ladder's nominal sensors with `geo-inspector ladder`.
   Select the actual ROOT node explicitly; measurement ladder names are **not**
   converted into ROOT indices or node names.
2. The reader accepts geo-inspector global v1 and node-relative v2 JSON. Lengths
   explicitly marked `cm` are converted to mm; `mm` is also accepted. Full local
   dimensions and sensor rotation matrices define oriented boxes. Exported AABBs
   are not used as substitutes for these boxes.
3. Set `measurement_frame` to `global` for global exports, or to the **exact
   `reference_node` string** for node-relative exports. This is an explicit
   assertion that the exported frame physically coincides with the pipeline's
   detector/C-frame coordinates. Merely matching the string cannot verify that
   physical assertion. The reader rejects inconsistent frame labels and never
   silently applies `reference_to_global`.
4. Pipeline inputs retain the existing scanner-coordinate contract: the final
   scanner-to-detector transform is applied to points and their covariance before
   registration. For data **already in C-frame coordinates**, use the standalone
   API below, or the pipeline's original scanner files; do not apply the C-frame
   transform to already-aligned files a second time.
5. In `automatic` mode, registration uses the **complete ladder source file**,
   not the automatic segmenter's selected subset. In `segmented_files` mode it
   concatenates every configured sensor file in configuration order. Every point
   must be a sensor-surface measurement. Both modes match against the union of
   nominal sensors; measured sensor names are not guessed to be geometry IDs.

The numerical registration implementation does not import ROOT. When
`root_geometry.enabled`, normal pipeline processing first invokes the independent
`geo_inspector` package in a configured external Python/ROOT environment; preview
and `mode: reuse` validate its published manifest without running ROOT. Optional
optical analysis likewise runs in its own configured interpreter.

## Nominal ROOT geometry and optical correction

For U3DL half-unit work, use either the dataset-local
**`PointCloudMetrology/input/u3dl/config.yaml`** or the equivalent workspace-oriented
**`sts_tools/config.yaml`**. Both contain the laser settings, `half_unit`,
`root_geometry`, and `ladder_metrology`; their relative paths are rebased to their
own directories but resolve to the same inputs and `output/U3DL` hierarchy. Other
configurations remain supported when either external stage is absent or disabled.

`RootGeometryDriver` executes `python -m geo_inspector ladder ...` through a
bounded, shell-free subprocess. The ROOT source file, node selector, frame,
interpreter, working directory, ROOT executable and output directory are all
explicit configuration. Extraction occurs in a sibling staging directory. The
driver validates the machine-readable receipt and every geo-inspector JSON with
the package's ROOT-independent nominal-geometry reader before replacing
`root_geo`. Its manifest records source/config fingerprints and per-file hashes.
Failed extraction leaves the last valid batch untouched. When enabled, each
`registration.geometry_file` must be one of the files in that validated manifest.
No manufacturing ladder name, ROOT copy index or node is inferred.

`LadderMetrologyDriver` executes `python -m sts_ladder_metrology.main --config ...`
through a bounded, shell-free subprocess. `execution.working_dir` controls where
that module is launched, while all input/output paths still resolve from the YAML.
The driver does not import the optical package, open its database, load its
fiducials or implement its fits. The optical producer owns inventory discovery,
grouping, fitting, origin/axes conversion, covariance and freshness checking. See
`../sts_ladder_metrology/README.md` for its schema.

The data boundary is explicit: the producer atomically writes
`ladder_metrology.result_file`, emits a JSON receipt naming that path and its
fingerprint/digest, and the consumer rejects any receipt, schema, identity, frame,
coverage or digest mismatch before preparing geometry. A normal consumer-driven
run stores combined producer stdout/stderr in `execution.log` next to the result.
The legacy `output_dir` form remains readable as
`output_dir/sensor_transforms.json`, but new configurations should use
`result_file` and must not specify both.

Normal processing executes ROOT inspection and optical production when their
blocks are enabled with `mode: run`; `reuse` validates existing artifacts without
regeneration. Preparation occurs even when ladder registration is disabled,
allowing a later preview to use the generated nominal and optical results. An
absent or disabled block preserves compatibility and does not execute that stage.

Preview **only checks/reuses** existing ROOT and optical results, regardless of
`mode`. It never runs ROOT extraction or fits optical, C-frame, sensor-plane or
ladder-registration parameters. Optical checking still needs its interpreter,
source files and current read-only assembly snapshot. Missing, stale, mismatched or
failed results cause errors, not fallback to uncorrected geometry.

The U3DL nominal-geometry block is structurally:

```yaml
root_geometry:
  enabled: true
  source_file: /path/to/sts_v26a.geo.root
  node: Unit04L_10
  frame: node
  output_dir: output/U3DL/root_geo
  mode: run
  execution:
    python: /path/to/python
    working_dir: sts_geo_utils
    root: /path/to/root
    timeout_seconds: 120
```

`frame: node` makes the actual selected ROOT node path the exported
`measurement_frame`; for the current file this is `/sts_v26a/Unit04L_10`.

Each corrected point-cloud ladder uses its configured ladder key as its STS8
assembly identity; `assembly_ladder` may override that key when different. Its
effective `registration` block must contain `optical_sensor_map`. An explicit per-module mapping associates
**every assembly module** one-to-one with **every ROOT sensor box**, with exactly
these fields:

- `node_path`: the exact ROOT sensor path from the geometry export.
- `sensor_to_geometry`: a 4x4 orthogonal coordinate-basis conversion, translation
  in mm, from the optical nominal sensor frame to that ROOT sensor box's local
  frame. Explicit axis reflections (determinant -1) are allowed here, since the
  legacy survey X/Y swap reverses parity. Scaling/shearing are rejected. The
  optical deviation itself and the corrected ROOT pose must remain proper rigid
  transforms; rotations are converted by conjugation, never by swapping vector
  components.

Alternatively, `optical_sensor_map: {mode: y_order, sensor_to_geometry: <4x4>}`
resolves module-to-node identities from sorted nominal ROOT sensor-centre Y. It
orders assembly `B` modules from outermost to centre (highest index to zero),
then `T` modules from centre outward (zero to highest index); the position digit
immediately after B/T is used even if subsequent name digits vary. It requires
consecutive indices on both sides, equal side counts, distinct ROOT Y coordinates,
and negative/positive Y on the corresponding sides. The resolved mapping is
recorded in `optical_metrology.sensor_mapping`. A 4x4 bridge here applies to
all sensors. For the U3DL centre-origin ladder-axis convention, use
`optical_sensor_map: {mode: y_order, sensor_to_geometry: ladder_aligned_box_center}`.
This derives each sensor's basis as `ROOT_box_rotation.T @ ladder_rotation`,
with zero translation. The top boxes define ladder axes; all top rotations must
agree, and bottom boxes must be rotated 180° around local Z relative to the top.
The preview fails if the exported geometry violates this convention. No
half-thickness offset is applied: the user-specified optical correction origin
is the ROOT box centre, although measured fiducials lie on its surface.

For other sensor-frame conventions, provide explicit per-module bridges; identity
is valid only when both origins and axes coincide. The Y-order association and
bridge do not by themselves validate the saved C-frame alignment.

For nominal ROOT sensor pose `T0`, optical sensor-local nominal-to-measured
transform `Delta`, and optical-local -> ROOT-box-local bridge `B`:

```text
Delta_box = B @ Delta @ inverse(B)
T_corrected = T0 @ Delta_box
T_final = G_ladder @ T_corrected
```

`LadderGeometryPreparation` always loads the original geometry JSON and returns
new oriented boxes. Dimensions stay fixed; centres, rotations, corners and bounds
reflect the sensor deviations. Repeated preparation cannot apply a correction
twice and never modifies the source export. The ladder-local origin is not
inferred and need not be reconstructed for this sensor-local composition.

Both `LadderRegistrationStage` and `RegistrationPreview` use this same preparation
component. For explicitly mapped ladders, preview's initial pose and registration's
optimized pose act **after** the optical correction, so their baseline/dashed
geometry is optically corrected. An unmapped preview instead uses the untouched
ROOT nominal geometry and records that optical correction was not applied. Their
JSON records `root_geometry` manifest/source/file provenance and
`optical_metrology`, including input/result digests and, when applied, mapping and
corrected poses. The fitted ladder matrix describes the additional
common placement of this corrected geometry, not a substitute for the individual
optical transformations. Optical covariance is retained in the producer artifact
but is **not propagated** into registration; those results are conditional on the
prepared geometry.

The workspace U3DL assembly inventory has 3 ladders (10, 10, 6 sensors).
Y-order association has been checked against those three exported ROOT ladders;
it does not by itself verify the detector/C-frame equivalence or the bridge.
These are deliberately not assigned to `L3DL*` manufacturing IDs. Registration
remains disabled, and corrected preview requires verified assembly/module/ROOT
mappings and valid geometry files before it can succeed.

## Inspect the initial pose without running registration

From the `PointCloudMetrology` directory:

```bash
python -m detector_alignment.registration_preview input/u3dl/config.yaml
```

This preview **never calls the C-frame fit, sensor-plane fits, ladder optimizer,
or pose-Jacobian computation**, even if registration is enabled in the YAML. It
loads the original scanner-coordinate ladder files and applies an existing
`output_dir/matrices/transform_matrix.npy` once. That saved scanner-to-detector
matrix must belong to the current measurement inputs, reference configuration
and **`cframe.normal` orientation / `cframe.facet_offset` target Z**; it is reused
explicitly, not recomputed or assumed to be current. If either field changes,
first recompute the C-frame alignment with ladder registration disabled, then
preview. The preview does not apply a new direction or offset to a cached matrix. A missing or
non-rigid saved matrix produces an error rather than an identity fallback.

Nominal geometry is shown at the effective configured `initial_transform`
(identity when omitted), with translucent blue sensor volumes, dashed original
nominal outlines, and measured points coloured by initial surface distance in
**mm**. XY, YZ and perspective appear in one image per ladder. No centroid matching,
frame conversion or fitted correction is introduced to make the objects overlap.
The title warns that geometry/measurement physical frame equivalence is not
verified. Each selected ladder must have valid geometry/frame fields; an explicit
preview ignores the registration `enabled` and `draw` switches without changing them.
If both `assembly_ladder` and `registration.optical_sensor_map` are configured,
preview applies the validated optical correction. With `--require-optical`, preview
fails rather than plotting uncorrected boxes; use this switch when inspecting
corrected nominal positions. Its dashed outlines show the original ROOT boxes
and its blue volumes show corrected boxes at the configured initial ladder pose;
`original_root_bounds_mm` and `initial_geometry_bounds_mm` are separately recorded.
If neither mapping is configured, the default preview shows uncorrected ROOT
nominal geometry and records
`optical_metrology.applied: false` with the reason. A partial mapping remains an
error; preview never guesses an assembly or ROOT sensor association.

Preview-specific defaults are **150 DPI** and **20,000 displayed points per
ladder**, independent of potentially expensive production rendering settings.
Bounds, RMS, containment counts and total-point counts use every source point;
sampling affects only display. Outputs are isolated in:

```text
output_dir/
├── images/registration/
│   └── <ladder>_initial_registration.png
└── registration/
    └── preview_summary.json
```

The JSON records the saved alignment matrix and path, initial poses, input files,
geometry identities, point counts, measured/initial-geometry bounds, initial
surface distances, and outside-point counts. It states `registration_performed:
false` and contains no fit acceptance claim. Existing `results.json`, fitted
matrices/images, transformed-cloud exports and pipeline logs are not overwritten.

Useful overrides (paths are relative to the configuration directory):

```bash
python -m detector_alignment.registration_preview input/u3dl/config.yaml \
  --ladder L3UL0 --dpi 100 --sample-size 10000
```

Use `--alignment-matrix` to select another saved `.npy` scanner-to-detector matrix,
`--output-dir` for a different preview-only directory, or repeat `--ladder` to
select multiple ladders. See the module's `--help` for all arguments.

## Configuration

Put shared settings in `ladder_analysis.registration`, and use a named ladder's
`registration` mapping for field-by-field overrides:

```yaml
ladder_analysis:
  enabled: true
  analyze_individual_sensors: false # Registration still runs.
  registration:
    enabled: true
    measurement_frame: global
    uncertainty_sigma: 3.0
    containment_tolerance_mm: 1.0e-6
    max_translation_mm: 5.0
    max_rotation_deg: 5.0
    max_iterations: 300
    draw: true
  L3DL0:
    file: data/L3DL0.txt
    segmentation:
      mode: automatic
    registration:
      geometry_file: data/L3DL0_geometry.json
      max_translation_mm: 2.0 # Overrides only this field; other fields are inherited.
```

**Use `global` only if ROOT global really is the measured detector frame.** For a
half-unit/C-frame export, replace it with that export's full `reference_node`.
Paths are resolved relative to the pipeline YAML file, as for other inputs.
Resolution order is **built-in option defaults → shared registration fields →
per-ladder registration fields**. A provided field replaces the shared value;
matrices such as `initial_transform` are replaced as whole values, not merged by
row. No source configuration mapping is mutated. Every registration field can be
shared or overridden, including `geometry_file`, `measurement_frame`, `enabled`
and `draw`; geometry paths normally belong in per-ladder blocks.

If a ladder omits its block, it inherits the shared block unchanged. If both
blocks are absent, registration is disabled, preserving existing configurations.
Set per-ladder `enabled: false` to opt out of a globally enabled stage, or
per-ladder `enabled: true` to opt into a globally disabled stage. An empty mapping
inherits fields; a YAML `null` registration block behaves like an absent block,
not an opt-out. A field explicitly set to `null` does **not** fall back to its
shared value and is rejected when that active field requires a concrete value.
An enabled effective configuration must supply valid `geometry_file` and
`measurement_frame`; shared defaults alone may omit these required fields if
per-ladder blocks supply them. Unknown keys and invalid values are rejected.

| Option                     | Default  | Meaning                                                                   |
| -------------------------- | -------- | ------------------------------------------------------------------------- |
| `geometry_file`            | required | One ladder geometry JSON                                                  |
| `measurement_frame`        | required | Explicit matching coordinate-frame identity                               |
| `initial_transform`        | identity | Nominal-to-as-built 4x4 rigid matrix, translation in mm                   |
| `uncertainty_sigma`        | `3.0`    | Positive multiplier on projected point standard deviation                 |
| `containment_tolerance_mm` | `1e-6`   | Positive numerical allowance in optimization and final containment check  |
| `max_translation_mm`       | `5.0`    | Per-component bound on pivot displacement relative to initial pose        |
| `max_rotation_deg`         | `5.0`    | Per-component bound on the incremental rotation vector, in degrees        |
| `max_iterations`           | `300`    | Warm-start evaluation limit and constrained-optimizer iteration limit     |
| `draw`                     | `true`   | Registration image switch, independent of existing residual-plot switches |

The pivot is the mean nominal sensor center, **not an inferred mechanical ladder
origin**. Rotation updates act about the initially transformed pivot; translation
bounds apply there rather than at a distant ROOT origin. Bounds are componentwise,
not limits on vector magnitude. Use a suitable `initial_transform` for larger
known displacements. Plot DPI, sample size and seed inherit the existing
`ladder_analysis.visualization` / top-level `visualization` settings.

Point uncertainty uses the existing file/global `uncertainty` configuration.
Covariances must be symmetric and positive semidefinite. All concatenated input
files must have covariance, or all must omit it; mixed availability is rejected.
Without covariance, boxes are unexpanded and a warning is recorded.

## Objective and hard containment

For a measured point `p`, nominal sensor center `c`, sensor orientation `S`, and
trial ladder correction `(R, T)`, compute sensor-local coordinates

```text
u = (R @ S).T @ (p - (R @ c + T))
q = abs(u) - dimensions / 2
signed_surface_distance = norm(max(q, 0)) + min(max(q), 0)
```

The objective is the **unweighted mean squared distance to the nearest sensor
box surface**, not to a sensor center plane or merely to its occupied volume.
Residuals are negative inside the nominal volume and positive outside it.
Sensor dimensions and relative placements are never optimization variables.

For point covariance `C`, let `A = R @ S` and

```text
padding[a] = uncertainty_sigma * sqrt((A.T @ C @ A)[a, a])
sensor_margin = min_a(dimensions[a] / 2 + padding[a] - abs(u[a]))
point_margin = max_over_sensors(sensor_margin)
```

Every point must have `point_margin >= -containment_tolerance_mm`. This is a hard
constraint in the final SLSQP solve and is independently checked afterward on
**all input points**. There is no outlier rejection or numerical subsampling.
A penalty-based least-squares fit is only an initializer, never the acceptance
criterion. The uncertainty envelope is an axiswise marginal k-sigma box, not an
exact covariance-ellipsoid intersection test or a simultaneous confidence bound
for an entire cloud. Gaussian tails, incorrect geometry, or nuisance points can
make all-point containment impossible at a selected k; the stage does not
silently discard such points or enlarge k.

This is a bounded **local** optimization with nearest-surface correspondence
changes, not a guarantee of the global minimum. Review initialization, bounds,
convergence and residuals for real measurements.

## Outputs and interpretation

The correction follows the column-vector convention

```text
p_as_built = R_delta @ p_nominal + T_delta
```

Both sides are in the declared measurement frame; translation is in mm. This is
**not** the inverse cloud-to-nominal transform. If the actual nominal
ladder-local-to-C-frame matrix `H_nominal` is available separately, its as-built
pose is `H_delta @ H_nominal`. No mechanical ladder origin is reconstructed from
sensor bounds. The JSON also reports the nominal pivot and its displacement to
make rotations about distant reference origins interpretable.

For each configured ladder `<name>`, the pipeline writes:

- `results.json["ladder_registrations"]["<name>"]`: rotation, translation,
  rotation vector, homogeneous correction, frame/node identity, options, input
  provenance, before/after RMS and maximum surface residual, outside-point count,
  maximum containment violation, nearest-surface sensor counts, convergence,
  acceptance, observability and warnings.
- `registration/<name>_nominal_to_as_built.npy`: the 4x4 correction, including a
  **diagnostic candidate** when the fit is not accepted. Always check the JSON
  status before using it downstream.
- `registration/<name>_point_residuals.csv`: one row per input point, preserving
  input-file/row order. Columns are point index, detector-frame XYZ in mm,
  nearest-surface sensor index, signed surface distance in mm, and containment
  margin in mm. Sensor indices refer to the geometry JSON sensor order and the
  JSON summary's sensor list. The nearest surface and the sensor with the best
  expanded-box containment need not be the same in overlap regions.
- `images/registration/<name>_ladder_registration.png`: **XY, YZ and perspective** in one image,
  with measured points coloured by surface distance, dashed nominal outlines,
  and translucent fitted sensor volumes. YZ uses labelled independent axis
  scales; perspective expands the Z display scale when necessary and labels it.
  Sampling limits display only; plot bounds, colour scale, fitting and containment
  checks use the full cloud. The colour scale has a 1 µm minimum span so exact
  synthetic fits remain readable.

`accepted` means that the constrained optimizer converged **and** every point
passed containment. It does **not** imply a unique pose. A failed fit remains an
explicitly labelled diagnostic candidate; other ladders can still be processed.
Malformed input/configuration raises an error instead. Disabled registration or
drawing removes its known stale products for still-configured ladders; outputs
for ladders removed entirely from configuration are not automatically deleted.

### Identifiability and uncertainty limits

A partial planar surface generally cannot uniquely measure in-plane translations
or rotation about its normal. Footprint containment may bound these parameters
without selecting a unique solution. Unlabelled thin boxes also permit front/back
surface ambiguity, and repeated sensor layouts can permit alternative matches.
No edge samples, face identities, point-to-CAD correspondences, or scan coverage
are assumed merely because a point is near a nominal sensor.

The result therefore includes a local surface-residual Jacobian rank, singular
values, and nullspace directions. Rotation parameters are scaled by a recorded
ladder lever arm to compare them with pivot translations. These describe the
scaled optimizer coordinates around the fitted pose, exclude active containment
inequalities, and are **not a six-parameter pose covariance**. Even local rank six
does not prove global uniqueness. Poorly constrained directions can depend on the
initializer and numerical path; do not interpret their fitted values as uniquely
measured deviations.

Containment uses measurement covariance conditional on the fixed C-frame fit;
common C-frame uncertainty and instrument systematics are not propagated into a
ladder pose covariance in this first implementation. Face labels, measured edges
or fiducials, explicit sensor correspondences, multi-start ambiguity studies, and
correlated frame/pose uncertainty propagation are useful subsequent developments.

## Standalone API for already-aligned clouds

```python
from detector_alignment.io import AsciiPointCloudLoader
from detector_alignment.nominal_geometry import NominalLadder
from detector_alignment.ladder_registration import LadderRegistrar, RegistrationOptions
from detector_alignment.registration_viz import RegistrationVisualizer

geometry = NominalLadder.from_json("ladder_geometry.json", expected_frame="global")
cloud = AsciiPointCloudLoader().load(
    "ladder_in_matching_frame.xyz",
    {"mode": "constant", "sigma_xyz_mm": [0.008, 0.008, 0.008]},
)
result = LadderRegistrar(RegistrationOptions()).fit(geometry, cloud.points, cloud.covariances)
summary = result.summary(geometry)
RegistrationVisualizer().plot(
    geometry, cloud.points, result, "ladder_registration.png", ladder_name="L3DL0"
)
# Check result.accepted, summary["observability"], and result.warnings before use.
```

## Validation

Run `python -m pytest tests/test_ladder_registration.py tests/test_registration_pipeline.py -q`.
The tests exercise exported-frame/unit handling, rigid pose recovery, rotated
boxes and anisotropic covariance, partial-plane ambiguity, infeasible all-point
constraints, initial-pose/pivot conventions, configuration failures, independent
pipeline integration, complete point exports, and translucent three-view PNGs.
Preview tests additionally forbid optimizer/alignment calls, verify both input
modes and non-identity saved/initial transformations, preserve existing fit
products, and exercise missing/non-rigid cached matrices and path/ladder selection.
Real ROOT exports and production scans still require dataset-specific validation;
synthetic recovery alone does not establish their six-degree-of-freedom identifiability.
