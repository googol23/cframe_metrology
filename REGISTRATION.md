# Nominal-to-as-built ladder registration

This opt-in stage fits **one rigid correction per ladder**, holding the relative
sensor positions, orientations and dimensions fixed **after mandatory optical
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
5. Registration requires `segmented_files`: exactly one measured cloud per ROOT
   sensor. Files are concatenated in configuration order, retaining every row.
   Cloud median Y and corrected sensor-center Y (at the initial pose) establish
   a fixed one-to-one association. Duplicate Y centers, missing/extra clouds and
   pairs beyond `pairing_tolerance_mm` are rejected. Manufacturing names are not
   guessed to be ROOT geometry IDs. The fit never reassigns points to nearer boxes.
6. Every enabled fit requires `ladder_metrology.enabled: true` and a complete
   `optical_sensor_map`. Uncorrected ROOT geometry is never a registration target.
   There is no `registration.mode` setting; legacy values, including
   `labeled_faces`, must be removed and are rejected even on disabled blocks.

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
own directories but resolve to the same inputs and `output/U3DL` hierarchy.
Their fit switches differ: the dataset config enables registration while the
workspace config keeps it disabled for preview preparation. C-frame processing
and preview remain available without optical corrections; enabled registration
requires them. Existing ROOT exports may be supplied without a managed ROOT stage.

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
Corrected preview and registration require complete assembly/module/ROOT
mappings and valid geometry files. The workspace-root configuration keeps fitting
disabled; enabling it does not bypass these requirements.

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
sampling affects only display. Preview distances are nearest-box diagnostics,
not assigned-facet residuals or fit acceptance tests. This evaluator has no
optimizer and accepts unsegmented inputs for inspection only. Outputs are isolated in:

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

Keep measured point-cloud inputs in `half_unit.ladders` and nominal geometry
and fit controls in the top-level `registration` stage. Per-ladder registration
fields override shared fields, independently of `ladder_analysis.enabled`:

```yaml
half_unit:
  ladders:
    L3DL0:
      assembly_ladder: L3DL100112
      segmentation:
        mode: segmented_files
      sensors:
        - {name: measured_sensor_0, file: data/sensor_0.txt}
        - {name: measured_sensor_1, file: data/sensor_1.txt}
# Optical producer settings must also be configured and enabled; see above.
registration:
  enabled: true
  measurement_frame: global
  uncertainty_sigma: 3.0
  containment_tolerance_mm: 1.0e-6
  footprint_penalty_weight: 100.0
  max_translation_mm: 5.0
  max_rotation_deg: 5.0
  max_iterations: 300
  draw: true
  ladders:
    L3DL0:
      geometry_file: data/L3DL0_geometry.json
      optical_sensor_map: {mode: y_order, sensor_to_geometry: ladder_aligned_box_center}
      max_translation_mm: 2.0 # Overrides only this field.
ladder_analysis:
  enabled: false # Registration and transformed cloud exports still run.
```

Legacy `ladder_analysis.registration` and per-ladder registration mappings
remain accepted for existing configurations, but cannot be mixed with the new
half-unit ladder inventory.

**Use `global` only if ROOT global really is the measured detector frame.** For a
half-unit/C-frame export, replace it with that export's full `reference_node`.
Paths are resolved relative to the pipeline YAML file, as for other inputs.
Resolution order is **built-in option defaults → shared registration fields →
per-ladder registration fields**. A provided field replaces the shared value;
matrices such as `initial_transform` are replaced as whole values, not merged by
row. No source configuration mapping is mutated. Every registration field can be
shared or overridden, including `geometry_file`, `measurement_frame`, `enabled`
and `draw`; geometry paths normally belong in per-ladder blocks.

If a ladder omits its registration block, it inherits the shared block unchanged.
If both blocks are absent, registration is disabled, preserving existing configurations.
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
| `containment_tolerance_mm` | `1e-6`   | Positive allowance shared by the XY penalty and final containment check   |
| `footprint_penalty_weight` | `100.0`  | Nonnegative dimensionless weight on squared XY excess; `0` disables only the penalty for diagnostic comparison |
| `max_translation_mm`       | `5.0`    | Per-component bound on pivot displacement relative to initial pose        |
| `max_rotation_deg`         | `5.0`    | Per-component bound on the incremental rotation vector, in degrees        |
| `max_iterations`           | `300`    | Maximum least-squares function evaluations                               |
| `facet_z_mm`               | `0.0`    | C-frame facet Z used to select each sensor's outward face                 |
| `outlier_sigma`            | `3.0`    | Robust sensor-plane filtering threshold in scaled MAD units               |
| `pairing_tolerance_mm`     | `15.0`   | Maximum measured/initial corrected sensor-center Y difference             |
| `diagnostic_interval_s`    | `15.0`   | Progress heartbeat interval                                               |
| `draw`                     | `true`   | Registration image switch, independent of existing residual-plot switches |

The pivot is the mean nominal sensor center, **not an inferred mechanical ladder
origin**. Rotation updates act about the initially transformed pivot; translation
bounds apply there rather than at a distant ROOT origin. Bounds are componentwise,
not limits on vector magnitude. Use a suitable `initial_transform` for larger
known displacements. Plot DPI, sample size and seed inherit the existing
`ladder_analysis.visualization` / top-level `visualization` settings.

Point uncertainty uses `half_unit.uncertainty` by default; a point-cloud entry
may override it. Legacy top-level `uncertainty` remains accepted.
Covariances must be symmetric and positive semidefinite. All concatenated input
files must have covariance, or all must omit it; mixed availability is rejected.
Without covariance, the face uncertainty band has zero width and a warning is
recorded; lateral footprints are unexpanded.

## Objective and hard containment

Each corrected sensor contributes exactly one outward local-Z facet, selected by
its position relative to `facet_z_mm`. Sensor local Z must align with C-frame Z
(`abs(S[2,2]) >= 0.99`); a center on the C-frame facet is ambiguous and rejected.
For fixed assigned sensor center `c`, orientation `S`, outward unit normal `n`,
face center `f`, and trial ladder correction `(R, T)`:

```text
u = (R @ S).T @ (p - (R @ c + T))
signed_surface_distance = dot(p - (R @ f + T), R @ n)
```

The objective combines **unweighted squared assigned-facet plane distances** with
**weighted squared XY footprint violations**, after per-cloud robust height/PCA
filtering. Exact sufficient statistics compress the surface term to four residuals
per sensor; the footprint term adds one residual per inlier. No numerical point
subsampling is used. All input rows, including excluded outliers, remain in the exported CSV.
Sensor labels and outward faces stay fixed throughout optimization. Neither
individual sensor geometry nor nearest-box assignments are optimized.

For point covariance `C`, let `A = R @ S` and

```text
padding[a] = uncertainty_sigma * sqrt((A.T @ C @ A)[a, a])
lateral_margin = min_a_in_XY(dimensions[a] / 2 + padding[a] - abs(u[a]))
normal_margin = padding[Z] - abs(signed_surface_distance)
point_margin = min(lateral_margin, normal_margin)
```

At every trial pose, the optimizer appends a fixed-length vector:

```text
excess_i = max(0, -lateral_margin_i - containment_tolerance_mm)
footprint_residual_i = sqrt(footprint_penalty_weight) * excess_i
objective = sum_i(surface_distance_i**2) + footprint_penalty_weight * sum_i(excess_i**2)
```

Each inlier contributes its worst-axis XY excess in mm; a point outside both X
and Y is not counted twice. Allowed points contribute zero. The covariance is
projected in the trial sensor frame, using the same calculation as final
validation. Excluded plane outliers do not contribute to either objective term.
The face-normal band is **not** an additional penalty: face distance remains the
surface objective, and normal-band containment remains a final acceptance gate.
The default weight `100` makes equal-depth XY violations contribute 100 times the
squared cost of face residuals. It is a tunable starting value, not a physical
prior or guarantee of feasibility. `0` is for plane-only diagnostic comparisons.
All weights leave acceptance unchanged. Use YAML numeric literals such as `0.1`
or `1.0e-1`; PyYAML reads `1e-1` as a string, which validation rejects.

Every **inlier** must still have `point_margin >= -containment_tolerance_mm` for
the same assigned sensor. This finite-footprint and face-band acceptance check
runs independently on full-resolution inliers after bounded least squares. The
XY penalty is soft, not a hard inequality or a union-of-boxes constraint. A
finite weight can leave violations, even when a feasible pose exists. Minimizing
squared depth also need not minimize the number of failing points. Optimizer
success alone does not establish acceptance; failed containment leaves a diagnostic candidate.
Outlier counts and fit-use flags are explicit; tolerances are never enlarged to
make a result pass. The axiswise marginal k-sigma envelope is not an exact
covariance-ellipsoid intersection or simultaneous whole-cloud confidence bound.

This remains a bounded **local** optimization, not a guarantee of a unique or
global solution. Review initialization, bounds, correspondence and residuals.

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
  maximum containment violation, assigned-sensor/inlier counts, convergence,
  acceptance, observability and warnings. Initial/final metrics also include
  `outside_footprint_points`, `max_footprint_excess_mm` (beyond tolerance), and
  `footprint_penalty_sum_sq_mm2`; per-sensor metrics appear under
  `sensor_surface_assignments`. `footprint_penalty_weight` and `objective` document
  the fit tradeoff. Surface RMS and surface Jacobian rank exclude penalty rows.
- `registration/<name>_nominal_to_as_built.npy`: the 4x4 correction, including a
  **diagnostic candidate** when the fit is not accepted. Always check the JSON
  status before using it downstream.
- `registration/<name>_point_residuals.csv`: one row per input point, preserving
  input-file/row order. Columns are point index, detector-frame XYZ in mm,
  `assigned_sensor_index`, `signed_surface_distance_mm`, `containment_margin_mm`,
  and `used_for_fit` (0 or 1). Sensor indices refer to the geometry JSON sensor
  order and the JSON summary's sensor list. Excluded outliers retain their original
  assignments and rows. Both residuals and containment use the assigned facet.
- `images/registration/<name>_ladder_registration.png`: **XY, YZ and perspective** in one image,
  with measured points coloured by their assigned sensor, matching each
  optically corrected sensor volume and its dashed pre-fit outline. A dotted
  outline in that same colour shows the fitted outward-facet acceptance envelope:
  the finite XY footprint and face-normal band expanded by the **largest inlier
  projected uncertainty per sensor** plus `containment_tolerance_mm`. With no
  covariance, it shows tolerance alone. Because uncertainty can differ per point,
  this is an outer visual guide, **not** the per-point acceptance test; a point
  inside the outline may still fail its own containment check. YZ uses labelled
  independent axis scales; perspective expands the Z display scale when necessary
  and labels it. Sampling limits
  display only; plot bounds use the full cloud, while fitting and containment
  use full-resolution inliers. Preview-only plots retain the surface-distance
  colour scale and colourbar.

The JSON retains `registration_mode: labeled_faces` as a **fixed output
compatibility marker**, not a configuration switch. Its
`registration_reference_geometry` is always `optically_corrected_nominal_sensors`.
The LadderRegistration bridge consumes `assigned_sensor_index` and retains every
row, including outliers, for its own downstream processing.

`accepted` means that the optimizer converged **and** every inlier
passed its assigned facet's containment check. It does **not** imply a unique pose. A failed fit remains an
explicitly labelled diagnostic candidate; other ladders can still be processed.
Malformed input/configuration raises an error instead. Disabled registration or
drawing removes its known stale products for still-configured ladders; outputs
for ladders removed entirely from configuration are not automatically deleted.

### Identifiability and uncertainty limits

A partial planar surface generally cannot uniquely measure in-plane translations
or rotation about its normal. Footprint containment may bound these parameters
without selecting a unique solution. Fixed sensor and outward-face associations
prevent nearest-box/front-back switching but do not establish scan coverage or
measured edges. Association still depends on a valid C-frame/ROOT mapping.

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
ladder pose covariance. Measured edges or fiducials, multi-start ambiguity studies,
and correlated frame/pose uncertainty propagation remain useful developments.

## Standalone API for already-aligned clouds

The former `LadderRegistrar.fit` nearest-box API has been removed. Use
`LabeledFaceRegistrar.fit(corrected_geometry, points, sensor_indices, covariances,
initial_transform)` for numerical work on already-aligned data. The caller must
prepare corrected geometry through `LadderGeometryPreparation` and establish a
complete set of ROOT sensor-list indices before calling it. Loading raw geometry
with `NominalLadder.from_json` alone does not apply optical corrections.

The pipeline stage enforces correction provenance; the low-level numerical API
cannot infer provenance from arrays. Check `result.accepted`, its observability
summary and warnings before using its homogeneous correction.

`SensorBoxEvaluator` supports preview diagnostics only and has no fitting API.

## Validation

Run `python -m pytest tests/test_labeled_registration.py tests/test_ladder_registration.py tests/test_registration_pipeline.py tests/test_stage_config.py -q`.
The tests exercise exported-frame/unit handling, assigned-facet rigid recovery,
preview box distances and anisotropic covariance, partial-plane ambiguity,
infeasible inlier footprints, initial-pose/pivot conventions, obsolete-mode and
missing-optical failures, fixed correspondence, complete point exports, and
translucent three-view PNGs.
Preview tests additionally forbid optimizer/alignment calls, verify both input
modes and non-identity saved/initial transformations, preserve existing fit
products, and exercise missing/non-rigid cached matrices and path/ladder selection.
Real ROOT exports and production scans still require dataset-specific validation;
synthetic recovery alone does not establish their six-degree-of-freedom identifiability.
