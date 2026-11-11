# XY footprint penalty: frozen-input comparison

## Scope and implementation

The fit now minimizes the existing sum of squared assigned-face distances plus
`footprint_penalty_weight * sum(max(0, -XY_margin - tolerance)**2)` over the same
fixed inliers. Each point contributes its worst X/Y violation, not a discontinuous
outside-point count. Default weight: **100**, a tunable starting value, not a
physically calibrated prior. Weight zero disables the penalty for diagnostics.
The face-normal band is still checked after fitting; it is not an additional
objective term. No acceptance tolerance, outlier rule, label or pose bound changed.
Surface-only Jacobian rank is calculated from the leading surface residual rows.

## Reproducibility

- `baseline/` contains the exact pre-edit registrar/options sources and original
  dataset YAML, including pre-existing uncommitted work.
- `config.numeric.json` explicitly converts the YAML tolerance string `1e-1` to
  numeric `0.1`. The live dataset YAML was not edited.
- `manifest.json` records configuration, validated cached ROOT/optical provenance,
  input file hashes, saved C-frame transform/hash, fit-array hashes and versions.
- Local `L3DL*.npz` snapshots contain the exact transformed points, covariance,
  fixed labels, corrected geometry and initial transform. These binary measurement
  snapshots are ignored by Git, but retained locally for replay. Preserve them
  with the manifest when transferring this experiment.
- Each `*.weight_*.json` report records the mask hash, fitted pose, optimizer status,
  face residuals, full containment, per-sensor XY footprints, and timing.
- All four comparisons for each ladder have **identical inlier-mask hashes**.
- No production output was replaced. External producers were only invoked in
  read-only validation (`--check`); no ROOT extraction or optical refit was run.
- These are current frozen inputs, not an assumed reconstruction of the old logged
  run. Nevertheless, the archived implementation reproduces the earlier RMS,
  bound-hitting poses and outside counts.

Preparation initially failed because the producer resolved the copied YAML's
relative paths from its new location. The comparison script was corrected to
validate the unchanged original YAML against its hash and use its original path
for the read-only producer check. That retry succeeded; no fit or source
configuration was changed to bypass a provenance check.

From `PointCloudMetrology`, the executed commands were:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py prepare
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py 10
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/footprint_penalty/compare.py 100
```

The script refuses to overwrite existing snapshots or reports. The baseline uses
an imported copy of the original registrar rather than changing working source.
Comparison hashes reflect source at execution time; subsequent removal of one
trailing space in the new registrar has no numerical effect.

## Results

All eight fits reported optimizer success. **None passed final containment.**
Outside counts below are full acceptance failures; at these final poses every
failure is also an XY footprint failure. Excess is measured beyond covariance
padding **and** the 0.1 mm tolerance.

| Ladder | Objective weight | Face RMS (mm) | Outside inliers | Maximum XY excess (mm) | Z rotation-vector component (degrees) |
|---|---:|---:|---:|---:|---:|
| L3DL100112 | Old plane-only | 0.0378692 | 59,565 | 28.23268 | -5.00000 |
| L3DL100112 | 1 | 0.0399312 | 194 | 1.04664 | -0.02910 |
| L3DL100112 | 10 | 0.0400092 | 154 | 1.04432 | -0.02783 |
| L3DL100112 | 100 | 0.0400595 | 154 | 1.04364 | -0.02758 |
| L3DL300112 | Old plane-only | 0.0387104 | 47,092 | 28.71076 | -5.00000 |
| L3DL300112 | 1 | 0.0412608 | 119 | 1.06120 | -0.04585 |
| L3DL300112 | 10 | 0.0412899 | 67 | 1.06547 | -0.03876 |
| L3DL300112 | 100 | 0.0413056 | 63 | 1.06594 | -0.03803 |

At weight 100, pivot XY displacement is (-0.3411, +0.2442) mm and
(-0.1065, -0.1993) mm respectively, instead of (-5, -5) mm for both old fits.
Maximum face residuals are 0.13353 and 0.12350 mm. The evaluated band allows
0.15 mm covariance padding plus 0.1 mm tolerance; passing this configured band
is not an independent validation of its physical justification.

Fit times for the old implementation were about 1.6 and 1.3 seconds; weight 100
fits took about 25.2 and 23.6 seconds. These are illustrative local timings, not a
controlled benchmark: some validation commands ran concurrently. The extra
full-resolution per-trial geometry/covariance evaluation is intentional.

### Per-sensor XY comparison

Sensor indices follow the frozen geometry sensor order, with exact node paths in
the reports. These counts exclude plane outliers. Initial means the original pose,
not the prior diagnostic refit holding selected parameters fixed.

| Ladder | Sensor index | Initial outside XY | Old fitted outside XY | Weight-100 outside XY | Weight-100 max excess (mm) |
|---|---:|---:|---:|---:|---:|
| L3DL100112 | 0 | 8 | 797 | 15 | 0.7446 |
| L3DL100112 | 1 | 19 | 173 | 7 | 0.6731 |
| L3DL100112 | 2 | 17 | 1,183 | 12 | 0.6691 |
| L3DL100112 | 3 | 12 | 4,358 | 18 | 0.7321 |
| L3DL100112 | 4 | 20 | 16,791 | 37 | 1.0436 |
| L3DL100112 | 5 | 6 | 1,532 | 6 | 0.4596 |
| L3DL100112 | 6 | 9 | 4,611 | 18 | 0.9242 |
| L3DL100112 | 7 | 4 | 5,297 | 7 | 0.4662 |
| L3DL100112 | 8 | 9 | 8,105 | 9 | 0.1998 |
| L3DL100112 | 9 | 15 | 16,718 | 25 | 0.7025 |
| L3DL300112 | 0 | 16 | 620 | 8 | 0.4834 |
| L3DL300112 | 1 | 11 | 86 | 6 | 0.6761 |
| L3DL300112 | 2 | 21 | 731 | 11 | 0.7605 |
| L3DL300112 | 3 | 15 | 3,721 | 14 | 1.0659 |
| L3DL300112 | 4 | 8 | 15,924 | 10 | 1.0165 |
| L3DL300112 | 5 | 5 | 1,081 | 4 | 0.6162 |
| L3DL300112 | 6 | 1 | 3,700 | 0 | 0.0000 |
| L3DL300112 | 7 | 9 | 4,046 | 6 | 0.3590 |
| L3DL300112 | 8 | 24 | 5,144 | 2 | 0.0156 |
| L3DL300112 | 9 | 32 | 12,039 | 2 | 0.1398 |

## Assessment and remaining acceptance blocker

**Verified:** The new objective suppresses the large rotation/lateral-translation
escape. It reduces squared violation depth substantially without changing inliers
or tolerance. However, residual violations near 1 mm persist across weights
1–100. Increasing weight does not monotonically reduce the largest violation or
the outside count because the objective is a sum of squared depths, not minimax
or a count objective. For the first ladder, weight 100 has more failing points
than the initial XY pose (154 versus 119), while reducing their total squared
excess. This is an expected tradeoff, not proof of accepted placement.

**Not established:** a feasible rigid pose exists for every retained point, or
that the configuration's tolerances and broad pose bounds have physical approval.
Do not use these fitted matrices as accepted registrations. The weight sweep was
stopped rather than raising weights indefinitely or widening tolerances.

The next investigation should inspect the surviving points against each assigned
sensor edge, verify their scan membership and fixed corrected sensor dimensions,
and check whether their covariance model covers those departures. They survived
height/PCA filtering; that is not proof of valid XY membership. If the measurements
and geometry are physically consistent but strict feasibility remains necessary,
evaluate a constrained feasibility solve separately. No lateral outliers were
silently discarded and no sensor labels were changed here.

## Automated validation

- Pre-edit targeted registration tests: **87 passed**.
- New penalty tests: **11 passed**, including competing poses, feasible lateral
  recovery, covariance projection, exact depth weighting, unchanged inlier mask,
  active-footprint versus surface-only rank, and invalid weights.
- Final targeted command: `.venv/bin/python -m pytest tests/test_footprint_penalty.py tests/test_labeled_registration.py tests/test_ladder_registration.py tests/test_registration_pipeline.py -q` — **98 passed**. Existing impossible-input and diagnostic-export tests still pass.
- `git --no-pager diff --check` passed. The live YAML and both frozen fit-array hashes were verified unchanged.
- Full project suite: **236 passed, 1 failed**. The failure is
  `test_dataset_config_uses_stage_sections_and_resolves_paths`: the live YAML
  contains string `1e-1`. The archived pre-edit `RegistrationOptions` independently
  rejects the same value with the same validation error. Production YAML was
  intentionally preserved; `containment_tolerance_mm: 0.1` is the required numeric
  spelling before using that configuration for a normal pipeline run.
