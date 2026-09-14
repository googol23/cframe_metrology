# Detector Alignment Package Context

## Purpose

`detector_alignment` computes a traceable rigid transformation from measured point clouds into a nominal detector coordinate system.

The alignment is established from:

- a reference plane, which determines detector tilt and the aligned Z origin;
- reference features, such as known-radius holes or directly measured points, which determine in-plane rotation and translation;
- measurement uncertainties, which are propagated into the uncertainty of the final transform.

The package also applies the resulting transform to point-cloud files and produces machine-readable results and diagnostic visualizations.

## Conceptual Pipeline

1. **Load and validate inputs**
   - Read configuration, point clouds, reference definitions, and optional measurement covariances.
   - Keep input/output handling separate from geometric processing.

2. **Estimate the reference plane**
   - Robustly reject outliers and fit the plane.
   - Determine the rotation that maps its normal to the aligned Z axis.
   - Estimate the plane-parameter covariance.

3. **Process in-plane references**
   - Level reference measurements using the fitted plane.
   - Extract reference locations from geometric features or direct measurements.
   - Preserve each reference's conditional uncertainty and sensitivity to the shared plane fit.

4. **Solve the rigid alignment**
   - Register measured reference locations to their nominal positions.
   - Combine plane leveling, in-plane registration, and plane offset into one rigid transform:
     `p_aligned = R @ p + T`.

5. **Propagate uncertainty**
   - Build a joint model that retains correlations introduced by the shared plane estimate.
   - Propagate it to the final rotation and translation parameters.

6. **Produce outputs**
   - Save the transform, covariance, fit summaries, and diagnostics.
   - Apply the transform to requested point clouds without requiring entire output datasets to remain in memory.

## Engineering Principles

This package is intended to function efficiently and reliably with large point clouds.

Implementations should prioritize:

- **Clarity:** keep I/O, fitting, registration, uncertainty propagation, transformation, and visualization as explicit, independently understandable stages.
- **Scalability:** avoid unnecessary copies and full-size intermediate arrays; use streaming, chunking, bounded sampling, and vectorized operations where appropriate.
- **Parallel readiness:** keep independent work units free of hidden shared state so expensive stages—such as processing separate references, transforming files, or generating diagnostics—can be multithreaded or otherwise parallelized when beneficial.
- **Determinism:** make randomized robust fitting and sampling reproducible through explicit seeds.
- **Numerical correctness:** use consistent coordinate conventions, units, covariance ordering, and transform composition throughout the package.
- **Separation of concerns:** visualization and serialization must not alter numerical results, and performance optimizations must not obscure the geometry.
- **Extensibility:** represent reference extraction as a replaceable stage so additional reference types and point-cloud formats can be introduced without redesigning the alignment core.

Prefer straightforward, testable algorithms first. Add concurrency only where profiling shows meaningful benefit, while preserving deterministic results and clear error reporting.


AGENTS are allow to read all project files but never to execute any test command on it. Only command that support code edition like git diff.