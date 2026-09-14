from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .math3d import (
    angles_from_normal,
    normal_from_angles,
    rotation_align_normal_to_z,
)
from .models import PlaneFitResult


class RobustPlaneFitter:
    """
    Robust plane fitter for a point cloud dominated by one planar facet.

    Parameters
    ----------
    ransac_iterations
        Number of random plane hypotheses generated during the initial
        robust orientation search.

    ransac_threshold_mm
        Minimum geometric inlier threshold in millimetres.

        IMPORTANT:
        In the previous implementation this value was used as a hard
        threshold. That could reject most of a valid facet if the actual
        measurement scatter was larger than this value.

        In this implementation it is treated as a MINIMUM threshold.
        The actual threshold can automatically increase according to
        the robustly estimated residual scatter.

    random_seed
        Seed used for reproducible random sampling.
    """

    # ------------------------------------------------------------------
    # Internal tuning constants
    # ------------------------------------------------------------------

    # Maximum number of points used to generate RANSAC hypotheses.
    _MAX_RANSAC_POOL = 250_000

    # Candidate models do not need to be scored against the complete
    # RANSAC pool. This keeps the initial search reasonably fast.
    _MAX_SCORE_POOL = 60_000

    # Quantile used to score the dominant plane population.
    #
    # Because the expected facet fraction is > 50% and normally > 90%,
    # a 70% quantile remains dominated by the facet while being much
    # less sensitive to contaminating geometry.
    _RANSAC_SCORE_QUANTILE = 0.70

    # MAD-to-standard-deviation factor for approximately Gaussian noise.
    _MAD_TO_SIGMA = 1.482602218505602

    # Final geometric inlier criterion:
    #
    #     |residual| <= sigma_clip * robust_sigma
    #
    # unless ransac_threshold_mm is larger.
    _SIGMA_CLIP = 4.5

    # Number of classify/refine cycles before the final fit.
    _MAX_REFINEMENT_CYCLES = 10

    # Stop iterating when less than this fraction of classifications
    # changes.
    _MASK_CONVERGENCE_FRACTION = 1e-6

    # Numerical floors
    _SIGMA_FLOOR_MM = 1e-9
    _VARIANCE_FLOOR = 1e-18

    def __init__(
        self,
        ransac_iterations=600,
        ransac_threshold_mm=0.05,
        random_seed=42,
    ):
        self.ransac_iterations = int(ransac_iterations)
        self.ransac_threshold_mm = float(ransac_threshold_mm)
        self.rng = np.random.default_rng(random_seed)

        if self.ransac_iterations <= 0:
            raise ValueError("ransac_iterations must be greater than zero.")

        if self.ransac_threshold_mm <= 0:
            raise ValueError("ransac_threshold_mm must be greater than zero.")

    # ==================================================================
    # Basic plane utilities
    # ==================================================================
    @staticmethod
    def _plane_from_three(p: np.ndarray) -> tuple[np.ndarray, float] | None:
        """
        Construct a plane from three points.

        Plane convention:

            n . p + d = 0

        where n is a unit normal.
        """
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]

        n = np.cross(v1, v2)

        norm = np.linalg.norm(n)

        if norm < 1e-12:
            return None

        n = n / norm
        d = -float(np.dot(n, p[0]))

        return n, d

    @staticmethod
    def _orthogonal_plane_fit(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Orthogonal least-squares plane fit using PCA/SVD.

        Returns
        -------
        normal
            Unit plane normal.

        d
            Plane offset in n.p + d = 0.

        centroid
            Centroid of the fitted points.
        """
        centroid = np.mean(
            points,
            axis=0,
        )

        centered = points - centroid

        _, _, vt = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        normal = vt[-1]

        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-15:
            raise RuntimeError("Degenerate point set encountered during plane fitting.")

        normal = normal / normal_norm

        # Keep normal orientation compatible with the existing code.
        if normal[2] < 0:
            normal = -normal

        d = -float(np.dot(normal, centroid))

        return normal, d, centroid

    # ==================================================================
    # Robust statistics
    # ==================================================================
    @classmethod
    def _robust_sigma(cls, residuals: np.ndarray) -> tuple[float, float]:
        """
        Robustly estimate residual center and standard deviation.

        Returns
        -------
        center
            Median residual.

        sigma
            Robust sigma estimated using MAD.
        """
        residuals = np.asarray(
            residuals,
            dtype=float,
        )

        finite = residuals[np.isfinite(residuals)]

        if len(finite) == 0:
            raise RuntimeError(
                "No finite residuals available for robust scale estimation."
            )

        center = float(np.median(finite))

        absolute_deviation = np.abs(finite - center)

        mad = float(np.median(absolute_deviation))

        sigma = cls._MAD_TO_SIGMA * mad

        sigma = max(
            sigma,
            cls._SIGMA_FLOOR_MM,
        )

        return center, sigma

    def _classify_inliers(self, residuals: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Classify points using an adaptive robust threshold.

        The user-supplied RANSAC threshold is retained as the minimum
        allowed threshold.

        Returns
        -------
        mask
            Boolean inlier mask.

        threshold_mm
            Geometric threshold actually used.

        robust_sigma_mm
            MAD-derived residual sigma.
        """
        center, sigma = self._robust_sigma(residuals)

        threshold = max(
            self.ransac_threshold_mm,
            self._SIGMA_CLIP * sigma,
        )

        centered_residuals = residuals - center

        mask = np.abs(centered_residuals) <= threshold

        return mask, threshold, sigma

    # ==================================================================
    # Initial robust model
    # ==================================================================
    def _initial_ransac_model(self, points: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Determine the initial dominant-plane orientation.

        Unlike the previous implementation, candidates are NOT selected
        exclusively by the number of points within a fixed small
        distance threshold.

        Instead, the residual distribution of the dominant population
        is evaluated robustly. This is important when the facet contains
        >50% of all points but its real measurement scatter is larger
        than ransac_threshold_mm.
        """
        npts = len(points)

        # --------------------------------------------------------------
        # RANSAC hypothesis pool
        # --------------------------------------------------------------
        if npts > self._MAX_RANSAC_POOL:
            pool_indices = self.rng.choice(
                npts,
                size=self._MAX_RANSAC_POOL,
                replace=False,
            )

            pool = points[pool_indices]
        else:
            pool = points

        # --------------------------------------------------------------
        # Smaller set for candidate scoring
        # --------------------------------------------------------------
        if len(pool) > self._MAX_SCORE_POOL:
            score_indices = self.rng.choice(
                len(pool),
                size=self._MAX_SCORE_POOL,
                replace=False,
            )

            score_points = pool[score_indices]
        else:
            score_points = pool

        best_normal = None
        best_d = None
        best_score = np.inf

        # --------------------------------------------------------------
        # Generate random plane hypotheses
        # --------------------------------------------------------------
        for _ in range(self.ransac_iterations):
            idx = self.rng.choice(len(pool), size=3, replace=False)
            model = self._plane_from_three(pool[idx])

            if model is None:
                continue

            normal, d = model

            residuals = score_points @ normal + d

            # ----------------------------------------------------------
            # Remove arbitrary offset while evaluating orientation.
            #
            # For the correct orientation, the dominant planar
            # population will form a narrow residual distribution even
            # if the sampled three-point plane is displaced slightly.
            # ----------------------------------------------------------
            residual_center = np.median(residuals)

            centered = np.abs(residuals - residual_center)

            score = float(np.quantile(centered, self._RANSAC_SCORE_QUANTILE))

            if score < best_score:
                best_score = score
                best_normal = normal.copy()

                # Shift the hypothesis so that the median residual of
                # the dominant point population is zero.
                best_d = float(d - residual_center)

        if best_normal is None:
            raise RuntimeError("RANSAC failed to find a valid plane.")

        # Standardize normal direction.
        if best_normal[2] < 0:
            best_normal = -best_normal
            best_d = -best_d

        return best_normal, best_d

    # ==================================================================
    # Nonlinear refinement
    # ==================================================================

    def _nonlinear_refine(
        self,
        points: np.ndarray,
        covariances: np.ndarray | None,
        initial_normal: np.ndarray,
        initial_d: float
    ):
        """
        Refine plane parameters using robust nonlinear least squares.
        """
        theta0, phi0 = angles_from_normal(initial_normal)

        x0 = np.array(
            [
                theta0,
                phi0,
                initial_d,
            ],
            dtype=float,
        )

        def residuals(x):
            theta, phi, d = x

            normal = normal_from_angles(
                theta,
                phi,
            )

            raw = points @ normal + d

            if covariances is None:
                return raw

            # Projection of the XYZ measurement covariance onto the
            # current plane-normal direction:
            #
            #     var(r_i) = n^T C_i n
            #
            variance = np.einsum(
                "i,nij,j->n",
                normal,
                covariances,
                normal,
            )

            sigma = np.sqrt(np.maximum(variance,self._VARIANCE_FLOOR))

            return raw / sigma

        result = least_squares(
            residuals,
            x0=x0,
            loss="soft_l1",
            f_scale=1.0,
            jac="2-point",
            max_nfev=300,
        )

        theta, phi, d = result.x

        normal = normal_from_angles(
            theta,
            phi,
        )

        # Preserve previous normal convention.
        if normal[2] < 0:
            normal = -normal
            d = -d

            theta, phi = angles_from_normal(normal)

        return (
            result,
            normal,
            float(d),
            float(theta),
            float(phi),
        )

    # ==================================================================
    # Main API
    # ==================================================================

    def fit(self, points: np.ndarray, covariances: np.ndarray | None = None) -> PlaneFitResult:
        """
        Fit the dominant planar C-frame facet robustly.

        Parameters
        ----------
        points
            XYZ point array with shape (N, 3).

        covariances
            Optional per-point covariance matrices with shape
            (N, 3, 3).

        Returns
        -------
        PlaneFitResult
            Same output structure as the previous implementation.
        """
        points = np.asarray(
            points,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Validate points
        # --------------------------------------------------------------
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")

        npts = len(points)

        if npts < 3:
            raise ValueError("At least 3 points are required for a plane fit.")

        if not np.all(np.isfinite(points)):
            raise ValueError("points contain NaN or infinite values.")

        # --------------------------------------------------------------
        # Validate covariance input
        # --------------------------------------------------------------
        if covariances is not None:
            covariances = np.asarray(
                covariances,
                dtype=float,
            )

            if covariances.shape != (
                npts,
                3,
                3,
            ):
                raise ValueError("covariances must have shape (N, 3, 3).")

            if not np.all(np.isfinite(covariances)):
                raise ValueError("covariances contain NaN or infinite values.")

        # ==============================================================
        # STEP 1
        # Robust RANSAC orientation estimate
        # ==============================================================
        initial_normal, initial_d = self._initial_ransac_model(points)

        # ==============================================================
        # STEP 2
        # Estimate actual facet scatter using ALL points
        # ==============================================================
        initial_residuals = points @ initial_normal + initial_d

        residual_center, _ = self._robust_sigma(initial_residuals)

        # Correct residual offset before classification.
        initial_d -= residual_center

        initial_residuals = points @ initial_normal + initial_d

        inlier_mask, _, _ = self._classify_inliers(initial_residuals)

        if np.sum(inlier_mask) < 3:
            raise RuntimeError(
                "Robust plane initialization produced fewer than three inliers."
            )

        current_normal = initial_normal

        current_d = initial_d

        # ==============================================================
        # STEP 3
        # Iterative classify -> orthogonal fit -> nonlinear refinement
        # ==============================================================
        for _ in range(self._MAX_REFINEMENT_CYCLES):
            previous_mask = inlier_mask.copy()

            Pin = points[inlier_mask]

            Cin = covariances[inlier_mask] if covariances is not None else None

            # ----------------------------------------------------------
            # Orthogonal SVD refinement first.
            #
            # This gives the nonlinear optimizer a very stable starting
            # orientation even when the RANSAC plane came from three
            # noisy points.
            # ----------------------------------------------------------
            svd_normal, svd_d, _ = self._orthogonal_plane_fit(Pin)

            # ----------------------------------------------------------
            # Robust nonlinear refinement including measurement
            # covariance when available.
            # ----------------------------------------------------------
            (
                _,
                current_normal,
                current_d,
                _,
                _,
            ) = self._nonlinear_refine(
                points=Pin,
                covariances=Cin,
                initial_normal=svd_normal,
                initial_d=svd_d,
            )

            # ----------------------------------------------------------
            # Recenter the refined plane using the median residual of the
            # complete point set.
            #
            # Because the C-frame facet is assumed to be the dominant
            # population, this is robust against the contaminating
            # geometry.
            # ----------------------------------------------------------
            all_residuals = points @ current_normal + current_d

            center, _ = self._robust_sigma(all_residuals)

            current_d -= center

            all_residuals = points @ current_normal + current_d

            # ----------------------------------------------------------
            # Adaptive reclassification
            # ----------------------------------------------------------
            (
                inlier_mask,
                _,
                _,
            ) = self._classify_inliers(all_residuals)

            n_inliers = int(np.sum(inlier_mask))

            if n_inliers < 3:
                raise RuntimeError("Plane refinement left fewer than three inliers.")

            # ----------------------------------------------------------
            # Check mask convergence
            # ----------------------------------------------------------
            changed = np.count_nonzero(inlier_mask != previous_mask)

            changed_fraction = changed / npts

            if changed_fraction <= self._MASK_CONVERGENCE_FRACTION:
                break

        # ==============================================================
        # STEP 4
        # Final nonlinear fit using the converged inlier population
        # ==============================================================
        Pin = points[inlier_mask]

        Cin = covariances[inlier_mask] if covariances is not None else None

        # Stable orthogonal starting solution.
        final_svd_normal, final_svd_d, _ = self._orthogonal_plane_fit(Pin)

        (
            opt,
            normal,
            d,
            theta,
            phi,
        ) = self._nonlinear_refine(
            points=Pin,
            covariances=Cin,
            initial_normal=final_svd_normal,
            initial_d=final_svd_d,
        )

        # ==============================================================
        # STEP 5
        # Final inlier classification against final plane
        # ==============================================================
        residuals_all = points @ normal + d

        center, _ = self._robust_sigma(residuals_all)

        d -= center

        residuals_all = points @ normal + d

        (
            final_mask,
            _,
            _,
        ) = self._classify_inliers(residuals_all)

        # If classification changed appreciably after the final fit,
        # perform one final optimization using exactly that population.
        mask_change_fraction = np.count_nonzero(final_mask != inlier_mask) / npts

        if (
            mask_change_fraction > self._MASK_CONVERGENCE_FRACTION
            and np.sum(final_mask) >= 3
        ):
            inlier_mask = final_mask

            Pin = points[inlier_mask]

            Cin = covariances[inlier_mask] if covariances is not None else None

            final_svd_normal, final_svd_d, _ = self._orthogonal_plane_fit(Pin)

            (
                opt,
                normal,
                d,
                theta,
                phi,
            ) = self._nonlinear_refine(
                points=Pin,
                covariances=Cin,
                initial_normal=final_svd_normal,
                initial_d=final_svd_d,
            )

        else:
            inlier_mask = final_mask

        # ==============================================================
        # STEP 6
        # Final residual statistics
        # ==============================================================
        Pin = points[inlier_mask]

        raw_resid = Pin @ normal + d

        rms = float(np.sqrt(np.mean(raw_resid**2)))

        # ==============================================================
        # STEP 7
        # Parameter covariance
        # ==============================================================
        #
        # scipy.optimize.least_squares returns the effective Jacobian
        # corresponding to the robust optimization.
    
        # ==============================================================
        J = np.asarray(
            opt.jac,
            dtype=float,
        )

        dof = max(
            len(Pin) - 3,
            1,
        )

        s2 = float(np.sum(opt.fun**2) / dof)

        information_matrix = J.T @ J

        cov_params = np.linalg.pinv(information_matrix) * s2

        # ==============================================================
        # STEP 8
        # Project inlier centroid onto the final plane
        # ==============================================================
        centroid_all = np.mean(
            Pin,
            axis=0,
        )

        centroid_proj = (
            centroid_all
            - (
                np.dot(
                    normal,
                    centroid_all,
                )
                + d
            )
            * normal
        )

        # ==============================================================
        # STEP 9
        # Rotation aligning plane normal with +Z
        # ==============================================================
        R = rotation_align_normal_to_z(normal)

        # ==============================================================
        # Return EXACTLY the same result structure as before
        # ==============================================================
        return PlaneFitResult(
            normal=normal,
            d=float(d),
            centroid=centroid_proj,
            inlier_mask=inlier_mask,
            rms_mm=rms,
            params=np.array(
                [
                    theta,
                    phi,
                    d,
                ],
                dtype=float,
            ),
            covariance_params=cov_params,
            rotation_to_xy=R,
        )