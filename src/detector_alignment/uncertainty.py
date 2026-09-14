from __future__ import annotations

import numpy as np

from .math3d import (
    normal_from_angles,
    rigid_2d_registration,
    rotation_align_normal_to_z,
    transform_params,
)

# ======================================================================
# Joint covariance
# ======================================================================


def build_joint_covariance(
    plane_cov: np.ndarray,
    references,
) -> np.ndarray:
    """
    Build the joint covariance matrix for

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

    Parameters
    ----------
    plane_cov
        Covariance matrix of the fitted plane parameters

            [theta, phi, d]

        with shape (3, 3).

    references
        Sequence of directly measured reference-point result objects.

        Each object must provide:

            covariance_center_conditional : (2, 2)
            jacobian_center_wrt_plane     : (2, 3)

    Notes
    -----
    Each reference-point XY coordinate depends on the same fitted plane.

    Therefore, even if the directly measured scanner points themselves
    are statistically independent, their leveled XY coordinates are
    correlated through the common plane uncertainty.

    For reference i:

        C_i =
            C_i_conditional
            + J_i C_plane J_i^T

    For two different references i and j:

        C_ij =
            J_i C_plane J_j^T

    The cross-covariance between the plane parameters and reference i is:

        C_plane,i =
            C_plane J_i^T
    """

    # ------------------------------------------------------------------
    # Validate plane covariance
    # ------------------------------------------------------------------

    plane_cov = np.asarray(
        plane_cov,
        dtype=float,
    )

    if plane_cov.shape != (3, 3):
        raise ValueError("plane_cov must have shape (3, 3).")

    if not np.all(np.isfinite(plane_cov)):
        raise ValueError("plane_cov contains non-finite values.")

    # Protect against tiny asymmetry caused by numerical calculations.
    plane_cov = 0.5 * (plane_cov + plane_cov.T)

    n_references = len(references)

    if n_references < 1:
        raise ValueError("At least one reference point is required.")

    # ------------------------------------------------------------------
    # Joint state dimension
    # ------------------------------------------------------------------
    #
    # 3 plane parameters
    # +
    # 2 XY coordinates per reference
    # ------------------------------------------------------------------

    m = 3 + 2 * n_references

    C = np.zeros(
        (m, m),
        dtype=float,
    )

    # Plane block.
    C[:3, :3] = plane_cov

    # Cache validated quantities to avoid repeatedly extracting them.
    jacobians = []
    conditional_covariances = []

    for i, ref in enumerate(references):
        J = np.asarray(
            ref.jacobian_center_wrt_plane,
            dtype=float,
        )

        C_cond = np.asarray(
            ref.covariance_center_conditional,
            dtype=float,
        )

        if J.shape != (2, 3):
            raise ValueError(
                f"Reference {i + 1}: "
                "jacobian_center_wrt_plane must have "
                f"shape (2, 3), got {J.shape}."
            )

        if C_cond.shape != (2, 2):
            raise ValueError(
                f"Reference {i + 1}: "
                "covariance_center_conditional must have "
                f"shape (2, 2), got {C_cond.shape}."
            )

        if not np.all(np.isfinite(J)):
            raise ValueError(
                f"Reference {i + 1}: "
                "jacobian_center_wrt_plane contains "
                "non-finite values."
            )

        if not np.all(np.isfinite(C_cond)):
            raise ValueError(
                f"Reference {i + 1}: "
                "covariance_center_conditional contains "
                "non-finite values."
            )

        # Protect against small numerical asymmetry.
        C_cond = 0.5 * (C_cond + C_cond.T)

        jacobians.append(J)

        conditional_covariances.append(C_cond)

    # ------------------------------------------------------------------
    # Fill reference blocks
    # ------------------------------------------------------------------

    for i in range(n_references):
        Ji = jacobians[i]

        si = slice(
            3 + 2 * i,
            3 + 2 * i + 2,
        )

        # --------------------------------------------------------------
        # Plane <-> reference cross-covariance
        # --------------------------------------------------------------
        #
        # ref = f(plane, scanner point)
        #
        # Cov(ref, plane) = J C_plane
        # --------------------------------------------------------------

        cross_plane = Ji @ plane_cov

        C[
            si,
            :3,
        ] = cross_plane

        C[
            :3,
            si,
        ] = cross_plane.T

        # --------------------------------------------------------------
        # Reference diagonal covariance
        # --------------------------------------------------------------

        C[
            si,
            si,
        ] = conditional_covariances[i] + Ji @ plane_cov @ Ji.T

        # --------------------------------------------------------------
        # Reference-reference cross-covariance
        # --------------------------------------------------------------
        #
        # Their direct scanner uncertainties are independent.
        #
        # They are nevertheless correlated because both use the same
        # fitted plane orientation.
        # --------------------------------------------------------------

        for j in range(i):
            Jj = jacobians[j]

            sj = slice(
                3 + 2 * j,
                3 + 2 * j + 2,
            )

            cross = Ji @ plane_cov @ Jj.T

            C[
                si,
                sj,
            ] = cross

            C[
                sj,
                si,
            ] = cross.T

    # ------------------------------------------------------------------
    # Final numerical symmetry
    # ------------------------------------------------------------------

    C = 0.5 * (C + C.T)

    if not np.all(np.isfinite(C)):
        raise ValueError("Joint covariance contains non-finite values.")

    return C


# ======================================================================
# Final transformation from state vector
# ======================================================================


def final_transform_from_q(
    q: np.ndarray,
    nominal_xy: np.ndarray,
):
    """
    Compute the final rigid transformation from the state vector.

    State convention
    ----------------
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

    Transformation convention
    -------------------------
        p_aligned = R @ p + t

    Plane contribution
    ------------------
    theta, phi
        Determine the rotation Rp that aligns the fitted plane normal
        with +Z.

    d
        Determines the Z translation.

        The original fitted plane is

            n . p + d = 0

        and after applying Rp:

            z = -d

        Therefore:

            tz = d

        maps the fitted plane onto z = 0.

    Reference contribution
    ----------------------
    The leveled reference XY coordinates determine:

        - yaw about leveled Z
        - X translation
        - Y translation
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    nominal_xy = np.asarray(
        nominal_xy,
        dtype=float,
    )

    if q.ndim != 1:
        raise ValueError("q must be a one-dimensional array.")

    if len(q) < 7:
        raise ValueError(
            "q must contain plane parameters and at least two reference XY coordinates."
        )

    if (len(q) - 3) % 2 != 0:
        raise ValueError("Reference portion of q must contain XY pairs.")

    n_references = (len(q) - 3) // 2

    if nominal_xy.shape != (
        n_references,
        2,
    ):
        raise ValueError(
            f"nominal_xy must have shape ({n_references}, 2), got {nominal_xy.shape}."
        )

    if n_references < 2:
        raise ValueError(
            "At least two reference points are required "
            "to determine in-plane rotation and translation."
        )

    # ------------------------------------------------------------------
    # Extract state
    # ------------------------------------------------------------------

    plane = q[:3]

    centers = q[3:].reshape(
        -1,
        2,
    )

    theta = float(plane[0])

    phi = float(plane[1])

    d = float(plane[2])

    # ------------------------------------------------------------------
    # Plane rotation
    # ------------------------------------------------------------------

    n = normal_from_angles(
        theta,
        phi,
    )

    Rp = rotation_align_normal_to_z(n)

    # ------------------------------------------------------------------
    # Remaining 2D rigid registration
    # ------------------------------------------------------------------
    #
    # The reference points are already represented in leveled XY.
    #
    # Solve:
    #
    #     xy_nominal = R2 @ xy_measured + t2
    #
    # ------------------------------------------------------------------

    R2, t2, yaw = rigid_2d_registration(
        centers,
        nominal_xy,
    )

    R2 = np.asarray(
        R2,
        dtype=float,
    )

    t2 = np.asarray(
        t2,
        dtype=float,
    )

    if R2.shape != (2, 2):
        raise ValueError(
            "rigid_2d_registration returned an invalid 2D rotation matrix."
        )

    if t2.shape != (2,):
        raise ValueError(
            "rigid_2d_registration returned an invalid 2D translation vector."
        )

    # ------------------------------------------------------------------
    # Embed the in-plane yaw into 3D
    # ------------------------------------------------------------------

    Rz = np.eye(
        3,
        dtype=float,
    )

    Rz[
        :2,
        :2,
    ] = R2

    # First level the plane, then apply the in-plane rotation.
    R = Rz @ Rp

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    #
    # X/Y come from reference-point registration.
    #
    # Z comes from the plane equation.
    #
    # After Rp:
    #
    #     z_plane = -d
    #
    # therefore:
    #
    #     tz = d
    #
    # Rz does not modify Z.
    # ------------------------------------------------------------------

    t = np.array(
        [
            t2[0],
            t2[1],
            d,
        ],
        dtype=float,
    )

    return (
        R,
        t,
        float(yaw),
    )


# ======================================================================
# Final transform covariance propagation
# ======================================================================


def propagate_final_transform_covariance(
    q0: np.ndarray,
    Cq: np.ndarray,
    nominal_xy: np.ndarray,
):
    """
    Propagate the complete state covariance into the final rigid
    transformation parameters.

    The returned covariance uses the parameter convention:

        [
            rx,
            ry,
            rz,
            tx,
            ty,
            tz
        ]

    where [rx, ry, rz] is the rotation vector representation produced
    by ``transform_params``.

    A numerical central-difference Jacobian is used because the state
    dimension is very small:

        3 + 2*N_reference

    so the computational cost is negligible compared with fitting the
    original point cloud.
    """

    q0 = np.asarray(
        q0,
        dtype=float,
    )

    Cq = np.asarray(
        Cq,
        dtype=float,
    )

    nominal_xy = np.asarray(
        nominal_xy,
        dtype=float,
    )

    if q0.ndim != 1:
        raise ValueError("q0 must be a one-dimensional array.")

    n_state = len(q0)

    if Cq.shape != (
        n_state,
        n_state,
    ):
        raise ValueError(f"Cq must have shape ({n_state}, {n_state}), got {Cq.shape}.")

    if not np.all(np.isfinite(q0)):
        raise ValueError("q0 contains non-finite values.")

    if not np.all(np.isfinite(Cq)):
        raise ValueError("Cq contains non-finite values.")

    # Protect against numerical covariance asymmetry.
    Cq = 0.5 * (Cq + Cq.T)

    # ------------------------------------------------------------------
    # Nominal transformation
    # ------------------------------------------------------------------

    R0, t0, yaw = final_transform_from_q(
        q0,
        nominal_xy,
    )

    # The nominal transform parameter vector is evaluated mainly to
    # establish the expected output shape.
    y0 = transform_params(
        R0,
        t0,
    )

    y0 = np.asarray(
        y0,
        dtype=float,
    )

    if y0.shape != (6,):
        raise ValueError("transform_params must return a vector with shape (6,).")

    # ------------------------------------------------------------------
    # Numerical Jacobian
    # ------------------------------------------------------------------

    J = np.zeros(
        (
            6,
            n_state,
        ),
        dtype=float,
    )

    # Standard deviations of state parameters.
    state_sigma = np.sqrt(
        np.maximum(
            np.diag(Cq),
            0.0,
        )
    )

    for j in range(n_state):
        # --------------------------------------------------------------
        # Parameter-specific minimum finite-difference steps
        # --------------------------------------------------------------
        #
        # q[0] theta : radians
        # q[1] phi   : radians
        # q[2] d     : millimetres
        # q[3:] refs : millimetres
        #
        # Using separate floors avoids treating the plane offset d as
        # though it were an angular parameter.
        # --------------------------------------------------------------

        if j < 2:
            # Angular plane parameters [rad].
            h_floor = 1.0e-8

        elif j == 2:
            # Plane offset [mm].
            h_floor = 1.0e-6

        else:
            # Reference XY coordinates [mm].
            h_floor = 1.0e-6

        # Use a fraction of the actual uncertainty when available.
        h = max(
            0.2 * state_sigma[j],
            h_floor,
        )

        qp = q0.copy()
        qm = q0.copy()

        qp[j] += h
        qm[j] -= h

        # --------------------------------------------------------------
        # Positive perturbation
        # --------------------------------------------------------------

        Rp, tp, _ = final_transform_from_q(
            qp,
            nominal_xy,
        )

        yp = np.asarray(
            transform_params(
                Rp,
                tp,
            ),
            dtype=float,
        )

        # --------------------------------------------------------------
        # Negative perturbation
        # --------------------------------------------------------------

        Rm, tm, _ = final_transform_from_q(
            qm,
            nominal_xy,
        )

        ym = np.asarray(
            transform_params(
                Rm,
                tm,
            ),
            dtype=float,
        )

        # --------------------------------------------------------------
        # Central derivative
        # --------------------------------------------------------------
        #
        # Rotation-vector coordinates are locally continuous for the
        # small detector alignment rotations expected here.
        # --------------------------------------------------------------

        J[
            :,
            j,
        ] = (yp - ym) / (2.0 * h)

    # ------------------------------------------------------------------
    # First-order covariance propagation
    # ------------------------------------------------------------------

    Cy = J @ Cq @ J.T

    # Numerical cleanup.
    Cy = 0.5 * (Cy + Cy.T)

    if not np.all(np.isfinite(Cy)):
        raise ValueError("Final transformation covariance contains non-finite values.")

    return (
        R0,
        t0,
        yaw,
        Cy,
    )
