#!/usr/bin/env python3

"""
ip_adc_lanczos.py

Validation of a symmetric Lanczos implementation for PySCF IP-ADC.

PySCF's restricted IP-ADC matrix representation is not symmetric
with respect to the ordinary Euclidean scalar product. Instead,

    A^T S = S A

with

    S = diag(I_1h, 2 I - P_jk)

where P_jk exchanges the two occupied indices of a 2h1p
amplitude r[a,j,k].

Since S is positive definite, define

    A_sym = S^(1/2) A S^(-1/2).

A_sym is symmetric and has exactly the same eigenvalues as A.

This script:
    1. Runs RHF and ADC(2)
    2. Constructs the IP-ADC matvec
    3. Implements S^(+/-1/2) matrix-free
    4. Constructs a symmetric transformed matvec
    5. Tests symmetry numerically
    6. Runs standard symmetric Lanczos
    7. Compares Lanczos Ritz values with PySCF Davidson
    8. Computes Ritz residual norms
    9. Checks Lanczos-basis orthogonality

The explicit ADC matrix is NOT constructed during Lanczos.
"""

import numpy as np

from pyscf import gto, scf, adc
from pyscf.adc import radc_ip


HARTREE2EV = 27.211386245988


# ============================================================
# Metric transformations
# ============================================================

def apply_sqrt_metric(x, nocc, nvir):
    """
    Apply S^(1/2) to an IP-ADC vector.

    The IP vector is

        x = (x_1h, x_2h1p)

    and

        S = diag(I, 2I - P_jk).

    For the 2h1p sector,

        S^(1/2) = a I + b P

    with

        a = (1 + sqrt(3))/2
        b = (1 - sqrt(3))/2.
    """

    x = np.asarray(x)
    y = np.array(x, copy=True)

    ns = nocc

    r2 = x[ns:].reshape(nvir, nocc, nocc)

    sqrt3 = np.sqrt(3.0)

    a = 0.5 * (1.0 + sqrt3)
    b = 0.5 * (1.0 - sqrt3)

    y[ns:] = (
        a * r2
        + b * r2.transpose(0, 2, 1)
    ).reshape(-1)

    return y


def apply_invsqrt_metric(x, nocc, nvir):
    """
    Apply S^(-1/2) to an IP-ADC vector.

    For the 2h1p sector,

        S^(-1/2) = a I + b P

    with

        a = (1 + 1/sqrt(3))/2
        b = (1 - 1/sqrt(3))/2.
    """

    x = np.asarray(x)
    y = np.array(x, copy=True)

    ns = nocc

    r2 = x[ns:].reshape(nvir, nocc, nocc)

    invsqrt3 = 1.0 / np.sqrt(3.0)

    a = 0.5 * (1.0 + invsqrt3)
    b = 0.5 * (1.0 - invsqrt3)

    y[ns:] = (
        a * r2
        + b * r2.transpose(0, 2, 1)
    ).reshape(-1)

    return y


# ============================================================
# Standard symmetric Lanczos
# ============================================================

def lanczos(matvec, v0, maxiter=100, tol=1.0e-12,
            reorthogonalize=True, verbose=True):
    """
    Standard symmetric Lanczos algorithm.

    Full two-pass reorthogonalization is used by default.

    Parameters
    ----------
    matvec : callable
        Function returning A @ x for a symmetric matrix A.

    v0 : ndarray
        Initial vector.

    maxiter : int
        Maximum Krylov-space dimension.

    tol : float
        Lanczos breakdown threshold.

    reorthogonalize : bool
        Perform full two-pass reorthogonalization.

    verbose : bool
        Print iteration information.

    Returns
    -------
    alpha : ndarray
        Diagonal elements of the Lanczos tridiagonal matrix.

    beta : ndarray
        Off-diagonal elements.

    Q : ndarray
        Orthonormal Lanczos vectors stored as columns.
    """

    q = np.asarray(v0, dtype=float).copy()

    norm = np.linalg.norm(q)

    if norm < tol:
        raise ValueError("Initial Lanczos vector has zero norm.")

    q /= norm

    n = q.size
    maxiter = min(maxiter, n)

    Q = np.zeros((n, maxiter), dtype=float)

    alpha = np.zeros(maxiter, dtype=float)
    beta = np.zeros(maxiter - 1, dtype=float)

    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    m = 0

    for j in range(maxiter):

        Q[:, j] = q

        # ----------------------------------------------
        # Matrix-vector product
        # ----------------------------------------------

        z = np.asarray(matvec(q), dtype=float)

        # ----------------------------------------------
        # alpha_j
        # ----------------------------------------------

        alpha[j] = np.dot(q, z)

        # ----------------------------------------------
        # Three-term recurrence
        # ----------------------------------------------

        z -= alpha[j] * q

        if j > 0:
            z -= beta_prev * q_prev

        # ----------------------------------------------
        # Full reorthogonalization
        #
        # Two passes provide robust numerical
        # orthogonality for validation purposes.
        # ----------------------------------------------

        if reorthogonalize:

            Qj = Q[:, :j + 1]

            for _ in range(2):

                coeff = Qj.T @ z
                z -= Qj @ coeff

        b = np.linalg.norm(z)

        m = j + 1

        if verbose:

            print(
                f"Iteration {j+1:4d}: "
                f"alpha = {alpha[j]:18.12f}   "
                f"beta = {b:12.5e}"
            )

        # ----------------------------------------------
        # Lanczos breakdown
        # ----------------------------------------------

        if b < tol:

            if verbose:
                print(
                    "\nLanczos breakdown/convergence "
                    f"after {m} iterations."
                )

            break

        if j == maxiter - 1:
            break

        beta[j] = b

        q_prev = q.copy()
        q = z / b
        beta_prev = b

    alpha = alpha[:m]

    if m > 1:
        beta = beta[:m - 1]
    else:
        beta = np.empty(0)

    Q = Q[:, :m]

    return alpha, beta, Q


# ============================================================
# Construct Lanczos tridiagonal matrix
# ============================================================

def make_tridiagonal(alpha, beta):
    """
    Construct the symmetric tridiagonal Lanczos matrix.
    """

    m = len(alpha)

    T = np.diag(alpha.copy())

    if m > 1:

        idx = np.arange(m - 1)

        T[idx, idx + 1] = beta
        T[idx + 1, idx] = beta

    return T

# ============================================================
# Symmetric Block-Lanczos
# ============================================================

def block_lanczos(
    matvec,
    V0,
    max_blocks=20,
    tol=1.0e-12,
    reorthogonalize=True,
    verbose=True,
):
    """
    Symmetric Block-Lanczos with full two-pass
    reorthogonalization.

    Parameters
    ----------
    matvec : callable
        Function returning H @ x, where H is symmetric.

    V0 : ndarray, shape (n, p)
        Initial block.

    max_blocks : int
        Maximum number of Lanczos blocks.

    tol : float
        Breakdown threshold.

    reorthogonalize : bool
        Perform full two-pass block reorthogonalization.

    verbose : bool
        Print iteration information.

    Returns
    -------
    A_blocks : list of ndarray
        Diagonal blocks A_j.

    B_blocks : list of ndarray
        Off-diagonal blocks B_{j+1}.

        B_blocks[j] couples block j and j+1.

    Q_blocks : list of ndarray
        Orthonormal Lanczos blocks.
    """

    V0 = np.asarray(V0, dtype=float)

    if V0.ndim != 2:
        raise ValueError("V0 must be a two-dimensional array.")

    n, p = V0.shape

    # --------------------------------------------------------
    # Orthonormalize initial block
    # --------------------------------------------------------

    Q0, R0 = np.linalg.qr(V0, mode="reduced")

    if np.min(np.abs(np.diag(R0))) < tol:
        raise ValueError(
            "Initial block is rank deficient."
        )

    Q_blocks = [Q0]

    A_blocks = []
    B_blocks = []

    Q_prev = None
    B_prev = None

    for j in range(max_blocks):

        Qj = Q_blocks[j]

        # ----------------------------------------------------
        # H @ Q_j
        #
        # Apply matrix-vector product column by column.
        # ----------------------------------------------------

        Z = np.column_stack(
            [
                matvec(Qj[:, k])
                for k in range(p)
            ]
        )

        # ----------------------------------------------------
        # Diagonal block
        #
        # A_j = Q_j^T H Q_j
        # ----------------------------------------------------

        Aj = Qj.T @ Z

        # Numerical symmetrization only at roundoff level.
        #
        # H is already mathematically symmetric.
        # This merely removes ~machine-precision noise.

        Aj = 0.5 * (Aj + Aj.T)

        A_blocks.append(Aj)

        # ----------------------------------------------------
        # Three-term block recurrence
        # ----------------------------------------------------

        W = Z - Qj @ Aj

        if j > 0:
            W -= Q_prev @ B_prev.T

        # ----------------------------------------------------
        # Full block reorthogonalization
        #
        # For validation we orthogonalize against ALL
        # previously generated Lanczos blocks.
        # ----------------------------------------------------

        if reorthogonalize:

            Qall = np.hstack(Q_blocks)

            for _ in range(2):

                C = Qall.T @ W
                W -= Qall @ C

        # ----------------------------------------------------
        # QR factorization:
        #
        #     W = Q_{j+1} B_{j+1}
        #
        # ----------------------------------------------------

        Qnext, Bnext = np.linalg.qr(
            W,
            mode="reduced"
        )

        # ----------------------------------------------------
        # Diagnose possible block breakdown
        # ----------------------------------------------------

        singular_values = np.linalg.svd(
            W,
            compute_uv=False
        )

        sigma_min = singular_values[-1]
        sigma_max = singular_values[0]

        if verbose:

            print()
            print(
                f"Block iteration {j+1:4d}"
            )

            print(
                "  ||A_j - A_j^T|| = "
                f"{np.linalg.norm(Aj - Aj.T):12.5e}"
            )

            print(
                "  singular values(W): "
                f"max = {sigma_max:12.5e}   "
                f"min = {sigma_min:12.5e}"
            )

        # ----------------------------------------------------
        # Complete block breakdown
        # ----------------------------------------------------

        if sigma_max < tol:

            if verbose:
                print(
                    "\nComplete Block-Lanczos "
                    "breakdown/convergence."
                )

            break

        # ----------------------------------------------------
        # Partial breakdown
        #
        # For this FIRST validation implementation,
        # stop rather than deflate.
        # ----------------------------------------------------

        if sigma_min < tol:

            if verbose:
                print(
                    "\nPartial block breakdown detected."
                )
                print(
                    "Deflation is not implemented in "
                    "this validation version."
                )

            break

        # ----------------------------------------------------
        # Last requested block
        # ----------------------------------------------------

        if j == max_blocks - 1:
            break

        B_blocks.append(Bnext)

        Q_prev = Qj
        B_prev = Bnext

        Q_blocks.append(Qnext)

    return A_blocks, B_blocks, Q_blocks


# ============================================================
# Construct block-tridiagonal Lanczos matrix
# ============================================================

def make_block_tridiagonal(A_blocks, B_blocks):

    nb = len(A_blocks)

    if nb == 0:
        raise ValueError("No Lanczos blocks.")

    p = A_blocks[0].shape[0]

    T = np.zeros(
        (nb * p, nb * p),
        dtype=float
    )

    for j in range(nb):

        s = slice(j*p, (j+1)*p)

        T[s, s] = A_blocks[j]

        if j < nb - 1:

            sp = slice(
                (j+1)*p,
                (j+2)*p
            )

            B = B_blocks[j]

            # lower block
            T[sp, s] = B

            # upper block
            T[s, sp] = B.T

    return T


# ============================================================
# Main calculation
# ============================================================

def main():

    # ========================================================
    # Molecule
    # ========================================================

    mol = gto.M(
        atom="""
        O   0.000000   0.000000   0.000000
        H   0.000000  -0.757000   0.587000
        H   0.000000   0.757000   0.587000
        """,
        basis="cc-pvdz",
        verbose=4,
    )

    # ========================================================
    # RHF
    # ========================================================

    mf = scf.RHF(mol)
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("RHF calculation did not converge.")

    # ========================================================
    # ADC ground-state calculation
    # ========================================================

    myadc = adc.ADC(mf)

    myadc.method = "adc(2)"
    myadc.method_type = "ip"

    myadc.kernel_gs()

    # ========================================================
    # Construct IP-ADC object
    # ========================================================

    myadcip = radc_ip.RADCIP(myadc)

    nocc = myadcip._nocc
    nvir = myadcip._nvir

    # ========================================================
    # Integral transformation / intermediates
    #
    # This follows radc.kernel():
    #
    #   eris = adc.transform_integrals()
    #   imds = adc.get_imds(eris)
    #   matvec, diag = adc.gen_matvec(imds, eris)
    #
    # ========================================================

    eris = myadcip.transform_integrals()

    imds = myadcip.get_imds(eris)

    matvec, diag = myadcip.gen_matvec(imds, eris)

    dim = len(diag)

    print()
    print("========================================")
    print("IP-ADC dimensions")
    print("========================================")
    print(f"nocc             = {nocc}")
    print(f"nvir             = {nvir}")
    print(f"1h dimension     = {nocc}")
    print(f"2h1p dimension   = {nvir*nocc*nocc}")
    print(f"total dimension  = {dim}")

    # ========================================================
    # Test S^(1/2) S^(-1/2) = I
    # ========================================================

    rng = np.random.default_rng(12345)

    x = rng.standard_normal(dim)

    y = apply_invsqrt_metric(x, nocc, nvir)
    z = apply_sqrt_metric(y, nocc, nvir)

    metric_inverse_error = (
        np.linalg.norm(z - x) / np.linalg.norm(x)
    )

    print()
    print("========================================")
    print("Metric transformation test")
    print("========================================")

    print(
        "||sqrt(S) invsqrt(S) x - x|| / ||x|| =",
        metric_inverse_error
    )

    # ========================================================
    # Symmetrized matrix-vector product
    #
    #       A_sym = S^(1/2) A S^(-1/2)
    #
    # ========================================================

    def matvec_sym(x):

        y = apply_invsqrt_metric(
            x, nocc, nvir
        )

        y = matvec(y)

        y = apply_sqrt_metric(
            y, nocc, nvir
        )

        return y

    # ========================================================
    # Numerical symmetry test
    # ========================================================

    print()
    print("========================================")
    print("Symmetry test for transformed matvec")
    print("========================================")

    max_sym_error = 0.0

    for k in range(10):

        x = rng.standard_normal(dim)
        y = rng.standard_normal(dim)

        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)

        Ax = matvec_sym(x)
        Ay = matvec_sym(y)

        lhs = np.dot(x, Ay)
        rhs = np.dot(Ax, y)

        diff = abs(lhs - rhs)

        max_sym_error = max(
            max_sym_error,
            diff
        )

        print(
            f"{k:3d}  "
            f"x^T A_sym y = {lhs:20.12e}  "
            f"(A_sym x)^T y = {rhs:20.12e}  "
            f"diff = {lhs-rhs:12.5e}"
        )

    print()
    print(
        "Maximum symmetry error =",
        max_sym_error
    )

    # ========================================================
    # Initial Lanczos vector
    #
    # For validation, use the 1h configuration associated
    # with the smallest diagonal element.
    #
    # Since it lies entirely in the 1h sector,
    #
    #       S^(1/2) v0 = v0.
    #
    # ========================================================

    i0 = np.argmin(diag)

    v0 = np.zeros(dim)
    v0[i0] = 1.0

    print()
    print("========================================")
    print("Lanczos initial vector")
    print("========================================")

    print("Initial-vector index =", i0)
    print(
        "Corresponding diagonal element =",
        diag[i0]
    )

    # ========================================================
    # Lanczos calculation
    # ========================================================

    maxiter = min(100, dim)

    print()
    print("========================================")
    print("Starting symmetric Lanczos")
    print("========================================")

    alpha, beta, Q = lanczos(
        matvec_sym,
        v0,
        maxiter=maxiter,
        tol=1.0e-12,
        reorthogonalize=True,
        verbose=True,
    )

    # ========================================================
    # Tridiagonal matrix
    # ========================================================

    T = make_tridiagonal(
        alpha,
        beta
    )

    ritz_values, ritz_vectors = np.linalg.eigh(T)

    # ========================================================
    # Lanczos Ritz values
    # ========================================================

    print()
    print("========================================")
    print("Lowest Lanczos Ritz values")
    print("========================================")

    nprint = min(
        10,
        len(ritz_values)
    )

    for i in range(nprint):

        e = ritz_values[i]

        print(
            f"{i:3d}  "
            f"{e:18.12f} Eh   "
            f"{e*HARTREE2EV:14.8f} eV"
        )

    # ========================================================
    # Orthogonality test
    # ========================================================

    orth_error = np.linalg.norm(
        Q.T @ Q
        - np.eye(Q.shape[1])
    )

    print()
    print("========================================")
    print("Lanczos orthogonality")
    print("========================================")

    print(
        "||Q^T Q - I|| =",
        orth_error
    )

    # ========================================================
    # Ritz residuals
    #
    #       r_i = A_sym u_i - theta_i u_i
    #
    # ========================================================

    print()
    print("========================================")
    print("Lanczos Ritz residuals")
    print("========================================")

    ritz_residuals = []

    for i in range(nprint):

        theta = ritz_values[i]

        y = ritz_vectors[:, i]

        u = Q @ y

        residual = (
            matvec_sym(u)
            - theta * u
        )

        resnorm = np.linalg.norm(
            residual
        )

        ritz_residuals.append(
            resnorm
        )

        print(
            f"{i:3d}  "
            f"E = {theta:18.12f}  "
            f"residual = {resnorm:12.5e}"
        )

    # ========================================================
    # Davidson reference
    #
    # This uses the original PySCF IP-ADC representation.
    # A and A_sym are related by similarity transformation,
    # so their eigenvalues must be identical.
    # ========================================================

    nroots = min(100, dim)

    print()
    print("========================================")
    print("PySCF Davidson reference")
    print("========================================")

    E_dav, U_dav, P_dav, X_dav = (
        myadcip.kernel(
            nroots=nroots,
            eris=eris,
        )
    )

    # ========================================================
    # Davidson vs Lanczos
    # ========================================================

    print()
    print("========================================")
    print("Davidson vs Lanczos")
    print("========================================")

    print(
        f"{'root':>5s} "
        f"{'Davidson / Eh':>20s} "
        f"{'Lanczos / Eh':>20s} "
        f"{'difference / Eh':>20s} "
        f"{'residual':>14s}"
    )

    ncompare = min(
        nroots,
        len(ritz_values)
    )

    for i in range(ncompare):

        diff = (
            ritz_values[i]
            - E_dav[i]
        )

        if i < len(ritz_residuals):
            residual = ritz_residuals[i]
        else:
            residual = np.nan

        print(
            f"{i:5d} "
            f"{E_dav[i]:20.12f} "
            f"{ritz_values[i]:20.12f} "
            f"{diff:20.6e} "
            f"{residual:14.5e}"
        )

    # ========================================================
    # Optional:
    # Davidson eigenvectors transformed to symmetric basis
    #
    # If A U = E U, then
    #
    #     U_sym = S^(1/2) U
    #
    # satisfies
    #
    #     A_sym U_sym = E U_sym.
    #
    # Check this for the Davidson states.
    # ========================================================

    print()
    print("========================================")
    print("Transformed Davidson eigenvector test")
    print("========================================")

    for i in range(nroots):

        u_original = U_dav[:, i]

        u_sym = apply_sqrt_metric(
            u_original,
            nocc,
            nvir
        )

        # Normalize in the Euclidean metric of
        # the transformed representation.

        u_sym /= np.linalg.norm(u_sym)

        residual = (
            matvec_sym(u_sym)
            - E_dav[i] * u_sym
        )

        resnorm = np.linalg.norm(
            residual
        )

        overlap = abs(
            np.dot(v0, u_sym)
        )

        print(
            f"{i:3d}  "
            f"E = {E_dav[i]:18.12f}  "
            f"residual = {resnorm:12.5e}  "
            f"|<v0|U_sym>|^2 = {overlap**2:12.5e}"
        )


    # ============================================================
    # Initial Block-Lanczos block
    #
    # For validation use several 1h unit vectors.
    # ============================================================

#   block_size = min(3, nocc)
    block_size = nocc

    # Only inspect the 1h diagonal elements.
    idx = np.argsort(diag[:nocc])[:block_size]

    V0 = np.zeros((dim, block_size))
    V0[:nocc, :] = np.eye(nocc)

    max_blocks = 50

#   for k, i in enumerate(idx):
#       V0[i, k] = 1.0

    print()
    print("========================================")
    print("Block-Lanczos initial block")
    print("========================================")

    print("Block size =", block_size)
    print("1h indices =", idx)

    for k, i in enumerate(idx):

        print(
            f"column {k:3d}: "
            f"index = {i:4d}   "
            f"diag = {diag[i]:18.12f}"
        )

    print()
    print(
        "Initial-block orthogonality error =",
        np.linalg.norm(
            V0.T @ V0
            - np.eye(block_size)
        )
    )


    # ============================================================
    # Block-Lanczos
    # ============================================================

#   max_blocks = min(10, dim // block_size)

    print()
    print("========================================")
    print("Starting symmetric Block-Lanczos")
    print("========================================")

    A_blocks, B_blocks, Q_blocks = block_lanczos(
        matvec_sym,
        V0,
        max_blocks=max_blocks,
        tol=1.0e-12,
        reorthogonalize=True,
        verbose=True,
    )

    T_block = make_block_tridiagonal(
        A_blocks,
        B_blocks
    )

    ritz_values_block, ritz_vectors_block = (
        np.linalg.eigh(T_block)
    )


    Q_block = np.hstack(Q_blocks)

    orth_error_block = np.linalg.norm(
        Q_block.T @ Q_block
        - np.eye(Q_block.shape[1])
    )

    print()
    print("========================================")
    print("Block-Lanczos orthogonality")
    print("========================================")

    print(
        "Krylov dimension =",
        Q_block.shape[1]
    )

    print(
        "||Q^T Q - I|| =",
        orth_error_block
    )


    HQ = np.column_stack(
        [
            matvec_sym(Q_block[:, k])
            for k in range(Q_block.shape[1])
        ]
    )

    T_direct = Q_block.T @ HQ

    print()
    print("========================================")
    print("Block projection test")
    print("========================================")

    print(
        "||T_direct - T_direct^T|| =",
        np.linalg.norm(
            T_direct - T_direct.T
        )
    )

    print(
        "||T_direct - T_block|| =",
        np.linalg.norm(
            T_direct - T_block
        )
    )


    print()
    print("========================================")
    print("Lowest Block-Lanczos Ritz values")
    print("========================================")

    nprint = min(
        30,
        len(ritz_values_block)
    )

    for i in range(nprint):

        e = ritz_values_block[i]

        print(
            f"{i:3d}  "
            f"{e:18.12f} Eh   "
            f"{e*HARTREE2EV:14.8f} eV"
        )


    print()
    print("========================================")
    print("Block-Lanczos Ritz residuals")
    print("========================================")

    block_residuals = []

    for i in range(nprint):

        theta = ritz_values_block[i]

        y = ritz_vectors_block[:, i]

        u = Q_block @ y

        residual = (
            matvec_sym(u)
            - theta * u
        )

        resnorm = np.linalg.norm(
            residual
        )

        block_residuals.append(resnorm)

        print(
            f"{i:3d}  "
            f"E = {theta:18.12f}   "
            f"residual = {resnorm:12.5e}"
        )


    print()
    print("========================================")
    print("Davidson vs Block-Lanczos")
    print("========================================")

    print(
        f"{'BL':>5s} "
        f"{'Dav':>5s} "
        f"{'Davidson / Eh':>20s} "
        f"{'Block-Lanczos / Eh':>22s} "
        f"{'difference / Eh':>20s} "
        f"{'residual':>14s}"
    )

    ncompare = min(
        30,
        len(ritz_values_block)
    )

    for i in range(ncompare):

        theta = ritz_values_block[i]

        # Davidson state closest in energy
        j = np.argmin(
            np.abs(E_dav - theta)
        )

        diff = theta - E_dav[j]

        print(
            f"{i:5d} "
            f"{j:5d} "
            f"{E_dav[j]:20.12f} "
            f"{theta:22.12f} "
            f"{diff:20.6e} "
            f"{block_residuals[i]:14.5e}"
        )



# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()


