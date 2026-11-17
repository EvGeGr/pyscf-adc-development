#!/usr/bin/env python3
"""
Lanczos and block-Lanczos diagonalization for the symmetric IP-ADC matrix.

The code uses the PySCF ADC matrix only through

    adc_ip.gen_matvec()

so the full ADC matrix is never constructed explicitly.

Two algorithms are provided:

    lanczos()          standard symmetric Lanczos
    block_lanczos()   symmetric block Lanczos

The block-Lanczos implementation uses full reorthogonalization. This is
not the most memory-efficient implementation, but it is an excellent
first implementation for testing the algorithm against PySCF Davidson.

IMPORTANT:
    The initial block used here is an algorithmic test block. For a genuine
    Stieltjes photoionization calculation, it must eventually be replaced
    by the physically appropriate dipole-response starting vectors.
"""

import numpy as np


# ============================================================================
# Utility routines
# ============================================================================

def orthonormalize(X, tol=1e-12):
    """
    Orthonormalize the columns of X using SVD.

    Returns
    -------
    Q : ndarray
        Orthonormal columns.
    rank : int
        Numerical rank.
    """
    X = np.asarray(X)

    U, s, _ = np.linalg.svd(X, full_matrices=False)

    if len(s) == 0:
        return np.empty((X.shape[0], 0), dtype=X.dtype), 0

    cutoff = tol * max(s[0], 1.0)
    rank = np.count_nonzero(s > cutoff)

    return U[:, :rank], rank


def orthogonalize_against(X, Q):
    """
    Remove projections of X onto the columns of Q.

    Parameters
    ----------
    X : ndarray, shape (n, m)
    Q : ndarray, shape (n, k)

    Returns
    -------
    Xorth : ndarray
    """
    if Q is None or Q.shape[1] == 0:
        return X

    return X - Q @ (Q.T.conj() @ X)


# ============================================================================
# Standard symmetric Lanczos
# ============================================================================

def lanczos(matvec, q0, niter=50, reorthogonalize=True, tol=1e-12):
    """
    Standard symmetric Lanczos algorithm.

    Parameters
    ----------
    matvec : callable
        Function implementing y = A @ x.
    q0 : ndarray
        Initial vector.
    niter : int
        Maximum number of Lanczos iterations.
    reorthogonalize : bool
        If True, use full reorthogonalization.
    tol : float
        Breakdown threshold.

    Returns
    -------
    alpha : ndarray
        Diagonal elements of the tridiagonal Lanczos matrix.
    beta : ndarray
        Off-diagonal elements.
        beta[j] couples q_j and q_{j+1}.
    Q : ndarray
        Lanczos vectors.
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)

    norm = np.linalg.norm(q0)
    if norm < tol:
        raise ValueError("Initial Lanczos vector has zero norm.")

    q = q0 / norm

    Q = []
    alpha = []
    beta = []

    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    for iteration in range(niter):

        Q.append(q.copy())

        z = np.asarray(matvec(q))

        # Three-term Lanczos recurrence
        a = np.vdot(q, z).real
        z = z - a * q

        if iteration > 0:
            z = z - beta_prev * q_prev

        # Full reorthogonalization.
        #
        # This is intentionally used in the first implementation because
        # it makes comparison with Davidson much more robust.
        if reorthogonalize:
            Qmat = np.column_stack(Q)

            z = z - Qmat @ (Qmat.T.conj() @ z)

        b = np.linalg.norm(z)

        alpha.append(a)

        if b < tol:
            print(
                f"Lanczos breakdown after {iteration + 1} iterations "
                f"(beta = {b:.3e})"
            )
            break

        beta.append(b)

        q_prev = q
        q = z / b
        beta_prev = b

    return (
        np.asarray(alpha),
        np.asarray(beta),
        np.column_stack(Q),
    )


def make_tridiagonal(alpha, beta):
    """
    Construct the Lanczos tridiagonal matrix.
    """
    n = len(alpha)

    T = np.diag(alpha)

    if n > 1:
        T += np.diag(beta[:n - 1], 1)
        T += np.diag(beta[:n - 1], -1)

    return T


# ============================================================================
# Symmetric block Lanczos
# ============================================================================

def block_lanczos(
    matvec,
    B0,
    nblock=20,
    reorthogonalize=True,
    tol=1e-12,
    verbose=True,
):
    """
    Symmetric block Lanczos algorithm.

    Parameters
    ----------
    matvec : callable
        Function implementing Y = A @ X.

        It must accept either:
            - a vector of shape (N,)
            - or a block of shape (N, nblock)

        PySCF's ADC matvec normally accepts vectors one at a time, so this
        implementation automatically falls back to column-by-column
        application when necessary.

    B0 : ndarray
        Initial block, shape (N, block_size).

    nblock : int
        Maximum number of block Lanczos iterations.

    reorthogonalize : bool
        Use full reorthogonalization against all previous Lanczos vectors.

    tol : float
        Numerical breakdown threshold.

    verbose : bool
        Print iteration information.

    Returns
    -------
    A_blocks : list of ndarray
        Diagonal blocks A_j.

    B_blocks : list of ndarray
        Off-diagonal blocks B_j.

    Q_blocks : list of ndarray
        Orthonormal Lanczos blocks.

    T : ndarray
        Complete block-tridiagonal Lanczos matrix.

    Notes
    -----
    The block recurrence is

        W_j = A Q_j - Q_j A_j - Q_{j-1} B_j^T

    followed by

        W_j = Q_{j+1} B_{j+1}

    where

        A_j = Q_j^T A Q_j.

    Because A is assumed symmetric, ordinary symmetric block Lanczos
    is used. No left/right Lanczos vectors are required.
    """

    B0 = np.asarray(B0)

    if B0.ndim == 1:
        B0 = B0[:, None]

    # Initial orthonormalization
    Q0, rank = orthonormalize(B0, tol=tol)

    if rank == 0:
        raise ValueError("Initial block has zero numerical rank.")

    if rank != B0.shape[1]:
        print(
            f"Warning: initial block rank reduced from "
            f"{B0.shape[1]} to {rank}"
        )

    Q_blocks = [Q0]
    A_blocks = []
    B_blocks = []

    Q_previous = None
    B_previous = None

    for j in range(nblock):

        Qj = Q_blocks[j]

        # ------------------------------------------------------------
        # Matrix multiplication
        # ------------------------------------------------------------
        Z = apply_matvec(matvec, Qj)

        # ------------------------------------------------------------
        # Diagonal block
        # ------------------------------------------------------------
        Aj = Qj.T.conj() @ Z

        # Enforce numerical Hermiticity.
        Aj = 0.5 * (Aj + Aj.T.conj())

        A_blocks.append(Aj)

        # ------------------------------------------------------------
        # Block Lanczos residual
        # ------------------------------------------------------------
        W = Z - Qj @ Aj

        if Q_previous is not None:
            W -= Q_previous @ B_previous.T.conj()

        # ------------------------------------------------------------
        # Full reorthogonalization
        # ------------------------------------------------------------
        if reorthogonalize:

            # Reorthogonalize against all existing blocks.
            Qall = np.column_stack(Q_blocks)

            W -= Qall @ (Qall.T.conj() @ W)

            # Second pass is cheap compared with an ADC matrix-vector
            # operation and considerably improves numerical stability.
            W -= Qall @ (Qall.T.conj() @ W)

        # ------------------------------------------------------------
        # QR factorization
        # ------------------------------------------------------------
        Qnext, R = orthonormalize(W, tol=tol)

        if verbose:
            print(
                f"Block Lanczos iteration {j + 1:4d}: "
                f"block size = {Qj.shape[1]}, "
                f"next rank = {Qnext.shape[1] if Qnext.size else 0}"
            )

        # Breakdown
        if Qnext.shape[1] == 0:
            if verbose:
                print("Block Lanczos terminated: numerical breakdown.")
            break

        # The block size should normally remain constant.
        #
        # If the rank decreases, the algorithm still continues, but
        # the resulting block-tridiagonal matrix is no longer composed
        # of equal-size blocks.
        Bnext = R[:Qnext.shape[1], :Qnext.shape[1]]

        B_blocks.append(Bnext)

        Q_previous = Qj
        B_previous = Bnext

        Q_blocks.append(Qnext)

    # Construct complete block-tridiagonal matrix.
    T = assemble_block_tridiagonal(A_blocks, B_blocks)

    return A_blocks, B_blocks, Q_blocks, T


def apply_matvec(matvec, X):
    """
    Apply a PySCF-style matvec to a vector or block.

    PySCF ADC matvec functions are normally vector functions, so for a
    block we apply them column by column.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        return np.asarray(matvec(X))

    Y = np.empty_like(X, dtype=np.result_type(X, float))

    for i in range(X.shape[1]):
        Y[:, i] = matvec(X[:, i])

    return Y


def assemble_block_tridiagonal(A_blocks, B_blocks):
    """
    Assemble the block-tridiagonal Lanczos matrix.

    The matrix has the form

        A1       B1.T
        B1       A2       B2.T
                  B2      A3
                           ...

    """
    if len(A_blocks) == 0:
        return np.empty((0, 0))

    # Determine block dimensions.
    sizes = [A.shape[0] for A in A_blocks]

    offsets = np.zeros(len(sizes) + 1, dtype=int)

    for i, size in enumerate(sizes):
        offsets[i + 1] = offsets[i] + size

    dimension = offsets[-1]

    dtype = np.result_type(*[A.dtype for A in A_blocks])

    T = np.zeros((dimension, dimension), dtype=dtype)

    # Diagonal blocks
    for i, A in enumerate(A_blocks):
        s0 = offsets[i]
        s1 = offsets[i + 1]
        T[s0:s1, s0:s1] = A

    # Off-diagonal blocks
    for i, B in enumerate(B_blocks):

        r0 = offsets[i + 1]
        r1 = offsets[i + 2]

        c0 = offsets[i]
        c1 = offsets[i + 1]

        T[r0:r1, c0:c1] = B
        T[c0:c1, r0:r1] = B.T.conj()

    return T


# ============================================================================
# Ritz diagonalization
# ============================================================================

def diagonalize_lanczos_matrix(T, nroots=None):
    """
    Diagonalize the small Lanczos matrix.

    Returns the lowest nroots Ritz values and vectors.
    """
    from scipy.linalg import eigh

    eigenvalues, eigenvectors = eigh(T)

    if nroots is not None:
        eigenvalues = eigenvalues[:nroots]
        eigenvectors = eigenvectors[:, :nroots]

    return eigenvalues, eigenvectors


def recover_ritz_vectors(Q_blocks, ritz_vectors, nroots=None):
    """
    Transform Lanczos-space eigenvectors back into the original ADC space.

        V_Ritz = Q T_eigenvectors

    """
    Q = np.column_stack(Q_blocks)

    if nroots is not None:
        ritz_vectors = ritz_vectors[:, :nroots]

    return Q @ ritz_vectors


# ============================================================================
# PySCF ADC helper
# ============================================================================

def make_ip_adc_matvec(adc_ip):
    """
    Obtain the IP-ADC matrix-vector operation and diagonal from PySCF.

    Parameters
    ----------
    adc_ip
        A PySCF RADCIP object.

    Returns
    -------
    matvec : callable
    diag : ndarray
    """
    matvec, diag = adc_ip.gen_matvec()
    return matvec, np.asarray(diag)


def make_initial_block_from_adc(adc_ip, block_size=3):
    """
    Construct an initial block using PySCF's standard IP-ADC guesses.

    This is useful for testing the block-Lanczos machinery against
    Davidson.

    IMPORTANT:
        These are NOT yet the physical dipole starting vectors required
        for a Stieltjes photoionization calculation.
    """
    matvec, diag = adc_ip.gen_matvec()

    guesses = adc_ip.get_init_guess(
        nroots=block_size,
        diag=diag,
        ascending=True,
    )

    B = np.column_stack(guesses)

    return B


# ============================================================================
# Simple driver
# ============================================================================

def run_test(adc_ip, nroots=5, niter=20, block_size=3):
    """
    Run both standard Lanczos and block Lanczos.

    This is intended as the first validation test against PySCF Davidson.

    Parameters
    ----------
    adc_ip
        Initialized PySCF RADCIP object.

    nroots : int
        Number of low-energy roots to print.

    niter : int
        Number of Lanczos iterations.

    block_size : int
        Size of the block-Lanczos starting block.
    """

    import scipy.linalg

    matvec, diag = adc_ip.gen_matvec()

    # ------------------------------------------------------------
    # PySCF Davidson reference
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("PySCF Davidson reference")
    print("=" * 72)

    e_ref, v_ref, p_ref, x_ref = adc_ip.kernel(nroots=nroots)

    e_ref = np.atleast_1d(e_ref)

    print()
    print("Davidson IP energies:")
    for i, e in enumerate(e_ref):
        print(f"{i + 1:4d}  {e:18.10f}")

    # ------------------------------------------------------------
    # Standard Lanczos
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("Standard Lanczos")
    print("=" * 72)

    B = make_initial_block_from_adc(adc_ip, block_size=1)
    q0 = B[:, 0]

    alpha, beta, Q = lanczos(
        matvec,
        q0,
        niter=niter,
        reorthogonalize=True,
    )

    T = make_tridiagonal(alpha, beta)

    e_lanczos, u_lanczos = scipy.linalg.eigh(
        T,
        subset_by_index=(0, min(nroots, T.shape[0]) - 1),
    )

    print()
    print("Lanczos Ritz values:")
    for i, e in enumerate(e_lanczos):
        print(f"{i + 1:4d}  {e:18.10f}")

    # ------------------------------------------------------------
    # Block Lanczos
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(f"Block Lanczos  (block size = {block_size})")
    print("=" * 72)

    B0 = make_initial_block_from_adc(
        adc_ip,
        block_size=block_size,
    )

    A_blocks, B_blocks, Q_blocks, T_block = block_lanczos(
        matvec,
        B0,
        nblock=niter,
        reorthogonalize=True,
        verbose=True,
    )

    e_block, u_block = scipy.linalg.eigh(
        T_block,
        subset_by_index=(0, min(nroots, T_block.shape[0]) - 1),
    )

    print()
    print("Block-Lanczos Ritz values:")
    for i, e in enumerate(e_block):
        print(f"{i + 1:4d}  {e:18.10f}")

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("Comparison")
    print("=" * 72)

    ncompare = min(
        nroots,
        len(e_ref),
        len(e_lanczos),
        len(e_block),
    )

    print()
    print(
        f"{'root':>4s}"
        f"{'Davidson':>18s}"
        f"{'Lanczos':>18s}"
        f"{'Block Lanczos':>18s}"
    )

    for i in range(ncompare):
        print(
            f"{i + 1:4d}"
            f"{e_ref[i]:18.10f}"
            f"{e_lanczos[i]:18.10f}"
            f"{e_block[i]:18.10f}"
        )

    return {
        "e_ref": e_ref,
        "e_lanczos": e_lanczos,
        "e_block": e_block,
        "T_lanczos": T,
        "T_block": T_block,
        "Q_lanczos": Q,
        "Q_blocks": Q_blocks,
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":

    from pyscf import gto, scf, adc

    # ------------------------------------------------------------
    # Small test molecule
    # ------------------------------------------------------------

    mol = gto.M(
        atom="""
        N  0.000000  0.000000  0.000000
        N  0.000000  0.000000  1.097700
        """,
        basis="cc-pVDZ",
        charge=0,
        spin=0,
        verbose=4,
    )

    mf = scf.RHF(mol).run()

    # ------------------------------------------------------------
    # Ground-state ADC object
    # ------------------------------------------------------------

   # myadc = adc.RADC(mf)
   # myadc.method = "adc(2)"
   # myadc.kernel()

    # ------------------------------------------------------------
    # IP-ADC object
    # ------------------------------------------------------------

    adc_ip = adc.RADC(mf)
    adc_ip.method = "adc(2)"

    # ------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------

    results = run_test(
        adc_ip,
        nroots=5,
        niter=20,
        block_size=3,
    )

