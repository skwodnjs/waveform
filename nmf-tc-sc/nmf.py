import numpy as np


def nmf_tc_sc(nonnegative_matrix, n_components, max_iter=1000, eps=1e-10):
    """
    Input:
        nonnegative_matrix: Nonnegative matrix to factorize.
        n_components: Number of components.
        max_iter: Number of iterations.
        eps: Small value to avoid division by zero.

    Output:
        W: Basis matrix.
        H: Activation matrix.
    """

    X = np.asarray(nonnegative_matrix, dtype=np.float64)

    if np.any(X < 0):
        raise ValueError("nonnegative_matrix must be nonnegative.")

    F, T = X.shape
    J = n_components

    # ============================================================
    # 1. Initialization
    # ============================================================

    rng = np.random.default_rng(0)

    W = rng.random((F, J)) + eps
    H = rng.random((J, T)) + eps

    # ============================================================
    # 2. Multiplicative Updates
    # ============================================================

    for _ in range(max_iter):

        # --------------------------------------------------------
        # Update H
        # --------------------------------------------------------

        numerator_H = W.T @ X
        denominator_H = W.T @ W @ H

        H *= numerator_H / (denominator_H + eps)

        # --------------------------------------------------------
        # Update W
        # --------------------------------------------------------

        numerator_W = X @ H.T
        denominator_W = W @ H @ H.T

        W *= numerator_W / (denominator_W + eps)

    return W, H