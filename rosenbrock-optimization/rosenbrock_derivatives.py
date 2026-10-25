
"""Rosenbrock objective and exact first/second derivatives."""
import numpy as np


def objective(x):
    # Convert input to a floating-point NumPy vector.
    x = np.asarray(x, dtype=float)

    # Rosenbrock is defined here for vectors of length at least 2.
    if x.ndim != 1 or x.size < 2:
        raise ValueError('x must be a 1D vector of length >= 2')

    # r[i] = x[i+1] - x[i]^2 for each neighboring pair.
    r = x[1:] - x[:-1] ** 2

    # Add all Rosenbrock terms.
    return float(np.sum(100.0 * r**2 + (1.0 - x[:-1])**2))


def gradient(x):
    # Convert input and check its dimensions.
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError('x must be a 1D vector of length >= 2')

    # Neighboring residuals, reused in the derivative formulas.
    r = x[1:] - x[:-1] ** 2

    # Allocate one partial derivative for every coordinate.
    g = np.zeros_like(x)

    # Contribution from each term's left variable x[i].
    g[:-1] += -400.0 * x[:-1] * r + 2.0 * (x[:-1] - 1.0)

    # Contribution from each term's right variable x[i+1].
    # Interior coordinates receive contributions from both sides.
    g[1:] += 200.0 * r

    return g


def hessian(x):
    # Convert input and check its dimensions.
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError('x must be a 1D vector of length >= 2')

    # H is n-by-n because there are n optimization variables.
    n = x.size
    H = np.zeros((n, n))

    # Indices corresponding to the n-1 neighboring pairs.
    idx = np.arange(n - 1)

    # Second derivatives with respect to left variables.
    H[idx, idx] += 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0

    # Second derivatives with respect to right variables.
    H[idx + 1, idx + 1] += 200.0

    # Mixed derivatives: neighboring coordinates interact.
    H[idx, idx + 1] = -400.0 * x[:-1]

    # Hessian symmetry: H[i+1,i] = H[i,i+1].
    H[idx + 1, idx] = -400.0 * x[:-1]

    # The resulting Hessian is symmetric and tridiagonal.
    return H


def alternating_start(n):
    # Construct (-1.2, 1, -1.2, 1, ...).
    x = np.ones(n)
    x[::2] = -1.2
    return x
