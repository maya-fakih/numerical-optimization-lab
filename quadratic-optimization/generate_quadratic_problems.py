"""Exercise 2: reproducibly generate paired original/modified quadratic instances."""
from pathlib import Path
import numpy as np

# The homework fixes the dimension and seeds.
DIMENSION = 50
SEEDS = range(1, 6)
BASE = Path(__file__).resolve().parent


def make_instance(seed, modified=False):
    """Return Q, q, minimizer and spectrum; same B/q for paired instances."""
    # A fresh generator with this seed guarantees paired runs use identical B, u, q.
    rng = np.random.default_rng(seed)
    # B has independent standard normal entries.
    B = rng.normal(0.0, 1.0, size=(DIMENSION, DIMENSION))
    # A = B B^T is symmetric positive semidefinite.
    A = B @ B.T
    # eigvalsh is for real symmetric matrices and returns ascending eigenvalues.
    lam0 = float(np.linalg.eigvalsh(A)[0])
    # Draw u even in the modified case, keeping the random stream identical.
    u = float(rng.uniform(0.0, 1.0))
    # N(0,100) means VARIANCE 100, hence standard deviation 10.
    q = rng.normal(loc=0.0, scale=10.0, size=DIMENSION)
    # Q2.1 uses lambda0+u; Q2.2 changes only the shift to lambda0+10.
    shift = lam0 + (10.0 if modified else u)
    Q = A + shift * np.eye(DIMENSION)
    # Symmetric eigenvalues: mu smallest, L largest, kappa their ratio.
    eigs = np.linalg.eigvalsh(Q)
    mu, L = float(eigs[0]), float(eigs[-1])
    kappa = L / mu
    # Solve Q x* = -q instead of explicitly constructing Q inverse.
    x_star = np.linalg.solve(Q, -q)
    # The assignment requires the SAME zero initial vector for all methods.
    x0 = np.zeros(DIMENSION)
    return dict(seed=seed, modified=modified, B=B, A=A, lam0=lam0,
                u=u, Q=Q, q=q, x0=x0, x_star=x_star,
                mu=mu, L=L, kappa=kappa)


def objective(Q, q, x):
    """F(x) = (1/2)x^T Q x + q^T x."""
    return float(0.5 * x @ Q @ x + q @ x)


def gradient(Q, q, x):
    """Gradient Qx+q because Q is symmetric."""
    return Q @ x + q


def relative_distance(x, x_star, x0):
    """Assignment's relative error: denominator ||x0-x*||, not ||x*||."""
    return float(np.linalg.norm(x - x_star) / np.linalg.norm(x0 - x_star))


if __name__ == '__main__':
    # Confirm both sets have positive eigenvalues and accurate minimizers.
    for modified in (False, True):
        print('\nMODIFIED' if modified else '\nORIGINAL')
        for seed in SEEDS:
            p = make_instance(seed, modified)
            residual = np.linalg.norm(gradient(p['Q'], p['q'], p['x_star']))
            assert np.allclose(p['Q'], p['Q'].T)
            assert p['mu'] > 0
            assert residual < 1e-9
            print(f"seed={seed}  mu={p['mu']:.5g}  L={p['L']:.5g}  "
                  f"kappa={p['kappa']:.5g}  ||grad(x*)||={residual:.2e}")
