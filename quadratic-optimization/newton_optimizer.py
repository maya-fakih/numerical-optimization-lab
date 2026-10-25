"""Exercise 2: full-step Newton method for a strictly convex quadratic."""
import numpy as np
from generate_quadratic_problems import gradient


def newton_step(x, Q, q):
    # For F(x)=1/2 x^T Q x + q^T x, the Hessian is the constant matrix Q.
    g = gradient(Q, q, x)
    # Solve Q*s=-g, without explicitly computing Q^{-1}.
    s = np.linalg.solve(Q, -g)
    # A full Newton step has coefficient 1, and reaches x* in exact arithmetic.
    return x + s
