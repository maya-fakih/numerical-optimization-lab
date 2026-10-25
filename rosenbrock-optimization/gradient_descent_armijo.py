
"""Gradient descent with Armijo backtracking."""

import time
import numpy as np
from rosenbrock_derivatives import objective, gradient


def optimize(x0, grad_tol=1e-6, max_iter=2000, alpha0=1.0,
             beta=0.5, c=1e-4):
    """
    x0       = initial point
    grad_tol = gradient stopping tolerance
    max_iter = maximum outer iterations
    alpha0   = initial trial step length
    beta     = backtracking factor (0.5 means halve alpha)
    c        = Armijo sufficient-decrease constant
    """

    # Start the runtime clock.
    start = time.perf_counter()

    # Create a working copy of the initial point.
    x = np.asarray(x0, dtype=float).copy()

    # Record the starting state, iteration 0.
    history = [{
        'iteration': 0,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'x': x.copy()
    }]

    # Default stopping reason.
    reason = 'max_iterations'

    # Each outer iteration attempts one gradient-descent move.
    for k in range(1, max_iter + 1):

        # Compute gradient at the current point.
        g = gradient(x)

        # Stop when the gradient is sufficiently small.
        if np.linalg.norm(g) <= grad_tol:
            reason = 'gradient_tolerance'
            break

        # Start every iteration with alpha = 1.
        alpha = alpha0

        # Cache current objective value for line-search tests.
        fx = objective(x)

        # Test the Armijo sufficient-decrease condition:
        #
        # f(x-alpha*g) <= f(x)-c*alpha*||g||^2
        #
        # While it fails, shrink alpha without moving x.
        while objective(x - alpha * g) > fx - c * alpha * (g @ g):

            # Try half the previous step length.
            alpha *= beta

            # Stop if numerical step length becomes too small.
            if alpha < 1e-16:
                reason = 'line_search_failure'
                break

        # If backtracking failed, stop the outer loop too.
        if reason == 'line_search_failure':
            break

        # Armijo passed, so actually update x.
        x -= alpha * g

        # Record the accepted point and selected step length.
        history.append({
            'iteration': k,
            'f': objective(x),
            'grad_norm': float(np.linalg.norm(gradient(x))),
            'x': x.copy(),
            'alpha': alpha
        })

    # Check if the last permitted step reached the tolerance.
    if np.linalg.norm(gradient(x)) <= grad_tol:
        reason = 'gradient_tolerance'

    # Return final results and all accepted points.
    return {
        'method': 'gd_armijo',
        'x': x,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'iterations': len(history) - 1,
        'rejections': 0,
        'runtime_sec': time.perf_counter() - start,
        'stop_reason': reason,
        'history': history
    }
