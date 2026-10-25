
"""Cubic regularization with numerically minimized local models."""

import time
import numpy as np
from scipy.optimize import minimize
from rosenbrock_derivatives import objective, gradient, hessian


def solve_cubic_subproblem(g, H, sigma):
    """
    INNER SOLVER.

    g     = gradient at current x
    H     = Hessian at current x
    sigma = current cubic penalty

    Returns a proposed step s minimizing the cubic model approximately.
    """

    # Number of variables.
    n = len(g)

    # Cubic model without constant f(x), which doesn't affect minimization.
    def model(s):
        norm = np.linalg.norm(s)
        return float(
            g @ s + 0.5 * s @ H @ s + sigma / 3.0 * norm**3
        )

    # Exact derivative of the cubic model with respect to s.
    def jac(s):
        return g + H @ s + sigma * np.linalg.norm(s) * s

    # BFGS is a local optimizer, so we try several initial guesses.
    starts = [np.zeros(n)]

    # A unit step in the negative-gradient direction.
    gn = np.linalg.norm(g)
    if gn > 0:
        starts.append(-g / gn)

    # Examine the Hessian for negative curvature.
    eigvals, eigvecs = np.linalg.eigh(H)

    if eigvals[0] < 0:
        # Scale the negative-curvature trial according to sigma.
        radius = max(1.0, -eigvals[0] / sigma)

        # Eigenvector corresponding to the smallest eigenvalue.
        v = eigvecs[:, 0]

        # Try both directions because one may reduce the model more.
        starts.extend([radius * v, -radius * v])

    # Retain the best starting guess as a fallback.
    best = min(starts, key=model)

    # Run SciPy's BFGS from each starting guess.
    for x0 in starts:
        result = minimize(
            model,
            x0,
            jac=jac,
            method='BFGS',
            options={
                'gtol': 1e-9,    # Inner gradient tolerance
                'maxiter': 400  # Inner iteration limit
            }
        )

        # Keep the result if it is finite and improves the model.
        if (np.all(np.isfinite(result.x))
                and model(result.x) < model(best)):
            best = result.x

    # Return the best approximate step found.
    # Global optimality is not guaranteed.
    return best


def optimize(x0, grad_tol=1e-6, max_iter=2000,
             sigma0=1.0, eta=0.1):
    """
    OUTER CUBIC-REGULARIZATION ALGORITHM.

    x0       = initial point
    grad_tol = stopping threshold for gradient norm
    max_iter = maximum outer trials, including rejections
    sigma0   = initial cubic penalty, required to be 1
    eta      = acceptance threshold, required to be 0.1
    """

    # Start runtime measurement.
    start = time.perf_counter()

    # Working copy of the initial point.
    x = np.asarray(x0, dtype=float).copy()

    # Current penalty parameter.
    sigma = float(sigma0)

    # Record the initial state for later plots.
    history = [{
        'iteration': 0,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'x': x.copy()
    }]

    # Count rejected outer trials.
    rejected = 0

    # Default stopping reason.
    reason = 'max_iterations'

    # One iteration = one proposed cubic step and its evaluation.
    for k in range(1, max_iter + 1):

        # Compute derivatives at current x.
        g, H = gradient(x), hessian(x)

        # Stop if gradient norm meets the assignment tolerance.
        if np.linalg.norm(g) <= grad_tol:
            reason = 'gradient_tolerance'
            break

        # Find a proposed step using the inner solver.
        s = solve_cubic_subproblem(g, H, sigma)

        # Predicted reduction m(0)-m(s).
        # Constant f(x) cancels.
        predicted = -(
            g @ s
            + 0.5 * s @ H @ s
            + sigma / 3.0 * np.linalg.norm(s)**3
        )

        # Actual reduction in the real objective.
        # x+s is still only a trial point.
        actual = objective(x) - objective(x + s)

        # Nonpositive or tiny predicted reduction means rejection.
        rho = actual / predicted if predicted > 1e-14 else -np.inf

        # Accept if rho >= 0.1.
        accepted = np.isfinite(rho) and rho >= eta

        if accepted:
            # Move to the trial point.
            x += s
        else:
            # Keep x unchanged and count the rejection.
            rejected += 1

        # Update sigma for the next iteration.
        #
        # Poor model agreement: increase the penalty.
        if rho < 0.25:
            sigma = min(2.0 * sigma, 1e12)

        # Excellent agreement: decrease the penalty.
        elif rho > 0.75:
            sigma = max(0.5 * sigma, 1e-10)

        # Otherwise, keep sigma unchanged.

        # Record this iteration, including rejected trials.
        history.append({
            'iteration': k,
            'f': objective(x),
            'grad_norm': float(np.linalg.norm(gradient(x))),
            'x': x.copy(),
            'accepted': bool(accepted),
            'rho': float(rho),
            'sigma': sigma
        })

    else:
        # Executed if the for loop finishes without a break.
        k = max_iter

    # Check whether the last permitted step reached convergence.
    if np.linalg.norm(gradient(x)) <= grad_tol:
        reason = 'gradient_tolerance'

    # Return final results and trajectory.
    return {
        'method': 'cubic_regularization',
        'x': x,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'iterations': len(history) - 1,
        'rejections': rejected,
        'runtime_sec': time.perf_counter() - start,
        'stop_reason': reason,
        'history': history
    }
