
"""Trust-region method with a numerically solved quadratic subproblem."""

import time
import numpy as np
from scipy.optimize import minimize
from rosenbrock_derivatives import objective, gradient, hessian


def solve_trust_subproblem(g, H, delta):
    """
    INNER SOLVER.

    Inputs:
      g     = gradient at current x
      H     = Hessian at current x
      delta = maximum permitted step length

    Output:
      s = approximately optimal step inside ||s|| <= delta

    Uses SLSQP with analytical derivatives and several starting
    guesses. This is an approximate numerical subproblem solver.
    """

    # The proposed step has the same dimension as x.
    n = len(g)

    # Quadratic model without constant f(x).
    # The constant does not affect which step minimizes the model.
    def model(s):
        return float(g @ s + 0.5 * s @ H @ s)

    # Gradient of the quadratic model with respect to s.
    def jac(s):
        return g + H @ s

    # SciPy's inequality constraint must be nonnegative.
    # delta^2 - ||s||^2 >= 0 means ||s|| <= delta.
    def constraint(s):
        return delta**2 - float(s @ s)

    # Gradient of the constraint with respect to s.
    def constraint_jac(s):
        return -2.0 * s

    # Different initial guesses improve robustness.
    candidates = []

    # First possible guess: negative gradient direction.
    norm_g = np.linalg.norm(g)

    if norm_g > 0:
        # Scale the direction to the trust-region boundary.
        candidates.append(-delta * g / norm_g)

    # Examine Hessian eigenvalues for negative curvature.
    eigvals, eigvecs = np.linalg.eigh(H)

    if eigvals[0] < 0:
        # Smallest eigenvalue gives strongest negative curvature.
        v = eigvecs[:, 0]

        # Try both directions along that eigenvector.
        candidates.extend([delta * v, -delta * v])

    # Zero is always feasible, so it is a useful fallback.
    candidates.append(np.zeros(n))

    # Choose the initial candidate with the lowest model value.
    best = min(candidates, key=model)

    # Run the constrained inner solver from each candidate.
    for start in candidates:
        result = minimize(
            model,
            start,
            jac=jac,
            method='SLSQP',
            constraints=[{
                'type': 'ineq',
                'fun': constraint,
                'jac': constraint_jac
            }],
            options={
                'ftol': 1e-12,  # Inner numerical tolerance
                'maxiter': 250  # Maximum inner solver iterations
            }
        )

        # Only consider finite steps inside the trust region,
        # allowing a small tolerance for numerical rounding.
        if (np.all(np.isfinite(result.x))
                and np.linalg.norm(result.x) <= delta * (1 + 1e-7)):

            # Keep this result if its model value is better.
            if model(result.x) < model(best):
                best = result.x

    # Return the best step found.
    # This is not guaranteed to be the global subproblem minimizer.
    return best


def optimize(x0, grad_tol=1e-6, max_iter=2000, delta0=1.0,
             eta=0.1, delta_max=100.0):
    """
    OUTER TRUST-REGION ALGORITHM.

    x0        = initial point
    grad_tol  = gradient stopping tolerance
    max_iter  = maximum outer trials, including rejections
    delta0    = initial radius (assignment requires 1)
    eta       = acceptance threshold (assignment requires 0.1)
    delta_max = upper bound on the trust radius
    """

    # Start the runtime clock.
    start = time.perf_counter()

    # Working copy of the initial point.
    x = np.asarray(x0, dtype=float).copy()

    # Current trust-region radius.
    delta = float(delta0)

    # Record iteration 0 before taking any steps.
    history = [{
        'iteration': 0,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'x': x.copy()
    }]

    # Count rejected outer trial steps.
    rejected = 0

    # Default reason, changed if convergence occurs.
    reason = 'max_iterations'

    # Each iteration proposes one step and judges its quality.
    # Rejected trials also count toward max_iter.
    for k in range(1, max_iter + 1):

        # Compute derivatives at the current point.
        g, H = gradient(x), hessian(x)

        # Stop if the gradient is sufficiently small.
        if np.linalg.norm(g) <= grad_tol:
            reason = 'gradient_tolerance'
            break

        # Ask the inner solver for a step satisfying ||s|| <= delta.
        s = solve_trust_subproblem(g, H, delta)

        # Predicted decrease q(0) - q(s).
        # Constant f(x) cancels from the subtraction.
        predicted = -(g @ s + 0.5 * s @ H @ s)

        # Actual decrease in the real Rosenbrock function.
        # x+s is only a trial point; x is unchanged here.
        actual = objective(x) - objective(x + s)

        # Reject steps with nonpositive or numerically tiny
        # predicted reduction to avoid unsafe division.
        rho = actual / predicted if predicted > 1e-14 else -np.inf

        # The assignment requires rho >= 0.1 for acceptance.
        accepted = np.isfinite(rho) and rho >= eta

        if accepted:
            # Accept: move to the trial point.
            x += s
        else:
            # Reject: keep x unchanged.
            rejected += 1

        # Update radius for the NEXT trial.
        #
        # rho < 0.25: model prediction was poor; shrink.
        # This includes accepted trials with 0.1 <= rho < 0.25.
        if rho < 0.25:
            delta = max(0.25 * delta, 1e-12)

        # rho > 0.75 and step near boundary:
        # model worked well, so permit a larger region.
        elif rho > 0.75 and np.linalg.norm(s) >= 0.99 * delta:
            delta = min(2.0 * delta, delta_max)

        # Otherwise, delta remains unchanged.

        # Record the result of this trial.
        # Even rejected trials are recorded.
        history.append({
            'iteration': k,
            'f': objective(x),
            'grad_norm': float(np.linalg.norm(gradient(x))),
            'x': x.copy(),
            'accepted': bool(accepted),
            'rho': float(rho),
            'delta': delta
        })

    else:
        # for-else executes when the loop ends without break.
        k = max_iter

    # Check whether the final permitted step reached tolerance.
    if np.linalg.norm(gradient(x)) <= grad_tol:
        reason = 'gradient_tolerance'

    # Return summary and trajectory for the experiment runner.
    return {
        'method': 'trust_region',
        'x': x,
        'f': objective(x),
        'grad_norm': float(np.linalg.norm(gradient(x))),
        'iterations': len(history) - 1,
        'rejections': rejected,
        'runtime_sec': time.perf_counter() - start,
        'stop_reason': reason,
        'history': history
    }
