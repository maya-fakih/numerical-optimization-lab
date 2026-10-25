"""Exercise 2: four gradient-descent step-size rules."""
import numpy as np
from generate_quadratic_problems import objective, gradient


def gradient_descent_step(x, Q, q, L, iteration, method):
    """Perform one GD step; return new x and the accepted scalar step size."""
    # The negative gradient is the steepest-descent direction.
    g = gradient(Q, q, x)
    if method == 'GD: exact line search':
        # Differentiate F(x-alpha*g) and set its derivative to zero.
        denominator = float(g @ (Q @ g))
        # Avoid division by zero if already at a stationary point.
        alpha = float((g @ g) / denominator) if denominator > 0 else 0.0
    elif method == 'GD: Armijo':
        # Assignment: restart the line search at alpha=1 EACH iteration.
        alpha = 1.0
        f_current = objective(Q, q, x)
        # Armijo sufficient-decrease constant.
        c = 1e-4
        # Trial values use the actual quadratic, not a Taylor approximation.
        while objective(Q, q, x - alpha * g) > f_current - c * alpha * (g @ g):
            alpha *= 0.5
            # Guard against an infinite loop from floating-point stagnation.
            if alpha < 1e-16:
                break
    elif method == 'GD: diminishing':
        # iteration starts at 0, so the first step is 0.1.
        alpha = 0.1 / (iteration + 1)
    elif method == 'GD: constant':
        # Largest Hessian eigenvalue is the smoothness constant L.
        alpha = 1.0 / L
    else:
        raise ValueError(f'Unknown GD method: {method}')
    # Update the vector using the chosen step length.
    return x - alpha * g, alpha


GD_METHODS = ('GD: exact line search', 'GD: Armijo',
              'GD: diminishing', 'GD: constant')
