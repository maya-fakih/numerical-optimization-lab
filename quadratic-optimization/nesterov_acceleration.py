"""Exercise 2: Nesterov's accelerated gradient and periodic momentum restart."""
import numpy as np
from generate_quadratic_problems import gradient


def initial_state(x0):
    # x is the current accepted iterate; y is the look-ahead/extrapolated point.
    # t=1 makes the initial extrapolation coefficient zero.
    return dict(x=x0.copy(), y=x0.copy(), t=1.0)


def nesterov_step(state, Q, q, L):
    # Read current values before modifying the state.
    x_old, y_old, t_old = state['x'], state['y'], state['t']
    # Unlike GD, the gradient is evaluated at y, not at x.
    x_new = y_old - gradient(Q, q, y_old) / L
    # Nesterov's prescribed acceleration parameter recurrence.
    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_old**2)) / 2.0
    # Extrapolate in the direction of the latest displacement.
    momentum = (t_old - 1.0) / t_new
    y_new = x_new + momentum * (x_new - x_old)
    # Keep all state for the next iteration.
    return dict(x=x_new, y=y_new, t=t_new)


def restart(state):
    # KEEP the current iterate x; only remove accumulated momentum.
    x = state['x'].copy()
    return dict(x=x, y=x.copy(), t=1.0)
