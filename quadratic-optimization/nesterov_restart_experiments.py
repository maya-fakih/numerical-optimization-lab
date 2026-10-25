"""Exercise 2.3: seed 1 modified Q, four Nesterov restart schedules."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Work on servers without a GUI.
import matplotlib.pyplot as plt
from generate_quadratic_problems import make_instance, relative_distance
from nesterov_acceleration import initial_state, nesterov_step, restart

BASE = Path(__file__).resolve().parent
FIGURES = BASE / 'results' / 'figures'
TABLES = BASE / 'results' / 'tables'
K = 1000
STOP_TOL = 1e-12
DISPLAY_FLOOR = 1e-16


def run_nesterov(problem, period=None):
    """Return errors for iterations 0..1000, with optional periodic restart."""
    state = initial_state(problem['x0'])
    errors = [1.0]  # By definition ||x0-x*|| / ||x0-x*|| = 1.
    status = 'completed'
    stop = K
    for k in range(1, K + 1):
        # Execute ONE full Nesterov iteration.
        state = nesterov_step(state, problem['Q'], problem['q'], problem['L'])
        # Measure the actual iterate BEFORE any momentum reset.
        err = relative_distance(state['x'], problem['x_star'], problem['x0'])
        errors.append(err)
        if not np.isfinite(err):
            status, stop = 'numerical_failure', k
            break
        # Preserve the early-stopping rule from 2.1.
        if err <= STOP_TOL:
            status, stop = 'tolerance', k
            break
        # After EVERY T completed iterations, reset y=x and t=1.
        if period is not None and k % period == 0:
            state = restart(state)
    # The required plot has k=0..1000, including carried-forward points.
    errors.extend([errors[-1]] * (K + 1 - len(errors)))
    return np.array(errors), status, stop


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    # The homework specifically requires the MODIFIED instance for SEED 1.
    p = make_instance(seed=1, modified=True)
    root_kappa = np.sqrt(p['kappa'])
    # Ceiling is essential: restart periods must be integer numbers of steps.
    choices = [('No restart', None),
               ('T=ceil(sqrt(kappa))', int(np.ceil(root_kappa))),
               ('T=ceil(5sqrt(kappa))', int(np.ceil(5 * root_kappa))),
               ('T=ceil(20sqrt(kappa))', int(np.ceil(20 * root_kappa)))]
    print(f'Modified seed 1: kappa={p["kappa"]:.6f}, L={p["L"]:.6f}')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for name, period in choices:
        errors, status, stop = run_nesterov(p, period)
        # The 1e-16 floor is for DISPLAY ONLY, never change stored errors.
        log_errors = np.log10(np.maximum(errors, DISPLAY_FLOOR))
        ax.plot(np.arange(K + 1), log_errors, label=name + (f' (T={period})' if period else ''))
        final_log = float(log_errors[-1]) if np.isfinite(log_errors[-1]) else np.nan
        rows.append([name, '' if period is None else period, final_log, stop, status])
        print(f'{name:26} period={str(period):>4} epsilon1000={final_log:9.5f} '
              f'stop={stop} status={status}')
    ax.set_xlabel('Iteration k')
    ax.set_ylabel('log10(max(relative distance, 1e-16))')
    ax.set_title('Nesterov restart comparison — modified Q, seed 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output = FIGURES / 'nesterov_restarts_modified_seed1.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    with (TABLES / 'nesterov_restart_summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'Period', 'epsilon_1000', 'stop_iteration', 'status'])
        writer.writerows(rows)
    print('Saved:', output)


if __name__ == '__main__':
    main()
