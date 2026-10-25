"""Run Exercise 2.1 and 2.2; save both six-method tables and trajectories."""
from pathlib import Path
import csv
import numpy as np
from generate_quadratic_problems import (SEEDS, make_instance, relative_distance)
from gradient_descent_variants import GD_METHODS, gradient_descent_step
from newton_optimizer import newton_step
from nesterov_acceleration import initial_state, nesterov_step

BASE = Path(__file__).resolve().parent
TABLES = BASE / 'results' / 'tables'
RUNS = BASE / 'results' / 'runs'
METHODS = (*GD_METHODS, 'Newton', 'Nesterov')
K = 1000
STOP_TOL = 1e-12
DISPLAY_FLOOR = 1e-16


def run_method(problem, method, K=K):
    """Return exactly K+1 relative errors, carrying forward after early stop."""
    Q, q, L = problem['Q'], problem['q'], problem['L']
    x0, x_star = problem['x0'], problem['x_star']
    x = x0.copy()
    state = initial_state(x0) if method == 'Nesterov' else None
    errors = [relative_distance(x, x_star, x0)]  # k=0 error = 1
    status = 'completed'
    stop_iteration = K
    for k in range(K):
        # Apply the requested algorithm; all share exactly the same problem.
        if method in GD_METHODS:
            x, _alpha = gradient_descent_step(x, Q, q, L, k, method)
        elif method == 'Newton':
            x = newton_step(x, Q, q)
        elif method == 'Nesterov':
            state = nesterov_step(state, Q, q, L)
            x = state['x']
        else:
            raise ValueError(method)
        # Report numerical failures rather than hiding NaN/Inf.
        if not np.all(np.isfinite(x)):
            status = 'numerical_failure'
            stop_iteration = k + 1
            errors.append(np.nan)
            break
        err = relative_distance(x, x_star, x0)
        errors.append(err)
        if not np.isfinite(err):
            status = 'numerical_failure'
            stop_iteration = k + 1
            break
        # The assignment explicitly allows stopping at relative error 1e-12.
        if err <= STOP_TOL:
            status = 'tolerance'
            stop_iteration = k + 1
            break
    # If stopped early, carry the same iterate/error to iteration 1000.
    errors.extend([errors[-1]] * (K + 1 - len(errors)))
    final = errors[-1]
    # Floor is ONLY for the logarithmic display/metric; raw errors remain intact.
    epsilon = float(np.log10(max(final, DISPLAY_FLOOR))) if np.isfinite(final) else np.nan
    return dict(method=method, errors=np.array(errors), epsilon=epsilon,
                status=status, stop_iteration=stop_iteration, final_x=x)


def write_table(problems, results, filename):
    # One row for each method, then the condition-number row, as requested.
    columns = ['Method / quantity', *[f'Seed {s}' for s in SEEDS], 'Mean']
    rows = []
    for method in METHODS:
        values = [results[(p['seed'], method)]['epsilon'] for p in problems]
        rows.append([method, *values, float(np.mean(values))])
    kappas = [p['kappa'] for p in problems]
    rows.append(['Condition number kappa', *kappas, float(np.mean(kappas))])
    with (TABLES / filename).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print('\n' + filename)
    print('Method'.ljust(24), *(f'Seed {s:>2}' for s in SEEDS), '    Mean')
    for row in rows:
        print(str(row[0]).ljust(24), *(f'{v:8.3f}' for v in row[1:]))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    for modified in (False, True):
        label = 'modified' if modified else 'original'
        problems = [make_instance(seed, modified) for seed in SEEDS]
        results = {}
        for p in problems:
            for method in METHODS:
                result = run_method(p, method)
                results[(p['seed'], method)] = result
                # Save the unmodified relative-error history for later plots.
                tag = method.lower().replace(': ', '_').replace(' ', '_')
                np.savez_compressed(RUNS / f'{label}_seed{p["seed"]}_{tag}.npz',
                                    relative_error=result['errors'],
                                    epsilon_1000=result['epsilon'],
                                    status=result['status'],
                                    stop_iteration=result['stop_iteration'])
                print(f'{label} seed {p["seed"]} {method}: '
                      f'epsilon={result["epsilon"]:.5f}, status={result["status"]}, '
                      f'stop={result["stop_iteration"]}')
        write_table(problems, results, f'quadratic_{label}_table.csv')
    print('\nFinished Exercise 2.1 and 2.2.')


if __name__ == '__main__':
    main()
