
"""Run all Q1 Rosenbrock experiments and export figures and tables."""

from pathlib import Path
import csv
import numpy as np
import matplotlib

# Save plots as PNG files instead of opening interactive windows.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import objective and the alternating starting-vector constructor.
from rosenbrock_derivatives import objective, alternating_start

# Import each optimizer under a different name.
from trust_region_optimizer import optimize as trust_region
from cubic_regularization_optimizer import optimize as cubic_regularization
from gradient_descent_armijo import optimize as gradient_descent


# Use the current script's directory as the base path.
BASE = Path(__file__).resolve().parent

# Output folders.
FIGURES = BASE / 'results' / 'figures'
TABLES = BASE / 'results' / 'tables'
RUNS = BASE / 'results' / 'runs'

# Algorithms to run for every experiment.
METHODS = [trust_region, cubic_regularization, gradient_descent]


def plot_convergence(results, filename):
    """
    Plot objective error and gradient norm against iteration number.

    results = outputs from the three optimizers for one experiment.
    filename = destination PNG filename.
    """

    # Two plots: objective error and gradient norm.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Plot one curve for each optimization algorithm.
    for result in results:
        h = result['history']

        # Outer iteration numbers from the saved history.
        its = [z['iteration'] for z in h]

        # Since f*=0, objective error is simply f(x).
        # Floor values at 1e-16 so log10 remains defined.
        axes[0].plot(
            its,
            np.log10(np.maximum([z['f'] for z in h], 1e-16)),
            label=result['method']
        )

        # Plot the logarithm of the gradient norm.
        axes[1].plot(
            its,
            np.log10(np.maximum(
                [z['grad_norm'] for z in h], 1e-16
            )),
            label=result['method']
        )

    # Mathematical labels for the vertical axes.
    axes[0].set_ylabel(r'$\log_{10}(f(x_k)-f^*)$')
    axes[1].set_ylabel(r'$\log_{10}\|\nabla f(x_k)\|$')

    # Shared plot formatting.
    for ax in axes:
        ax.set_xlabel('Outer iteration (rejections included)')
        ax.grid(alpha=0.25)
        ax.legend()

    # Prevent labels from overlapping.
    fig.tight_layout()

    # Save the figure to the results directory.
    fig.savefig(FIGURES / filename, dpi=180)

    # Close the figure to release memory.
    plt.close(fig)


def plot_contours(results, filename):
    """
    For n=2, plot Rosenbrock contours and the three optimizer paths.
    """

    # Build a grid of coordinates for the contour plot.
    a = np.linspace(-2, 2, 300)
    b = np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(a, b)

    # Evaluate the 2D Rosenbrock function at each grid point.
    Z = 100 * (Y - X**2)**2 + (1 - X)**2

    # Create the figure.
    fig, ax = plt.subplots(figsize=(7, 6))

    # Log-spaced contours reveal the narrow Rosenbrock valley.
    ax.contour(
        X, Y, Z,
        levels=np.logspace(-1, 3.5, 22),
        norm=matplotlib.colors.LogNorm(),
        linewidths=0.8
    )

    # Overlay the trajectory of each optimizer.
    for result in results:

        # Collect all recorded points into one array.
        path = np.array([h['x'] for h in result['history']])

        # Connect successive points to visualize the trajectory.
        ax.plot(
            path[:, 0], path[:, 1],
            lw=1.3,
            label=result['method']
        )

        # Mark the initial point.
        ax.scatter(*path[0], s=30)

    # Mark the known global minimizer.
    ax.plot(1, 1, 'k*', ms=12, label='global minimizer')

    # Label and restrict the plot axes.
    ax.set(
        xlabel='$x_1$',
        ylabel='$x_2$',
        xlim=(-2, 2),
        ylim=(-1, 3)
    )

    ax.legend()
    fig.tight_layout()

    # Save and close the figure.
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def main():
    """
    ENTRY POINT.

    Run 3 algorithms x 2 dimensions x 2 starting points = 12 runs.
    Save trajectories, comparison table, and figures.
    """

    # Create output folders if they do not exist.
    for folder in (FIGURES, TABLES, RUNS):
        folder.mkdir(parents=True, exist_ok=True)

    # Collect one comparison-table row per experiment.
    rows = []

    # Test dimensions n=2 and n=10.
    for n in (2, 10):

        # Test both required starting points.
        for start_name, x0 in [
            ('alternating', alternating_start(n)),
            ('zero', np.zeros(n))
        ]:

            # Collect three results for plotting together.
            results = []

            # Execute each algorithm from the same initial point.
            for algorithm in METHODS:

                # Run the complete optimization algorithm.
                result = algorithm(x0)

                # Save the result for comparison plots.
                results.append(result)

                # Add final measurements to the comparison table.
                rows.append({
                    'dimension': n,
                    'start': start_name,
                    'method': result['method'],
                    'final_objective': result['f'],
                    'final_grad_norm': result['grad_norm'],
                    'iterations': result['iterations'],
                    'rejections': result['rejections'],
                    'runtime_sec': result['runtime_sec'],
                    'stop_reason': result['stop_reason']
                })

                # Unique name identifying dimension, start, and method.
                tag = f'n{n}_{start_name}_{result["method"]}'

                # Save trajectory arrays for future analysis.
                np.savez_compressed(
                    RUNS / f'{tag}.npz',
                    path=np.array([
                        h['x'] for h in result['history']
                    ]),
                    objective=np.array([
                        h['f'] for h in result['history']
                    ]),
                    grad_norm=np.array([
                        h['grad_norm'] for h in result['history']
                    ])
                )

                # Print a short summary to the terminal.
                print(
                    tag,
                    result['stop_reason'],
                    'f=', f'{result["f"]:.3e}',
                    'grad=', f'{result["grad_norm"]:.3e}'
                )

            # Plot convergence for the three methods.
            plot_convergence(
                results,
                f'convergence_n{n}_{start_name}.png'
            )

            # Only n=2 trajectories can be plotted in the x1-x2 plane.
            if n == 2:
                plot_contours(
                    results,
                    f'contours_n2_{start_name}.png'
                )

    # Write the complete numerical comparison table to CSV.
    with (TABLES / 'rosenbrock_comparison.csv').open(
        'w', newline=''
    ) as file:

        # Use dictionary keys as the column names.
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


# Run main() only when this file is executed directly.
if __name__ == '__main__':
    main()
