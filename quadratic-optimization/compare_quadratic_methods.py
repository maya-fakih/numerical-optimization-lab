"""Convenient single entry point for every Exercise 2 experiment."""
from compare_conditioning_experiments import main as run_comparison
from nesterov_restart_experiments import main as run_restarts

if __name__ == '__main__':
    run_comparison()  # Exercises 2.1 and 2.2: two CSV tables.
    run_restarts()    # Exercise 2.3: restart plot and summary.
