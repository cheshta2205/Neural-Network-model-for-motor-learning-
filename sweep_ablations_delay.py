import argparse
from motor_structure_learning.experiments.sweeps.feedforward_delay import run_named_sweep_delay, FIG_DIR
from motor_structure_learning.experiments.sweeps.common import CONFIG_PRESETS

if __name__ == "__main__":
    # Run the full named sweep (instructed-delay variant) with default settings
    # (5 seeds, outputs to figs/sweeps/feedforward/delay).
    parser = argparse.ArgumentParser(description="Run EC/CB ablation sweep for the instructed-delay motor task.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run for each config.")
    parser.add_argument("--base_fig_dir", type=str, default=FIG_DIR, help="Base directory to save figures and aggregated results.")
    parser.add_argument("--save_agg", action="store_true", default=True, help="Whether to save aggregated results across seeds.")
    args = parser.parse_args()

    run_named_sweep_delay(CONFIG_PRESETS, seeds=args.seeds, base_fig_dir=args.base_fig_dir, save_agg=args.save_agg)
