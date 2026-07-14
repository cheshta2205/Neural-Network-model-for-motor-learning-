import argparse
from motor_structure_learning.experiments._ablations_sweep import run_named_sweep, CONFIG_PRESETS

if __name__ == "__main__":
    # Run the full named sweep with default settings (5 seeds, outputs to figs/mc_ablations)
    parser = argparse.ArgumentParser(description="Run EC/CB ablation sweep for motor learning experiments.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run for each config.")
    parser.add_argument("--base_fig_dir", type=str, default="figs/mc_ablations", help="Base directory to save figures and aggregated results.")
    parser.add_argument("--save_agg", action="store_true", default=True, help="Whether to save aggregated results across seeds.")
    args = parser.parse_args()

    run_named_sweep(CONFIG_PRESETS, seeds=args.seeds, base_fig_dir=args.base_fig_dir, save_agg=args.save_agg)