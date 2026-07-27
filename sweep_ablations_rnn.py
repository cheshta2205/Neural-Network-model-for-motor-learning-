import argparse
from motor_structure_learning.experiments.sweeps.recurrent_base import run_named_sweep_rnn, FIG_DIR
from motor_structure_learning.experiments.sweeps.common import CONFIG_PRESETS

if __name__ == "__main__":
    # Run the full named sweep (recurrent-actor variant) with default settings
    # (5 seeds, outputs to figs/sweeps/recurrent/base).
    parser = argparse.ArgumentParser(description="Run EC/CB ablation sweep with a recurrent (GRUCell) actor.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run for each config.")
    parser.add_argument("--base_fig_dir", type=str, default=FIG_DIR, help="Base directory to save figures and aggregated results.")
    parser.add_argument("--save_agg", action="store_true", default=True, help="Whether to save aggregated results across seeds.")
    args = parser.parse_args()

    run_named_sweep_rnn(CONFIG_PRESETS, seeds=args.seeds, base_fig_dir=args.base_fig_dir, save_agg=args.save_agg)
