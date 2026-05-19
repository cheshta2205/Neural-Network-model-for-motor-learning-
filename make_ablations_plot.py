"""Plot ablation sweep results with curve fitting and parameter distributions."""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from typing import Optional
from scipy.optimize import curve_fit
from scipy.stats import linregress
from statannotations.Annotator import Annotator
import pdb

from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params

# Settings
set_plt_params(style='fast')
BASE_FIG_DIR = "figs/mc_ablations"
PLOT_OUTPUT_DIR = os.path.join(BASE_FIG_DIR, "plots")
METRICS = ["success", "return"]

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Seaborn styling
sns.set_style("whitegrid")

CONFIG_COLORS = {
    "baseline": "#0e3b5c",
    "cb_only": "#5e4f6d",
    "full_end_to_end": "#9f6671",
    "full": "#d8897b",
    "oracle": "#f4ac7b",
}

CONFIG_ORDERS = {
    "baseline": 0,
    "cb_only": 1,
    "oracle": 2,
    "full": 3,
    "full_end_to_end": 4,
}

CONFIG_ORDER = [config for config, _ in sorted(CONFIG_ORDERS.items(), key=lambda item: item[1])]

PHASE_COLORS = {
    0: "#d2e3e1",
    1: "#a6c8c3",
    2: "#79aca6",
    3: "#4d9188",
    4: "#20756a",
}


def get_config_order(data: Optional[pd.DataFrame] = None) -> list:
    """Return the canonical config order, optionally filtered to configs present in data."""
    if data is None or data.empty or "config" not in data.columns:
        return CONFIG_ORDER

    present = [config for config in CONFIG_ORDER if config in set(data["config"].dropna().astype(str))]
    extras = sorted(set(data["config"].dropna().astype(str)) - set(present))
    return present + extras

def logistic(x, k, x0):
    """Generalized logistic curve: 1 / (1 + exp(-k(x - x0)))
    
    Parameters:
    - k: learning rate (steepness; higher k = faster learning)
    - x0: inflection point (episode at 50% of rise)
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

def hill(x, a, L, n, K):
    """Hill curve: a + (L - a) * x^n / (K^n + x^n)
    
    Parameters:
    - a: baseline (lower asymptote)
    - L: maximum (upper asymptote)
    - n: Hill coefficient (cooperativity; n=1 is logistic-like)
    - K: half-maximum (x at 50% of rise)
    """
    return a + (L - a) * (x**n) / (K**n + x**n)


def fit_learning_curve(episodes, values, metric_name):
    """Fit Hill function to learning data.
    
    Hill equation: a + (L - a) * x^n / (K^n + x^n)
    - a: baseline (lower asymptote)
    - L: maximum (upper asymptote)
    - n: Hill coefficient (cooperativity/shape flexibility)
    - K: half-maximum point
    """
    valid_idx = ~np.isnan(values)
    x = episodes[valid_idx].astype(float)
    y = values[valid_idx].astype(float)
    
    if len(x) < 4:
        return None
    
    # Normalize data to [0, 1]
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    
    try:
        # Initial guess
        a_guess = np.clip(y_norm[0], 0, 1)  # start at first value
        L_guess = np.clip(np.max(y_norm), 0.5, 1)  # end at max
        K_guess = len(x) / 2  # midpoint
        n_guess = 1.0  # logistic-like initially
        
        popt, _ = curve_fit(
            hill,
            x, y_norm,
            p0=[a_guess, L_guess, n_guess, K_guess],
            maxfev=5000,
            ftol=1e-6,
            bounds=([0, 0, 0.1, 0], [1, 1, 10, len(x) * 2])
        )
        
        a, L, n, K = popt
        return {"model": "hill", "a": a, "L": L, "n": n, "K": K}
    except Exception as e:
        print(f"Fitting failed for {metric_name}: {e}")
        return None
    

def load_all_configs(base_dir: str) -> pd.DataFrame:
    """Load and concatenate all aggregated CSVs into a single DataFrame."""
    dfs = []
    
    csv_paths = glob.glob(os.path.join(base_dir, "*/aggregated_results.csv"))
    
    for csv_path in sorted(csv_paths):
        config_name = os.path.basename(os.path.dirname(csv_path))
        if config_name in CONFIG_ORDERS:
            try:
                df = pd.read_csv(csv_path)
                df['config'] = config_name
                dfs.append(df)
                print(f"Loaded {config_name}: {len(df)} rows, {df['seed'].nunique()} seeds")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def add_phase_occurrence(data: pd.DataFrame, label_col: str = "label", phase_col: str = "phase") -> pd.DataFrame:
    if label_col not in data.columns or phase_col not in data.columns:
        raise KeyError(f"Expected '{label_col}' and '{phase_col}' in the CSV")

    data = data.sort_values(["config", "seed", "ep"]).copy()
    data["phase_occurrence"] = 0
    pdb.set_trace()
    for (config, seed), group in data.groupby(["config", "seed"], sort=False):
        label_counts = {}
        prev_phase = None
        prev_label = None

        # iterate episode-by-episode to avoid counting multiple rows within same episode
        for ep, ep_group in group.groupby("ep", sort=False):
            # choose representative label/phase for that episode (first row)
            lbl = ep_group[label_col].iloc[0]
            ph = ep_group[phase_col].iloc[0]

            if lbl not in label_counts:
                label_counts[lbl] = 0
            else:
                # increment only when the phase actually changed since previous episode
                if ph != prev_phase:
                    label_counts[lbl] += 1

            data.loc[ep_group.index, "phase_occurrence"] = label_counts[lbl]
            prev_label = lbl
            prev_phase = ph

    # human-readable occurrence label
    data["phase_label"] = data[label_col] + " (x" + (data["phase_occurrence"] + 1).astype(str) + ")"
    data["occurrence_label"] = "x" + (data["phase_occurrence"] + 1).astype(str)
    pdb.set_trace()
    return data

def fit_curves_per_phase(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Fit learning curves for each (config, seed, phase, occurrence) and extract parameters.

    Each fit uses episodes within that phase-occurrence (ordered by global 'ep').
    Returns rows keyed by config, seed, phase, phase_occurrence, phase_label.
    """
    results = []

    group_keys = ['config', 'seed', 'phase', 'occurrence', 'phase_occurrence']  # include phase_occurrence for tracking
    for keys, group in data.groupby(group_keys):
        config, seed, phase, occurrence, phase_occurrence = keys
        group = group.sort_values('ep')
        n = len(group)
        if n < 3:
            # not enough points to fit reliably
            continue

        # episodes are local index within the phase-occurrence
        episodes = np.arange(n).astype(float)
        # We'll let fit_learning_curve normalize episodes internally (it normalizes).
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].values
            fit_params = fit_learning_curve(episodes, values, metric)
            row = {
                'config': config,
                'seed': seed,
                'phase': int(phase),
                'occurrence': int(occurrence),
                'phase_occurrence': phase_occurrence,
            }

            if fit_params:
                row[f"{metric}_a"] = fit_params["a"]
                row[f"{metric}_L"] = fit_params["L"]
                row[f"{metric}_n"] = fit_params["n"]
                row[f"{metric}_K"] = fit_params["K"]
            else:
                row[f"{metric}_a"] = np.nan
                row[f"{metric}_L"] = np.nan
                row[f"{metric}_n"] = np.nan
                row[f"{metric}_K"] = np.nan
            results.append(row)

    if not results:
        return pd.DataFrame(columns=['config','seed','phase','phase_occurrence'] + [f"{m}_{p}" for m in metrics for p in ['a','L','n','K']])
    return pd.DataFrame(results)

def fit_curves_per_occurrence(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    results = []

    group_keys = ["config", "seed", "occurrence"]
    for keys, group in data.groupby(group_keys):
        config, seed, occurrence = keys
        group = group.sort_values("ep")
        n = len(group)
        if n < 3:
            continue

        episodes = np.arange(n).astype(float)

        for metric in metrics:
            if metric not in group.columns:
                continue

            values = group[metric].values
            fit_params = fit_learning_curve(episodes, values, metric)
            row = {
                "config": config,
                "seed": seed,
                "occurrence": int(occurrence),
            }

            if fit_params:
                row[f"{metric}_a"] = fit_params["a"]
                row[f"{metric}_L"] = fit_params["L"]
                row[f"{metric}_n"] = fit_params["n"]
                row[f"{metric}_K"] = fit_params["K"]
            else:
                row[f"{metric}_a"] = np.nan
                row[f"{metric}_L"] = np.nan
                row[f"{metric}_n"] = np.nan
                row[f"{metric}_K"] = np.nan

            results.append(row)

    if not results:
        return pd.DataFrame(
            columns=["config", "seed", "occurrence"]
            + [f"{m}_{p}" for m in metrics for p in ["a", "L", "n", "K"]]
        )
    return pd.DataFrame(results)


def estimate_criterion_crossing_episode(
    episodes: np.ndarray,
    values: np.ndarray,
    target: float,
    smooth_window: int = 5,
) -> float:
    """Estimate the first episode where the trajectory crosses a fixed target."""
    valid_idx = ~np.isnan(values)
    x = np.asarray(episodes[valid_idx], dtype=float)
    y = np.asarray(values[valid_idx], dtype=float)

    if len(x) < 4:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if smooth_window and smooth_window > 1:
        window = min(int(smooth_window), len(y))
        y = pd.Series(y).rolling(window=window, center=True, min_periods=1).median().to_numpy()

    if np.isnan(target):
        return np.nan

    increasing = y[-1] >= y[0]
    crossings = np.where(y >= target)[0] if increasing else np.where(y <= target)[0]

    if len(crossings) == 0:
        return x.max()
    
    idx = int(crossings[0])
    if idx == 0:
        return float(x[0])

    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])

    if np.isclose(y0, y1):
        return x1

    frac = (target - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return x0 + frac * (x1 - x0)


def build_criterion_crossing_table(
    data: pd.DataFrame,
    metrics: list,
    criterion_fraction: float = 0.8,
) -> pd.DataFrame:
    """Build a per-seed episode-to-criterion table for each metric.

    The criterion is global per metric: criterion_fraction * max(metric across all
    configs/seeds/occurrences).
    """
    rows = []

    global_targets = {}
    for metric in metrics:
        if metric in data.columns:
            metric_values = data[metric].to_numpy(dtype=float)
            global_max = np.nanmax(metric_values)
            global_targets[metric] = criterion_fraction * global_max if np.isfinite(global_max) else np.nan

    for (config, seed, occurrence), group in data.groupby(["config", "seed", "occurrence"]):
        group = group.sort_values("ep")
        episodes = np.arange(len(group), dtype=float)

        for metric in metrics:
            if metric not in group.columns:
                continue

            target = global_targets.get(metric, np.nan)
            t50 = estimate_criterion_crossing_episode(episodes, group[metric].values, target=target)
            rows.append(
                {
                    "config": config,
                    "seed": seed,
                    "occurrence": int(occurrence),
                    "metric": metric,
                    "criterion_target": target,
                    "criterion_episode": t50,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["config", "seed", "occurrence", "metric", "criterion_target", "criterion_episode"]
        )

    return pd.DataFrame(rows)


def build_criterion_crossing_phase_table(
    data: pd.DataFrame,
    metrics: list,
    criterion_fraction: float = 0.8,
) -> pd.DataFrame:
    """Build a per-seed episode-to-criterion table split by phase occurrence."""
    rows = []

    global_targets = {}
    for metric in metrics:
        if metric in data.columns:
            metric_values = data[metric].to_numpy(dtype=float)
            global_max = np.nanmax(metric_values)
            global_targets[metric] = criterion_fraction * global_max if np.isfinite(global_max) else np.nan

    if "phase_occurrence" not in data.columns:
        raise KeyError("Expected a 'phase_occurrence' column for the phase-aware criterion plot")

    for (config, seed, phase_occurrence), group in data.groupby(["config", "seed", "phase_occurrence"]):
        group = group.sort_values("ep")
        episodes = np.arange(len(group), dtype=float)

        for metric in metrics:
            if metric not in group.columns:
                continue

            target = global_targets.get(metric, np.nan)
            criterion_episode = estimate_criterion_crossing_episode(
                episodes,
                group[metric].values,
                target=target,
            )

            labels = group["label"].unique()
            label = labels[0] if len(labels) > 0 else "Unknown"

            occurrences = group["occurrence"].unique()
            occurrence = occurrences[0] if len(occurrences) > 0 else -1
            rows.append(
                {
                    "config": config,
                    "seed": seed,
                    "phase_occurrence": phase_occurrence,
                    "metric": metric,
                    "criterion_target": target,
                    "criterion_episode": criterion_episode,
                    "label": label,
                    "occurrence": occurrence,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "config",
                "seed",
                "phase_occurrence",
                "metric",
                "criterion_target",
                "criterion_episode",
            ]
        )

    return pd.DataFrame(rows)


def plot_criterion_crossing(data: pd.DataFrame, metrics: list, output_dir: str, criterion_fraction: float = 0.8):
    """Plot model-free episode-to-criterion estimates with a consistent global threshold."""
   
    config_order = get_config_order(data)
    shared_seeds = get_shared_seeds(data, config_order)

    restricted_data = data[data["seed"].isin(shared_seeds) & data["config"].isin(config_order)].copy()

    crossing_table = build_criterion_crossing_phase_table(restricted_data, metrics, criterion_fraction=criterion_fraction)
    
    for metric in metrics:
        plot_data = crossing_table[crossing_table["metric"] == metric].dropna(subset=["criterion_episode"])

        facet_data = plot_data.copy()
        facet_data["occurrence"] = pd.to_numeric(facet_data["occurrence"], errors="coerce")
        facet_data = facet_data.dropna(subset=["occurrence"])
        facet_data["occurrence"] = facet_data["occurrence"].astype(int)
        
        if plot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(3, 3))

        sns.violinplot(
            data=plot_data,
            x="occurrence",
            y="criterion_episode",
            hue="config",
            palette=CONFIG_COLORS,
            hue_order=get_config_order(plot_data),
            ax=ax,
            inner="box",
            inner_kws={"box_width": 1, "whis_width": 0.5, "color": "black"},
            width=1
        )

        pairs = []
        present = set(zip(facet_data["occurrence"], facet_data["config"]))

        occurrence_order = sorted(data["occurrence"].dropna().unique())
        config_order = get_config_order(data)

        for occ in occurrence_order:
            baseline_pair = (occ, "baseline")
            if baseline_pair not in present:
                continue
            for cfg in config_order:
                if cfg == "baseline":
                    continue
                if (occ, cfg) in present:
                    pairs.append((baseline_pair, (occ, cfg)))

        if pairs:
            annot = Annotator(
                ax,
                pairs,
                data=facet_data,
                x="occurrence",
                y="criterion_episode",
                hue="config",
                order=occurrence_order,
                hue_order=config_order,
            )
            annot.configure(
                test="Wilcoxon",
                text_format="star",
                comparisons_correction="holm",
                correction_format='replace',   # Replaces old label with corrected label
                verbose=False,
            )
            annot.apply_and_annotate()

        ax.set_xlabel("Previous Experiences")
        ax.set_ylabel(f"Episodes to Criterion")
        ax.set_title(f"Criterion Crossing Episode: {metric.replace('_', ' ').title()}")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=15)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        output_path = os.path.join(output_dir, f"criterion_crossing_{metric}")
        plt.tight_layout()
        make_axis_nice(fig)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def get_shared_seeds(data: pd.DataFrame, configs: list[str]) -> set[int]:
    seed_sets = []
    for config in configs:
        seeds = set(data.loc[data["config"] == config, "seed"].dropna().astype(int).unique())
        seed_sets.append(seeds)

    if not seed_sets:
        return set()

    return set.intersection(*seed_sets)

def plot_criterion_crossing_by_phase_occurrence(
    data: pd.DataFrame,
    metrics: list,
    output_dir: str,
    criterion_fraction: float = 0.8,
):
    """Plot episode-to-criterion estimates separated by phase occurrence."""

    config_order = get_config_order(data)
    shared_seeds = get_shared_seeds(data, config_order)

    restricted_data = data[data["seed"].isin(shared_seeds) & data["config"].isin(config_order)].copy()

    crossing_table = build_criterion_crossing_phase_table(restricted_data, metrics, criterion_fraction=criterion_fraction)

    for metric in metrics:
        plot_data = crossing_table[crossing_table["metric"] == metric].dropna(subset=["criterion_episode"])
        if plot_data.empty:
            continue

        g = sns.FacetGrid(
            plot_data,
            col="label",
            col_wrap=4,
            height=3,
            aspect=1.0,
        )

        def plot_func(data, **kwargs):
            ax = plt.gca()

            facet_data = data.copy()
            facet_data["occurrence"] = pd.to_numeric(facet_data["occurrence"], errors="coerce")
            facet_data = facet_data.dropna(subset=["occurrence"])
            facet_data["occurrence"] = facet_data["occurrence"].astype(int)

            occurrence_order = sorted(data["occurrence"].dropna().unique())
            config_order = get_config_order(data)
            
            sns.violinplot(
                data=data,
                x="occurrence",
                y="criterion_episode",
                hue="config",
                palette=CONFIG_COLORS,
                hue_order=config_order,
                order=occurrence_order,
                inner="box",
                inner_kws={"box_width": 1, "whis_width": 0.5, "color": "black"},
                **kwargs,
            )

            pairs = []
            present = set(zip(facet_data["occurrence"], facet_data["config"]))
            for occ in occurrence_order:
                baseline_pair = (occ, "baseline")
                if baseline_pair not in present:
                    continue
                for cfg in config_order:
                    if cfg == "baseline":
                        continue
                    if (occ, cfg) in present:
                        pairs.append((baseline_pair, (occ, cfg)))

            if pairs:
                annot = Annotator(
                    ax,
                    pairs,
                    data=facet_data,
                    x="occurrence",
                    y="criterion_episode",
                    hue="config",
                    order=occurrence_order,
                    hue_order=config_order,
                )
                annot.configure(
                    test="Wilcoxon",
                    text_format="star",
                    comparisons_correction="holm",
                    correction_format='replace',   # Replaces old label with corrected label
                    verbose=False,
                )

                annot.apply_and_annotate()

        g.map_dataframe(plot_func)
        g.figure.set_layout_engine("constrained")

        g.set_axis_labels("Previous Experiences", f"Episodes to Criterion")
        g.set(ylim=(0, restricted_data['ep'].max()))

        g.figure.suptitle(f"Criterion Crossing by Phase Occurrence: {metric.replace('_', ' ').title()}", fontsize=14)

        output_path = os.path.join(output_dir, f"criterion_crossing_by_phase_occurrence_{metric}")
        make_axis_nice(g.figure)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves with seaborn."""
    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: {metric} not in data")
            continue
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=data,
            x="ep",
            y=metric,
            hue="config",
            errorbar=("se", 1),
            linewidth=1,
            palette=CONFIG_COLORS,
            hue_order=get_config_order(data),
        )
        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Learning Curve: {metric.replace('_', ' ').title()}")
        plt.legend(title="Config", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        make_axis_nice(plt.gca())
        output_path = os.path.join(output_dir, f"learning_curve_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_episodic_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str,
                                  phase_length: int = 500):
    """Plot learning curves with seaborn."""
    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: {metric} not in data")
            continue
        
        plt.figure(figsize=(12, 2))

        x_max = data['global_ep'].max()

        # Add vertical lines for phase transitions
        for phase_start in range(0, int(x_max) + phase_length, phase_length):
            plt.axvline(x=phase_start, color='black', linestyle='--', alpha=0.5)

        sns.lineplot(
            data=data,
            x="global_ep",
            y=metric,
            hue="config",
            errorbar=("se", 1),
            linewidth=1,
            palette=CONFIG_COLORS,
        )

        ax = plt.gca()

        plt.xlim(0, data['global_ep'].max())
        ax.xaxis.set_major_locator(MultipleLocator(500))  # minor ticks every 100 episodes
        ax.xaxis.set_minor_locator(MultipleLocator(100))  # minor ticks every 100 episodes
        
        if metric == 'return':
            plt.ylim(-35, 5)
        else:
            plt.ylim(0, 1)

        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Learning Curve: {metric.replace('_', ' ').title()}")
        plt.legend(title="Config", bbox_to_anchor=(1.05, 1), loc="upper left")
        make_axis_nice(plt.gca())
        output_path = os.path.join(output_dir, f"stepwise_learning_curve_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_learning_curves_by_phase_occurrence(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves colored by phase occurrence."""
    for metric in metrics:
        if metric not in data.columns:
            continue
        
        fig, axes = plt.subplots(1, len(data['config'].unique()), figsize=(6 * len(data['config'].unique()), 6))
        if len(data['config'].unique()) == 1:
            axes = [axes]
        
        for ax_idx, config in enumerate(get_config_order(data)):
            config_data = data[data['config'] == config]
            
            sns.lineplot(
                data=config_data,
                x="ep",
                y=metric,
                hue="phase_occurrence",
                palette="tab10",
                errorbar=("se", 1),
                linewidth=1,
                ax=axes[ax_idx],
            )
            axes[ax_idx].set_title(f"{config} - {metric.replace('_', ' ').title()}")
            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel(metric.replace("_", " ").title())
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(title="Phase", fontsize=9, loc="best")
        
        make_axis_nice(plt.gca())
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"by_phase_occurrence_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_learning_curves_by_occurrence(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves colored by phase occurrence."""
    for metric in metrics:
        if metric not in data.columns:
            continue
        
        fig, axes = plt.subplots(1, len(data['config'].unique()), figsize=(6 * len(data['config'].unique()), 6))
        if len(data['config'].unique()) == 1:
            axes = [axes]
        
        for ax_idx, config in enumerate(get_config_order(data)):
            config_data = data[data['config'] == config]
            sns.lineplot(
                data=config_data,
                x="ep",
                y=metric,
                hue="occurrence",
                palette=PHASE_COLORS,
                errorbar=("se", 1),
                linewidth=1,
                ax=axes[ax_idx],
            )
            axes[ax_idx].set_title(f"{config} - {metric.replace('_', ' ').title()}")
            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel(metric.replace("_", " ").title())
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(title="Phase", fontsize=9, loc="best")
        
        plt.tight_layout()
        make_axis_nice(fig)
        output_path = os.path.join(output_dir, f"by_occurrence_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_phase_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves by phase using FacetGrid."""
    for metric in metrics:
        if metric not in data.columns:
            continue
                
        g = sns.FacetGrid(
            data,
            col="label",
            row="occurrence",
            height=5,
            aspect=1.2,
        )
        
        def plot_func(data, **kwargs):
            sns.lineplot(
                data=data,
                x="ep",
                y=metric,
                hue="config",
                errorbar=("se", 1),
                palette=CONFIG_COLORS,
                linewidth=1,
                hue_order=get_config_order(data),
                **kwargs,
            )
        
        g.map_dataframe(plot_func)

        # Add one shared legend for configs across all facets.
        config_order = sorted(data["config"].dropna().unique())
        config_order = get_config_order(data)
        handles = [
            Line2D(
                [0],
                [0],
                color=CONFIG_COLORS.get(cfg, "#333333"),
                marker="o",
                linewidth=1,
                markersize=3,
                label=cfg,
            )
            for cfg in config_order
        ]
        g.figure.legend(handles=handles, title="Config", loc="upper right")
        
        g.set_axis_labels("Episode", metric.replace("_", " ").title())
        
        output_path = os.path.join(output_dir, f"phase_learning_curve_{metric}")
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        make_axis_nice(plt.gcf())
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_occurrence_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves by occurrence using FacetGrid."""
    for metric in metrics:
        if metric not in data.columns:
            continue
                
        g = sns.FacetGrid(
            data,
            col="occurrence",
            height=5,
            aspect=1.2,
        )
        
        def plot_func(data, **kwargs):
            sns.lineplot(
                data=data,
                x="ep",
                y=metric,
                hue="config",
                errorbar=("se", 1),
                linewidth=1,
                **kwargs,
            )
        
        g.map_dataframe(plot_func)
        
        g.set_axis_labels("Episode", metric.replace("_", " ").title())
        
        output_path = os.path.join(output_dir, f"occurrence_learning_curve_{metric}")
        plt.tight_layout()
        make_axis_nice(plt.gcf())
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")

def plot_curve_parameters(fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot distributions of fitted curve parameters for each metric."""
    param_names = {"n": "Hill Coefficient", "K": "Half-Maximum", "L": "Asymptote", "a": "Baseline"}
    
    for metric in metrics:
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            
            if param_col not in fit_data.columns:
                continue
            
            # Filter out NaN values
            plot_data = fit_data[[param_col, 'config', 'phase_occurrence']].dropna()
            
            if plot_data.empty:
                continue
            
            plot_data.columns = ['value', 'config', 'phase_occurrence']  # rename for plotting
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.violinplot(
                data=plot_data,
                x="config",
                y="value",
                hue="phase_occurrence",
                # palette=CONFIG_COLORS,
                order=get_config_order(plot_data),
                ax=ax,
            )
            
            ax.set_xlabel("Config")
            ax.set_ylabel(param_label)
            ax.set_title(f"{param_label} Distribution: {metric.replace('_', ' ').title()}")
            ax.tick_params(axis="x", rotation=15)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            
            output_path = os.path.join(output_dir, f"param_{metric}_{param}")
            plt.tight_layout()
            make_axis_nice(fig)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_path}")

def plot_curve_parameters_per_occurrence(fit_data: pd.DataFrame, metrics: list, output_dir: str, comparison_table: Optional[pd.DataFrame] = None):
    """Plot distributions of fitted curve parameters for each metric."""
    param_names = {"n": "Hill Coefficient", "K": "Half-Maximum", "L": "Asymptote", "a": "Baseline"}
    
    for metric in metrics:
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            
            if param_col not in fit_data.columns:
                continue
            
            # Filter out NaN values
            plot_data = fit_data[[param_col, 'config', 'occurrence']].dropna()
            
            if plot_data.empty:
                continue
            
            plot_data.columns = ['value', 'config', 'occurrence']  # rename for plotting
            
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.violinplot(
                data=plot_data,
                x="config",
                y="value",
                hue="occurrence",
                palette=PHASE_COLORS,
                order=get_config_order(plot_data),
                ax=ax,
            )

            ax.set_xlabel("Config")
            ax.set_ylabel(param_label)
            ax.set_title(f"{param_label} Distribution: {metric.replace('_', ' ').title()}")
            ax.tick_params(axis="x", rotation=15)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            
            output_path = os.path.join(output_dir, f"param_{metric}_{param}_occurrence")
            plt.tight_layout()
            make_axis_nice(fig)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_path}")


def plot_fitted_curves(data: pd.DataFrame, fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot raw data with overlaid fitted curves to inspect fit quality.
    
    Creates a grid of subplots, one per (config, metric) combination.
    Within each subplot, shows multiple phase occurrences with raw data points 
    and fitted curves overlaid.
    """
    fit_subdir = os.path.join(output_dir, "fit_quality")
    os.makedirs(fit_subdir, exist_ok=True)
    
    for metric in metrics:
        if metric not in data.columns:
            continue
        
        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]
        
        for ax_idx, config in enumerate(configs):
            ax = axes[ax_idx]
            config_data = data[data['config'] == config]
            config_fit = fit_data[fit_data['config'] == config]
            
            # Get sorted phase occurrences and assign colors
            phase_occs = sorted(config_data['phase_occurrence'].unique())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(phase_occs)))
            
            # Track legend entries for custom labels with R²
            handles, labels = [], []
            
            # Plot raw data for this config, grouped by phase_occurrence
            for phase_idx, phase_occ in enumerate(phase_occs):
                phase_data = config_data[config_data['phase_occurrence'] == phase_occ]
                color = colors[phase_idx]
                
                # Aggregate across seeds: plot mean ± std
                ep_grouped = phase_data.groupby('ep')[metric].agg(['mean', 'std', 'count'])
                ep_grouped['sem'] = ep_grouped['std'] / np.sqrt(ep_grouped['count'])
                
                x_vals = ep_grouped.index.values.astype(float)
                y_vals = ep_grouped['mean'].values
                y_err = ep_grouped['sem'].values
                
                # Normalize to [0, 1] for fitting comparison
                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                y_norm = (y_vals - y_min) / (y_max - y_min + 1e-8)
                
                # Sample data points for cleaner visualization
                sample_idx = np.random.RandomState(42).choice(
                    len(x_vals), size=max(1, int(len(x_vals) * .1)), replace=False
                )
                sample_idx = np.sort(sample_idx)
                
                # Plot sampled raw data with color
                ax.errorbar(x_vals[sample_idx], y_norm[sample_idx], yerr=y_err[sample_idx] / (y_max - y_min + 1e-8), 
                           fmt='o', alpha=0.6, color=color, markersize=5)
                
                # Find and plot fitted curve
                phase_occ_fits = config_fit[config_fit['phase_occurrence'] == phase_occ]
                r_squared = np.nan
                if not phase_occ_fits.empty:
                    # Average fit parameters across seeds
                    a_mean = phase_occ_fits[f'{metric}_a'].mean()
                    L_mean = phase_occ_fits[f'{metric}_L'].mean()
                    n_mean = phase_occ_fits[f'{metric}_n'].mean()
                    K_mean = phase_occ_fits[f'{metric}_K'].mean()
                    
                    if all(~np.isnan([a_mean, L_mean, n_mean, K_mean])):
                        # Generate smooth curve
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_smooth = hill(x_smooth, a_mean, L_mean, n_mean, K_mean)
                        
                        # Calculate R²
                        y_pred = hill(x_vals, a_mean, L_mean, n_mean, K_mean)
                        ss_res = np.sum((y_norm - y_pred) ** 2)
                        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                        
                        # Plot fitted curve with matching color
                        ax.plot(x_smooth, y_smooth, '--', linewidth=2, alpha=0.8, color=color)
                
                # Create label with R² value
                if np.isnan(r_squared):
                    label_text = f'{phase_occ}'
                else:
                    label_text = f'{phase_occ} (R²={r_squared:.3f})'
                
                # Add to custom legend tracking
                handles.append(Line2D([0], [0], color=color, marker='o', linestyle='--', linewidth=2, markersize=5, label=label_text))
            
            ax.set_title(f'{config}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(f'{metric.title()} (normalized)')
            ax.grid(True, alpha=0.3)
            ax.legend(handles=handles, fontsize=8, loc='best')
        
        plt.suptitle(f'Fit Quality: {metric.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(fit_subdir, f'fit_quality_{metric}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {output_path}')


def plot_fitted_curves_by_occurrence(data: pd.DataFrame, fit_data_occur: pd.DataFrame, metrics: list, output_dir: str):
    """Plot occurrence-level raw data with overlaid Hill fits to inspect fit quality."""
    fit_subdir = os.path.join(output_dir, "fit_quality")
    os.makedirs(fit_subdir, exist_ok=True)

    for metric in metrics:
        if metric not in data.columns:
            continue

        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]

        for ax_idx, config in enumerate(configs):
            ax = axes[ax_idx]
            config_data = data[data['config'] == config].copy()
            config_fit = fit_data_occur[fit_data_occur['config'] == config]

            occs = sorted(config_data['occurrence'].dropna().unique())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(occs)))
            handles = []

            for occ_idx, occurrence in enumerate(occs):
                occ_data = config_data[config_data['occurrence'] == occurrence].copy()
                if occ_data.empty:
                    continue

                color = colors[occ_idx]
                color = PHASE_COLORS.get(occurrence, color)

                # Align x-axis with fitting: local episode index within each seed/occurrence sequence.
                occ_data = occ_data.sort_values(['seed', 'ep'])
                occ_data['local_ep'] = occ_data.groupby('seed').cumcount().astype(float)

                grouped = occ_data.groupby('local_ep')[metric].agg(['mean', 'std', 'count'])
                grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

                x_vals = grouped.index.values.astype(float)
                y_vals = grouped['mean'].values
                y_err = grouped['sem'].fillna(0).values

                if len(x_vals) == 0:
                    continue

                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                denom = y_max - y_min + 1e-8
                y_norm = (y_vals - y_min) / denom

                sample_idx = np.random.RandomState(42).choice(
                    len(x_vals), size=max(1, int(len(x_vals) * 0.1)), replace=False
                )
                sample_idx = np.sort(sample_idx)

                ax.errorbar(
                    x_vals[sample_idx],
                    y_norm[sample_idx],
                    yerr=y_err[sample_idx] / denom,
                    fmt='o',
                    alpha=0.6,
                    color=color,
                    markersize=5,
                )

                occ_fit = config_fit[config_fit['occurrence'] == occurrence]
                r_squared = np.nan

                if not occ_fit.empty:
                    a_mean = occ_fit[f'{metric}_a'].mean()
                    L_mean = occ_fit[f'{metric}_L'].mean()
                    n_mean = occ_fit[f'{metric}_n'].mean()
                    K_mean = occ_fit[f'{metric}_K'].mean()

                    if all(~np.isnan([a_mean, L_mean, n_mean, K_mean])):
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_smooth = hill(x_smooth, a_mean, L_mean, n_mean, K_mean)
                        y_pred = hill(x_vals, a_mean, L_mean, n_mean, K_mean)

                        ss_res = np.sum((y_norm - y_pred) ** 2)
                        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                        ax.plot(x_smooth, y_smooth, '--', linewidth=2, alpha=0.8, color=color)

                if np.isnan(r_squared):
                    label_text = f'occ {int(occurrence)}'
                else:
                    label_text = f'occ {int(occurrence)} (R²={r_squared:.3f})'

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker='o',
                        linestyle='--',
                        linewidth=2,
                        markersize=5,
                        label=label_text,
                    )
                )

            ax.set_title(f'{config}')
            ax.set_xlabel('Local Episode (within occurrence)')
            ax.set_ylabel(f'{metric.title()} (normalized)')
            ax.grid(True, alpha=0.3)
            if handles:
                ax.legend(handles=handles, fontsize=8, loc='best')

        plt.suptitle(f'Fit Quality by Occurrence: {metric.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(fit_subdir, f'fit_quality_occurrence_{metric}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {output_path}')


def plot_parameter_summary(fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Create a summary heatmap of mean curve parameters."""
    param_names = {"a": "Baseline", "L": "Asymptote", "n": "Hill Coeff", "K": "Half-Max"}
    
    for metric in metrics:
        # Aggregate: mean of parameters per config
        summary_data = {}
        
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            if param_col in fit_data.columns:
                summary_data[param_label] = fit_data.groupby("config")[param_col].mean()
        
        if not summary_data:
            continue
        
        summary_df = pd.DataFrame(summary_data)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            summary_df.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            cbar_kws={"label": "Parameter Value"},
        )
        plt.title(f"Curve Parameters Summary: {metric.replace('_', ' ').title()}")
        plt.xlabel("Config")
        output_path = os.path.join(output_dir, f"param_summary_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def main():
    sns.set_style("ticks")

    print(f"Loading aggregated results from {BASE_FIG_DIR}...")
    data = load_all_configs(BASE_FIG_DIR)
    
    if data is None or data.empty:
        print(f"No aggregated_results.csv files found in {BASE_FIG_DIR}")
        return
    
    if data is not None and not data.empty:
        # choose occurrence numeric column name (try a few common variants)
        occ_col = None
        for c in ("occurrence", "phase_occurrence", "occurrence_label", "occ"):
            if c in data.columns:
                occ_col = c
                break

        if occ_col is None:
            raise KeyError("Expected an 'occurrence' column (e.g. 'occurrence' or 'phase_occurrence') in the CSVs")

        # normalize numeric occurrence (if occurrence_label like 'x1' is present, extract digits)
        if data[occ_col].dtype == object:
            occ_numeric = pd.to_numeric(data[occ_col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce").fillna(0).astype(int)
        else:
            occ_numeric = data[occ_col].fillna(0).astype(int)

        # create the combined phase_occurrence as "<phase> (xN)" where N is 1-based
        data["phase_occurrence"] = data["label"].astype(str) + " (x" + (occ_numeric + 1).astype(str) + ")"
    
    print("Adding phase occurrence tracking...")
    
    print("Fitting learning curves...")
    fit_data = fit_curves_per_phase(data, METRICS)
    fit_data_occur = fit_curves_per_occurrence(data, METRICS)

    print(f"\nCreating plots in {PLOT_OUTPUT_DIR}...")
    # plot_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_learning_curves_by_phase_occurrence(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_learning_curves_by_occurrence(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_episodic_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)

    # plot_phase_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)
    
    # plot_fitted_curves(data, fit_data, METRICS, PLOT_OUTPUT_DIR)
    # plot_fitted_curves_by_occurrence(data, fit_data_occur, METRICS, PLOT_OUTPUT_DIR)
    
    # plot_curve_parameters(fit_data, METRICS, PLOT_OUTPUT_DIR)
    plot_criterion_crossing(data, METRICS, PLOT_OUTPUT_DIR)
    plot_criterion_crossing_by_phase_occurrence(data, METRICS, PLOT_OUTPUT_DIR)

    # plot_curve_parameters_per_occurrence(fit_data_occur, METRICS, PLOT_OUTPUT_DIR,)
    print(f"\nDone! All plots saved to {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()