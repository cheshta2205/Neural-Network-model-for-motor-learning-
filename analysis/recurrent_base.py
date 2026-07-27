"""
Analyze dynamics of the motor cortex module during learning, for the
recurrent-actor variant (see `sweep_ablations_rnn.py`).

Reuses everything generic from `feedforward_base.py` -- the shared
center-out eval dataset, checkpoint discovery, and the low-level
participation-ratio / subspace-alignment primitives -- and defines only
what's specific to a recurrent actor: reconstructing `RecurrentActor` instead
of the feedforward `Actor`, and reading MC activity directly from the hidden
state `get_action` already returns -- no forward hook needed, since the
hidden state is an explicit output of a recurrent cell rather than something
an internal `backbone` submodule happens to produce.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import seaborn as sns

from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional
import pdb

from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params
from motor_structure_learning.experiments import _mc_sac_impl as sac_base
from motor_structure_learning.experiments.sweeps import common as sweep
from motor_structure_learning.core.ec import EntorhinalCortex

from feedforward_base import (
    base_path,
    PLOT_OUTPUT_DIR,
    discover_checkpoints,
    load_or_create_eval_dataset,
    CONFIG_PRESETS_BY_NAME,
    _hidden_activity_path,
    participation_ratio,
    ALIGNMENT_K,
    _pca_subspace,
    alignment_index,
    plot_geometry_3d,
    _reconstruct_frozen_pretrained_ec_cb,
)

# Recurrent-actor variant of the pipeline: same shared eval dataset and
# checkpoint format, but checkpoints come from `figs/sweeps/recurrent/base/`
# and the actor is `sac_base.RecurrentActor` (GRUCell-based) for every config.
CHECKPOINT_ROOT_RNN = Path("figs/sweeps/recurrent/base")
HIDDEN_ACTIVITY_DIR_RNN = base_path / "sweeps" / "recurrent" / "base" / "hidden_activity"


def load_actor_for_row_rnn(row: pd.Series, device: str = "cpu") -> dict:
    """Same as `feedforward_base.load_actor_for_row`, but reconstructs a
    `RecurrentActor` instead of the feedforward `Actor`. Obs dim computation
    is unaffected by recurrence (`sweep._compute_aug_obs_dim` as before);
    only the actor's internal architecture differs.
    """
    preset = CONFIG_PRESETS_BY_NAME[row["config"]]
    use_ec, use_cb, use_oracle = preset["use_ec"], preset["use_cb"], preset["use_oracle"]
    aug_obs_dim = sweep._compute_aug_obs_dim(use_ec, use_cb, use_oracle)

    ckpt = torch.load(row["path"], map_location=device)
    actor = sac_base.RecurrentActor(
        obs_dim=aug_obs_dim, action_dim=sac_base.ACTION_DIM, hidden_dim=sac_base.HIDDEN_DIM
    ).to(device)
    actor.load_state_dict(ckpt["trainer"]["models"]["actor"])
    actor.eval()

    ec = cb = cb_proj = None
    ec_cb_state = ckpt.get("ec_cb_trainer")
    if ec_cb_state is not None:
        if use_ec:
            ec = EntorhinalCortex(input_dim=8, hidden_dim=sweep.EC_HIDDEN, latent_dim=sweep.EC_LATENT).to(device)
            ec.load_state_dict(ec_cb_state["models"]["ec"])
            ec.eval()
        if use_cb:
            cb = sweep.CerebellumCB(latent_dim=sweep.EC_LATENT, hidden=sweep.CB_HIDDEN).to(device)
            cb.load_state_dict(ec_cb_state["models"]["cb"])
            cb.eval()
            cb_proj = ec_cb_state["cb_proj"].to(device)
    elif use_ec or use_cb:
        if not preset.get("use_pretrained_ec_cb"):
            raise RuntimeError(
                f"checkpoint {row['path']} has no ec_cb_trainer state but config "
                f"{row['config']!r} uses EC/CB and isn't pretrained -- weights are lost."
            )
        ec, cb, cb_proj = _reconstruct_frozen_pretrained_ec_cb(preset, row["seed"], use_ec, use_cb, device)

    return {
        "actor": actor,
        "ec": ec,
        "cb": cb,
        "cb_proj": cb_proj,
        "use_ec": use_ec,
        "use_cb": use_cb,
        "use_oracle": use_oracle,
        "config": row["config"],
        "seed": row["seed"],
        "phase_idx": row["phase_idx"],
        "phase_label": row["phase_label"],
        "global_step": ckpt.get("global_step"),
        "global_ep": ckpt.get("global_ep"),
    }


def rollout_actor_rnn(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    ctx_idx: int,
    device: str = "cpu",
    max_steps: int = sac_base.MAX_STEPS,
    deterministic: bool = True,
) -> dict:
    """Like `feedforward_base.rollout_actor`, but for a `RecurrentActor`:
    MC hidden activity is the actor's own recurrent state, taken directly
    from `get_action`'s returned `h_new` each step -- no forward hook needed.
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    n_trials = init_pos.shape[0]
    hidden_dim = actor.hidden_dim

    hidden_hist = np.full((n_trials, max_steps, hidden_dim), np.nan, dtype=np.float32)
    pos_hist = np.full((n_trials, max_steps, 2), np.nan, dtype=np.float32)
    action_hist = np.full((n_trials, max_steps, 2), np.nan, dtype=np.float32)
    dist_hist = np.full((n_trials, max_steps), np.nan, dtype=np.float32)
    success = np.zeros(n_trials, dtype=bool)
    steps = np.zeros(n_trials, dtype=np.int64)

    for i in range(n_trials):
        pos = init_pos[i].copy()
        vel = np.zeros(2, dtype=np.float32)
        g = goal[i]
        prev_action = np.zeros(2, dtype=np.float32)
        h_ec = torch.zeros(1, 1, sweep.EC_HIDDEN, device=device) if use_ec else None
        h_actor = actor.init_hidden(batch_size=1, device=device)

        for t in range(max_steps):
            d = float(np.linalg.norm(pos - g))
            dist_hist[i, t] = d
            if d < sac_base.GOAL_RADIUS:
                success[i] = True
                steps[i] = t
                break

            base_obs = sac_base.build_obs(pos, g, prev_action)
            if use_ec or use_cb:
                z_ec, z_cb = sweep._current_features(
                    ec, cb, cb_proj, h_ec, pos, prev_action, device, use_ec, use_cb
                )
            else:
                z_ec, z_cb = None, None
            obs_np = sweep._build_aug_obs(base_obs, z_ec, z_cb, use_ec, use_cb, use_oracle, ctx_one_hot)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                y_t, _, mean_action_t, h_actor_new = actor.get_action(obs_t, h_actor)
            action_t = mean_action_t if deterministic else y_t
            action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            hidden_hist[i, t] = h_actor_new.squeeze(0).cpu().numpy()
            pos_hist[i, t] = pos
            action_hist[i, t] = action_np
            steps[i] = t + 1
            h_actor = h_actor_new

            next_pos, next_vel = sac_base.env_step_np(pos, vel, action_np, beta, theta, gamma_curl, g_dyn, bx, by)
            if use_ec:
                h_ec, _ = sweep._update_ec_hidden(ec, cb, h_ec, pos, action_np, next_pos, device, use_ec)
            pos, vel, prev_action = next_pos, next_vel, action_np

    return {
        "hidden": hidden_hist,
        "pos": pos_hist,
        "action": action_hist,
        "dist": dist_hist,
        "success": success,
        "steps": steps,
        "ctx_idx": ctx_idx,
        "ctx_label": ctx_label,
    }


def evaluate_checkpoint_rnn(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    device: str = "cpu",
    **rollout_kwargs,
) -> dict:
    """Recurrent-actor counterpart to `feedforward_base.evaluate_checkpoint`."""
    n_ctx = init_pos.shape[0]
    return {
        ctx_idx: rollout_actor_rnn(bundle, init_pos[ctx_idx], goal[ctx_idx], ctx_idx, device=device, **rollout_kwargs)
        for ctx_idx in range(n_ctx)
    }


def aggregate_hidden_activity_rnn(
    df_ckpts: pd.DataFrame,
    init_pos: np.ndarray,
    goal: np.ndarray,
    out_dir: Path = HIDDEN_ACTIVITY_DIR_RNN,
    device: str = "cpu",
    skip_existing: bool = True,
    **rollout_kwargs,
) -> pd.DataFrame:
    """Recurrent-actor counterpart to
    `feedforward_base.aggregate_hidden_activity`: same resumable,
    one-file-per-checkpoint caching, via `load_actor_for_row_rnn` /
    `evaluate_checkpoint_rnn`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    n_ckpts = len(df_ckpts)

    for i, (_, row) in enumerate(df_ckpts.iterrows(), start=1):
        ckpt_path = _hidden_activity_path(row, out_dir)

        if skip_existing and ckpt_path.exists():
            print(f"[{i}/{n_ckpts}] skip (cached): {ckpt_path.name}")
            with np.load(ckpt_path) as data:
                per_ctx = {
                    ctx_idx: {"success": data[f"ctx{ctx_idx}_success"], "steps": data[f"ctx{ctx_idx}_steps"]}
                    for ctx_idx in range(init_pos.shape[0])
                }
        else:
            print(f"[{i}/{n_ckpts}] evaluating: {row['config']} seed={row['seed']} "
                  f"phase={row['phase_idx']} ({row['phase_label']})")
            bundle = load_actor_for_row_rnn(row, device=device)
            results = evaluate_checkpoint_rnn(bundle, init_pos, goal, device=device, **rollout_kwargs)
            save_data = {
                f"ctx{ctx_idx}_{key}": r[key]
                for ctx_idx, r in results.items()
                for key in ("hidden", "pos", "action", "dist", "success", "steps")
            }
            np.savez_compressed(ckpt_path, **save_data)
            per_ctx = results

        for ctx_idx, r in per_ctx.items():
            index_rows.append({
                "config": row["config"],
                "seed": row["seed"],
                "phase_idx": row["phase_idx"],
                "phase_label": row["phase_label"],
                "ctx_idx": ctx_idx,
                "ctx_label": sweep.PHASE_DATA[ctx_idx]["label"],
                "success_rate": float(np.mean(r["success"])),
                "mean_steps": float(np.mean(r["steps"])),
                "path": ckpt_path,
            })

    return pd.DataFrame(index_rows)


def load_cached_result_rnn(index_row: pd.Series) -> dict:
    """Load one context's full arrays back out of an
    `aggregate_hidden_activity_rnn()` index row's cache file.
    """
    ctx_idx = index_row["ctx_idx"]
    with np.load(index_row["path"]) as data:
        return {key: data[f"ctx{ctx_idx}_{key}"] for key in ("hidden", "pos", "action", "dist", "success", "steps")}


def compute_cross_context_participation_ratio_rnn(index_df: pd.DataFrame) -> pd.DataFrame:
    """Same as `feedforward_base.compute_cross_context_participation_ratio`,
    but reads recurrent-actor cache files via `load_cached_result_rnn`.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        pooled = []
        for _, row in group.iterrows():
            r = load_cached_result_rnn(row)
            hidden, steps = r["hidden"], r["steps"]
            for i, s in enumerate(steps):
                pooled.append(hidden[i, :s])
        pooled = np.concatenate(pooled, axis=0)
        records.append({
            "config": config,
            "seed": seed,
            "phase_idx": phase_idx,
            "phase_label": phase_label,
            "n_samples": pooled.shape[0],
            "participation_ratio": participation_ratio(pooled),
        })

    return pd.DataFrame(records).sort_values(["config", "seed", "phase_idx"]).reset_index(drop=True)


def context_alignment_matrix_rnn(group: pd.DataFrame, k: int = ALIGNMENT_K) -> np.ndarray:
    """Same as `feedforward_base.context_alignment_matrix`, but reads
    recurrent-actor cache files via `load_cached_result_rnn`.
    """
    group = group.sort_values("ctx_idx")
    covs, Qs = [], []
    for _, row in group.iterrows():
        r = load_cached_result_rnn(row)
        X = np.concatenate([r["hidden"][i, :s] for i, s in enumerate(r["steps"])], axis=0)
        Q, cov = _pca_subspace(X, k)
        covs.append(cov)
        Qs.append(Q)

    n = len(covs)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a_ij = alignment_index(covs[i], Qs[i], Qs[j])
            a_ji = alignment_index(covs[j], Qs[j], Qs[i])
            M[i, j] = M[j, i] = 0.5 * (a_ij + a_ji)
    return M


def compute_cross_context_alignment_rnn(index_df: pd.DataFrame, k: int = ALIGNMENT_K) -> pd.DataFrame:
    """Same as `feedforward_base.compute_cross_context_alignment`, but for
    recurrent-actor checkpoints.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix_rnn(group, k=k)
        off_diag = M[~np.eye(M.shape[0], dtype=bool)]
        records.append({
            "config": config,
            "seed": seed,
            "phase_idx": phase_idx,
            "phase_label": phase_label,
            "alignment_matrix": M,
            "mean_offdiag_alignment": float(off_diag.mean()),
        })

    return pd.DataFrame(records).sort_values(["config", "seed", "phase_idx"]).reset_index(drop=True)


def plot_context_geometry_rnn(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Same as `feedforward_base.plot_context_geometry`, but for
    recurrent-actor checkpoints.
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result_rnn(row)
        hidden, steps = r["hidden"], r["steps"]
        per_ctx_trajs[row["ctx_idx"]] = {
            "trajs": [hidden[i, :s] for i, s in enumerate(steps)],
            "angle": angle[row["ctx_idx"]],
        }

    title = f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']}) -- recurrent actor"
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components, out_path_panels=out_panels, out_path_combined=out_combined
    )


if __name__ == "__main__":
    set_plt_params(style='fast')

    df_ckpts = discover_checkpoints(root=CHECKPOINT_ROOT_RNN)
    print(f"Found {len(df_ckpts)} checkpoints across {df_ckpts['config'].nunique()} configs and {df_ckpts['seed'].nunique()} seeds.")

    init_pos, goal, radius, angle = load_or_create_eval_dataset()

    df_index = aggregate_hidden_activity_rnn(df_ckpts, init_pos, goal, device="cpu", skip_existing=True)
    print(f"Aggregated hidden activity for {len(df_index)} (config, seed, phase, context) rows.")

    pr_df = compute_cross_context_participation_ratio_rnn(df_index)
    ai_df = compute_cross_context_alignment_rnn(df_index, k=ALIGNMENT_K)

    print(f"Computed cross-context participation ratios for {len(pr_df)} checkpoints.")

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=pr_df, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    plt.title("Cross-Context Participation Ratio Over Learning Phases (recurrent actor)")
    plt.xlabel("Block Index")
    plt.ylabel("Participation Ratio")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=ai_df, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    plt.title("Cross-Context Alignment Over Learning Phases (recurrent actor)")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Off-Diagonal Alignment")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    geometry_seed = 0
    last_phase = sweep.N_PHASES - 1
    for config in df_index["config"].unique():
        plot_context_geometry_rnn(
            df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
            out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_rnn_geometry",
        )

    plt.pause(1)
