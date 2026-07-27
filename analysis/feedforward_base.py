"""
Analyze dynamics of the motor cortex module during learning.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional
from scipy.optimize import curve_fit
from scipy.stats import linregress
from statannotations.Annotator import Annotator
import pdb

from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params
from motor_structure_learning.experiments import _mc_sac_impl as sac_base
from motor_structure_learning.experiments.sweeps import common as sweep
from motor_structure_learning.core.ec import EntorhinalCortex

# Settings
BASE_FIG_DIR = "paper/neural_activity_comparisons/mc_dynamics"
base_path = Path(BASE_FIG_DIR)
base_path.mkdir(parents=True, exist_ok=True)
checkpoint_dir = "figs/mc_ablations"
CHECKPOINT_ROOT = Path(checkpoint_dir)
PLOT_OUTPUT_DIR = base_path / "plots"

# Shared evaluation dataset: a deterministic center-out reaching layout (every
# trial starts at the grid center; targets sit on concentric circles at evenly
# spaced angles). No sampling involved, so the same targets apply to every
# config, seed, checkpoint, AND context.
RADII = (0.5, 1.0, 2.0)
N_DIRECTIONS = 12
N_CTX = 4
EVAL_DATASET_PATH = base_path / "eval_dataset.npz"


# Checkpoint discovery: `<root>/<config>/seed_<n>/checkpoints/phase_NN_<label>.pt`
PHASE_FILE_RE = re.compile(r"^phase_(\d+)_(.+)\.pt$")


def generate_eval_dataset(
    radii: tuple = RADII,
    n_directions: int = N_DIRECTIONS,
    grid_size: float = sac_base.GRID_SIZE,
    n_ctx: int = N_CTX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center-out reaching layout: every trial starts at the grid center, with
    targets on `len(radii)` concentric circles at `n_directions` equally spaced
    angles each. Purely deterministic (no sampling), so the same targets apply
    to every context. Returns `(init_pos, goal, radius, angle)`, each shaped
    `(n_ctx, n_trials, ...)` with `n_trials = len(radii) * n_directions`.
    """
    center = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    radius_grid, angle_grid = np.meshgrid(np.asarray(radii, dtype=np.float32), angles, indexing="ij")
    radius_flat = radius_grid.reshape(-1).astype(np.float32)
    angle_flat = angle_grid.reshape(-1).astype(np.float32)
    offsets = np.stack([np.cos(angle_flat), np.sin(angle_flat)], axis=-1) * radius_flat[:, None]

    n_trials = len(radii) * n_directions
    goal = center[None, :] + offsets
    init_pos = np.tile(center, (n_trials, 1))

    init_pos = np.tile(init_pos[None].astype(np.float32), (n_ctx, 1, 1))
    goal = np.tile(goal[None].astype(np.float32), (n_ctx, 1, 1))
    radius = np.tile(radius_flat[None], (n_ctx, 1))
    angle = np.tile(angle_flat[None], (n_ctx, 1))
    return init_pos, goal, radius, angle


def load_or_create_eval_dataset(
    path: Path = EVAL_DATASET_PATH, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the cached shared eval dataset, generating and caching it on first
    use so repeated runs of this script (across configs/seeds) evaluate every
    model on identical targets.
    """
    if path.exists():
        print(f"Loading cached eval dataset from {path}")
        data = np.load(path)
        return data["init_pos"], data["goal"], data["radius"], data["angle"]
    init_pos, goal, radius, angle = generate_eval_dataset(**kwargs)
    np.savez(path, init_pos=init_pos, goal=goal, radius=radius, angle=angle)
    return init_pos, goal, radius, angle


def discover_checkpoints(root: Path = CHECKPOINT_ROOT) -> pd.DataFrame:
    """Scan the ablation-sweep output tree and return one row per checkpoint
    file, with config/seed/phase parsed out of the path. Seeds that don't have
    a full set of `sweep.N_PHASES` checkpoints (still training, or crashed) are
    dropped so downstream evaluation only ever sees complete runs.
    """
    rows = []
    for ckpt_path in sorted(root.glob("*/seed_*/checkpoints/phase_*.pt")):
        match = PHASE_FILE_RE.match(ckpt_path.name)
        if match is None:
            continue
        rows.append({
            "config": ckpt_path.parents[2].name,
            "seed": int(ckpt_path.parents[1].name.removeprefix("seed_")),
            "phase_idx": int(match.group(1)),
            "phase_label": match.group(2),
            "path": ckpt_path,
        })

    df = pd.DataFrame(rows, columns=["config", "seed", "phase_idx", "phase_label", "path"])
    if df.empty:
        return df

    n_ckpts = df.groupby(["config", "seed"])["phase_idx"].transform("count")
    complete = n_ckpts == sweep.N_PHASES
    if (~complete).any():
        dropped = sorted(set(map(tuple, df.loc[~complete, ["config", "seed"]].to_numpy().tolist())))
        print(f"discover_checkpoints: dropping {len(dropped)} incomplete seed(s): {dropped}")
    df = df.loc[complete]

    return df.sort_values(["config", "seed", "phase_idx"]).reset_index(drop=True)


# Model reconstruction: rebuild the actor (+ EC/CB, if the config uses them)
# from a config's flags, then load one checkpoint row's weights into it.
CONFIG_PRESETS_BY_NAME = {preset["name"]: preset for preset in sweep.CONFIG_PRESETS}


def _reconstruct_frozen_pretrained_ec_cb(
    preset: dict, seed: int, use_ec: bool, use_cb: bool, device: str = "cpu",
) -> tuple:
    """Reconstruct EC/CB directly from the pretrained EC/CB checkpoint plus a
    seed-derived `cb_proj`, for configs whose EC/CB is frozen immediately
    after loading (currently only `full`: `use_pretrained_ec_cb=True`,
    `freeze_pretrained_ec_cb=True`).

    Training (`run_babab`/`run_babab_rnn`/`run_babab_delay`/`run_babab_delay_rnn`)
    only builds an `ECCBTrainer` -- and hence only saves EC/CB into the
    per-phase checkpoint -- when EC/CB has at least one *trainable*
    parameter (`ec_cb_params = [p for p in ec.parameters() if p.requires_grad] + ...`).
    A fully frozen EC/CB has none, so `ec_cb_trainer` ends up `None` and the
    per-phase checkpoint never contains EC/CB weights at all, even though
    they were actively used throughout training. Since they're frozen and
    pretrained, though, they're identical every phase/seed and fully
    recoverable from the original sources: `sweep.EC_CB_CKPT` for the
    weights, and a `np.random.default_rng(seed)`-seeded draw for `cb_proj`
    (`cb_proj` is the very first random draw in `run_babab*`, so reseeding
    identically and redrawing immediately reproduces it exactly).
    """
    ec = cb = cb_proj = None
    if use_ec:
        ec = EntorhinalCortex(input_dim=8, hidden_dim=sweep.EC_HIDDEN, latent_dim=sweep.EC_LATENT).to(device)
    if use_cb:
        cb = sweep.CerebellumCB(latent_dim=sweep.EC_LATENT, hidden=sweep.CB_HIDDEN).to(device)
    sweep._load_ec_cb_if_available(ec, cb, sweep.EC_CB_CKPT, device)
    if use_ec:
        ec.eval()
    if use_cb:
        cb.eval()
        rng = np.random.default_rng(seed)
        cb_proj_np = rng.standard_normal((sweep.CB_HIDDEN, sweep.CB_PROJ_DIM))
        cb_proj = torch.tensor(cb_proj_np, dtype=torch.float32, device=device)
        cb_proj = cb_proj / (cb_proj.norm(dim=0, keepdim=True) + 1e-8)
    return ec, cb, cb_proj


def load_actor_for_row(row: pd.Series, device: str = "cpu") -> dict:
    """Reconstruct the actor for one `discover_checkpoints()` row and load its
    saved weights, along with EC/CB (if the row's config uses them).

    Returns a dict with the actor (in eval mode), optional `ec`/`cb` modules and
    `cb_proj` projection matrix, the `use_ec`/`use_cb`/`use_oracle` flags needed
    to build augmented observations exactly as at train time, and checkpoint
    metadata (phase/seed/global step).
    """
    preset = CONFIG_PRESETS_BY_NAME[row["config"]]
    use_ec, use_cb, use_oracle = preset["use_ec"], preset["use_cb"], preset["use_oracle"]
    aug_obs_dim = sweep._compute_aug_obs_dim(use_ec, use_cb, use_oracle)

    ckpt = torch.load(row["path"], map_location=device)
    actor = sac_base.Actor(obs_dim=aug_obs_dim, action_dim=sac_base.ACTION_DIM).to(device)
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


# Rollout: replay a loaded actor bundle on the shared eval trials under one
# dynamics context, recording MC (actor backbone) activity at every step.
def rollout_actor(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    ctx_idx: int,
    device: str = "cpu",
    max_steps: int = sac_base.MAX_STEPS,
    deterministic: bool = True,
) -> dict:
    """Roll out one actor bundle (from `load_actor_for_row`) on a batch of
    (init_pos, goal) trials under `sweep.PHASE_DATA[ctx_idx]`'s dynamics,
    recording the MC (actor backbone) hidden activity at every step alongside
    position/action/distance-to-goal. EC/CB features (if the config uses them)
    are recomputed online exactly as at train time, with a fresh EC hidden
    state per trial. `ctx_idx` is independent of the checkpoint's own training
    phase, so the same bundle can be evaluated under every context.
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    n_trials = init_pos.shape[0]
    hidden_dim = sac_base.HIDDEN_DIM

    hidden_hist = np.full((n_trials, max_steps, hidden_dim), np.nan, dtype=np.float32)
    pos_hist = np.full((n_trials, max_steps, 2), np.nan, dtype=np.float32)
    action_hist = np.full((n_trials, max_steps, 2), np.nan, dtype=np.float32)
    dist_hist = np.full((n_trials, max_steps), np.nan, dtype=np.float32)
    success = np.zeros(n_trials, dtype=bool)
    steps = np.zeros(n_trials, dtype=np.int64)

    hidden_capture: dict = {}

    def _capture_hidden(_module, _inp, out):
        hidden_capture["h"] = out.detach()

    handle = actor.backbone.register_forward_hook(_capture_hidden)
    try:
        for i in range(n_trials):
            pos = init_pos[i].copy()
            vel = np.zeros(2, dtype=np.float32)
            g = goal[i]
            prev_action = np.zeros(2, dtype=np.float32)
            h_ec = torch.zeros(1, 1, sweep.EC_HIDDEN, device=device) if use_ec else None

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
                    y_t, _, mean_action_t = actor.get_action(obs_t)
                action_t = mean_action_t if deterministic else y_t
                action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

                hidden_hist[i, t] = hidden_capture["h"].squeeze(0).cpu().numpy()
                pos_hist[i, t] = pos
                action_hist[i, t] = action_np
                steps[i] = t + 1

                next_pos, next_vel = sac_base.env_step_np(pos, vel, action_np, beta, theta, gamma_curl, g_dyn, bx, by)
                if use_ec:
                    h_ec, _ = sweep._update_ec_hidden(ec, cb, h_ec, pos, action_np, next_pos, device, use_ec)
                pos, vel, prev_action = next_pos, next_vel, action_np
    finally:
        handle.remove()

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


def evaluate_checkpoint(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    device: str = "cpu",
    **rollout_kwargs,
) -> dict:
    """Roll `bundle` out under every context in `sweep.base.PHASE_DATA`, using
    that context's slice of the shared eval dataset (`init_pos`/`goal` shaped
    `(n_ctx, n_trials, 2)`). Returns `{ctx_idx: rollout_actor(...) result}`.
    """
    n_ctx = init_pos.shape[0]
    return {
        ctx_idx: rollout_actor(bundle, init_pos[ctx_idx], goal[ctx_idx], ctx_idx, device=device, **rollout_kwargs)
        for ctx_idx in range(n_ctx)
    }


# Aggregation: evaluate every discovered checkpoint and cache its hidden
# activity to disk, one file per checkpoint (resumable, memory-bounded).
HIDDEN_ACTIVITY_DIR = base_path / "hidden_activity"


def _hidden_activity_path(row: pd.Series, out_dir: Path) -> Path:
    return out_dir / f"{row['config']}_seed{row['seed']:02d}_phase{row['phase_idx']:02d}.npz"


def aggregate_hidden_activity(
    df_ckpts: pd.DataFrame,
    init_pos: np.ndarray,
    goal: np.ndarray,
    out_dir: Path = HIDDEN_ACTIVITY_DIR,
    device: str = "cpu",
    skip_existing: bool = True,
    **rollout_kwargs,
) -> pd.DataFrame:
    """Evaluate every checkpoint in `df_ckpts` on the shared eval dataset across
    all contexts, caching the resulting hidden activity (+ trajectories) to one
    `.npz` file per checkpoint under `out_dir`.

    Returns an index DataFrame with one row per (config, seed, phase, context)
    pointing at that checkpoint's cache file, plus summary stats (success_rate,
    mean_steps) for quick filtering without reloading the full arrays.

    Resumable: existing cache files are skipped by default (`skip_existing`),
    so re-running as more checkpoints appear (e.g. training still in progress)
    only evaluates what's new.
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
            bundle = load_actor_for_row(row, device=device)
            results = evaluate_checkpoint(bundle, init_pos, goal, device=device, **rollout_kwargs)
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


def load_cached_result(index_row: pd.Series) -> dict:
    """Load one context's full hidden/pos/action/dist/success/steps arrays back
    out of an `aggregate_hidden_activity()` index row's cache file.
    """
    ctx_idx = index_row["ctx_idx"]
    with np.load(index_row["path"]) as data:
        return {key: data[f"ctx{ctx_idx}_{key}"] for key in ("hidden", "pos", "action", "dist", "success", "steps")}


def participation_ratio(X: np.ndarray) -> float:
    """Participation ratio of `X`'s (n_samples, n_features) covariance
    eigenspectrum: PR = (sum(eigvals))^2 / sum(eigvals^2). Ranges from 1 (all
    variance along one dimension) to n_features (variance spread equally
    across every dimension) -- a standard effective-dimensionality measure.
    """
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float((eigvals.sum() ** 2 / np.sum(eigvals ** 2)) / X.shape[1])


def compute_cross_context_participation_ratio(index_df: pd.DataFrame) -> pd.DataFrame:
    """For each checkpoint (config, seed, phase), pool the valid (pre-success or
    pre-timeout) MC hidden-activity vectors across every evaluated context and
    compute the participation ratio of the pooled activity -- one cross-context
    dimensionality estimate per checkpoint.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        pooled = []
        for _, row in group.iterrows():
            r = load_cached_result(row)
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


# Subspace alignment: for each checkpoint, how much do the different contexts'
# MC activity subspaces overlap with each other (Elsayed et al. 2016 style).
ALIGNMENT_K = 8


def _pca_subspace(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k PCA subspace of `X` (n_samples, n_features): returns `(Q, cov)`,
    where `Q` is an `(n_features, k)` orthonormal basis of the leading k
    eigenvectors of `X`'s covariance, and `cov` is that covariance matrix.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    Q = eigvecs[:, order[:k]]
    return Q, cov


def alignment_index(cov_i: np.ndarray, Q_i: np.ndarray, Q_j: np.ndarray) -> float:
    """Directional subspace alignment of context `i` onto context `j`'s
    subspace: the fraction of `i`'s variance captured by `Q_j`, relative to
    the max possible (the variance `i`'s own top-k subspace `Q_i` captures).
    `trace(Qj^T cov_i Qj) / trace(Qi^T cov_i Qi)`, ranges 0 (orthogonal) to 1
    (identical subspace). Asymmetric: `alignment_index` of i onto j need not
    equal j onto i.
    """
    own = np.trace(Q_i.T @ cov_i @ Q_i)
    cross = np.trace(Q_j.T @ cov_i @ Q_j)
    return float(cross / own)


def context_alignment_matrix(group: pd.DataFrame, k: int = ALIGNMENT_K) -> np.ndarray:
    """Symmetrized k-dim subspace alignment matrix across the contexts present
    in `group` (one checkpoint's rows from an `aggregate_hidden_activity()`
    index, one per context). Entry `[i, j]` is
    `0.5 * (alignment_index(i onto j) + alignment_index(j onto i))`; the
    diagonal is 1 by construction.
    """
    group = group.sort_values("ctx_idx")
    covs, Qs = [], []
    for _, row in group.iterrows():
        r = load_cached_result(row)
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


def compute_cross_context_alignment(index_df: pd.DataFrame, k: int = ALIGNMENT_K) -> pd.DataFrame:
    """For each checkpoint (config, seed, phase), compute the symmetrized k-dim
    subspace alignment matrix across contexts and summarize it as the mean of
    its off-diagonal entries -- one cross-context alignment score per
    checkpoint, alongside the full matrix for closer inspection.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix(group, k=k)
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


# Geometry visualization: MC hidden-activity trajectories in 3D PC space,
# colored by target angle -- a 2x2 per-context panel grid for detail, plus one
# combined plot overlaying every context (so translations/rotations between
# contexts' activity clouds are directly visible). Shared by every
# `plot_*_geometry*` function across the base/delay/recurrent variants.
CONTEXT_LINESTYLES = ["-", "--", ":", "-."]
CONTEXT_MARKERS = ["o", "s", "^", "D"]


def plot_geometry_3d(
    per_ctx_trajs: dict,
    title: str,
    n_components: int = 3,
    out_path_panels: Optional[Path] = None,
    out_path_combined: Optional[Path] = None,
    cmap_name: str = "twilight",
    mark_end: bool = False,
    end_label: str = "final (pre-go)",
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Shared 3D geometry visualization core.

    `per_ctx_trajs` is `{ctx_idx: {"trajs": [(n_steps_i, hidden_dim), ...],
    "angle": [angle_i, ...]}}` -- already-extracted per-trial trajectories
    (whatever epoch/context filtering the caller needed is done by then) with
    each trial's target angle in radians. PCA is fit jointly across every
    context's pooled trajectories, so every panel and the combined plot share
    one coordinate frame.

    Context identity is double-encoded on the combined plot (linestyle AND
    marker shape, with a small legend) since color is already used for target
    angle (`cmap_name`, cyclic since angle wraps at 360 degrees -- `twilight`
    and `hsv` both work).

    If `mark_end`, each trajectory's *last* point also gets a fixed "X"
    marker (same angle color, not context-varying) -- e.g. the delay-epoch
    plots use this to show each trial's final location right before the
    go-cue permits movement. `end_label` customizes what that marker is
    called in the legend/caption.

    Returns `(panels_figure, combined_figure)`.
    """
    ctx_idxs = sorted(per_ctx_trajs.keys())
    pooled = np.concatenate(
        [traj for ctx_idx in ctx_idxs for traj in per_ctx_trajs[ctx_idx]["trajs"]], axis=0
    )
    pca = PCA(n_components=n_components)
    pca.fit(pooled)

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
    angle_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    angle_labels = ["0", "90", "180", "270", "360"]

    # --- per-context panels ---
    fig1 = plt.figure(figsize=(11, 11))
    for pos, ctx_idx in enumerate(ctx_idxs, start=1):
        ax = fig1.add_subplot(2, 2, pos, projection="3d")
        d = per_ctx_trajs[ctx_idx]
        for traj, ang in zip(d["trajs"], d["angle"]):
            proj = pca.transform(traj)
            color = cmap(norm(ang % (2 * np.pi)))
            ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color=color, lw=1.0, alpha=0.85)
            ax.scatter(proj[0, 0], proj[0, 1], proj[0, 2], color=color, s=14)
            if mark_end:
                ax.scatter(proj[-1, 0], proj[-1, 1], proj[-1, 2], color=color, marker="X", s=40,
                           edgecolors="black", linewidths=0.5)
        ax.set_title(sweep.PHASE_DATA[ctx_idx]["label"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig1.colorbar(sm, ax=fig1.axes, fraction=0.03, pad=0.06, ticks=angle_ticks, shrink=0.7)
    cbar.ax.set_yticklabels(angle_labels)
    cbar.set_label("target angle (deg)")
    if mark_end:
        marker_legend = [
            Line2D([0], [0], color="gray", lw=0, marker="o", markersize=6, label="start"),
            Line2D([0], [0], color="gray", lw=0, marker="X", markersize=7,
                   markeredgecolor="black", label=end_label),
        ]
        fig1.legend(handles=marker_legend, loc="upper right", fontsize=8, frameon=False)
    fig1.suptitle(f"{title} -- per-context panels")
    if out_path_panels is not None:
        fig1.savefig(out_path_panels, dpi=150)

    # --- combined overlay ---
    # Every trial starts at (essentially) the same location regardless of
    # context -- e.g. every center-out trial starts at the grid center -- so
    # a context-varying marker there can't actually distinguish anything.
    # Trajectories only spread apart by context over time, so the
    # context-shape marker goes at the END instead; the start gets a plain,
    # non-varying marker.
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111, projection="3d")
    for i, ctx_idx in enumerate(ctx_idxs):
        d = per_ctx_trajs[ctx_idx]
        ls = CONTEXT_LINESTYLES[i % len(CONTEXT_LINESTYLES)]
        mk = CONTEXT_MARKERS[i % len(CONTEXT_MARKERS)]
        for traj, ang in zip(d["trajs"], d["angle"]):
            proj = pca.transform(traj)
            color = cmap(norm(ang % (2 * np.pi)))
            ax2.plot(proj[:, 0], proj[:, 1], proj[:, 2], color=color, lw=1.0, ls=ls, alpha=0.8)
            ax2.scatter(proj[0, 0], proj[0, 1], proj[0, 2], color=color, marker="o", s=14)
            ax2.scatter(proj[-1, 0], proj[-1, 1], proj[-1, 2], color=color, marker=mk, s=40,
                        edgecolors="black", linewidths=0.5)

    legend_handles = [
        Line2D(
            [0], [0], color="black", lw=1.5,
            ls=CONTEXT_LINESTYLES[i % len(CONTEXT_LINESTYLES)],
            marker=CONTEXT_MARKERS[i % len(CONTEXT_MARKERS)], markersize=6,
            label=sweep.PHASE_DATA[ctx_idx]["label"],
        )
        for i, ctx_idx in enumerate(ctx_idxs)
    ]
    legend_handles.append(
        Line2D([0], [0], color="gray", lw=0, marker="o", markersize=6, label="start (shared)")
    )
    if mark_end:
        legend_handles.append(
            Line2D([0], [0], color="gray", lw=0, marker="*", markersize=9, label=f"end marker = {end_label}")
        )
    ax2.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False)
    sm2 = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar2 = fig2.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.1, ticks=angle_ticks, shrink=0.7)
    cbar2.ax.set_yticklabels(angle_labels)
    cbar2.set_label("target angle (deg)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title(f"{title} -- combined")
    if out_path_combined is not None:
        fig2.savefig(out_path_combined, dpi=150)

    return fig1, fig2


def plot_context_geometry(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """PCA-project one checkpoint's MC hidden activity (fit jointly across all
    evaluated contexts) into 3D and visualize it via `plot_geometry_3d`:
    per-context panels plus one combined overlay for comparing contexts
    directly. `angle` is the `(n_ctx, n_trials)` array from
    `load_or_create_eval_dataset()` (identical across contexts by construction).
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result(row)
        hidden, steps = r["hidden"], r["steps"]
        per_ctx_trajs[row["ctx_idx"]] = {
            "trajs": [hidden[i, :s] for i, s in enumerate(steps)],
            "angle": angle[row["ctx_idx"]],
        }

    title = f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']})"
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components, out_path_panels=out_panels, out_path_combined=out_combined
    )


if __name__ == "__main__":
    set_plt_params(style='fast')

    # Discover all checkpoints in the ablation sweep output tree.
    df_ckpts = discover_checkpoints()
    print(f"Found {len(df_ckpts)} checkpoints across {df_ckpts['config'].nunique()} configs and {df_ckpts['seed'].nunique()} seeds.")

    # Generate or load the shared evaluation dataset (init_pos, goal) for all contexts.
    init_pos, goal, radius, angle = load_or_create_eval_dataset()

    # Aggregate hidden activity for every checkpoint, caching to disk and returning an index DataFrame.
    df_index = aggregate_hidden_activity(df_ckpts, init_pos, goal, device="cpu", skip_existing=True)
    print(f"Aggregated hidden activity for {len(df_index)} (config, seed, phase, context) rows.")

    pr_df = compute_cross_context_participation_ratio(df_index)
    ai_df = compute_cross_context_alignment(df_index, k=ALIGNMENT_K)


    print(f"Computed cross-context participation ratios for {len(pr_df)} checkpoints.")

    plt.figure(figsize=(6, 4))

    sns.lineplot(data=pr_df, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    plt.title("Cross-Context Participation Ratio Over Learning Phases")
    plt.xlabel("Block Index")
    plt.ylabel("Participation Ratio")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    plt.figure(figsize=(6, 4))

    sns.lineplot(data=ai_df, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    plt.title("Cross-Context Alignment Over Learning Phases")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Off-Diagonal Alignment")
    plt.ylim(0.5, 1.0)
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    # MC activity geometry in PC space, colored by target angle -- one figure
    # per config, for a single seed at the final training block.
    geometry_seed = 0
    last_phase = sweep.N_PHASES - 1
    for config in df_index["config"].unique():
        plot_context_geometry(df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
                              out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_geometry")

    plt.pause(1)
    pdb.set_trace()