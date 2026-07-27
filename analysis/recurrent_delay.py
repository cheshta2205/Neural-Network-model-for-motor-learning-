"""
Analyze dynamics of the motor cortex module during learning, for the
recurrent-actor instructed-delay variant (see `sweep_ablations_delay_rnn.py`).

Combines `recurrent_base`'s recurrent-actor handling (hidden state
read directly from `get_action`'s return, no forward hook) with
`feedforward_delay`'s task mechanics (fixed evaluation delay length,
go-cue observation, epoch tagging). Reuses everything generic from
`feedforward_base.py` directly.
"""
import pdb
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

from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params
from motor_structure_learning.experiments import _mc_sac_impl as sac_base
from motor_structure_learning.experiments.sweeps import common as sweep
from motor_structure_learning.experiments.sweeps import _delay_task as delay_task
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
from feedforward_delay import EVAL_DELAY_LEN, EC_PRIOR_SKIP_STEPS

# Recurrent-actor + instructed-delay variant: same shared eval dataset,
# checkpoints from `figs/sweeps/recurrent/delay/`, actor is
# `sac_base.RecurrentActor` with the extra go-cue observation dim, and MC
# hidden activity is read directly from `get_action`'s returned hidden state
# (no forward hook), tagged by `epoch` (delay vs. movement) exactly like the
# non-recurrent delay pipeline.
CHECKPOINT_ROOT_DELAY_RNN = Path("figs/sweeps/recurrent/delay")
HIDDEN_ACTIVITY_DIR_DELAY_RNN = base_path / "sweeps" / "recurrent" / "delay" / "hidden_activity"
# Separate cache dir for `warm_start_ec=True` runs (see `compute_ec_context_prior_rnn`)
# -- the cached arrays mean something different (EC seeded from a per-context
# prior, not zero-init), so they must not share a cache path with cold-start runs.
HIDDEN_ACTIVITY_DIR_DELAY_RNN_ECWARM = base_path / "sweeps" / "recurrent" / "delay" / "hidden_activity_ecwarm"


def load_actor_for_row_delay_rnn(row: pd.Series, device: str = "cpu") -> dict:
    """Same as `recurrent_base.load_actor_for_row_rnn`, but with the
    extra go-cue observation dim (`delay_task.compute_aug_obs_dim_delay`).
    """
    preset = CONFIG_PRESETS_BY_NAME[row["config"]]
    use_ec, use_cb, use_oracle = preset["use_ec"], preset["use_cb"], preset["use_oracle"]
    aug_obs_dim = delay_task.compute_aug_obs_dim_delay(use_ec, use_cb, use_oracle)

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


def _run_trial_chained_rnn(
    bundle: dict,
    pos0: np.ndarray,
    goal0: np.ndarray,
    ctx_idx: int,
    h_ec_in: Optional[torch.Tensor],
    delay_len: int = EVAL_DELAY_LEN,
    device: str = "cpu",
    max_steps: int = sac_base.MAX_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent-actor counterpart to
    `feedforward_delay._run_trial_chained`: runs one full
    instructed-delay trial in context `ctx_idx` with EC's hidden state
    starting from `h_ec_in` instead of zero. Returns `(h_ec_final,
    h_ec_trajectory)`: EC's hidden state at trial end, used to chain into the
    next trial -- the actor's own recurrent state (`h_actor`) still resets to
    zero at the start of this trial as usual, only EC's memory is chained
    across trials, since only EC has no other way to represent context before
    movement reveals it -- and the full per-timestep trajectory of EC hidden
    states across this trial, shape `(T, hidden_dim)` (used to build the
    context prior from EC's whole state trajectory rather than just its
    end-of-trial snapshot -- see `compute_ec_context_prior_rnn`).

    Whenever `h_ec_in` is given (a carried-forward, non-amnesiac state -- not
    the fresh-chain zero start), EC's hidden state is held fixed for the
    whole delay period instead of being updated every step -- see
    `feedforward_delay._run_trial_chained` for the full rationale (delay
    transitions are context-blind, so updating on them just relaxes the
    state back toward a shared fixed point, erasing the seed before movement
    even starts -- confirmed empirically to decay by ~3 orders of magnitude
    over one delay period).
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    pos = pos0.copy()
    vel = np.zeros(2, dtype=np.float32)
    prev_action = np.zeros(2, dtype=np.float32)
    h_ec = h_ec_in.clone() if h_ec_in is not None else torch.zeros(1, 1, sweep.EC_HIDDEN, device=device)
    h_actor = actor.init_hidden(batch_size=1, device=device)
    freeze_during_delay = h_ec_in is not None
    h_ec_trace = []

    episode_budget = delay_len + max_steps
    for t in range(episode_budget):
        in_delay = t < delay_len
        go_cue = 0.0 if in_delay else 1.0

        d = float(np.linalg.norm(pos - goal0))
        if not in_delay and d < sac_base.GOAL_RADIUS:
            break

        base_obs = delay_task.build_obs_delay(pos, goal0, prev_action, go_cue)
        z_ec, z_cb = sweep._current_features(ec, cb, cb_proj, h_ec, pos, prev_action, device, use_ec, use_cb)
        obs_np = sweep._build_aug_obs(base_obs, z_ec, z_cb, use_ec, use_cb, use_oracle, ctx_one_hot)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            _, _, mean_action_t, h_actor_new = actor.get_action(obs_t, h_actor)
        action_np = mean_action_t.squeeze(0).cpu().numpy().astype(np.float32)
        h_actor = h_actor_new

        h_ec_trace.append(h_ec.squeeze(0).squeeze(0).clone())  # state as used at this step, pre-update

        if in_delay:
            next_pos, next_vel = pos, vel
        else:
            next_pos, next_vel = sac_base.env_step_np(pos, vel, action_np, beta, theta, gamma_curl, g_dyn, bx, by)
        if not (in_delay and freeze_during_delay):
            h_ec, _ = sweep._update_ec_hidden(ec, cb, h_ec, pos, action_np, next_pos, device, use_ec)
        pos, vel, prev_action = next_pos, next_vel, action_np

    return h_ec, torch.stack(h_ec_trace, dim=0)


def compute_ec_context_prior_rnn(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    ctx_idx: int,
    delay_len: int = EVAL_DELAY_LEN,
    skip_steps: int = EC_PRIOR_SKIP_STEPS,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Recurrent-actor counterpart to
    `feedforward_delay.compute_ec_context_prior` -- see that docstring for
    the full rationale: the mean of a Gaussian fit over the natural
    distribution of EC hidden states this context produces during
    independent, zero-init cold rollouts, pooling every trial's
    movement-epoch states (success or timeout alike -- filtering out
    timeouts was tried and dropped, since it shrinks the sample pool
    hardest on exactly the weak phase/context combinations that most need
    a stable prior) after discarding each trial's first `skip_steps`
    movement steps (adaptively reduced so a trial that ends in only a few
    ballistic steps still contributes rather than being dropped entirely).
    Returns `None` if this config doesn't use EC, or in the degenerate case
    where no trial has any movement steps at all.
    """
    if not bundle["use_ec"]:
        return None
    n_trials = init_pos.shape[1]
    samples = []
    for j in range(n_trials):
        _, h_traj = _run_trial_chained_rnn(
            bundle, init_pos[ctx_idx, j], goal[ctx_idx, j], ctx_idx, None, delay_len=delay_len, device=device
        )
        move_len = h_traj.shape[0] - delay_len
        skip = min(skip_steps, max(move_len - 1, 0))
        move = h_traj[delay_len + skip:]
        if move.shape[0] == 0:
            continue
        samples.append(move.numpy())
    if not samples:
        print(f"compute_ec_context_prior_rnn: no movement-epoch steps for ctx_idx={ctx_idx}, "
              f"config={bundle.get('config')!r} phase={bundle.get('phase_label')!r} -- no prior fit, warm-start skipped.")
        return None
    mu = np.concatenate(samples, axis=0).mean(axis=0)
    return torch.tensor(mu, dtype=torch.float32, device=device).view(1, 1, -1)


def compute_ec_context_priors_rnn(bundle: dict, init_pos: np.ndarray, goal: np.ndarray, **kwargs) -> dict:
    """`compute_ec_context_prior_rnn` for every context. Values are `None`
    for every context if this config doesn't use EC.
    """
    n_ctx = init_pos.shape[0]
    return {
        ctx_idx: compute_ec_context_prior_rnn(bundle, init_pos, goal, ctx_idx, **kwargs) for ctx_idx in range(n_ctx)
    }


def rollout_actor_delay_rnn(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    ctx_idx: int,
    delay_len: int = EVAL_DELAY_LEN,
    device: str = "cpu",
    max_steps: int = sac_base.MAX_STEPS,
    deterministic: bool = True,
    h_ec_init: Optional[torch.Tensor] = None,
) -> dict:
    """Combines `recurrent_base.rollout_actor_rnn` (hidden state read
    directly from `get_action`, no hook) with
    `feedforward_delay.rollout_actor_delay`'s task mechanics (fixed
    delay hold, go-cue, epoch tagging, diagnostic premature-movement flag).

    `h_ec_init`, if given, seeds every trial's EC hidden state instead of the
    zero-init default -- see `compute_ec_context_prior_rnn`. Whenever it's
    given, EC's hidden state is also held fixed for the whole delay period
    instead of updating every step -- see `feedforward_delay.rollout_actor_delay`
    for the full rationale.
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    n_trials = init_pos.shape[0]
    hidden_dim = actor.hidden_dim
    episode_budget = delay_len + max_steps

    hidden_hist = np.full((n_trials, episode_budget, hidden_dim), np.nan, dtype=np.float32)
    pos_hist = np.full((n_trials, episode_budget, 2), np.nan, dtype=np.float32)
    action_hist = np.full((n_trials, episode_budget, 2), np.nan, dtype=np.float32)
    dist_hist = np.full((n_trials, episode_budget), np.nan, dtype=np.float32)
    epoch_hist = np.full((n_trials, episode_budget), np.nan, dtype=np.float32)
    premature_hist = np.zeros((n_trials, episode_budget), dtype=bool)
    success = np.zeros(n_trials, dtype=bool)
    steps = np.zeros(n_trials, dtype=np.int64)

    for i in range(n_trials):
        pos = init_pos[i].copy()
        vel = np.zeros(2, dtype=np.float32)
        g = goal[i]
        prev_action = np.zeros(2, dtype=np.float32)
        if use_ec:
            h_ec = h_ec_init.clone() if h_ec_init is not None else torch.zeros(1, 1, sweep.EC_HIDDEN, device=device)
        else:
            h_ec = None
        freeze_ec_during_delay = use_ec and (h_ec_init is not None)
        h_actor = actor.init_hidden(batch_size=1, device=device)

        for t in range(episode_budget):
            in_delay = t < delay_len
            go_cue = 0.0 if in_delay else 1.0

            d = float(np.linalg.norm(pos - g))
            dist_hist[i, t] = d
            if not in_delay and d < sac_base.GOAL_RADIUS:
                success[i] = True
                steps[i] = t
                break

            base_obs = delay_task.build_obs_delay(pos, g, prev_action, go_cue)
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
            epoch_hist[i, t] = go_cue
            steps[i] = t + 1
            h_actor = h_actor_new

            if in_delay:
                premature_hist[i, t] = np.linalg.norm(action_np) > delay_task.PREMATURE_ACTION_THRESHOLD
                next_pos, next_vel = pos, vel
            else:
                next_pos, next_vel = sac_base.env_step_np(
                    pos, vel, action_np, beta, theta, gamma_curl, g_dyn, bx, by
                )
            if use_ec and not (in_delay and freeze_ec_during_delay):
                h_ec, _ = sweep._update_ec_hidden(ec, cb, h_ec, pos, action_np, next_pos, device, use_ec)
            pos, vel, prev_action = next_pos, next_vel, action_np

    return {
        "hidden": hidden_hist,
        "pos": pos_hist,
        "action": action_hist,
        "dist": dist_hist,
        "epoch": epoch_hist,
        "premature": premature_hist,
        "success": success,
        "steps": steps,
        "ctx_idx": ctx_idx,
        "ctx_label": ctx_label,
    }


def evaluate_checkpoint_delay_rnn(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    device: str = "cpu",
    warm_start_ec: bool = False,
    ec_prior_kwargs: Optional[dict] = None,
    **rollout_kwargs,
) -> dict:
    """Recurrent-actor delay-task counterpart to
    `feedforward_base.evaluate_checkpoint`.

    If `warm_start_ec` (and `bundle["use_ec"]`), each context's trials are
    seeded with that context's warm EC prior (`compute_ec_context_priors_rnn`)
    instead of the amnesiac zero-init default.
    """
    n_ctx = init_pos.shape[0]
    h_ec_priors = (
        compute_ec_context_priors_rnn(bundle, init_pos, goal, device=device, **(ec_prior_kwargs or {}))
        if (warm_start_ec and bundle["use_ec"])
        else {}
    )
    return {
        ctx_idx: rollout_actor_delay_rnn(
            bundle, init_pos[ctx_idx], goal[ctx_idx], ctx_idx, device=device,
            h_ec_init=h_ec_priors.get(ctx_idx), **rollout_kwargs,
        )
        for ctx_idx in range(n_ctx)
    }


def aggregate_hidden_activity_delay_rnn(
    df_ckpts: pd.DataFrame,
    init_pos: np.ndarray,
    goal: np.ndarray,
    out_dir: Path = HIDDEN_ACTIVITY_DIR_DELAY_RNN,
    device: str = "cpu",
    skip_existing: bool = True,
    **rollout_kwargs,
) -> pd.DataFrame:
    """Recurrent-actor delay-task counterpart to
    `feedforward_base.aggregate_hidden_activity`.

    Pass `warm_start_ec=True` (forwarded via `rollout_kwargs` to
    `evaluate_checkpoint_delay_rnn`) to seed EC from the per-context warm
    prior instead of a zero-init; when doing so, pass a distinct `out_dir`
    (e.g. `HIDDEN_ACTIVITY_DIR_DELAY_RNN_ECWARM`) so the differently-computed
    cache doesn't collide with cold-start results.
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
            bundle = load_actor_for_row_delay_rnn(row, device=device)
            results = evaluate_checkpoint_delay_rnn(bundle, init_pos, goal, device=device, **rollout_kwargs)
            save_data = {
                f"ctx{ctx_idx}_{key}": r[key]
                for ctx_idx, r in results.items()
                for key in ("hidden", "pos", "action", "dist", "epoch", "premature", "success", "steps")
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


def load_cached_result_delay_rnn(index_row: pd.Series) -> dict:
    """Load one context's full arrays back out of an
    `aggregate_hidden_activity_delay_rnn()` index row's cache file, including
    the delay-task-specific `epoch` and `premature` arrays.
    """
    ctx_idx = index_row["ctx_idx"]
    with np.load(index_row["path"]) as data:
        return {
            key: data[f"ctx{ctx_idx}_{key}"]
            for key in ("hidden", "pos", "action", "dist", "epoch", "premature", "success", "steps")
        }


# Delay-epoch-restricted comparison analyses (see
# `feedforward_delay`'s equivalents for the non-recurrent actor).
def _delay_epoch_hidden_rnn(
    row: pd.Series, remove_condition_invariant: bool = True, cis: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pooled MC hidden-activity vectors from just the delay (preparatory)
    epoch of one (checkpoint, context) row (`epoch == 0.0`), across trials.

    If `remove_condition_invariant` (default), subtracts a condition-invariant
    signal (CIS) -- the trial-averaged activity at each delay timestep, i.e.
    whatever's common across every target/condition -- so what remains is
    purely the condition-DEPENDENT (target-specific) component. This is the
    standard preprocessing before comparing preparatory subspaces across
    contexts/conditions (cf. Kaufman et al. 2014's condition-invariant
    signal). Every trial runs the exact same fixed `delay_len`, so timesteps
    are directly comparable across trials -- no time-warping needed.

    If `cis` is given (a `(delay_len, hidden_dim)` array), it's used directly
    instead of being computed from this row's own trials -- e.g. a CIS
    pooled across *all* contexts (see `_compute_global_cis_rnn`), for
    `cis_scope='global'`. When `cis=None` (default), the CIS is computed from
    this row alone (`cis_scope='context'`).
    """
    r = load_cached_result_delay_rnn(row)
    hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]

    if remove_condition_invariant:
        delay_len = int(np.sum(epoch[0] == 0.0))
        if cis is None:
            cis = hidden[:, :delay_len].mean(axis=0)  # (delay_len, hidden_dim)
        hidden = hidden.copy()
        hidden[:, :delay_len] -= cis[np.newaxis]

    chunks = []
    for i, s in enumerate(steps):
        trial_hidden = hidden[i, :s]
        mask = epoch[i, :s] == 0.0
        chunks.append(trial_hidden[mask])
    return np.concatenate(chunks, axis=0)


def _compute_global_cis_rnn(group: pd.DataFrame) -> np.ndarray:
    """Condition-invariant signal pooled across every context in `group`
    (not just one), for `cis_scope='global'`. Returns `(delay_len, hidden_dim)`.
    """
    all_delay_hidden = []
    for _, row in group.iterrows():
        r = load_cached_result_delay_rnn(row)
        hidden, epoch = r["hidden"], r["epoch"]
        delay_len = int(np.sum(epoch[0] == 0.0))
        all_delay_hidden.append(hidden[:, :delay_len])
    return np.concatenate(all_delay_hidden, axis=0).mean(axis=0)


def compute_delay_epoch_participation_ratio_rnn(
    index_df: pd.DataFrame, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `feedforward_delay.compute_delay_epoch_participation_ratio`,
    for recurrent-actor checkpoints. See that function for what `cis_scope` does.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        global_cis = (
            _compute_global_cis_rnn(group) if (remove_condition_invariant and cis_scope == "global") else None
        )
        pooled = np.concatenate(
            [
                _delay_epoch_hidden_rnn(row, remove_condition_invariant, cis=global_cis)
                for _, row in group.iterrows()
            ],
            axis=0,
        )
        records.append({
            "config": config,
            "seed": seed,
            "phase_idx": phase_idx,
            "phase_label": phase_label,
            "n_samples": pooled.shape[0],
            "participation_ratio": participation_ratio(pooled),
        })

    return pd.DataFrame(records).sort_values(["config", "seed", "phase_idx"]).reset_index(drop=True)


def context_alignment_matrix_delay_epoch_rnn(
    group: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> np.ndarray:
    """Same as `feedforward_delay.context_alignment_matrix_delay_epoch`,
    for recurrent-actor checkpoints.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    group = group.sort_values("ctx_idx")
    global_cis = _compute_global_cis_rnn(group) if (remove_condition_invariant and cis_scope == "global") else None
    covs, Qs = [], []
    for _, row in group.iterrows():
        X = _delay_epoch_hidden_rnn(row, remove_condition_invariant, cis=global_cis)
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


def compute_delay_epoch_alignment_rnn(
    index_df: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `feedforward_delay.compute_delay_epoch_alignment`, for
    recurrent-actor checkpoints.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix_delay_epoch_rnn(
            group, k=k, remove_condition_invariant=remove_condition_invariant, cis_scope=cis_scope
        )
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


def plot_delay_epoch_geometry_rnn(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
    remove_global_cis: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Same as `feedforward_delay.plot_delay_epoch_geometry`, for
    recurrent-actor checkpoints. Same `remove_global_cis` option and rationale
    as the non-recurrent version -- raw activity by default (so cross-context
    baseline shifts show as translations), optionally with the globally
    (not per-context) pooled condition-invariant signal subtracted first.
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    global_cis = _compute_global_cis_rnn(group) if remove_global_cis else None

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result_delay_rnn(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        if remove_global_cis:
            delay_len = global_cis.shape[0]
            hidden = hidden.copy()
            hidden[:, :delay_len] -= global_cis[np.newaxis]
        trajs = [hidden[i, :s][epoch[i, :s] == 0.0] for i, s in enumerate(steps)]
        per_ctx_trajs[row["ctx_idx"]] = {"trajs": trajs, "angle": angle[row["ctx_idx"]]}

    title = (
        f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']}) -- "
        "recurrent actor, delay epoch only"
    )
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components, out_path_panels=out_panels, out_path_combined=out_combined,
        mark_end=True, end_label="final (pre-go)",
    )


# Execution-epoch-restricted comparison analyses (see `feedforward_delay`'s
# equivalents for the non-recurrent actor -- same "common execution length"
# rationale for CIS removal applies here unchanged).
def _execution_epoch_hidden_rnn(
    row: pd.Series, remove_condition_invariant: bool = True, cis: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pooled MC hidden-activity vectors from just the execution (movement)
    epoch of one (checkpoint, context) row (`epoch == 1.0`), across trials --
    the execution-epoch counterpart to `_delay_epoch_hidden_rnn`.
    """
    r = load_cached_result_delay_rnn(row)
    hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
    delay_len = int(np.sum(epoch[0] == 0.0))

    if remove_condition_invariant:
        if cis is not None:
            common_exec_len = cis.shape[0]
        else:
            common_exec_len = int((steps - delay_len).min())
            if common_exec_len > 0:
                cis = hidden[:, delay_len:delay_len + common_exec_len].mean(axis=0)
        if common_exec_len > 0:
            hidden = hidden.copy()
            hidden[:, delay_len:delay_len + common_exec_len] -= cis[np.newaxis]

    chunks = []
    for i, s in enumerate(steps):
        trial_hidden = hidden[i, :s]
        mask = epoch[i, :s] == 1.0
        chunks.append(trial_hidden[mask])
    return np.concatenate(chunks, axis=0)


def _compute_global_cis_execution_rnn(group: pd.DataFrame) -> np.ndarray:
    """Condition-invariant signal for the execution epoch, pooled across
    every context in `group` (not just one), for `cis_scope='global'` -- the
    execution-epoch counterpart to `_compute_global_cis_rnn`.
    """
    cached = []
    for _, row in group.iterrows():
        r = load_cached_result_delay_rnn(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        delay_len = int(np.sum(epoch[0] == 0.0))
        cached.append((hidden, delay_len, int((steps - delay_len).min())))
    common_exec_len = min(c[2] for c in cached)
    all_exec_hidden = [h[:, dl:dl + common_exec_len] for h, dl, _ in cached]
    return np.concatenate(all_exec_hidden, axis=0).mean(axis=0)


def compute_execution_epoch_participation_ratio_rnn(
    index_df: pd.DataFrame, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `compute_delay_epoch_participation_ratio_rnn`, but restricted
    to each trial's execution (movement) epoch only.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        global_cis = (
            _compute_global_cis_execution_rnn(group)
            if (remove_condition_invariant and cis_scope == "global")
            else None
        )
        pooled = np.concatenate(
            [
                _execution_epoch_hidden_rnn(row, remove_condition_invariant, cis=global_cis)
                for _, row in group.iterrows()
            ],
            axis=0,
        )
        records.append({
            "config": config,
            "seed": seed,
            "phase_idx": phase_idx,
            "phase_label": phase_label,
            "n_samples": pooled.shape[0],
            "participation_ratio": participation_ratio(pooled),
        })

    return pd.DataFrame(records).sort_values(["config", "seed", "phase_idx"]).reset_index(drop=True)


def context_alignment_matrix_execution_epoch_rnn(
    group: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> np.ndarray:
    """Same as `context_alignment_matrix_delay_epoch_rnn`, but each context's
    subspace is fit on its execution (movement) epoch activity only.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    group = group.sort_values("ctx_idx")
    global_cis = (
        _compute_global_cis_execution_rnn(group) if (remove_condition_invariant and cis_scope == "global") else None
    )
    covs, Qs = [], []
    for _, row in group.iterrows():
        X = _execution_epoch_hidden_rnn(row, remove_condition_invariant, cis=global_cis)
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


def compute_execution_epoch_alignment_rnn(
    index_df: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `compute_delay_epoch_alignment_rnn`, but restricted to each
    trial's execution (movement) epoch only.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix_execution_epoch_rnn(
            group, k=k, remove_condition_invariant=remove_condition_invariant, cis_scope=cis_scope
        )
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


def plot_execution_epoch_geometry_rnn(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
    remove_global_cis: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Same as `plot_delay_epoch_geometry_rnn`, but restricted to each
    trial's execution (movement) epoch: PCA is fit on pooled execution-epoch
    activity across contexts (its own fit, independent of the delay-epoch
    one), and only the execution-epoch portion of each trajectory is drawn.
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    global_cis = _compute_global_cis_execution_rnn(group) if remove_global_cis else None

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result_delay_rnn(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        if remove_global_cis:
            delay_len = int(np.sum(epoch[0] == 0.0))
            common_exec_len = global_cis.shape[0]
            hidden = hidden.copy()
            hidden[:, delay_len:delay_len + common_exec_len] -= global_cis[np.newaxis]
        trajs = [hidden[i, :s][epoch[i, :s] == 1.0] for i, s in enumerate(steps)]
        per_ctx_trajs[row["ctx_idx"]] = {"trajs": trajs, "angle": angle[row["ctx_idx"]]}

    title = (
        f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']}) -- "
        "recurrent actor, execution epoch only"
    )
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components, out_path_panels=out_panels, out_path_combined=out_combined,
        mark_end=True, end_label="final (movement)",
    )


if __name__ == "__main__":
    set_plt_params(style='fast')

    ### Load checkpoints and the shared evaluation dataset (initial positions, goals, target angles/radii) ###
    df_ckpts = discover_checkpoints(root=CHECKPOINT_ROOT_DELAY_RNN)
    print(f"Found {len(df_ckpts)} checkpoints across {df_ckpts['config'].nunique()} configs and {df_ckpts['seed'].nunique()} seeds.")
    
    init_pos, goal, radius, angle = load_or_create_eval_dataset()

    ### Get delay period activity over the full sweep of configs for the evaluation dataset

    # `full` and `full_end_to_end` are the only configs that use EC, and EC
    # has no way to represent context before movement starts unless it's
    # warm-started (see `compute_ec_context_prior_rnn`) -- their cold-start
    # delay-epoch activity is provably context-invariant (bit-identical
    # across contexts), so there's no informative cold view of them to plot.
    # Only their warm-started activity is ever computed; every other config
    # (nothing to warm-start) uses its ordinary cold rollout. Both are
    # combined into one `df_index` so every config lands in the same plots
    # below instead of a separate parallel set of "(EC warm-started)" figures.
    ec_configs = [name for name, preset in CONFIG_PRESETS_BY_NAME.items() if preset["use_ec"]]
    df_ckpts_cold = df_ckpts[~df_ckpts["config"].isin(ec_configs)]
    df_ckpts_ec = df_ckpts[df_ckpts["config"].isin(ec_configs)]

    df_index_cold = aggregate_hidden_activity_delay_rnn(df_ckpts_cold, init_pos, goal, device="cpu", skip_existing=True)
    df_index_ecwarm = aggregate_hidden_activity_delay_rnn(
        df_ckpts_ec, init_pos, goal, out_dir=HIDDEN_ACTIVITY_DIR_DELAY_RNN_ECWARM,
        device="cpu", skip_existing=True, warm_start_ec=True,
    )
    df_index = pd.concat([df_index_cold, df_index_ecwarm], ignore_index=True)
    print(f"Aggregated hidden activity for {len(df_index)} (config, seed, phase, context) rows "
          f"({len(df_index_cold)} cold + {len(df_index_ecwarm)} EC warm-started).")

    pr_df = compute_delay_epoch_participation_ratio_rnn(df_index)
    ai_df = compute_delay_epoch_alignment_rnn(df_index, k=ALIGNMENT_K)

    print(f"Computed delay-epoch participation ratios for {len(pr_df)} checkpoints.")

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=pr_df, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    plt.title("Delay-Epoch Participation Ratio Over Learning Phases (recurrent actor)")
    plt.xlabel("Block Index")
    plt.ylabel("Participation Ratio")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=ai_df, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    plt.title("Delay-Epoch Alignment Over Learning Phases (recurrent actor)")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Off-Diagonal Alignment")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    # MC delay-epoch activity geometry in PC space, colored by target angle --
    # one figure per config (full/full_end_to_end reflect warm-started
    # activity, every other config its ordinary cold rollout -- see above),
    # for a single seed at the final training block.
    geometry_seed = 0
    last_phase = sweep.N_PHASES - 1
    for config in df_index["config"].unique():
        plot_delay_epoch_geometry_rnn(
            df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
            out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_delay_rnn_geometry",
        )

    # Execution-epoch (movement-period) counterpart, so preparatory and
    # execution trajectories/subspaces can both be examined side by side.
    # pr_df_exec = compute_execution_epoch_participation_ratio_rnn(df_index)
    # ai_df_exec = compute_execution_epoch_alignment_rnn(df_index, k=ALIGNMENT_K)

    # plt.figure(figsize=(6, 4))
    # sns.lineplot(data=pr_df_exec, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    # plt.title("Execution-Epoch Participation Ratio Over Learning Phases (recurrent actor)")
    # plt.xlabel("Block Index")
    # plt.ylabel("Participation Ratio")
    # plt.xlim(0, sweep.N_PHASES - 1)
    # make_axis_nice(plt.gca())

    # plt.figure(figsize=(6, 4))
    # sns.lineplot(data=ai_df_exec, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    # plt.title("Execution-Epoch Alignment Over Learning Phases (recurrent actor)")
    # plt.xlabel("Block Index")
    # plt.ylabel("Mean Off-Diagonal Alignment")
    # plt.xlim(0, sweep.N_PHASES - 1)
    # make_axis_nice(plt.gca())

    # for config in df_index["config"].unique():
    #     plot_execution_epoch_geometry_rnn(
    #         df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
    #         out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_execution_rnn_geometry",
    #     )

    plt.pause(1)
    pdb.set_trace()