"""
Analyze dynamics of the motor cortex module during learning, for the
instructed-delay task variant (see `sweep_ablations_delay.py`).

Reuses everything generic from `feedforward_base.py` -- the shared
center-out eval dataset, checkpoint discovery, and the low-level
participation-ratio / subspace-alignment primitives -- and defines only what's
specific to this task variant: checkpoints have one extra go-cue observation
dim, and every trial holds at center for a fixed delay before movement is
permitted, so hidden activity gets tagged by epoch (delay vs. movement).
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

CHECKPOINT_ROOT_DELAY = Path("figs/sweeps/feedforward/delay")
EVAL_DELAY_LEN = 20
HIDDEN_ACTIVITY_DIR_DELAY = base_path / "sweeps" / "feedforward" / "delay" / "hidden_activity"
# Separate cache dir for `warm_start_ec=True` runs (see `compute_ec_context_prior`) --
# the cached arrays mean something different (EC seeded from a per-context prior,
# not zero-init), so they must not share a cache path with the cold-start runs.
HIDDEN_ACTIVITY_DIR_DELAY_ECWARM = base_path / "sweeps" / "feedforward" / "delay" / "hidden_activity_ecwarm"


def load_actor_for_row_delay(row: pd.Series, device: str = "cpu") -> dict:
    """Same as `feedforward_base.load_actor_for_row`, but for
    instructed-delay checkpoints: the actor is reconstructed with the extra
    go-cue observation dim (`delay_task.compute_aug_obs_dim_delay`).
    Everything else -- EC/CB reconstruction, checkpoint format -- is
    identical.
    """
    preset = CONFIG_PRESETS_BY_NAME[row["config"]]
    use_ec, use_cb, use_oracle = preset["use_ec"], preset["use_cb"], preset["use_oracle"]
    aug_obs_dim = delay_task.compute_aug_obs_dim_delay(use_ec, use_cb, use_oracle)

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


def _run_trial_chained(
    bundle: dict,
    pos0: np.ndarray,
    goal0: np.ndarray,
    ctx_idx: int,
    h_ec_in: Optional[torch.Tensor],
    delay_len: int = EVAL_DELAY_LEN,
    device: str = "cpu",
    max_steps: int = sac_base.MAX_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one full instructed-delay trial in context `ctx_idx`, with EC's
    hidden state starting from `h_ec_in` instead of zero. Returns
    `(h_ec_final, h_ec_trajectory)`: the EC hidden state at trial end -- used
    by `compute_ec_context_prior` to chain several trials back-to-back,
    carrying EC's memory forward between them the way it would behave
    sustained inside one context block, rather than the amnesiac zero reset
    every isolated eval trial otherwise gets -- and the full per-timestep
    trajectory of EC hidden states across this trial, shape `(T, hidden_dim)`
    (used to build the context prior from EC's whole state trajectory rather
    than just its end-of-trial snapshot -- see `compute_ec_context_prior`).

    Whenever `h_ec_in` is given (a carried-forward, non-amnesiac state -- not
    the fresh-chain zero start), EC's hidden state is held fixed for the
    whole delay period instead of being updated every step. Delay-period
    transitions are degenerate (position frozen, so the same context-blind
    "nothing happened" signal regardless of context) and don't carry any
    information EC should actually update on -- letting the GRU update on
    them anyway just relaxes the state back toward that shared degenerate
    fixed point, quietly erasing whatever context information the incoming
    state carried before movement even starts (confirmed empirically: the
    difference between a warm- and zero-seeded trial decays by ~3 orders of
    magnitude over one 20-step delay). EC only resumes updating once the
    go-cue permits movement and real, context-dependent feedback arrives.
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    pos = pos0.copy()
    vel = np.zeros(2, dtype=np.float32)
    prev_action = np.zeros(2, dtype=np.float32)
    h_ec = h_ec_in.clone() if h_ec_in is not None else torch.zeros(1, 1, sweep.EC_HIDDEN, device=device)
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
            _, _, mean_action_t = actor.get_action(obs_t)
        action_np = mean_action_t.squeeze(0).cpu().numpy().astype(np.float32)

        h_ec_trace.append(h_ec.squeeze(0).squeeze(0).clone())  # state as used at this step, pre-update

        if in_delay:
            next_pos, next_vel = pos, vel
        else:
            next_pos, next_vel = sac_base.env_step_np(pos, vel, action_np, beta, theta, gamma_curl, g_dyn, bx, by)
        if not (in_delay and freeze_during_delay):
            h_ec, _ = sweep._update_ec_hidden(ec, cb, h_ec, pos, action_np, next_pos, device, use_ec)
        pos, vel, prev_action = next_pos, next_vel, action_np

    return h_ec, torch.stack(h_ec_trace, dim=0)


EC_PRIOR_SKIP_STEPS = 10  # see `compute_ec_context_prior`


def compute_ec_context_prior(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    ctx_idx: int,
    delay_len: int = EVAL_DELAY_LEN,
    skip_steps: int = EC_PRIOR_SKIP_STEPS,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Per-context 'warm' EC hidden state: the mean of a Gaussian fit over
    the natural distribution of EC hidden states this context actually
    produces, used as the warm-start prior instead of a single ad-hoc
    trajectory snapshot/average.

    Built from `init_pos.shape[1]` independent, zero-init (cold, not
    chained) rollouts in context `ctx_idx`, pooling every trial's
    movement-epoch states (success or timeout alike -- an earlier version
    also filtered out timed-out trials, on the reasoning that they spend
    most of their length wandering/failing rather than executing confident,
    context-appropriate behavior; dropped since it made the fit depend on
    an unrelated axis, how good the checkpoint currently is at the task,
    which shrinks the sample pool hardest exactly on the weak
    phase/context combinations that most need a stable prior). Within
    each trial, the first `skip_steps` movement steps are discarded --
    early movement is still context-ambiguous (curl-type contexts only
    diverge from each other via `gamma_curl * (curl @ vel)`, which is
    exactly zero until velocity has built up over several steps), so only
    the settled tail is kept. This is adaptive per trial
    (`min(skip_steps, movement_len - 1)`): a trial that ends in only a few
    ballistic steps still contributes its last step or two rather than
    being dropped entirely, which a fixed skip would do to every
    fast/well-learned context (measured directly: a fixed 10-step skip
    left zero usable samples for several phase/context combinations with
    >90% success, because the model had learned to solve them in well
    under 10 steps).

    Returns `None` if this config doesn't use EC (`bundle["use_ec"]`
    False), or in the degenerate case where no trial has any movement
    steps at all.

    This replaced an earlier trajectory-tail-averaging heuristic (chain N
    trials back-to-back, average the last few trials' last few states) after
    a systemic check across `full`'s 12 phases (seed 0) found this posterior
    mean roughly halves both the mean and worst-case success-rate drop from
    warm-starting (mean delta -0.059 -> -0.005 to -0.026 depending on the
    exact filter tried, worst -0.556 -> -0.17 to -0.31, and eliminates every
    >30-percentage-point catastrophic drop the old heuristic produced).
    Sampling from the fitted covariance (instead of just using the mean) was
    also tested as a way to get naturalistic per-context noise, but performed
    *worse* than the deterministic mean on average -- a sample can be
    statistically unremarkable (not an outlier by the natural distribution's
    own spread) and still land on a `z_ec = ec.readout(h_ec)` the actor
    responds to badly, since the actor is only sensitive to certain
    directions of `h_ec` and isotropic Gaussian noise doesn't respect that.
    So only the mean is used here; the covariance isn't currently kept.
    """
    if not bundle["use_ec"]:
        return None
    n_trials = init_pos.shape[1]
    samples = []
    for j in range(n_trials):
        _, h_traj = _run_trial_chained(
            bundle, init_pos[ctx_idx, j], goal[ctx_idx, j], ctx_idx, None, delay_len=delay_len, device=device
        )
        move_len = h_traj.shape[0] - delay_len
        skip = min(skip_steps, max(move_len - 1, 0))
        move = h_traj[delay_len + skip:]
        if move.shape[0] == 0:
            continue
        samples.append(move.numpy())
    if not samples:
        print(f"compute_ec_context_prior: no movement-epoch steps for ctx_idx={ctx_idx}, "
              f"config={bundle.get('config')!r} phase={bundle.get('phase_label')!r} -- no prior fit, warm-start skipped.")
        return None
    mu = np.concatenate(samples, axis=0).mean(axis=0)
    return torch.tensor(mu, dtype=torch.float32, device=device).view(1, 1, -1)


def compute_ec_context_priors(bundle: dict, init_pos: np.ndarray, goal: np.ndarray, **kwargs) -> dict:
    """`compute_ec_context_prior` for every context. Values are `None` for
    every context if this config doesn't use EC.
    """
    n_ctx = init_pos.shape[0]
    return {ctx_idx: compute_ec_context_prior(bundle, init_pos, goal, ctx_idx, **kwargs) for ctx_idx in range(n_ctx)}


def rollout_actor_delay(
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
    """Like `feedforward_base.rollout_actor`, but for instructed-delay
    checkpoints: every trial holds at `init_pos` for a fixed `delay_len` steps
    (go-cue=0, action has no effect on the environment) before the go-cue
    permits movement (go-cue=1) for up to `max_steps` more. Records the same
    per-step arrays as `rollout_actor`, plus `epoch` (0.0 = delay,
    1.0 = movement) so downstream analyses can separate preparatory from
    movement-epoch activity, and `premature` (whether the action exceeded the
    training-time premature-movement tolerance during delay) -- purely
    diagnostic here, eval doesn't intervene on it the way training's soft
    penalty does.

    `h_ec_init`, if given, seeds every trial's EC hidden state instead of the
    zero-init default -- see `compute_ec_context_prior` for why (EC otherwise
    has no way to represent context before movement reveals it). Whenever
    it's given, EC's hidden state is also held fixed for the whole delay
    period instead of updating every step, since delay transitions carry no
    real (context-dependent) information and updating on them just relaxes
    the seeded state back toward a shared, context-blind fixed point --
    quietly erasing the seed before movement even starts. See
    `_run_trial_chained` for the full rationale and the measurement behind it.
    """
    actor, ec, cb, cb_proj = bundle["actor"], bundle["ec"], bundle["cb"], bundle["cb_proj"]
    use_ec, use_cb, use_oracle = bundle["use_ec"], bundle["use_cb"], bundle["use_oracle"]
    beta, theta, gamma_curl, g_dyn, bx, by, ctx_label = sweep.PHASE_DATA[ctx_idx]["params"]
    ctx_one_hot = sweep._context_one_hot(ctx_idx) if use_oracle else None

    n_trials = init_pos.shape[0]
    hidden_dim = sac_base.HIDDEN_DIM
    episode_budget = delay_len + max_steps

    hidden_hist = np.full((n_trials, episode_budget, hidden_dim), np.nan, dtype=np.float32)
    pos_hist = np.full((n_trials, episode_budget, 2), np.nan, dtype=np.float32)
    action_hist = np.full((n_trials, episode_budget, 2), np.nan, dtype=np.float32)
    dist_hist = np.full((n_trials, episode_budget), np.nan, dtype=np.float32)
    epoch_hist = np.full((n_trials, episode_budget), np.nan, dtype=np.float32)
    premature_hist = np.zeros((n_trials, episode_budget), dtype=bool)
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
            if use_ec:
                h_ec = h_ec_init.clone() if h_ec_init is not None else torch.zeros(1, 1, sweep.EC_HIDDEN, device=device)
            else:
                h_ec = None
            freeze_ec_during_delay = use_ec and (h_ec_init is not None)

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
                    y_t, _, mean_action_t = actor.get_action(obs_t)
                action_t = mean_action_t if deterministic else y_t
                action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

                hidden_hist[i, t] = hidden_capture["h"].squeeze(0).cpu().numpy()
                pos_hist[i, t] = pos
                action_hist[i, t] = action_np
                epoch_hist[i, t] = go_cue
                steps[i] = t + 1

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
    finally:
        handle.remove()

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


def evaluate_checkpoint_delay(
    bundle: dict,
    init_pos: np.ndarray,
    goal: np.ndarray,
    device: str = "cpu",
    warm_start_ec: bool = False,
    ec_prior_kwargs: Optional[dict] = None,
    **rollout_kwargs,
) -> dict:
    """Delay-task counterpart to `feedforward_base.evaluate_checkpoint`:
    rolls `bundle` out under every context via `rollout_actor_delay`.

    If `warm_start_ec` (and `bundle["use_ec"]`), each context's trials are
    seeded with that context's warm EC prior (`compute_ec_context_priors`)
    instead of the amnesiac zero-init default.
    """
    n_ctx = init_pos.shape[0]
    h_ec_priors = (
        compute_ec_context_priors(bundle, init_pos, goal, device=device, **(ec_prior_kwargs or {}))
        if (warm_start_ec and bundle["use_ec"])
        else {}
    )
    return {
        ctx_idx: rollout_actor_delay(
            bundle, init_pos[ctx_idx], goal[ctx_idx], ctx_idx, device=device,
            h_ec_init=h_ec_priors.get(ctx_idx), **rollout_kwargs,
        )
        for ctx_idx in range(n_ctx)
    }


def aggregate_hidden_activity_delay(
    df_ckpts: pd.DataFrame,
    init_pos: np.ndarray,
    goal: np.ndarray,
    out_dir: Path = HIDDEN_ACTIVITY_DIR_DELAY,
    device: str = "cpu",
    skip_existing: bool = True,
    **rollout_kwargs,
) -> pd.DataFrame:
    """Delay-task counterpart to `feedforward_base.aggregate_hidden_activity`:
    same resumable, one-file-per-checkpoint caching, but via
    `load_actor_for_row_delay` / `evaluate_checkpoint_delay`, and additionally
    caching `epoch` and `premature` per step.

    Pass `warm_start_ec=True` (forwarded via `rollout_kwargs` to
    `evaluate_checkpoint_delay`) to seed EC from the per-context warm prior
    instead of a zero-init; when doing so, pass a distinct `out_dir` (e.g.
    `HIDDEN_ACTIVITY_DIR_DELAY_ECWARM`) so the differently-computed cache
    doesn't collide with cold-start results.
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
            bundle = load_actor_for_row_delay(row, device=device)
            results = evaluate_checkpoint_delay(bundle, init_pos, goal, device=device, **rollout_kwargs)
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


def load_cached_result_delay(index_row: pd.Series) -> dict:
    """Load one context's full arrays back out of an
    `aggregate_hidden_activity_delay()` index row's cache file, including the
    delay-task-specific `epoch` and `premature` arrays.
    """
    ctx_idx = index_row["ctx_idx"]
    with np.load(index_row["path"]) as data:
        return {
            key: data[f"ctx{ctx_idx}_{key}"]
            for key in ("hidden", "pos", "action", "dist", "epoch", "premature", "success", "steps")
        }


# Delay-epoch-restricted comparison analyses: PR / alignment / geometry pooled
# from just each trial's delay (preparatory) epoch, using the `epoch` tag
# `aggregate_hidden_activity_delay` cached -- rather than the whole trial
# including movement.
def _delay_epoch_hidden(
    row: pd.Series, remove_condition_invariant: bool = True, cis: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pooled MC hidden-activity vectors from just the delay (preparatory)
    epoch of one (checkpoint, context) row (`epoch == 0.0`), across trials.

    If `remove_condition_invariant` (default), subtracts a condition-invariant
    signal (CIS) -- the trial-averaged activity at each delay timestep, i.e.
    whatever's common across every target/condition -- so what remains is
    purely the condition-DEPENDENT (target-specific) component. Standard
    preprocessing before comparing preparatory subspaces across
    contexts/conditions (cf. Kaufman et al. 2014's condition-invariant
    signal). Every trial runs the same fixed `delay_len`, so timesteps are
    directly comparable across trials.

    If `cis` is given (a `(delay_len, hidden_dim)` array), it's used directly
    instead of being computed from this row's own trials -- e.g. a CIS
    pooled across *all* contexts (see `_compute_global_cis`), for
    `cis_scope='global'` in `compute_delay_epoch_participation_ratio` /
    `compute_delay_epoch_alignment`. When `cis=None` (default), the CIS is
    computed from this row alone (`cis_scope='context'`).
    """
    r = load_cached_result_delay(row)
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


def _compute_global_cis(group: pd.DataFrame) -> np.ndarray:
    """Condition-invariant signal pooled across every context in `group`
    (not just one), for `cis_scope='global'`. Returns `(delay_len, hidden_dim)`.
    """
    all_delay_hidden = []
    for _, row in group.iterrows():
        r = load_cached_result_delay(row)
        hidden, epoch = r["hidden"], r["epoch"]
        delay_len = int(np.sum(epoch[0] == 0.0))
        all_delay_hidden.append(hidden[:, :delay_len])
    return np.concatenate(all_delay_hidden, axis=0).mean(axis=0)


def compute_delay_epoch_participation_ratio(
    index_df: pd.DataFrame, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `feedforward_base.compute_cross_context_participation_ratio`,
    but restricted to each trial's delay (preparatory) epoch only.

    `cis_scope` ('context' or 'global') controls how the condition-invariant
    signal is computed when `remove_condition_invariant`: per-context (each
    context's own trials only) or pooled across all contexts. See
    `_delay_epoch_hidden`/`_compute_global_cis` docstrings for the trade-off
    -- 'context' is the default and generally the better choice for this
    per-context-subspace analysis.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        global_cis = _compute_global_cis(group) if (remove_condition_invariant and cis_scope == "global") else None
        pooled = np.concatenate(
            [_delay_epoch_hidden(row, remove_condition_invariant, cis=global_cis) for _, row in group.iterrows()],
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


def context_alignment_matrix_delay_epoch(
    group: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> np.ndarray:
    """Same as `feedforward_base.context_alignment_matrix`, but each
    context's subspace is fit on its delay (preparatory) epoch activity only.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    group = group.sort_values("ctx_idx")
    global_cis = _compute_global_cis(group) if (remove_condition_invariant and cis_scope == "global") else None
    covs, Qs = [], []
    for _, row in group.iterrows():
        X = _delay_epoch_hidden(row, remove_condition_invariant, cis=global_cis)
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


def compute_delay_epoch_alignment(
    index_df: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `feedforward_base.compute_cross_context_alignment`, but
    restricted to each trial's delay (preparatory) epoch only. See
    `compute_delay_epoch_participation_ratio` for what `cis_scope` does.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix_delay_epoch(
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


def plot_delay_epoch_geometry(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
    remove_global_cis: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Same as `feedforward_base.plot_context_geometry`, but restricted to
    each trial's delay (preparatory) epoch: PCA is fit on pooled delay-epoch
    activity across contexts, and only the delay-epoch portion of each
    trajectory is drawn.

    By default (`remove_global_cis=False`) this uses the *raw* activity, so
    cross-context baseline shifts are directly visible as translations
    between context clusters. If `remove_global_cis`, the condition-invariant
    signal pooled across *all* contexts (`_compute_global_cis` -- not each
    context's own mean) is subtracted first: this declutters the PCA of
    whatever's identical everywhere (e.g. a shared settling transient) while
    still leaving each context's own deviation from that shared baseline
    fully intact, so translations remain visible. Per-context CIS removal
    would erase them instead (each context gets recentered on itself), which
    is why that option isn't offered here -- use `remove_condition_invariant`
    on `compute_delay_epoch_participation_ratio`/`compute_delay_epoch_alignment`
    if you want that for the quantitative analysis.
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    global_cis = _compute_global_cis(group) if remove_global_cis else None

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result_delay(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        if remove_global_cis:
            delay_len = global_cis.shape[0]
            hidden = hidden.copy()
            hidden[:, :delay_len] -= global_cis[np.newaxis]
        trajs = [hidden[i, :s][epoch[i, :s] == 0.0] for i, s in enumerate(steps)]
        per_ctx_trajs[row["ctx_idx"]] = {"trajs": trajs, "angle": angle[row["ctx_idx"]]}

    title = f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']}) -- delay epoch only"
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components,
        out_path_panels=out_panels, out_path_combined=out_combined, cmap_name="hsv",
        mark_end=True, end_label="final (pre-go)",
    )


# Execution-epoch-restricted comparison analyses: the movement-period
# counterpart to the delay-epoch analyses above, so preparatory and execution
# trajectories/subspaces can both be examined. Every eval trial's delay
# period is exactly `EVAL_DELAY_LEN` steps (fixed, unlike training's
# randomized hazard-based delay), so movement always starts at the same
# absolute timestep -- but trials end at different times (faster/slower
# successes, or timing out), so the execution epoch itself is variable-length
# per trial, unlike the delay epoch's fixed length. That only matters for
# condition-invariant-signal (CIS) removal, which needs a common time axis to
# average across trials: it's restricted to each trial's first `min`
# execution length across the trials being pooled (the longest window common
# to all of them). Pooling for PR/alignment doesn't need cross-trial
# alignment and still uses each trial's full (ragged) execution-epoch
# activity.
def _execution_epoch_hidden(
    row: pd.Series, remove_condition_invariant: bool = True, cis: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pooled MC hidden-activity vectors from just the execution (movement)
    epoch of one (checkpoint, context) row (`epoch == 1.0`), across trials --
    the execution-epoch counterpart to `_delay_epoch_hidden`.

    If `cis` is given (a `(common_exec_len, hidden_dim)` array), it's used
    directly instead of being computed from this row's own trials -- e.g. a
    CIS pooled across *all* contexts (see `_compute_global_cis_execution`),
    for `cis_scope='global'`. When `cis=None` (default), the CIS -- and the
    common execution length it's computed over -- comes from this row alone
    (`cis_scope='context'`).
    """
    r = load_cached_result_delay(row)
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


def _compute_global_cis_execution(group: pd.DataFrame) -> np.ndarray:
    """Condition-invariant signal for the execution epoch, pooled across
    every context in `group` (not just one), for `cis_scope='global'`.
    Returns `(common_exec_len, hidden_dim)`, where `common_exec_len` is the
    shortest execution length across every trial in every context in `group`
    -- the execution-epoch counterpart to `_compute_global_cis`.
    """
    cached = []
    for _, row in group.iterrows():
        r = load_cached_result_delay(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        delay_len = int(np.sum(epoch[0] == 0.0))
        cached.append((hidden, delay_len, int((steps - delay_len).min())))
    common_exec_len = min(c[2] for c in cached)
    all_exec_hidden = [h[:, dl:dl + common_exec_len] for h, dl, _ in cached]
    return np.concatenate(all_exec_hidden, axis=0).mean(axis=0)


def compute_execution_epoch_participation_ratio(
    index_df: pd.DataFrame, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `compute_delay_epoch_participation_ratio`, but restricted to
    each trial's execution (movement) epoch only.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        global_cis = (
            _compute_global_cis_execution(group) if (remove_condition_invariant and cis_scope == "global") else None
        )
        pooled = np.concatenate(
            [_execution_epoch_hidden(row, remove_condition_invariant, cis=global_cis) for _, row in group.iterrows()],
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


def context_alignment_matrix_execution_epoch(
    group: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> np.ndarray:
    """Same as `context_alignment_matrix_delay_epoch`, but each context's
    subspace is fit on its execution (movement) epoch activity only.
    """
    assert cis_scope in ("context", "global"), f"cis_scope must be 'context' or 'global', got {cis_scope!r}"
    group = group.sort_values("ctx_idx")
    global_cis = (
        _compute_global_cis_execution(group) if (remove_condition_invariant and cis_scope == "global") else None
    )
    covs, Qs = [], []
    for _, row in group.iterrows():
        X = _execution_epoch_hidden(row, remove_condition_invariant, cis=global_cis)
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


def compute_execution_epoch_alignment(
    index_df: pd.DataFrame, k: int = ALIGNMENT_K, remove_condition_invariant: bool = True, cis_scope: str = "context"
) -> pd.DataFrame:
    """Same as `compute_delay_epoch_alignment`, but restricted to each
    trial's execution (movement) epoch only.
    """
    records = []
    group_cols = ["config", "seed", "phase_idx", "phase_label"]
    for (config, seed, phase_idx, phase_label), group in index_df.groupby(group_cols):
        M = context_alignment_matrix_execution_epoch(
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


def plot_execution_epoch_geometry(
    index_df: pd.DataFrame,
    config: str,
    seed: int,
    phase_idx: int,
    angle: np.ndarray,
    n_components: int = 3,
    out_path: Optional[Path] = None,
    remove_global_cis: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
    """Same as `plot_delay_epoch_geometry`, but restricted to each trial's
    execution (movement) epoch: PCA is fit on pooled execution-epoch activity
    across contexts (its own fit, independent of the delay-epoch one), and
    only the execution-epoch portion of each trajectory is drawn.
    """
    group = index_df[
        (index_df["config"] == config) & (index_df["seed"] == seed) & (index_df["phase_idx"] == phase_idx)
    ].sort_values("ctx_idx")
    if group.empty:
        raise ValueError(f"no cached rows for config={config!r}, seed={seed}, phase_idx={phase_idx}")

    global_cis = _compute_global_cis_execution(group) if remove_global_cis else None

    per_ctx_trajs = {}
    for _, row in group.iterrows():
        r = load_cached_result_delay(row)
        hidden, epoch, steps = r["hidden"], r["epoch"], r["steps"]
        if remove_global_cis:
            delay_len = int(np.sum(epoch[0] == 0.0))
            common_exec_len = global_cis.shape[0]
            hidden = hidden.copy()
            hidden[:, delay_len:delay_len + common_exec_len] -= global_cis[np.newaxis]
        trajs = [hidden[i, :s][epoch[i, :s] == 1.0] for i, s in enumerate(steps)]
        per_ctx_trajs[row["ctx_idx"]] = {"trajs": trajs, "angle": angle[row["ctx_idx"]]}

    title = f"{config}, seed {seed}, phase {phase_idx} ({group.iloc[0]['phase_label']}) -- execution epoch only"
    out_panels = out_path.with_name(out_path.name + "_panels.png") if out_path is not None else None
    out_combined = out_path.with_name(out_path.name + "_combined.png") if out_path is not None else None
    return plot_geometry_3d(
        per_ctx_trajs, title, n_components=n_components,
        out_path_panels=out_panels, out_path_combined=out_combined, cmap_name="hsv",
        mark_end=True, end_label="final (movement)",
    )


if __name__ == "__main__":
    set_plt_params(style='fast')

    # Discover all delay-task checkpoints in the ablation sweep output tree.
    df_ckpts = discover_checkpoints(root=CHECKPOINT_ROOT_DELAY)
    print(f"Found {len(df_ckpts)} checkpoints across {df_ckpts['config'].nunique()} configs and {df_ckpts['seed'].nunique()} seeds.")

    # Same shared evaluation dataset as the base pipeline (delay task doesn't
    # change the center-out targets, only what happens before movement).
    init_pos, goal, radius, angle = load_or_create_eval_dataset()

    # `full` and `full_end_to_end` are the only configs that use EC, and EC
    # has no way to represent context before movement starts unless it's
    # warm-started (see `compute_ec_context_prior`) -- their cold-start
    # delay-epoch activity is provably context-invariant (bit-identical
    # across contexts), so there's no informative cold view of them to plot.
    # Only their warm-started activity is ever computed; every other config
    # (nothing to warm-start) uses its ordinary cold rollout. Both are
    # combined into one `df_index` so every config lands in the same plots
    # below instead of a separate parallel set of "(EC warm-started)" figures.
    ec_configs = [name for name, preset in CONFIG_PRESETS_BY_NAME.items() if preset["use_ec"]]
    df_ckpts_cold = df_ckpts[~df_ckpts["config"].isin(ec_configs)]
    df_ckpts_ec = df_ckpts[df_ckpts["config"].isin(ec_configs)]

    df_index_cold = aggregate_hidden_activity_delay(df_ckpts_cold, init_pos, goal, device="cpu", skip_existing=True)
    df_index_ecwarm = aggregate_hidden_activity_delay(
        df_ckpts_ec, init_pos, goal, out_dir=HIDDEN_ACTIVITY_DIR_DELAY_ECWARM,
        device="cpu", skip_existing=True, warm_start_ec=True,
    )
    df_index = pd.concat([df_index_cold, df_index_ecwarm], ignore_index=True)
    print(f"Aggregated hidden activity for {len(df_index)} (config, seed, phase, context) rows "
          f"({len(df_index_cold)} cold + {len(df_index_ecwarm)} EC warm-started).")

    pr_df = compute_delay_epoch_participation_ratio(df_index)
    ai_df = compute_delay_epoch_alignment(df_index, k=ALIGNMENT_K)

    print(f"Computed delay-epoch participation ratios for {len(pr_df)} checkpoints.")

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=pr_df, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    plt.title("Delay-Epoch Participation Ratio Over Learning Phases")
    plt.xlabel("Block Index")
    plt.ylabel("Participation Ratio")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=ai_df, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    plt.title("Delay-Epoch Alignment Over Learning Phases")
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
        plot_delay_epoch_geometry(
            df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
            out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_delay_epoch_geometry",
        )

    # Execution-epoch (movement-period) counterpart, so preparatory and
    # execution trajectories/subspaces can both be examined side by side.
    pr_df_exec = compute_execution_epoch_participation_ratio(df_index)
    ai_df_exec = compute_execution_epoch_alignment(df_index, k=ALIGNMENT_K)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=pr_df_exec, x="phase_idx", y="participation_ratio", hue="config", marker="o")
    plt.title("Execution-Epoch Participation Ratio Over Learning Phases")
    plt.xlabel("Block Index")
    plt.ylabel("Participation Ratio")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=ai_df_exec, x="phase_idx", y="mean_offdiag_alignment", hue="config", marker="o")
    plt.title("Execution-Epoch Alignment Over Learning Phases")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Off-Diagonal Alignment")
    plt.xlim(0, sweep.N_PHASES - 1)
    make_axis_nice(plt.gca())

    for config in df_index["config"].unique():
        plot_execution_epoch_geometry(
            df_index, config=config, seed=geometry_seed, phase_idx=last_phase, angle=angle,
            out_path=PLOT_OUTPUT_DIR / f"{config}_seed{geometry_seed}_phase{last_phase}_execution_epoch_geometry",
        )

    plt.pause(1)
    pdb.set_trace()