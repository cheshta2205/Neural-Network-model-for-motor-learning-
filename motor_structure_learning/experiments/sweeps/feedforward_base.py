"""Feedforward-actor, base-task ablation sweep: the original motor-learning
SAC experiment with EC+CB augmented observations.

This variant extends the base SAC loop by adding:
- EC latent features from a recurrent transition encoder (optional via use_ec)
- CB hidden features conditioned on state, previous action, and EC latent (optional via use_cb)

Both EC and CB can be independently toggled and optionally initialized from a
pretrained checkpoint. Everything generic (EC/CB augmentation, replay buffer,
checkpoint format, CONFIG_PRESETS) lives in `sweeps.common`, imported here as
`sweep`; only the feedforward `Actor` + base-task episode loop are specific to
this file.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import pandas as pd

from motor_structure_learning.core.ec import EntorhinalCortex
from motor_structure_learning.core.trainers import SACTrainer, ECCBTrainer, ECCBBatch

from motor_structure_learning.experiments import _mc_sac_impl as base
from motor_structure_learning.experiments.sweeps import common as sweep

# Original location -- left untouched (git-tracked, with pre-existing
# uncommitted changes unrelated to this reorg) even though this file itself
# moved into sweeps/.
FIG_DIR = "figs/mc_ablations"


def run_babab(
    seed: int = 0,
    use_ec: bool = True,
    use_cb: bool = True,
    use_oracle: bool = False,
    use_pretrained_ec_cb: bool = sweep.USE_PRETRAINED_EC_CB,
    freeze_pretrained_ec_cb: bool = sweep.FREEZE_PRETRAINED_EC_CB,
    ec_cb_lr: float = 1e-4,
    ec_cb_ckpt: str = sweep.EC_CB_CKPT,
    priority_gamma_t: float = sweep.PRIORITY_GAMMA_T,
    priority_gamma_td: float = sweep.PRIORITY_GAMMA_TD,
    priority_mix: float = sweep.PRIORITY_MIX,
    checkpoint_dir: Optional[str] = None,
    config_name: Optional[str] = None,
):
    device = sweep.DEVICE
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Compute dynamic obs dimension
    aug_obs_dim = sweep._compute_aug_obs_dim(use_ec, use_cb, use_oracle)

    # Initialize EC and CB (may be None if not used)
    ec = None
    cb = None
    cb_proj = None

    if use_ec:
        ec = EntorhinalCortex(input_dim=8, hidden_dim=sweep.EC_HIDDEN, latent_dim=sweep.EC_LATENT).to(device)

    if use_cb:
        cb = sweep.CerebellumCB(latent_dim=sweep.EC_LATENT, hidden=sweep.CB_HIDDEN).to(device)
        cb_proj = torch.tensor(
            rng.standard_normal((sweep.CB_HIDDEN, sweep.CB_PROJ_DIM)), dtype=torch.float32, device=device
        )
        cb_proj = cb_proj / (cb_proj.norm(dim=0, keepdim=True) + 1e-8)

    if use_pretrained_ec_cb and (ec is not None or cb is not None):
        sweep._load_ec_cb_if_available(ec, cb, ec_cb_ckpt, device)

    if freeze_pretrained_ec_cb:
        if ec is not None:
            sweep._freeze_module(ec)
        if cb is not None:
            sweep._freeze_module(cb)

    actor = base.Actor(obs_dim=aug_obs_dim, action_dim=base.ACTION_DIM).to(device)
    qf1 = base.SoftQNetwork(obs_dim=aug_obs_dim, action_dim=base.ACTION_DIM).to(device)
    qf2 = base.SoftQNetwork(obs_dim=aug_obs_dim, action_dim=base.ACTION_DIM).to(device)
    qf1_target = base.SoftQNetwork(obs_dim=aug_obs_dim, action_dim=base.ACTION_DIM).to(device)
    qf2_target = base.SoftQNetwork(obs_dim=aug_obs_dim, action_dim=base.ACTION_DIM).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=base.LR_ACTOR)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=base.LR_Q)

    ec_cb_params = []
    if ec is not None:
        ec_cb_params += [p for p in ec.parameters() if p.requires_grad]
    if cb is not None:
        ec_cb_params += [p for p in cb.parameters() if p.requires_grad]

    # Build the trainer wrapper whenever EC/CB exist at all, even if every
    # param is frozen (`ec_cb_params` empty) -- a frozen-but-present EC/CB is
    # still actively used to generate observations throughout training, and
    # `_save_phase_checkpoint` only ever persists EC/CB weights through this
    # trainer's `state_dict()`. Gating construction on "has trainable params"
    # meant frozen pretrained EC/CB (e.g. the `full` config) was silently
    # never saved into any checkpoint at all. Only the optimizer itself
    # (needed only when there's something to train) stays conditional --
    # `ECCBTrainer.update()` already no-ops cleanly when `optimizer=None`.
    if ec is not None or cb is not None:
        ec_cb_optimizer = torch.optim.Adam(ec_cb_params, lr=ec_cb_lr) if ec_cb_params else None
        ec_cb_trainer = ECCBTrainer(
            ec=ec,
            cb=cb,
            cb_proj=cb_proj,
            optimizer=ec_cb_optimizer,
            base_obs_dim=base.OBS_DIM,
            ec_latent_dim=sweep.EC_LATENT,
            cb_proj_dim=sweep.CB_PROJ_DIM,
            ec_hidden_dim=sweep.EC_HIDDEN,
            use_ec=use_ec,
            use_cb=use_cb,
            device=device,
            grad_clip_norm=base.GRAD_CLIP_NORM,
        )
    else:
        ec_cb_trainer = None

    trainer = SACTrainer(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        alpha=0.2,
        gamma=base.GAMMA,
        tau=base.TAU,
        target_entropy=-float(base.ACTION_DIM),
        autotune=True,
        device=device,
        grad_clip_norm=base.GRAD_CLIP_NORM,
    )

    replay_buffer = sweep.TimeTDEPrioritizedReplayBuffer(
        capacity=base.BUFFER_CAPACITY,
        gamma_t=priority_gamma_t,
        gamma_td=priority_gamma_td,
        mix=priority_mix,
        seed=seed,
        ec_hidden_dim=sweep.EC_HIDDEN,
    )

    curve = {
        "ep": [],
        "global_ep": [],
        "phase": [],
        "occurrence": [],
        "label": [],
        "success": [],
        "steps": [],
        "dist_norm": [],
        "angle": [],
        "first_angle": [],
        "return": [],
        "q_loss": [],
        "actor_loss": [],
        "alpha": [],
        "q_target_mean": [],
        "log_pi_mean": [],
        "ec_cb_loss": [],
    }

    global_step = 0
    global_ep = 0

    for phase_idx in range(base.N_PHASES):
        phase_id = phase_idx % len(base.PHASE_DATA)
        occurrence = phase_idx // len(base.PHASE_DATA)
        ctx = base.PHASE_DATA[phase_id]["params"]
        ctx_oracle = sweep._context_one_hot(phase_id)
        name = base.PHASE_DATA[phase_id]["label"]
        print(f"\nPhase {name} ({base.N_PHASE_EP} episodes)...")

        for local_ep in range(1, base.N_PHASE_EP + 1):
            global_ep += 1

            beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx
            pos_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)
            goal_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(pos_np - goal_np) < 0.5:
                goal_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)

            vel_np = np.zeros(2, dtype=np.float32)
            prev_action_np = np.zeros(2, dtype=np.float32)
            base_obs_np = base.build_obs(pos_np, goal_np, prev_action_np)
            init_dist = float(np.linalg.norm(pos_np - goal_np))

            h_ec = torch.zeros(1, 1, sweep.EC_HIDDEN, device=device)
            z_ec_t, z_cb_t = sweep._current_features(ec, cb, cb_proj, h_ec, pos_np, prev_action_np, device, use_ec, use_cb)
            obs_np = sweep._build_aug_obs(base_obs_np, z_ec_t, z_cb_t, use_ec, use_cb, use_oracle, ctx_oracle)
            rewards = []
            dist_seq = []
            angle_seq = []
            first_angle: Optional[float] = None
            success = False
            steps = 0
            latest_update = {
                "q_loss": 0.0,
                "actor_loss": 0.0,
                "alpha": trainer.current_alpha,
                "q_target_mean": 0.0,
                "log_pi_mean": 0.0,
            }

            for _ in range(base.MAX_STEPS):
                d = float(np.linalg.norm(pos_np - goal_np))
                dist_seq.append(d)
                if d < base.GOAL_RADIUS:
                    success = True
                    break

                obs_t = sweep._to_tensor(obs_np, device)
                if global_step < base.LEARNING_STARTS or len(replay_buffer) < base.BATCH_SIZE:
                    action_np = rng.uniform(-1.0, 1.0, size=base.ACTION_DIM).astype(np.float32)
                else:
                    with torch.no_grad():
                        action_t, _, _ = trainer.act(obs_t)
                    action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

                s_next_np, vel_new_np = base.env_step_np(
                    pos_np,
                    vel_np,
                    action_np,
                    beta,
                    theta,
                    gamma_curl,
                    g_dyn,
                    bx,
                    by,
                    sdn_scale=base.SDN_COEFF,
                    rng=rng,
                )
                done = bool(np.linalg.norm(s_next_np - goal_np) < base.GOAL_RADIUS)
                reward = -0.1 * float(np.linalg.norm(s_next_np - goal_np))
                if done:
                    reward += base.SUCCESS_BONUS
                rewards.append(reward)

                gd = goal_np - pos_np
                cross = gd[0] * action_np[1] - gd[1] * action_np[0]
                dot = action_np[0] * gd[0] + action_np[1] * gd[1]
                alpha = float(np.degrees(np.arctan2(cross, dot)))
                angle_seq.append(alpha)
                if first_angle is None:
                    first_angle = alpha

                h_ec_prev = h_ec.clone()
                h_ec, pred_err = sweep._update_ec_hidden(ec, cb, h_ec, pos_np, action_np, s_next_np, device, use_ec)
                next_base_obs_np = base.build_obs(s_next_np, goal_np, action_np)
                next_z_ec_t, next_z_cb_t = sweep._current_features(ec, cb, cb_proj, h_ec, s_next_np, action_np, device, use_ec, use_cb)
                next_obs_np = sweep._build_aug_obs(next_base_obs_np, next_z_ec_t, next_z_cb_t, use_ec,
                                              use_cb, use_oracle, ctx_oracle)
                replay_buffer.add(
                    obs_np,
                    action_np,
                    reward,
                    next_obs_np,
                    done,
                    t_step=global_step,
                    s_t=pos_np,
                    s_t_next=s_next_np,
                    pred_err_t=pred_err.squeeze(),
                    h_ec=h_ec_prev.squeeze(0),  # Store the current h_ec before updating
                )

                if (
                    len(replay_buffer) >= base.BATCH_SIZE
                    and global_step >= base.LEARNING_STARTS
                    and global_step % base.UPDATE_EVERY == 0
                ):
                    batch, idxs, transition_data = replay_buffer.sample(base.BATCH_SIZE, device, t_now=global_step)
                    s_t, a_t, s_t_next, pred_err_t, h_ec_prev = transition_data
                    with torch.no_grad():
                        q_target = trainer.compute_q_target(batch)
                        q1 = trainer.qf1(batch.observations, batch.actions).view(-1)
                        q2 = trainer.qf2(batch.observations, batch.actions).view(-1)
                        abs_tde = 0.5 * (torch.abs(q1 - q_target) + torch.abs(q2 - q_target))

                    latest_update = trainer.update(batch)

                    if ec_cb_trainer is not None:
                        ec_cb_batch = ECCBBatch(
                            observations=batch.observations,
                            actions=batch.actions,
                            next_observations=batch.next_observations,
                            s_t=s_t,
                            a_t=a_t,
                            s_t_next=s_t_next,
                            pred_err_t=pred_err_t,
                            h_ec_prev=h_ec_prev
                        )
                        ec_cb_update = ec_cb_trainer.update(ec_cb_batch)
                        latest_update.update(ec_cb_update)
                    replay_buffer.update_abs_tde(idxs, abs_tde.detach().cpu().numpy())

                pos_np = s_next_np
                vel_np = vel_new_np
                prev_action_np = action_np
                obs_np = next_obs_np
                global_step += 1
                steps += 1

                if done:
                    success = True
                    break

            ep_return = float(sum(rewards))
            dist_norm = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
            mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
            first_angle = first_angle if first_angle is not None else 0.0

            curve["ep"].append(local_ep)
            curve["global_ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["occurrence"].append(occurrence)
            curve["label"].append(name)
            curve["success"].append(float(success))
            curve["steps"].append(float(steps))
            curve["dist_norm"].append(float(dist_norm))
            curve["angle"].append(float(mean_angle))
            curve["first_angle"].append(float(first_angle))
            curve["return"].append(float(ep_return))
            curve["q_loss"].append(float(latest_update["q_loss"]))
            curve["actor_loss"].append(float(latest_update["actor_loss"]))
            curve["alpha"].append(float(latest_update["alpha"]))
            curve["q_target_mean"].append(float(latest_update["q_target_mean"]))
            curve["log_pi_mean"].append(float(latest_update["log_pi_mean"]))
            curve["ec_cb_loss"].append(float(latest_update.get("ec_cb_loss", np.nan)))

            if local_ep % 50 == 0:
                w = min(50, len(curve["success"]))
                if ec_cb_trainer is not None:
                    print(
                        f"  ep {local_ep:4d} | "
                        f" global_ep {global_ep:5d} | "
                        f"succ={np.mean(curve['success'][-w:]):.2f} | "
                        f"ret={np.mean(curve['return'][-w:]):.1f} | "
                        f"q_loss={np.mean(curve['q_loss'][-w:]):.4f} | "
                        f"actor={np.mean(curve['actor_loss'][-w:]):.4f} | "
                        f"alpha={np.mean(curve['alpha'][-w:]):.3f} | "
                        f"ec_cb={np.mean(curve['ec_cb_loss'][-w:]):.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  ep {local_ep:4d} | "
                        f" global_ep {global_ep:5d} | "
                        f"succ={np.mean(curve['success'][-w:]):.2f} | "
                        f"ret={np.mean(curve['return'][-w:]):.1f} | "
                        f"q_loss={np.mean(curve['q_loss'][-w:]):.4f} | "
                        f"actor={np.mean(curve['actor_loss'][-w:]):.4f} | "
                        f"alpha={np.mean(curve['alpha'][-w:]):.3f}",
                        flush=True,
                    )

        if checkpoint_dir is not None:
            ckpt_path = sweep._save_phase_checkpoint(
                checkpoint_dir,
                phase_idx,
                name,
                trainer,
                ec_cb_trainer,
                seed,
                config_name,
                global_step,
                global_ep,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    return curve


def run_named_sweep(presets, seeds: int, base_fig_dir: str, save_agg: bool = True):
    for cfg in presets:
        cfg_name = cfg.get("name", "baseline")
        cfg_fig_dir = os.path.join(base_fig_dir, cfg_name)
        os.makedirs(cfg_fig_dir, exist_ok=True)
        print(f"\nRunning config: {cfg_name} -> saving to {cfg_fig_dir}")
        main(
            seeds=seeds,
            use_ec=cfg.get("use_ec", True),
            use_cb=cfg.get("use_cb", True),
            use_oracle=cfg.get("use_oracle", False),
            use_pretrained_ec_cb=cfg.get("use_pretrained_ec_cb", sweep.USE_PRETRAINED_EC_CB),
            freeze_pretrained_ec_cb=cfg.get("freeze_pretrained_ec_cb", sweep.FREEZE_PRETRAINED_EC_CB),
            ec_cb_ckpt=cfg.get("ec_cb_ckpt", sweep.EC_CB_CKPT),
            fig_dir=cfg_fig_dir,
            priority_gamma_t=cfg.get("priority_gamma_t", sweep.PRIORITY_GAMMA_T),
            priority_gamma_td=cfg.get("priority_gamma_td", sweep.PRIORITY_GAMMA_TD),
            priority_mix=cfg.get("priority_mix", sweep.PRIORITY_MIX),
            save_agg=save_agg,
            config_name=cfg_name,
        )


def main(
    seeds: int = 5,
    use_ec: bool = True,
    use_cb: bool = True,
    use_oracle: bool = False,
    use_pretrained_ec_cb: bool = sweep.USE_PRETRAINED_EC_CB,
    freeze_pretrained_ec_cb: bool = sweep.FREEZE_PRETRAINED_EC_CB,
    ec_cb_ckpt: str = sweep.EC_CB_CKPT,
    fig_dir: str = FIG_DIR,
    priority_gamma_t: float = sweep.PRIORITY_GAMMA_T,
    priority_gamma_td: float = sweep.PRIORITY_GAMMA_TD,
    priority_mix: float = sweep.PRIORITY_MIX,
    save_agg: bool = True,
    config_name: Optional[str] = None,
):
    print(f"Device: {sweep.DEVICE}")
    os.makedirs(fig_dir, exist_ok=True)

    if config_name is None:
        config_name = "custom"

    print(
        "Config | "
        f"seeds={seeds}, "
        f"use_ec={use_ec}, "
        f"use_cb={use_cb}, "
        f"use_oracle={use_oracle}, "
        f"use_pretrained_ec_cb={use_pretrained_ec_cb}, "
        f"freeze_pretrained_ec_cb={freeze_pretrained_ec_cb}, "
        f"ec_cb_ckpt={ec_cb_ckpt}, "
        f"fig_dir={fig_dir}, "
        f"priority_gamma_t={priority_gamma_t}, "
        f"priority_gamma_td={priority_gamma_td}, "
        f"priority_mix={priority_mix}"
    )

    for seed in range(seeds):
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        seed_dir = os.path.join(fig_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        curve_path = os.path.join(seed_dir, "curve.npz")

        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        else:
            curve = run_babab(
                seed=seed,
                use_ec=use_ec,
                use_cb=use_cb,
                use_oracle=use_oracle,
                use_pretrained_ec_cb=use_pretrained_ec_cb,
                freeze_pretrained_ec_cb=freeze_pretrained_ec_cb,
                ec_cb_ckpt=ec_cb_ckpt,
                priority_gamma_t=priority_gamma_t,
                priority_gamma_td=priority_gamma_td,
                priority_mix=priority_mix,
                checkpoint_dir=os.path.join(seed_dir, "checkpoints"),
                config_name=config_name,
            )
            np.savez(curve_path, curve=np.array(curve, dtype=object))

        key_params = {
            'use_ec': use_ec,
            'use_cb': use_cb,
            'use_pretrained_ec_cb': use_pretrained_ec_cb,
            'freeze_pretrained_ec_cb': freeze_pretrained_ec_cb,
            'priority_gamma_t': priority_gamma_t,
            'priority_gamma_td': priority_gamma_td,
            'priority_mix': priority_mix,
            'use_oracle': use_oracle,
            'config_name': config_name,
        }
        base.plot_seed(seed, curve, base.GOAL_RADIUS, seed_dir, key_params=key_params)

        # Convert per-seed curve dict to DataFrame and annotate with config
        try:
            df = pd.DataFrame(curve)
        except Exception:
            # Fallback: coerce into DataFrame via items
            df = pd.DataFrame({k: np.asarray(v) for k, v in curve.items()})
        for k, v in key_params.items():
            df[k] = v
        df['seed'] = int(seed)
        if 'all_runs' not in locals():
            all_runs = []
        all_runs.append(df)

    # Aggregate across seeds and save
    if 'all_runs' in locals() and all_runs:
        agg_df = pd.concat(all_runs, ignore_index=True)
        if save_agg:
            csv_path = os.path.join(fig_dir, "aggregated_results.csv")
            pkl_path = os.path.join(fig_dir, "aggregated_results.pkl")
            agg_df.to_csv(csv_path, index=False)
            agg_df.to_pickle(pkl_path)
            print(f"Aggregated results saved to {csv_path} and {pkl_path}")

    print(f"\nDone. Figures saved to {fig_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with optional EC/CB-augmented observations.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run.")
    parser.add_argument(
        "--use-ec",
        dest="use_ec",
        action="store_true",
        default=True,
        help="Enable EC features.",
    )
    parser.add_argument(
        "--no-use-ec",
        dest="use_ec",
        action="store_false",
        help="Disable EC features.",
    )
    parser.add_argument(
        "--use-cb",
        dest="use_cb",
        action="store_true",
        default=True,
        help="Enable CB features.",
    )
    parser.add_argument(
        "--no-use-cb",
        dest="use_cb",
        action="store_false",
        help="Disable CB features.",
    )

    parser.set_defaults(use_oracle=False)
    parser.add_argument(
        "--use-oracle",
        dest="use_oracle",
        action="store_true",
        help="Enable oracle context features (one-hot phase_id).",
    )

    parser.add_argument(
        "--no-use-oracle",
        dest="use_oracle",
        action="store_false",
        help="Disable oracle context features.",
    )

    parser.add_argument(
        "--ec-cb-ckpt",
        type=str,
        default=sweep.EC_CB_CKPT,
        help="Path to checkpoint with optional 'ec' and 'cb' state dicts.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=FIG_DIR,
        help="Output directory for figures and cached curves.",
    )
    parser.add_argument(
        "--priority-gamma-t",
        type=float,
        default=sweep.PRIORITY_GAMMA_T,
        help="Recency temperature in exp(-(dt/gamma_t + |tde|/gamma_td)).",
    )
    parser.add_argument(
        "--priority-gamma-td",
        type=float,
        default=sweep.PRIORITY_GAMMA_TD,
        help="TD-error temperature in exp(-(dt/gamma_t + |tde|/gamma_td)).",
    )
    parser.add_argument(
        "--priority-mix",
        type=float,
        default=sweep.PRIORITY_MIX,
        help="Uniform-mixture fraction for replay probabilities.",
    )

    parser.set_defaults(use_pretrained_ec_cb=sweep.USE_PRETRAINED_EC_CB)
    parser.add_argument(
        "--use-pretrained-ec-cb",
        dest="use_pretrained_ec_cb",
        action="store_true",
        help="Enable loading EC/CB from checkpoint.",
    )
    parser.add_argument(
        "--no-use-pretrained-ec-cb",
        dest="use_pretrained_ec_cb",
        action="store_false",
        help="Disable loading EC/CB from checkpoint.",
    )

    parser.set_defaults(freeze_pretrained_ec_cb=sweep.FREEZE_PRETRAINED_EC_CB)
    parser.add_argument(
        "--freeze-pretrained-ec-cb",
        dest="freeze_pretrained_ec_cb",
        action="store_true",
        help="Freeze EC/CB parameters during SAC training.",
    )
    parser.add_argument(
        "--no-freeze-pretrained-ec-cb",
        dest="freeze_pretrained_ec_cb",
        action="store_false",
        help="Allow EC/CB parameters to remain trainable (not optimized by SAC by default).",
    )

    parser.set_defaults(save_agg=True)
    parser.add_argument(
        "--no-save-agg",
        dest="save_agg",
        action="store_false",
        help="Do not save aggregated DataFrame (CSV/pickle) after runs.",
    )

    parser.set_defaults(sweep=False)
    parser.add_argument(
        "--sweep",
        dest="sweep",
        action="store_true",
        help="Run the named config sweep (uses CONFIG_PRESETS).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if getattr(args, "sweep", False):
        run_named_sweep(sweep.CONFIG_PRESETS, seeds=args.seeds, base_fig_dir=args.fig_dir, save_agg=args.save_agg)
    else:
        main(
            seeds=args.seeds,
            use_ec=args.use_ec,
            use_cb=args.use_cb,
            use_pretrained_ec_cb=args.use_pretrained_ec_cb,
            freeze_pretrained_ec_cb=args.freeze_pretrained_ec_cb,
            ec_cb_ckpt=args.ec_cb_ckpt,
            fig_dir=args.fig_dir,
            priority_gamma_t=args.priority_gamma_t,
            priority_gamma_td=args.priority_gamma_td,
            priority_mix=args.priority_mix,
            save_agg=args.save_agg,
            use_oracle=args.use_oracle,
        )
