"""
MC_A2C_pretrain.py

Same BABAB protocol as MC_A2C_babab_v2.py, but prefixed with a PRE-TRAIN phase
at relaxed goal radius (r=1.2) in a null environment (no rotation, no curl).
Purpose: give the critic a warm value surface and teach basic navigation before
exposing it to the harder r=0.8 BABAB, without biasing toward either perturbation.

Structure:
    [Pre-train @ r=1.2, null env]  →  [B1 A1 B2 A2 B3 @ r=0.8]

No weights reset between pre-train and BABAB. Actor + critic inherit pre-train
state. Critic is NOT reset at phase boundaries within BABAB (matching noreset
variant of babab_v2).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import MC_A2C as base

# v2 hyperparameter override (must happen before anything uses ACTION_STD_MIN)
base.ACTION_STD_MIN = -1.5

from MC_A2C import (
    EntorhinalCortex, CerebellumCB, MotorCortex,
    CTX_A, CTX_B,
    LR_MC, BATCH_EPS, DEVICE, CKPT,
)

# Re-use v2's run_episode (which drops per-episode advantage normalization).
# run_episode reads GOAL_RADIUS from babab_v2's module namespace, so we mutate
# v2.GOAL_RADIUS before each training segment.
import MC_A2C_babab_v2 as v2


# ── Schedule ─────────────────────────────────────────────────────────────────
PRETRAIN_RADIUS = 1.2
BABAB_RADIUS    = 0.8
N_PRETRAIN_EP   = 3500      # total pre-train episodes (null env)
N_PHASE_EP      = 3500      # per BABAB phase

# Null context: no rotation, no curl — just basic navigation
CTX_NULL = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, "null")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["B1: curl+", "A1: rot+60°", "B2: curl+", "A2: rot+60°", "B3: curl+"]
A1_IDX, B2_IDX, A2_IDX = 1, 2, 3

FIG_DIR = "figs/actor_critic_mc_pretrain_null_r1.2_babab_r0.8"


# ── Training loop ────────────────────────────────────────────────────────────
def run_pretrain_then_babab(seed=0):
    device = DEVICE
    rng    = np.random.default_rng(seed)

    ckpt = torch.load(CKPT, map_location=device)
    ec        = EntorhinalCortex().to(device)
    cb_frozen = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb_frozen.load_state_dict(ckpt["cb"])
    ec.eval(); cb_frozen.eval()
    for p in ec.parameters():        p.requires_grad_(False)
    for p in cb_frozen.parameters(): p.requires_grad_(False)

    mc           = MotorCortex().to(device)
    mc_optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)

    curve = {
        "ep": [], "phase": [], "success": [], "steps": [],
        "dist_norm": [], "angle": [], "first_angle": [], "return": [],
        "L_actor": [], "L_critic": [], "L_total": [], "adv_std": [],
        "radius": [],
    }

    global_ep = 0

    # ── Pre-train phase (radius 1.2, alternating A/B) ────────────────────────
    v2.GOAL_RADIUS = PRETRAIN_RADIUS
    print(f"\nPre-train @ r={PRETRAIN_RADIUS} ({N_PRETRAIN_EP} eps, null env)...")
    mc_optimizer.zero_grad()
    for local_ep in range(1, N_PRETRAIN_EP + 1):
        global_ep += 1
        ctx = CTX_NULL

        mc.train()
        out = v2.run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True)
        (out["mc_loss"] / BATCH_EPS).backward()
        if local_ep % BATCH_EPS == 0:
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            mc_optimizer.step()
            mc_optimizer.zero_grad()

        curve["ep"].append(global_ep)
        curve["phase"].append(-1)  # pre-train marker
        curve["success"].append(float(out["success"]))
        curve["steps"].append(float(out["steps"]))
        curve["dist_norm"].append(float(out["dist_norm"]))
        curve["angle"].append(float(out["angle"]))
        curve["first_angle"].append(float(out["first_angle"]))
        curve["return"].append(float(out["ep_return"]))
        curve["L_actor"].append(out["L_actor"])
        curve["L_critic"].append(out["L_critic"])
        curve["L_total"].append(float(out["mc_loss"].detach().item()))
        curve["adv_std"].append(out["adv_std"])
        curve["radius"].append(PRETRAIN_RADIUS)

        if local_ep % 100 == 0:
            w = min(100, len(curve["success"]))
            print(
                f"  pre ep {local_ep:5d} | "
                f"succ={np.mean(curve['success'][-w:]):.2f} | "
                f"ret={np.mean(curve['return'][-w:]):.1f} | "
                f"angle={np.mean(curve['angle'][-w:]):+.1f}° | "
                f"L_crit={np.mean(curve['L_critic'][-w:]):.3f}",
                flush=True,
            )

    # ── BABAB phase (radius 0.8, no critic reset) ───────────────────────────
    v2.GOAL_RADIUS = BABAB_RADIUS
    for phase_idx, ctx in enumerate(PHASES):
        print(f"\nPhase {PHASE_LABELS[phase_idx]} ({N_PHASE_EP} eps) @ r={BABAB_RADIUS}...")

        mc_optimizer.zero_grad()
        for local_ep in range(1, N_PHASE_EP + 1):
            global_ep += 1

            mc.train()
            out = v2.run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True)
            (out["mc_loss"] / BATCH_EPS).backward()
            if local_ep % BATCH_EPS == 0:
                torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
                mc_optimizer.step()
                mc_optimizer.zero_grad()

            curve["ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["success"].append(float(out["success"]))
            curve["steps"].append(float(out["steps"]))
            curve["dist_norm"].append(float(out["dist_norm"]))
            curve["angle"].append(float(out["angle"]))
            curve["first_angle"].append(float(out["first_angle"]))
            curve["return"].append(float(out["ep_return"]))
            curve["L_actor"].append(out["L_actor"])
            curve["L_critic"].append(out["L_critic"])
            curve["L_total"].append(float(out["mc_loss"].detach().item()))
            curve["adv_std"].append(out["adv_std"])
            curve["radius"].append(BABAB_RADIUS)

            if local_ep % 100 == 0:
                w = min(100, len(curve["success"]))
                print(
                    f"  ep {local_ep:5d} | "
                    f"succ={np.mean(curve['success'][-w:]):.2f} | "
                    f"ret={np.mean(curve['return'][-w:]):.1f} | "
                    f"angle={np.mean(curve['angle'][-w:]):+.1f}° | "
                    f"L_act={np.mean(curve['L_actor'][-w:]):.3f} | "
                    f"L_crit={np.mean(curve['L_critic'][-w:]):.3f} | "
                    f"adv_std={np.mean(curve['adv_std'][-w:]):.3f}",
                    flush=True,
                )

    return curve


# ── Plotting ─────────────────────────────────────────────────────────────────
SW = 100


def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def split_babab_phases(arr, phase_arr):
    """Return list of arrays, one per BABAB phase (phase_idx 0..4)."""
    arr = np.array(arr); phase_arr = np.array(phase_arr)
    return [arr[phase_arr == i] for i in range(len(PHASES))]


def babab_boundaries(phase_arr):
    """Global episode indices where each BABAB phase ends (for vlines)."""
    phase_arr = np.array(phase_arr)
    bounds = []
    for i in range(len(PHASES)):
        idxs = np.where(phase_arr == i)[0]
        if len(idxs):
            bounds.append(idxs[-1] + 1)
    return bounds


def add_phase_labels(ax, phase_arr):
    phase_arr = np.array(phase_arr)
    # pretrain / babab split line
    pretrain_end = np.where(phase_arr == -1)[0]
    if len(pretrain_end):
        ax.axvline(pretrain_end[-1] + 1, color="black", ls="-", lw=1.2)
        ax.text(pretrain_end[-1] / 2, 1.04, "Pre-train null @ r=1.2",
                ha="center", transform=ax.get_xaxis_transform(), fontsize=7)

    for i in range(len(PHASES)):
        idxs = np.where(phase_arr == i)[0]
        if not len(idxs): continue
        start, end = idxs[0], idxs[-1]
        ax.axvline(end + 1, color="gray", ls="--", lw=0.7)
        ax.text((start + end) / 2, 1.04, PHASE_LABELS[i],
                ha="center", transform=ax.get_xaxis_transform(), fontsize=7)


def plot_seed(seed, curve, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    ep_g  = np.arange(1, len(curve["success"]) + 1)
    ep_sm = ep_g[SW - 1:] if len(ep_g) >= SW else ep_g

    # 1) Success (full)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ep_sm, smooth(curve["success"]), lw=1.6, color="steelblue")
    add_phase_labels(ax, curve["phase"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("Success")
    ax.set_title(f"Pretrain→BABAB success (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "success_curve.png"), dpi=150)
    plt.close()

    # 2) Angle
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ep_sm, smooth(curve["angle"]), lw=1.5, color="darkorange")
    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    add_phase_labels(ax, curve["phase"])
    ax.set_ylim(-110, 60)
    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("Mean angle (°)")
    ax.set_title(f"Pretrain→BABAB angle (seed {seed})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "angle_curve.png"), dpi=150)
    plt.close()

    # 3) A1 vs A2 success — savings
    phases = split_babab_phases(curve["success"], curve["phase"])
    ep_local = np.arange(1, N_PHASE_EP + 1)
    ep_local_sm = ep_local[SW - 1:] if len(ep_local) >= SW else ep_local
    fig, ax = plt.subplots(figsize=(8, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y = smooth(np.array(phases[ph_idx]))
        ax.plot(ep_local_sm, y, lw=2, color=col, label=lbl)
    ax.set_xlim(1, N_PHASE_EP); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Episode within phase"); ax.set_ylabel("Success")
    ax.set_title(f"Pretrain→BABAB — A1 vs A2 success (seed {seed})")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_success.png"), dpi=150)
    plt.close()

    # 4) A1 vs A2 angle
    angle_phases = split_babab_phases(curve["angle"], curve["phase"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y = smooth(np.array(angle_phases[ph_idx]))
        ax.plot(ep_local_sm, y, color=col, lw=2, label=lbl)
    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, N_PHASE_EP); ax.set_ylim(-110, 60)
    ax.set_xlabel("Episode within phase"); ax.set_ylabel("Mean angle (°)")
    ax.set_title(f"Pretrain→BABAB — A1 vs A2 angle (seed {seed})")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_angle.png"), dpi=150)
    plt.close()

    print(f"Seed {seed}: figures saved to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Pre-train: {N_PRETRAIN_EP} eps @ r={PRETRAIN_RADIUS}")
    print(f"BABAB: {N_PHASE_EP} eps/phase × 5 @ r={BABAB_RADIUS}")
    os.makedirs(FIG_DIR, exist_ok=True)

    for seed in range(1):
        print(f"\n{'=' * 60}\nSeed {seed} — Pretrain → BABAB\n{'=' * 60}")
        seed_dir   = os.path.join(FIG_DIR, f"seed_{seed}")
        curve_path = os.path.join(seed_dir, "curve.npz")
        os.makedirs(seed_dir, exist_ok=True)

        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        else:
            curve = run_pretrain_then_babab(seed=seed)
            np.savez(curve_path, curve=curve)

        plot_seed(seed, curve, seed_dir)

    print(f"\nDone. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
