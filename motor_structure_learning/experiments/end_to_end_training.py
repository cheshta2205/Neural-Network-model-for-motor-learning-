"""
/experiments/end_to_end_training.py

This script trains a forward model to predict the next state givent the current state, action, and a latent context.
The latent context is learned end-to-end with the forward model, and is regularized to be low-dimensional.
In addition, we train a policy and critic to solve the task of reaching a goal state, using the learned forward model and latent context.
"""
import torch
import os
import argparse
import pdb

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.distributions import Normal
from motor_structure_learning.core.models import EntorhinalCortex, Hippocampus, CerebellumForwardModelContinuous, MotorCortex

class ReplayBuffer():
    """Simple replay buffer for storing episodes of experience.
    Each episode is a list of transitions, where each transition is a tuple of 
    (state, action, reward, next_state, done, MC State, MC Prev State, EC State, EC Prev State)."""
    pass

class EndToEndTrainer():
    def __init__(self, ec_module, hc_module, cb_module, mc_module, optimizers, device):
        self.ec = ec_module
        self.hc = hc_module
        self.cb = cb_module
        self.mc = mc_module
        self.opt_ec_cb = optimizers['ec_cb']
        self.opt_hc = optimizers['hc']
        self.opt_mc = optimizers['mc']
        self.device = device

    def update_networks(self, episodes):
        # Update the EC, HC, CB, and MC networks using the collected episodes
        pass

    def update_ec_cb(self, episodes):
        # Update the entorhinal cortex and cerebellum forward model using the collected episodes
        pass

    def update_hc(self, episodes):
        # Update the hippocampus (long term memory) using the collected episodes
        pass

    def update_mc(self, episodes):
        # Update the motor cortex (policy and critic) using the collected episodes
        pass

def run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True):
    beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx

    pos_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(pos_np - goal_np) < 0.5:
        goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    vel_np = np.zeros(2, dtype=np.float32)
    g_t = torch.tensor(goal_np, dtype=torch.float32, device=device)
    init_dist = float(np.linalg.norm(pos_np - goal_np))

    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)
    z_t = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    rewards = []
    log_probs = []
    entropies = []
    values = []
    dist_seq = []
    angle_seq = []

    success = False
    steps = 0
    first_angle = None

    for t in range(MAX_STEPS):
        d = float(np.linalg.norm(pos_np - goal_np))
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True
            break

        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            z_cb_t = cb_frozen.hidden(pos_t.unsqueeze(0), prev_a, z_t)

        if train:
            a_t, log_prob_t, entropy_t, V_t, h_mc = mc.sample_action(
                pos_t.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
            )
            log_probs.append(log_prob_t.squeeze(0))
            entropies.append(entropy_t.squeeze(0))
            values.append(V_t.squeeze(0).squeeze(0))
        else:
            with torch.no_grad():
                a_t, h_mc = mc.greedy_action(
                    pos_t.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
                )

        a_np = a_t.squeeze(0).detach().cpu().numpy()
        s_next_np, vel_new_np = env_step_np(pos_np, vel_np, a_np, beta, theta, gamma_curl, g_dyn, bx, by)

        reached = bool(np.linalg.norm(s_next_np - goal_np) < GOAL_RADIUS)
        r = -0.1 * float(np.linalg.norm(s_next_np - goal_np))
        if reached:
            r += SUCCESS_BONUS
        rewards.append(r)

        gd = goal_np - pos_np
        cross = gd[0] * a_np[1] - gd[1] * a_np[0]
        dot = a_np[0] * gd[0] + a_np[1] * gd[1]
        alpha = float(np.degrees(np.arctan2(cross, dot)))
        angle_seq.append(alpha)
        if t == 0:
            first_angle = alpha

        with torch.no_grad():
            s_next_t = torch.tensor(s_next_np, dtype=torch.float32, device=device)
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos_t.unsqueeze(0)
            cb_pred = cb_frozen(pos_t.unsqueeze(0), a_t.detach(), z_t)
            eps = s_next_t.unsqueeze(0) - cb_pred
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos_t.unsqueeze(0), eps], dim=-1), h_ec
            )

        prev_a = a_t.detach()
        pos_np = s_next_np
        vel_np = vel_new_np
        steps += 1

        if reached:
            success = True
            break

    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)
    returns = list(reversed(returns))

    if train and len(returns) > 0:
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)

        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        log_probs_t = torch.stack(log_probs)
        L_actor = -(advantages * log_probs_t).mean()
        L_critic = F.mse_loss(values_t, returns_t)
        entropies_t = torch.stack(entropies)
        L_entropy = entropies_t.mean()

        mc_loss = L_actor + CRITIC_W * L_critic - ENTROPY_W * L_entropy
        adv_std = float(advantages.std().item()) if len(advantages) > 1 else 0.0
    else:
        mc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        L_actor = mc_loss
        L_critic = mc_loss
        L_entropy = mc_loss
        adv_std = 0.0

    ep_return = float(sum(rewards))
    dist_norm = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    return {
        "mc_loss": mc_loss,
        "L_actor": float(L_actor.detach().item()) if train else 0.0,
        "L_critic": float(L_critic.detach().item()) if train else 0.0,
        "adv_std": adv_std,
        "success": float(success),
        "steps": steps,
        "dist_norm": dist_norm,
        "angle": mean_angle,
        "first_angle": first_angle,
        "ep_return": ep_return,
        "G0": returns[0] if returns else 0.0,
    }

class Env():
    def __init__(self, grid_size, max_steps, device):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.device = device

    def run_episode(self,):
        # initialize goal location and initial position
        pos_np = np.random.uniform(0, self.grid_size, size=2).astype(np.float32)
        goal_np = np.random.uniform(0, self.grid_size, size=2).astype(np.float32)

        while np.linalg.norm(pos_np - goal_np) < 0.15:
            goal_np = np.random.uniform(0, self.grid_size, size=2).astype(np.float32)

        # initialize velocity
        vel_np = np.zeros(2, dtype=np.float32)

        for t in range(self.max_steps):
            # compute action using current policy
            # step the environment using the action and context parameters
            # store transition in replay buffer
            # update the EC, HC, CB, and MC networks using the collected episodes
            pass


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def main(latent_dim: int, ec_hidden: int, cb_hidden: int, ec_input: int, cb_proj_dim: int,
        mc_hidden: int, hc_dim: int, max_steps: int, n_phase_ep: int, goal_radius: float,
        grid_size: float, dt: float, lr_mc: float, lr_ec_cb: float, lr_hcb: float, gamma: float,
        entropy_w: float, action_std_min: float, action_std_max: float, critic_w: float, batch_ep: int,
        success_bonus: float, n_phases: int, ckpt: str, fig_dir: str, seed: int, buffer_size: int, **kwargs):
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up the environment, model, and training loop here
    
    ctx_params = {
        'A: rot+60': (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0),
        'B: curl+': (1.0, 0.0, +0.4, 0.0, 0.0, 0.0)
    }

    phase_data = {}
    n_contexts = len(ctx_params)

    for i in range(n_phases):
        ctx_idx = i % n_contexts
        ctx_name = list(ctx_params.keys())[ctx_idx]
        name = ctx_name + f" (phase {i // n_contexts})"
        phase_data[i] = {
            "params": ctx_params[ctx_name],
            "label": name,
            "index": ctx_idx
        }

    # Construct ec, hc, cb, and mc models here, and set up optimizers

    ec_module = EntorhinalCortex(input_dim=ec_input, hidden_dim=ec_hidden,
                                latent_dim=latent_dim)
    cb_module = CerebellumForwardModelContinuous(input_dim=latent_dim + cb_proj_dim, hidden_dim=cb_hidden)
    hc_module = Hippocampus(latent_dim=latent_dim, hc_dim=hc_dim)

    mc_module = MotorCortex(latent_dim=latent_dim, ec_hidden=ec_hidden, cb_proj_dim=cb_proj_dim,
                            mc_hidden=mc_hidden, action_std_min=action_std_min, action_std_max=action_std_max)

    optimizers = {}
    optimizers['ec_cb'] = optim.Adam(list(ec_module.parameters()) + list(cb_module.parameters()), lr=lr_ec_cb)
    optimizers['hc'] = optim.Adam(hc_module.parameters(), lr=lr_hcb)
    optimizers['mc'] = optim.Adam(mc_module.parameters(), lr=lr_mc)


    trainer = EndToEndTrainer(ec_module, hc_module, cb_module, mc_module, optimizers, device=get_device())

    pdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="End-to-end training of forward model and policy")
    
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimensionality of the latent context signal")
    parser.add_argument("--ec-hidden", type=int, default=256, help="Hidden size of the enthorhinal cortex")
    parser.add_argument("--cb-hidden", type=int, default=256, help="Hidden size of the cerebellum")
    parser.add_argument("--ec-input", type=int, default=8, help="Dimensionality of the enthorhinal cortex input (states + action)")
    parser.add_argument("--cb-proj-dim", type=int, default=32, help="Dimensionality of the cerebellar projection to the policy and critic")
    parser.add_argument("--mc-hidden", type=int, default=256, help="Hidden size of the motor cortex")
    parser.add_argument("--hc-dim", type=int, default=128, help="Dimensionality of the hippocampus (long term memory) output")

    parser.add_argument("--max-steps", type=int, default=40, help="Maximum number of steps per episode")
    parser.add_argument("--n-phase-ep", type=int, default=4_000, help="Number of episodes to train in each phase")
    parser.add_argument("--goal-radius", type=float, default=8e-1, help="Radius around the goal state that counts as success")
    parser.add_argument("--grid-size", type=float, default=4.0, help="Size of the environment grid")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step for the environment dynamics")

    parser.add_argument("--lr-mc", type=float, default=3e-4, help="Learning rate for the motor cortex (policy and critic)")
    parser.add_argument("--lr-ec-cb", type=float, default=1e-3, help="Learning rate for the enthorhinal cortex and cerebellum")
    parser.add_argument("--lr-hcb", type=float, default=1e-3, help="Learning rate for the hippocampus (long term memory)")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor for rewards")
    parser.add_argument("--entropy-w", type=float, default=0.01, help="Weight for the policy entropy bonus")
    parser.add_argument("--action-std-min", type=float, default=-2.5, help="Minimum log standard deviation for the action distribution")
    parser.add_argument("--action-std-max", type=float, default=-0.2, help="Maximum log standard deviation for the action distribution")
    parser.add_argument("--critic-w", type=float, default=1.0, help="Weight for the critic loss in the overall loss function")
    parser.add_argument("--batch-ep", type=int, default=16, help="Number of episodes to include in each training batch")
    parser.add_argument("--success-bonus", type=float, default=5.0, help="Reward bonus to add to successful episodes")
    parser.add_argument("--n-phases", type=int, default=5, help="Number of training phases (each with a different context)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Maximum number of transitions to store in the replay buffer")

    parser.add_argument("--ckpt", type=str, default="checkpoints/e2e/", help="Directory to save model checkpoints")
    parser.add_argument("--fig-dir", type=str, default="figs/e2e/", help="Directory to save training curves and visualizations")

    args = parser.parse_args()
    main(**vars(args))