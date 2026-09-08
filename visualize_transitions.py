import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from pathlib import Path
from typing import Optional
from src.PlottingTools.plotting_functions import set_plt_params, make_axis_nice

import pdb

def generate_state_sequences(action: np.ndarray, dynamics_params: Optional[dict] = None, n_steps: int = 100, dt: float = 0.01,
                             speed_threshold: float = 1.0, distance_threshold: float = 5.0):
    """Generate state sequences given an action sequence and dynamics parameters."""
    
    if dynamics_params is None:
        dynamics_params = {
            "beta": 1.0,
            "theta": 0.0,
            "gamma_curl": 0.0,
            "g_dyn": 0.0,
            "bx": 0.0,
            "by": 0.0
        }

    n_actions = action.shape[1]
    positions = np.zeros((n_steps, 2, n_actions), dtype=np.float32)
    velocities = np.zeros((n_steps, 2, n_actions), dtype=np.float32)
    pos = np.zeros((2, n_actions), dtype=np.float32)
    vel = np.zeros((2, n_actions), dtype=np.float32)

    rotation_matrix = np.array([[np.cos(dynamics_params["theta"]), -np.sin(dynamics_params["theta"])],
                                [np.sin(dynamics_params["theta"]), np.cos(dynamics_params["theta"])]], dtype=np.float32)
    
    curl_matrix = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    drift_vector = np.array([dynamics_params["bx"], dynamics_params["by"]], dtype=np.float32)
    drift_vector = np.expand_dims(drift_vector, axis=1)  # Shape (2, 1) for broadcasting

    for t in range(n_steps):
        distance = np.linalg.norm(pos, axis=0)

        if (distance > distance_threshold).any():
            continue

        positions[t] = pos
        velocities[t] = vel

        F_env = dynamics_params["beta"] * rotation_matrix @ action + \
                dynamics_params["gamma_curl"] * curl_matrix @ vel + \
                drift_vector * dynamics_params["g_dyn"]
        
        vel += dt * F_env
        pos += dt * vel
        pos = np.clip(pos, -25, 25)  # Keep positions within bounds for visualization
        distance = np.linalg.norm(pos, axis=0)

        speed = np.linalg.norm(vel, axis=0)

        pos[:, distance > distance_threshold] = pos[:, distance > distance_threshold] * distance_threshold / distance[distance > distance_threshold]
        vel[:, speed > speed_threshold] = vel[:, speed > speed_threshold] *  speed_threshold/ speed[speed > speed_threshold]

    return positions

if __name__ == "__main__":
    set_plt_params(style='fast', )
    fig_folder = Path().cwd() / "figs" / "rlc"
    fig_folder.mkdir(parents=True, exist_ok=True)

    # Example usage:
    n_steps = 600
    dt = 0.01
    action_scale = 0.25
    n_actions = 16
    thetas = np.linspace(0, 2 * np.pi, n_actions)  # Varying rotation angles from 0 to 90 degrees
    actions = action_scale * np.array([[np.cos(theta), np.sin(theta)] for theta in thetas], dtype=np.float32)  # Unit actions in different directions
    actions = actions.T
    dynamics_params = {
        "beta": 1.0,
        "theta": 0,  # 45 degree rotation
        "gamma_curl": 0,
        "g_dyn": 0,
        "bx": 0,
        "by": 0
    }

    dynamics_positives = [
        ({"beta": 1.0, "theta": np.pi / 3, "gamma_curl": 0.0, "g_dyn": 0.0, "bx": 1.0, "by": 0.0}, r'rot (+60°) ($\theta = 60^\circ$)'),
        ({"beta": 1.0, "theta": 0, "gamma_curl": 0.5, "g_dyn": 0, "bx": 1.0, "by": 0}, r'curl (+) ($\gamma = 0.5$)'),
        ({"beta": 1.0, "theta": 0, "gamma_curl": 0.0, "g_dyn": .5 * action_scale, "bx": 1.0, "by": 0.0}, r'drift (+x) ($g = %.2f$)' % (.5 * action_scale)),
    ]

    dynamics_negatives = [
        ({"beta": 1.0, "theta": -np.pi / 3, "gamma_curl": 0.0, "g_dyn": 0.0, "bx": 1.0, "by": 0.0}, r'rot (-60°) ($\theta=-60^\circ$)'),
        ({"beta": 1.0, "theta": 0, "gamma_curl": -0.5, "g_dyn": 0, "bx": 1.0, "by": 0}, r'curl (-) ($\gamma = -0.5$)'),
        ({"beta": 1.0, "theta": 0, "gamma_curl": 0.0, "g_dyn": -.5 * action_scale, "bx": 1.0, "by": 0.0}, r'drift (-x) ($g = %.2f$)' % (-.5 * action_scale)),
    ]

    norm = colors.Normalize(vmin=thetas.min(), vmax=thetas.max())
    cmap = plt.get_cmap('hsv')
    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    for tidx, theta in enumerate(thetas):
        color = cmap(norm(theta))
        ax.plot([0, actions[0, tidx]], [0, actions[1, tidx]], color=color)
    
    make_axis_nice(ax)
    ax.set_title("Action Directions")
    ax.set_xlabel("Action X")
    ax.set_ylabel("Action Y")
    fig_path = fig_folder / "action_directions"
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)


    # Plotting the trajectory
    fig, ax = plt.subplots(2, len(dynamics_positives),
                           figsize=(1.5 * len(dynamics_positives), 2),
                           sharex=True, sharey=True)

    for idx, (dyn_params, label) in enumerate(dynamics_positives):
        pos = generate_state_sequences(actions, dyn_params, n_steps, dt)
        for tidx, theta in enumerate(thetas):
            color = cmap(norm(theta))
            ax[0, idx].plot(pos[:, 0, tidx], pos[:, 1, tidx], color=color)

        ax[0, idx].set_title(label)

    for idx, (dyn_params, label) in enumerate(dynamics_negatives):
        pos = generate_state_sequences(actions, dyn_params, n_steps, dt)
        for tidx, theta in enumerate(thetas):
            color = cmap(norm(theta))
            ax[1, idx].plot(pos[:, 0, tidx], pos[:, 1, tidx], color=color)

        ax[1, idx].set_title(label)

    make_axis_nice(fig)
    fig_path = fig_folder / "dynamics_effects"
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    plt.pause(1)
    pdb.set_trace()