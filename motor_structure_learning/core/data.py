"""Data generation functions used across training and evaluation scripts."""

import numpy as np


def simulate_state_action_and_next_state(
    n_runs=300,
    T=200,
    H=5,
    W=5,
    action_set="4",
    slip_prob=0.0,
    seed=0,
    beta_prob=0.7,
    beta_range=(-0.5, 0.5),
    gamma_prob=0.7,
    gamma_range=(-0.5, 0.5),
    context_mask=(1, 1),
    zero_disabled_in_context=True,
):
    """Simulate state-action-next_state tuples with context-dependent dynamics.

    Returns arrays for states, actions, next_states, and context values.
    """
    rng = np.random.default_rng(seed)
    m_beta, m_gamma = [int(x) for x in context_mask]

    if action_set == "4":
        actions = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
    elif action_set == "8":
        actions = [
            (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
            (0.0, -1.0),              (0.0, 1.0),
            (1.0, -1.0),  (1.0, 0.0),  (1.0, 1.0),
        ]
    else:
        raise ValueError("action_set must be '4' or '8'")

    states, acts, next_states, contexts = [], [], [], []
    num_actions = len(actions)

    for _ in range(n_runs):
        x = float(int(rng.integers(0, H)))
        y = float(int(rng.integers(0, W)))

        for _t in range(T):
            action_id = int(rng.integers(0, num_actions))
            dx, dy = actions[action_id]

            if slip_prob > 0.0 and rng.random() < slip_prob:
                action_id = int(rng.integers(0, num_actions))
                dx, dy = actions[action_id]

            beta = float(rng.uniform(beta_range[0], beta_range[1])) if (rng.random() < beta_prob) else 0.0
            gamma = float(rng.uniform(gamma_range[0], gamma_range[1])) if (rng.random() < gamma_prob) else 0.0

            beta_eff = m_beta * beta
            gamma_eff = m_gamma * gamma

            a = np.array([dx, dy], dtype=np.float32)
            a_norm = float(np.linalg.norm(a))

            if a_norm > 0.0:
                a_hat = a / a_norm
                a_hat_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)
            else:
                a_hat = np.zeros(2, dtype=np.float32)
                a_hat_perp = np.zeros(2, dtype=np.float32)

            f_env = beta_eff * a_hat + gamma_eff * a_hat_perp

            x_next = float(np.clip(x + a[0] + f_env[0], 0.0, float(H - 1)))
            y_next = float(np.clip(y + a[1] + f_env[1], 0.0, float(W - 1)))

            if zero_disabled_in_context:
                c_beta = beta_eff
                c_gamma = gamma_eff
            else:
                c_beta, c_gamma = beta, gamma

            states.append((x, y))
            acts.append((dx, dy))
            next_states.append((x_next, y_next))
            contexts.append((c_beta, c_gamma))

            x, y = x_next, y_next

    return (
        np.array(states, dtype=np.float32),
        np.array(acts, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(contexts, dtype=np.float32),
    )


__all__ = ["simulate_state_action_and_next_state"]
