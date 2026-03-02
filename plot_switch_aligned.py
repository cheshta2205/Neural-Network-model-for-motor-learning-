"""
plot_switch_aligned.py

Generates switch-aligned context decodability plots for both:
  - ECtoCB_MSE (shifted)
  - ECtoCB_MSE + NCE

For each sequence, time is aligned to its own switch point (t_rel = t - switch).
This removes the smearing effect of averaging over variable switch points.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from ec_module import EntorhinalCortex
from ECtoCB_MSEandTIMEclustering import generate_sequences

LATENT_DIM = 16
BATCH_SIZE = 128
N_VAL      = 200
SEQ_LEN    = 40


def get_z_and_labels(ckpt_path, device):
    """Load checkpoint, run EC on val set, return z per timestep aligned to switch."""
    ec = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ec.load_state_dict(ckpt["ec"])
    ec.eval()

    va_trans, va_ctx, va_switch = generate_sequences(N_VAL, seed=99)

    # Collect z per sequence
    all_z, all_ctype, all_switch = [], [], []
    with torch.no_grad():
        for i in range(0, len(va_trans), BATCH_SIZE):
            trans = va_trans[i:i+BATCH_SIZE].to(device)
            z, _  = ec(trans)
            all_z.append(z.cpu())
            all_ctype.append(va_ctx[i:i+BATCH_SIZE, :, 2])
            all_switch.append(va_switch[i:i+BATCH_SIZE])

    all_z      = torch.cat(all_z).numpy()       # (N, T, 16)
    all_ctype  = torch.cat(all_ctype).numpy()   # (N, T)
    all_switch = torch.cat(all_switch).numpy()  # (N,)
    return all_z, all_ctype, all_switch


def switch_aligned_accuracy(all_z, all_ctype, all_switch):
    """
    For each sequence, shift time so t_rel = t - switch_point.
    Group z_t by t_rel, run logistic regression at each t_rel.
    Returns: t_rel values, accuracies, sample counts.
    """
    # Collect (z, c_type) per relative timestep
    bins = defaultdict(lambda: ([], []))
    N, T, D = all_z.shape
    for i in range(N):
        sw = int(all_switch[i])
        for t in range(T):
            t_rel = t - sw
            bins[t_rel][0].append(all_z[i, t])
            bins[t_rel][1].append(all_ctype[i, t])

    t_rels, accs, counts = [], [], []
    for t_rel in sorted(bins.keys()):
        z_bin = np.array(bins[t_rel][0])
        c_bin = np.array(bins[t_rel][1]).astype(int)
        if len(np.unique(c_bin)) < 2 or len(z_bin) < 10:
            continue
        min_class = min(np.bincount(c_bin))
        cv = min(5, min_class)
        acc = cross_val_score(LogisticRegression(max_iter=500), z_bin, c_bin,
                              cv=cv, scoring="accuracy").mean()
        t_rels.append(t_rel)
        accs.append(acc)
        counts.append(len(z_bin))

    return np.array(t_rels), np.array(accs), np.array(counts)


def plot_both(t1, a1, t2, a2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, t_rels, accs, title in [
        (axes[0], t1, a1, "EC→CB MSE (shifted)"),
        (axes[1], t2, a2, "EC→CB MSE + NCE (λ=0.5)"),
    ]:
        ax.plot(t_rels, accs, color="steelblue", linewidth=2)
        ax.axhline(0.25, color="gray",  linestyle="--", label="chance (0.25)")
        ax.axvline(0,    color="tomato", linestyle="--", linewidth=2, label="switch point (t_rel=0)")
        ax.set_xlabel("Time relative to switch (t - switch_point)")
        ax.set_ylabel("Context decode accuracy")
        ax.set_title(f"Switch-aligned decodability\n{title}")
        ax.legend(); ax.set_ylim(0, 1)
        ax.set_xlim(-20, 20)

    plt.tight_layout()
    plt.savefig("figs/switch_aligned_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/switch_aligned_comparison.png")


if __name__ == "__main__":
    device = "cpu"

    print("Loading MSE shifted checkpoint...")
    z1, c1, sw1 = get_z_and_labels("checkpoints/ectocb_mse_shifted.pt", device)

    print("Loading MSE+NCE checkpoint...")
    z2, c2, sw2 = get_z_and_labels("checkpoints/ectocb_mse_nce.pt", device)

    print("Computing switch-aligned accuracy (MSE shifted)...")
    t1, a1, _ = switch_aligned_accuracy(z1, c1, sw1)

    print("Computing switch-aligned accuracy (MSE+NCE)...")
    t2, a2, _ = switch_aligned_accuracy(z2, c2, sw2)

    plot_both(t1, a1, t2, a2)
    print("Done.")
