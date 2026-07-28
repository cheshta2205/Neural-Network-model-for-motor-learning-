# Contextual Inference — paper code

Code and checkpoints to reproduce the figures of the MTL–cerebellar–motor-cortex
contextual-inference paper. The model: a recurrent **MTL** module infers a latent
**context** from a **cerebellar (CB)** forward model's prediction error; a
**motor-cortex (MC)** controller uses the inferred context to act.

Every script can **train from scratch** or **load the included checkpoint**
(`--skip-train`). Checkpoints are small (~1–2 MB) and committed, so all figures
reproduce out of the box.

## Setup

```bash
pip install -r requirements.txt
```

## Figure → script map

| Figure | Analysis | Script(s) | Checkpoint |
|--------|----------|-----------|------------|
| **1** | MTL infers latent context (decoding, confusion, decoder geometry) | `ECtoCB_train_MSE_12C.py`, `MLP_12c.py`, `replot_mtl_geometry.py` | `paper/checkpoints/ectocb_12ctx_seq80_z64/…seed0.pt` |
| **2** | Savings (SAC controller) | **External — see note below** | — |
| **3** | Structure learning / zero-shot generalization | `structure_learning_rot.py`, `structure_learning_drift.py` | `paper/checkpoints/structure_learning_{rot_0_45_90,rot_0_90,drift_0p4_0p8_1p2}.pt` |
| **4** | Context inference from reward-prediction error (SRO) | `context_inference_sro_rpe.py` | `checkpoints/context_inference_sro_rpe.pt` |
| **5** | MTL vs ventral hippocampus (motor task) | `geometry_section1_vhpc.py`, `make_context_3angle_section1.py` | (uses the 12ctx checkpoint) |
| **6** | MTL vs HPC/PFC geometry (SRO; Bernardi 2020) | `geometry_sro.py`, `geometry_sro_panels.py` | (uses the SRO checkpoint) |

Diagnostic: `within_context_decode_sro.py` — within-context vs pooled vs cross-context
(CCGP) decode, demonstrating the value/action XOR structure in the SRO task.

### Fig 2 (savings) — external
The savings experiment (Soft Actor–Critic controller, episodes-to-criterion across
re-exposures) lives in a separate repository:
`eamematsro/e2e/motor_structure_learning/` (`experiments/_ablations_sweep.py`,
`make_ablations_plot.py`). It is **not** included here; see that repo to reproduce
Figure 2.

## Reproduce the figures

```bash
# --- Fig 1: MTL infers latent context ---
python ECtoCB_train_MSE_12C.py            # loads the 12-context checkpoint, emits confusion/pca/etc.
python MLP_12c.py                         # GLM (linear) vs MLP (nonlinear) decode-over-time + confusion
python replot_mtl_geometry.py             # decoder-weight cosine heatmap + MDS

# --- Fig 3: structure learning ---
python structure_learning_rot.py   --skip-train --baselines --matched-oracle   # rotation (anchors 0,45,90)
python structure_learning_rot.py   --skip-train --train-angles 0 90            # leave-one-out (endpoints only)
python structure_learning_drift.py --skip-train --baselines --matched-oracle   # drift

# --- Fig 4: context inference from reward-prediction error ---
python context_inference_sro_rpe.py --skip-train

# --- Fig 5: MTL vs ventral hippocampus (motor) ---
python geometry_section1_vhpc.py
python make_context_3angle_section1.py

# --- Fig 6: MTL vs HPC/PFC geometry (SRO) ---
python geometry_sro.py
python geometry_sro_panels.py
python within_context_decode_sro.py       # (diagnostic: XOR / within-vs-pooled decode)
```

Or run everything: `bash make_all_figures.sh`.

The Fig-3 leave-one-out **panel** overlays the `0,45,90` and `0,90` runs (the two
`generalization_rot_*` curves); the composite panels in the paper are assembled in
Illustrator from these script outputs.

## Notes
- Figures are written under `Paper/neural_activity_comparisons/…` and `paper/…`
  (created automatically). The paper's final composite panels are assembled in
  Illustrator from these outputs.
- `MLP_12c.py` and `replot_mtl_geometry.py` dynamically import `ECtoCB_train_MSE_12C.py`;
  keep the filenames as-is (case-sensitive on Linux).

# Neural Network Model for Motor Learning

PyTorch codebase for motor structure learning experiments, including forward-model training, EC/HC/CB-style modular architectures, and reinforcement-learning pipelines (A2C, SAC, PPO-style variants).

## Repository Highlights

- Many historical experiment scripts remain at the repository root for reproducibility.
- Newer code is being consolidated under `motor_structure_learning/`.
- Stable core APIs now live in `motor_structure_learning.core`.
- Backward-compatible top-level shims keep older scripts working.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

## Project Structure

Key locations:

- `motor_structure_learning/core/`: reusable core models, data utilities, and trainer abstractions.
- `motor_structure_learning/experiments/`: runnable experiment entry points.
- `checkpoints/`: pretrained checkpoints.
- `figs/`: generated figures.

Legacy-compatible shims at repository root:

- `ec_module.py` -> `motor_structure_learning.core.ec.EntorhinalCortex`
- `hc_module.py` -> `motor_structure_learning.core.hc.Hippocampus`
- `cb_module.py` -> `motor_structure_learning.core.models.CerebellumForwardModelContinuous`
- `model.py` -> `motor_structure_learning.core.models.CerebellumForwardModelContinuous`
- `data.py` -> `motor_structure_learning.core.data.simulate_state_action_and_next_state`

## EC/CB Module Loop-In

To keep old and new code paths aligned, the EC and CB modules are exposed through both:

1. Canonical package path (`motor_structure_learning.core.*`)
2. Backward-compatible root-level module shims (`ec_module.py`, `cb_module.py`)

Recommended in new code:

```python
from motor_structure_learning.core.ec import EntorhinalCortex
from motor_structure_learning.core.models import CerebellumForwardModelContinuous
```

Still supported for older scripts:

```python
from ec_module import EntorhinalCortex
from cb_module import CerebellumForwardModelContinuous
```

## Common Commands

After editable install, console scripts are available via `pyproject.toml`.
Examples:

```bash
motor-mc-sac
motor-train-forward-model
motor-mc-savings-all
motor-olm-task
```

See [`OLM_TASK.md`](OLM_TASK.md) for the Object Location Memory task (object
egocentric features, static-object training, held-out displacement probe).

You can also run modules directly with Python:

```bash
python -m motor_structure_learning.experiments.mc_sac
```

## Notes

- This repository contains many experimental variants; not every script shares the same assumptions or interface.
- Prefer the `motor_structure_learning` package paths for new development.
