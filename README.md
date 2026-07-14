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
```

You can also run modules directly with Python:

```bash
python -m motor_structure_learning.experiments.mc_sac
```

## Notes

- This repository contains many experimental variants; not every script shares the same assumptions or interface.
- Prefer the `motor_structure_learning` package paths for new development.
