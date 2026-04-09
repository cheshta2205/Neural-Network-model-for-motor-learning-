# Incremental Refactor Plan

## Scope

Refactor sprawling top-level scripts into package-first entrypoints under:

- `motor_structure_learning.core`
- `motor_structure_learning.experiments`

while preserving current workflows via top-level compatibility shims.

## Conventions

1. New runtime entrypoints live in `motor_structure_learning/experiments/`.
2. Large migrated scripts use:
   - a package wrapper (`motor_structure_learning/experiments/<name>.py`)
   - a package-private implementation copy (`_..._impl.py`) when rewrite risk is high.
3. Original top-level files remain as thin shims so old commands still work.
4. Console scripts in `pyproject.toml` point to package wrappers.

## Recommended Import Style

```python
from motor_structure_learning.core import (
    simulate_state_action_and_next_state,
    CerebellumForwardModelContinuous,
    EntorhinalCortex,
    Hippocampus,
)
```

## Completed Phases

### Phase 2

- Core data/forward model moved into package namespace.
- Added top-level shims: `data.py`, `model.py`.
- Migrated runner: `eval_test.py`.

### Phase 3

- Migrated runner: `train.py`.
- Added console script: `motor-train-forward-model`.
- Kept `sklearn` import lazy inside training path.

### Phase 4

- Migrated runner: `train_ec_only.py`.
- Added console script: `motor-train-ec-only`.
- Kept plotting/sklearn imports lazy in EC-only path.

### Phase 5

- Added package-native EC/HC modules:
  - `motor_structure_learning/core/ec.py`
  - `motor_structure_learning/core/hc.py`
- Replaced top-level `ec_module.py` and `hc_module.py` with shims.

### Phase 6

- Migrated A2C runner: `MC_A2C.py` → package entrypoint.
- Kept existing debug behavior intact.

### Phase 7

- Hardened A2C imports by lazy-loading `scipy` in asymptote helper.

### Phase 8

- Migrated replay-heavy runner:
  - wrapper: `motor_structure_learning/experiments/mc_replay_hc.py`
  - impl: `motor_structure_learning/experiments/_mc_replay_hc_impl.py`
  - top-level shim: `mc_replay_hc.py`
- Added console script: `motor-mc-replay-hc`.

### Phase 9

- Migrated savings-all runner:
  - wrapper: `motor_structure_learning/experiments/mc_savings_all.py`
  - impl: `motor_structure_learning/experiments/_mc_savings_all_impl.py`
  - top-level shim: `MC_savings_all.py`
- Added console script: `motor-mc-savings-all`.

### Phase 10

- Migrated no-routing PPO savings runner:
  - wrapper: `motor_structure_learning/experiments/mc_norouting_savings_hc_ppo.py`
  - impl: `motor_structure_learning/experiments/_mc_norouting_savings_hc_ppo_impl.py`
  - top-level shim: `MC_norouting_savings_HC_PPO.py`
- Added console script: `motor-mc-norouting-savings-hc-ppo`.

### Phase 11

- Migrated no-routing HC+EC+CB savings runner:
  - wrapper: `motor_structure_learning/experiments/mc_norouting_savings_hc_ec_cb.py`
  - impl: `motor_structure_learning/experiments/_mc_norouting_savings_hc_ec_cb_impl.py`
  - top-level shim: `MC_norouting_savings_HC_EC_CB.py`
- Added console script: `motor-mc-norouting-savings-hc-ec-cb`.

## Remaining Work

1. Continue migrating remaining large orchestration scripts into package wrappers.
2. Replace private impl copies with true modular splits where worthwhile.
3. Remove top-level shims only after all external and internal invocations are updated.
