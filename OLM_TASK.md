# Object Location Memory (OLM) task

Extends the base EC→CB prediction-error-minimization task (`ECtoCB_MSE.py`,
`train_shifted()`) with two static objects in the environment. EC and CB are
trained exactly the same way as the base task — minimize MSE prediction error
of the next state under a random-action trajectory — except the predicted
state now includes egocentric perceptual features of both objects, not just
agent position.

The scientific payoff is a **displacement-detection probe**: after training
(on static objects only), feed the model matched control vs. object-relocated
episodes and check whether the cerebellar (CB) forward model's prediction
error spikes specifically for the relocated object. This is an **emergent**
novelty signal — the network is never trained on displacement, so a spike is
a genuine test of an unseen event, analogous to preferential exploration in
real object-location-memory experiments.

## Files

| File | Role |
|------|------|
| `motor_structure_learning/experiments/_olm_task_impl.py` | Implementation: sequence generator, model, training loop, displacement probe. |
| `motor_structure_learning/experiments/olm_task.py` | Thin package wrapper (`main()` → impl). |
| `olm_task.py` (repo root) | Backward-compatible root-level shim. |

No changes were made to `motor_structure_learning/core/*`. `EntorhinalCortex`
is reused unmodified (its `input_dim` is just instantiated wider). The
cerebellar forward model is **not** `CerebellumForwardModelContinuous`
(its output layer is hardcoded to 2 dims) — instead `_olm_task_impl.py`
defines a local `OLMForwardModel` with the same 2-hidden-layer ReLU MLP shape
but a configurable output dimension.

## Representation

- Agent state: `(x, y)`, continuous, bounded within `[-POS_BOUND, POS_BOUND]`
  per axis (default `POS_BOUND = 5.0`, i.e. a 10×10 region centered at the
  origin). Positions are clipped to this range after each step.
- Actions are continuous: a random direction (angle, uniform over `[0, 2π)`)
  and a random magnitude (uniform over `[0, MAX_STEP]`, default `MAX_STEP =
  1.0`), applied as a displacement vector. There's no fixed action set (the
  earlier 4-cardinal-direction scheme was replaced with this).
- Heading: the *sampled direction* of the last action, not the resulting
  displacement vector — this keeps heading well-defined even when the sampled
  magnitude is near zero (no normalize-by-zero edge case to handle). Defaults
  to `(1, 0)` before the first action. There is no separate orientation state
  — heading is derived from movement.
- **Objects**: exactly 2 per episode. Each gets a fresh random 8-dim identity
  vector (`rng.standard_normal`) and a fresh random location, both resampled
  every episode. Objects never move during training — the network only ever
  learns "objects are static." Locations/identities are *not* reused across
  episodes, so the circuit must bind identity↔location from within-episode
  experience alone (this is what makes it a memory task rather than a
  semantic-prior task).
- **Object feature** (egocentric, computed fresh every timestep from the
  agent's current position/heading):
  ```
  rel      = obj_pos - agent_pos
  distance = ||rel|| / diag                     # normalized to ~[0, 1]
  phi      = atan2(rel_y, rel_x)                 # world-frame bearing to object
  theta    = wrap_to_pi(phi - heading_angle)      # egocentric bearing
  feature  = [identity (8 dims), distance, sin(theta), cos(theta)]   # 11 dims
  ```
- **State vector** = `[agent_xy, object_0_feature, object_1_feature]` (24 dims
  with the defaults above). Objects are always "in the environment" — there is
  no occlusion/delay phase or presence flag; perceptibility is entirely a
  function of the (always-computed) distance/bearing features.

## Training

Mirrors `train_shifted()` in `ECtoCB_MSE.py`, generalized to the wider state:

1. Generate `(N, T, D)` sequences of `[state_t, action_t, state_{t+1}]` via
   `generate_olm_sequences`.
2. Run the whole sequence through `EntorhinalCortex`'s GRU in one shot:
   `z, _ = ec(trans)`.
3. Time-shift so EC can't trivially copy the answer: `z_prev = z[:, :-1]`
   (context computed only from steps `0..t`) is combined with
   `state_{t+1}, action_{t+1}` to predict `state_{t+2}`.
4. Loss: plain `F.mse_loss(s_pred, s_target)`, backprop jointly through
   `ec.parameters() + cb.parameters()` (Adam).

No replay buffer, no stored hidden states across calls — this is full-sequence
BPTT, not the detached single-step replay pattern used by `ECCBTrainer` in
`core/trainers/ec_cb.py` (that trainer's only real usage is a SAC/RL ablation
and doesn't fit this task; see the comment at the top of `_olm_task_impl.py`).

Checkpoint saved to `checkpoints/olm_task.pt` (`{"ec", "cb", "history"}`).

## Displacement probe

`evaluate_displacement` (no gradient, models in `.eval()`):

- For each of `n_pairs` probe episodes, generates **one** shared random-action
  trajectory + shared initial object identities/locations, then produces two
  variants identical up to `t_probe` (default: `seq_len // 2`):
  - **Control**: objects never move.
  - **Displaced**: at `t_probe`, one randomly chosen object is relocated to a
    new random position; everything else (trajectory, other object) is
    unchanged.
- Runs both variants through the trained EC/CB using the same time-shifted
  procedure as training, and computes per-timestep squared error restricted
  to each object's feature slot.
- Plots three curves — (moved slot, displaced), (same slot, control), (other
  slot, displaced) — vs. timestep, with a vertical line at `t_probe`, saved to
  `figs/olm_displacement_probe.png`.

Note on indexing: the array at position `j` (0-indexed) holds the error for
predicting `state(j+2)`, because of the time-shift in step 3 above — the
plotting code accounts for this (`t_axis = np.arange(2, seq_len + 1)`) so the
x-axis correctly reflects which timestep's state is being predicted. Get this
wrong and the plotted spike appears one step before `t_probe` instead of on it
(this was caught and fixed during development).

Expected result with default hyperparameters: the moved object's curve spikes
sharply at `t_probe`, then decays back to baseline over ~15-20 steps as the
agent re-observes the object at its new location; the control curve and the
unmoved object's curve stay flat throughout.

## Usage

```bash
# via console script (after `pip install -e .`)
motor-olm-task

# or directly
python olm_task.py

# or as a module
python -m motor_structure_learning.experiments.olm_task
```

By default this runs `train_olm()` with default hyperparameters (2000 train /
200 val sequences, 100 epochs, `seq_len=60`) followed by
`evaluate_displacement()`, and prints per-epoch train/val MSE.

`main(seeds=None, out_dir=MULTI_SEED_DIR, train_kwargs=None, eval_kwargs=None)`
— passing `seeds` switches to `train_olm_multi_seed` instead (see below), e.g.
`main(seeds=[0, 1, 2, 3, 4])`. There's no CLI flag for this yet (`motor-olm-task`
always runs the single-seed default) — call `main()`/`train_olm_multi_seed()`
directly from Python to use it. To customize the single-seed path, call the
functions directly:

```python
from motor_structure_learning.experiments._olm_task_impl import train_olm, evaluate_displacement

ec, cb, history, device = train_olm(n_train=500, epochs=30, seq_len=80, seed=0)
evaluate_displacement(ec, cb, n_pairs=200, seq_len=80, t_probe=40, device=device)
```

Key constants (top of `_olm_task_impl.py`): `POS_BOUND` (position bound),
`MAX_STEP` (max action magnitude), `SEQ_LEN`, `N_TRAIN`, `N_VAL`,
`BATCH_SIZE`, `EPOCHS`, `LR`, `LATENT_DIM`, `CB_HIDDEN`, `N_OBJECTS`,
`IDENTITY_DIM`.

### Training multiple seeds

`train_olm_multi_seed(seeds, out_dir=MULTI_SEED_DIR, train_kwargs=None,
eval_kwargs=None)` trains and evaluates one model per seed, useful for
checking whether the displacement-detection effect is robust across training
instances rather than a fluke of one seed. For each seed it calls
`train_olm(seed=seed, ...)` and `evaluate_displacement(..., seed=seed, ...)`,
and writes everything under `out_dir/seed_{seed}/`:

- `model.pt` — `{"ec", "cb", "history"}`
- `training_history.npz` — `train_mse`/`val_mse` per epoch (same arrays as
  `history`, saved separately so loss curves can be loaded without
  deserializing model weights) — and `training_curve.png` for a quick look.
- `displacement_probe.png`
- `displacement_probe_results.npz` — the dict `evaluate_displacement` returns
  (`mean_moved_displaced`, `mean_moved_control`, `mean_other_displaced`, and
  the corresponding `std_*` arrays, plus `t_probe`), so per-seed results can be
  reloaded and aggregated without retraining.

```python
from motor_structure_learning.experiments._olm_task_impl import train_olm_multi_seed

results = train_olm_multi_seed(
    seeds=[0, 1, 2, 3, 4],
    out_dir="checkpoints/olm_multi_seed",
    train_kwargs=dict(epochs=100),   # applied to every seed
    eval_kwargs=dict(n_pairs=200),   # applied to every seed
)
```

`train_kwargs`/`eval_kwargs` are applied identically to every seed — if you
override `seq_len` in one, override it in the other too, since
`evaluate_displacement`'s default `seq_len` won't otherwise match what the
model was trained on. Aggregating across the saved per-seed results (e.g.
mean/CI of the displacement spike across seeds) is left to the caller.

To regenerate just the probe figure from an existing checkpoint (no
retraining):

```python
import torch
from motor_structure_learning.experiments._olm_task_impl import (
    EntorhinalCortex, OLMForwardModel, evaluate_displacement,
    TRANS_DIM, STATE_DIM, LATENT_DIM, CB_HIDDEN, ACTION_DIM, CKPT_PATH,
)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
ec = EntorhinalCortex(input_dim=TRANS_DIM, hidden_dim=256, latent_dim=LATENT_DIM)
cb = OLMForwardModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN)
ec.load_state_dict(ckpt["ec"])
cb.load_state_dict(ckpt["cb"])
evaluate_displacement(ec, cb, device="cpu")
```

## Known limitations / follow-ups

- **No HC (hippocampus) in this first pass.** The circuit is EC→CB only,
  matching the base task. Wiring in `Hippocampus` (`core/hc.py`) as a slower
  trace over EC's latent stream, and comparing with/without HC, is the natural
  next step for actually testing the *hippocampal* contribution to spatial
  memory (vs. just EC/CB).
- Fixed at exactly 2 objects and a `POS_BOUND = 5.0` bounded region; both are
  constants, not derived from first principles — untested at other scales.
- No slip/force-field context perturbation (unlike the base task's
  `simulate_state_action_and_next_state`) — dropped deliberately since OLM
  isn't testing context/force-field inference.
- `main()` has no CLI flags; see the Python-API usage above for
  customization.
