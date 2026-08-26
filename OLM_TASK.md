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
| `motor_structure_learning/experiments/_olm_task_impl.py` | Implementation: sequence generator, model, training loop, probes, evaluation. |
| `motor_structure_learning/experiments/olm_task.py` | Thin package wrapper (`main()` → impl). |
| `olm_task.py` (repo root) | Backward-compatible root-level shim. |
| `motor_structure_learning/experiments/_olm_aggregate_eval_impl.py` | Multi-seed aggregation of the three probes' preference indices (and, optionally, causal-noise sweeps). |
| `motor_structure_learning/experiments/olm_aggregate_eval.py`, `olm_aggregate_eval.py` (repo root) | Wrapper/shim for the above, matching the `olm_task.py` convention. |
| `test_olm_objects.py` (repo root) | Sanity checks for polygon construction (`sample_polygon_object`) — vertex counts, per-shape/size area consistency, in-bounds, visualization. |
| `compute_olm_feature_variance.py` (repo root) | Empirically measures per-feature variance/std of the object representation over ordinary random-walk data — source of the `FEATURE_STD` normalization constant. |
| `test_olm_probes.py` (repo root) | Sanity checks for the probe generators — shapes/condition counts, control vs. manipulated identical-before/diverge-at `t_probe`, per-episode trajectory randomization, seed reproducibility. |

No changes were made to `motor_structure_learning/core/*`. `EntorhinalCortex`
is reused unmodified (its `input_dim` is just instantiated wider). The
cerebellar forward model is **not** `CerebellumForwardModelContinuous`
(its output layer is hardcoded to 2 dims) — instead `_olm_task_impl.py`
defines a local `OLMForwardModel` with the same 2-hidden-layer ReLU MLP shape
but a configurable output dimension.

## Representation

- Agent position: `(x, y)`, continuous, bounded within `[-POS_BOUND, POS_BOUND]`
  per axis (default `POS_BOUND = 5.0`, i.e. a 10×10 region centered at the
  origin). Positions are clipped to this range after each step. **Not part of
  the state** (see below) — it's tracked internally to generate the episode
  and compute egocentric object features, but never handed to the network
  directly; if the network needs it at all, it has to be inferred from
  integrating the action history.
- Actions are continuous: a turning direction/rate and a magnitude (uniform
  over `[0, MAX_STEP]`, default `MAX_STEP = 1.0`) — see "Heading" below for
  how the direction component is generated — applied as a displacement
  vector. There's no fixed action set (the earlier 4-cardinal-direction
  scheme was replaced with this).
- Heading: the *sampled direction* of the last action (not the resulting
  displacement, so it stays well-defined even for near-zero-magnitude
  actions), defaulting to a uniformly random angle before the first action.
  Serves as the agent's "camera direction" for the field-of-view object
  features below. Heading is **not** redrawn independently each step —
  `_sample_actions` instead evolves a turning rate (angular velocity)
  as a discrete-time Ornstein-Uhlenbeck process (`HEADING_OU_THETA`
  mean-reversion, `HEADING_OU_SIGMA` diffusion, exact discretization at
  `HEADING_DT` per step) and integrates it into a heading angle — i.e. the
  action's direction component is "which way to turn and how persistently,"
  not an unrelated fresh direction each frame, and the agent's turning has
  inertia the same way a real moving animal's gaze/orientation does. There
  is no absolute preferred heading (the turning *rate* mean-reverts to 0, not
  the angle itself) — over a long episode heading still sweeps the full
  circle, it just does so smoothly rather than teleporting. Measured: mean
  cos of the angle between consecutive headings ≈ 0.57 with the current
  `HEADING_OU_THETA = 1.0`/`HEADING_OU_SIGMA = 1.5` (vs. ≈ 0.014,
  statistically indistinguishable from 0, under the earlier fully
  independent-per-step design — see below).

  **This replaced an earlier design** where each timestep's direction was
  drawn fresh, independently and uniformly over `[0, 2π)` — a "teleporting
  gaze" snapping to a new random direction every frame, rather than a real
  moving animal's smoothly-turning one. That was known to be the direct
  cause of fast step-to-step jumps in the object features' direction
  component (measured on an earlier `[mean(dev), std(dev)]` version of the
  representation: ~49% of timesteps in ordinary training data had
  `|Δmean(dev)| > 1.0`, itself close to `mean(dev)`'s own overall std), but
  was *kept* for a long time regardless: retraining a model on that
  representation and inspecting predicted-vs-actual on a real trial
  (`plot_trial_prediction`) showed the network tracked the jumps almost
  exactly, since the heading driving each jump is exactly known to CB via
  the action it's given as input for that step (not exogenous noise from the
  model's perspective, just fast-varying but fully-recoverable) — so the
  independent-per-step design was judged not to corrupt training and was
  left as-is. That earlier justification is still correct as far as it goes,
  but it only ever answered "does this break training," not "is this the
  right thing to model" — the switch to a correlated OU turning rate here is
  about ecological plausibility for the object-location-memory analogy (the
  task's own framing leans on behavior "analogous to preferential
  exploration in real object-location-memory experiments"), not a response
  to any training failure. It's also orthogonal to a separate, independently
  discovered fact about this representation: the agent's absolute position
  is unrecoverable from `(state, action)` alone regardless of how heading is
  generated, since position requires integrating the *entire* action history
  from episode start and EC's GRU can recover the single most recent
  heading directly from the action it's given at each step either way (see
  the multimodality investigation below) — switching heading's generation
  process doesn't change that.
- **Objects**: exactly 2 shown per episode, each a regular polygon defined by
  `shape ∈ {triangle, square, pentagon}` (3/4/5 vertices) × `size ∈ {small,
  medium, large}` — `SHAPE_VERTICES`/`SIZE_AREA`. `size` sets the polygon's
  *area* directly (0.5/2.0/4.5 — chosen to match what a square's area was
  under an earlier fixed-radius scheme, so the overall scale is unchanged),
  the same value for every shape — so "small" means the same overall size for
  a triangle, square, or pentagon. Radius is then solved per-shape
  (`_polygon_radius_for_area`) to hit that target area; holding radius or
  edge length constant instead would both leave area growing with vertex
  count at a fixed "size" (e.g. a fixed-edge-length pentagon has ~4x a
  triangle's area; even fixed-radius only narrows that to ~1.8x).
  This gives exactly `POLYGON_TYPES` = 9 canonical objects (`C(9, 2) = 36`
  possible pairs shown together), drawn without replacement each episode.
  `sample_polygon_object(rng, shape, size)` then samples a random center +
  rotation such that every vertex stays within `[-POS_BOUND, POS_BOUND]²`
  *regardless of rotation* — since a regular polygon's vertices all lie on a
  circle of the given radius centered on it, sampling the center from a
  shrunk `[-(POS_BOUND - radius), POS_BOUND - radius]²` box guarantees this
  with no rejection sampling needed (see `_polygon_vertices`). Objects are
  static for the whole episode (center/rotation fixed once sampled).
- **Object feature** has two parts, computed by entirely different
  mechanisms (`_visual_feature`) — an ambiguous, field-of-view-limited
  perceptual readout, and a given identity code:
  ```
  for each vertex v_i of the object:
    rel_i   = v_i - agent_pos
    phi_i   = atan2(rel_i.y, rel_i.x)                    # world-frame angle to v_i
    theta_h = atan2(heading.y, heading.x)                # camera direction
    dev_i   = wrap_to_pi(phi_i - theta_h)                # egocentric deviation
    visible_i = |dev_i| < APERTURE_RAD / 2               # see APERTURE_DEG (a tunable constant --
                                                          # rerun compute_olm_feature_variance.py and
                                                          # update FEATURE_STD if you change it)

  # 1st trigonometric (circular Fourier) moment of the visible dev_i -- VIEW-dependent:
  C1, S1 = mean(cos(dev_i)), mean(sin(dev_i))
  R1          = |C1, S1|                    # resultant length ("concentration")
  visual_size = sqrt(-2 ln R1)              # DISTANCE: circular std -- spread/apparent size

  perceptual = [C1/R1, S1/R1, visual_size] / FEATURE_STD    # PERCEPTUAL_FEATURE_DIM = 3
  identity   = SHAPE_CODES[shape]                            # SHAPE_CODE_DIM = 8, fixed, given

  feature = concat(perceptual, identity)     # OBJ_FEATURE_DIM = 11
            (all zero if no vertices are visible)
  ```
  This factors into three orthogonal kinds of information, each handled by
  a different mechanism: **distance** (`visual_size`), **view/direction**
  (`cos(rel_angle)`, `sin(rel_angle)`), and **identity** (`SHAPE_CODES`, a
  fixed constant per shape, not derived from geometry or viewpoint at all).

  `cos(rel_angle)`/`sin(rel_angle)` (`= C1/R1, S1/R1`, i.e. the unit vector
  along the 1st moment's resultant direction) encode the object's egocentric
  bearing as a 2D unit vector instead of a raw angle — deliberately avoiding
  the `(-π, π]` wraparound discontinuity that a scalar bearing has (see the
  circular-statistics discussion below).

  **Shape identity is a given input, not something perceptually inferred —
  a deliberate assumption, not the original design.** An earlier version
  derived shape identity geometrically instead: `compactness` (isoperimetric
  quotient `4π·Area/Perimeter²`) and `vertical_asymmetry` (the lowest
  nonvanishing odd trigonometric moment of the vertex angles around their
  centroid — a triangle/pentagon-specific statistic, since even-vertex-count
  shapes have no nonvanishing odd harmonic by their exact 180°-point
  symmetry), both pure per-shape geometric constants computed directly from
  the polygon's own true vertex coordinates, independent of the agent's
  position or viewpoint. That version modeled shape recognition as something
  the network had to earn from raw geometry every time. The current version
  instead assumes an upstream visual system already resolves shape identity
  (analogous to ventral-stream object recognition) and hands it to the
  network as `SHAPE_CODES[shape]` — a fixed, arbitrary vector drawn once
  from a standard normal per shape (`SHAPE_NAMES`, a dedicated seed
  independent of any training/episode seed) and never re-derived. This was
  motivated by the ORM (shape/size-change) preference-index probes (see
  below) behaving largely independently of CB — suggesting identity
  resolution doesn't have to run through CB's forward-model pathway at all,
  the same way real shape/object recognition plausibly happens upstream of
  (and is much faster than) fine motor prediction.

  `SHAPE_CODES` is deliberately keyed by shape only, **not** by
  `(shape, size)` — a given shape's code is identical across all 3 sizes, so
  this channel carries *no size information whatsoever*. Size stays purely
  in the ambiguous, distance-confounded `visual_size` cue below: real vision
  doesn't resolve true size independent of viewing distance either, so if
  shape identity is a given, size still shouldn't be.

  **`visual_size` is its own coordinate, not a gain — after two intermediate
  designs that gained other features by it.** Every coordinate of the
  perceptual sub-vector is independent — none of them scales or gains any
  other. Two earlier versions (back when the shape channel was still the
  geometric `compactness`/`vertical_asymmetry` pair described above, not
  `SHAPE_CODES`) used a multiplicative-gain design instead, both
  reconsidered for concrete reasons that still motivate the current,
  ungained design:
  - **v1**: `visual_size` gained the *entire* feature vector (direction
    included), on a gain-control / population-vector-magnitude-as-confidence
    analogy (a weak, distant stimulus gives a weak, hard-to-localize signal
    overall). Reconsidered because it doesn't match how vision actually
    degrades with distance: you generally still know *where* a faint/distant
    object is even once you can no longer tell *what* it is or *how big* —
    bearing stays reliable long after apparent shape/size resolution breaks
    down.
  - **v2**: `visual_size` gained only the (then-geometric) shape pair,
    leaving direction ungained. This surfaced a real problem, caught
    empirically: on the shape-change probe (swap an object's shape,
    size/location held fixed), the manipulated object's true feature change
    was often tiny -- median squared difference ~11x *smaller* than a
    location-change probe's, and negligible (`<0.01`) in ~43% of episodes --
    because whenever the agent happened to be far from the object at the
    manipulation instant (common on an unconstrained random walk), the small
    `visual_size` gain shrank an already-modest shape-pair difference even
    further.
  - **v3 (current)**: making every coordinate independent and ungained fixed
    this directly -- rerunning the same shape-change-probe diagnostic
    afterward, the median squared difference went from 0.045 (v2, gained) to
    **2.65** (v3, ungained at the time — still true qualitatively with the
    current `SHAPE_CODES` channel, whose per-shape separation doesn't shrink
    with distance at all since it isn't derived from viewing geometry in the
    first place), with **0%** of episodes negligible (down from 43%).

  `visual_size` carries (confounded) information about apparent size — a
  `mean(1/distance)` term was deliberately dropped even though it would
  disambiguate size from distance, specifically *because* real vision has
  this size/distance ambiguity (a big object far away and a small one close
  up can produce the same angular spread) and `visual_size` is meant to
  stand in for it. There is no separate, unambiguous distance feature — just
  this one size/distance-confounded channel, and no location cue anywhere in
  the state either (see "Agent position" above) — only identity is given for
  free; everything else the network has to work out from ambiguous
  perceptual cues plus its own accumulated context, the same way recognizing
  *where* something is (unlike *what* it is) is not a given fact even when
  you already know what you're looking at.

  **Why circular statistics for direction.**
  `plot_feature_space_trajectories(seq_len, center=PROBE_CENTER,
  rotation=0.4, object_types=POLYGON_TYPES, seed=0)` is an illustrative (not
  evaluative) diagnostic for this feature: it holds ONE shared random-walk
  agent trajectory and object location/rotation fixed, swaps in each of the
  9 canonical `POLYGON_TYPES` in turn, and plots two views of the resulting
  per-object-type feature trajectories — each feature vs. time
  (`figs/olm_feature_space_trajectories.png`, plus a shared agent-to-object
  distance panel for context) and the same trajectories as paths through a
  2D slice of feature space directly, default `(shape_code_0,
  shape_code_1)` (`figs/olm_feature_space_state_space.png`, `state_dims`
  parameter) — since `SHAPE_CODES` doesn't depend on rotation, position, or
  size at all, this defaults to a static scatter (one fixed point per shape,
  all 3 sizes coinciding) rather than a moving trajectory; pick a
  `state_dims` including `visual_size` (index 2) or a direction dim (0/1)
  instead to see genuine time-varying structure.

  This diagnostic (run against an earlier, simpler `[mean(dev), std(dev)]`
  version of the representation, well before `SHAPE_CODES` existed) is what
  surfaced the finding that motivated separating direction from shape in the
  first place: at everyday range, a plain bearing average (`mean(dev)`)
  nearly collapses onto the same value for all 9 object types — it's close
  to a pure "look direction" signal, largely invariant to shape/size.
  Measured precisely (fix the agent at a range of distances from the object,
  sweep 200 random headings x all 9 object types, decompose feature variance
  into an object-identity component and a heading component): the
  identity-explained fraction of `mean(dev)`'s variance drops from ~37% at
  distance 0.3 to ~1% at distance 8. This is real and not fixable without
  changing what the feature measures -- a bearing to a small/distant object
  is fundamentally a directional signal, not an identity one (a large part
  of why identity was eventually given outright via `SHAPE_CODES` rather
  than left to be inferred from geometry at all).

  A related finding motivated using circular (not plain wrapped-angle)
  statistics for direction specifically: the same variance-decomposition
  test applied to a *plain* `std(dev)` (`.std()` over individually
  `(-π,π]`-wrapped `dev_i`, rather than a proper circular statistic) showed
  only 19-43% identity-explained variance, at every distance tested -- the
  rest was a **heading-dependent artifact**: whenever the object sits near
  directly behind the agent, some vertices' `dev_i` wrap past ±π while
  others don't, so a linear std over that mix reads as huge spread that has
  nothing to do with the object's actual size. The circular version
  (`R1`/`sqrt(-2 ln R1)` above) is invariant to heading by construction --
  rotating a whole set of vectors together (i.e. changing heading) doesn't
  change their spread -- and measured at 100% identity-explained variance,
  at every distance tested.

  The direction components (`cos`/`sin(rel_angle)`) still change step to
  step -- this is real, not an artifact: the "which way am I currently
  facing" component genuinely evolves as the agent turns (see "Heading"
  above). Investigated and confirmed, by retraining a model and inspecting
  predicted-vs-actual on a real trial (`plot_trial_prediction`), that the
  network tracks it closely (it's exactly recoverable given the next
  action, which CB always receives) — true both under the current
  correlated-heading design and the earlier fully independent-per-step one
  it replaced.

  A `count` feature (number of visible vertices) was tried early on and
  **dropped**: it's only informative when the aperture is narrow enough to
  gate visibility in a heading-dependent way. Once `APERTURE_DEG` was widened
  toward omnidirectional (see below), `count` became mostly just each
  object's own vertex count (3/4/5) — near heading-independent, so it stopped
  carrying much signal and mostly just added noise (and, while it was still
  in the representation, a very large one — see the normalization note
  below). `_visible_deviation` (factored out of `_visual_feature`) still
  computes the visibility mask internally and is available for anything
  (e.g. tests) that needs to check visibility directly without it being part
  of the network's actual input/target.
- **Feature normalization** (`FEATURE_STD`): only the `PERCEPTUAL_FEATURE_DIM
  = 3` perceptual columns (`cos(rel_angle)`, `sin(rel_angle)`, `visual_size`)
  are empirically normalized — `SHAPE_CODES` is a fixed draw from a standard
  normal, already ~unit variance per element by construction, and needs no
  additional scaling. The raw perceptual feature values have different
  natural scales (measured empirically, see `compute_olm_feature_variance.py`)
  — this mattered most dramatically for the now-dropped `count`, whose raw
  variance was ~78x `mean(dev)`'s and ~450x `std(dev)`'s at a narrow (60°)
  aperture, back when the representation was still `[mean(dev), std(dev)]`,
  and is why the imbalance was caught in the first place. Since MSE weights
  each dimension by its raw squared scale, training on the unnormalized
  representation would have the loss (and `_slot_error`/the preference
  index) dominated by whichever feature has the largest raw scale, almost
  regardless of how well the other is learned. `_visual_feature` divides its
  3-dim perceptual output by `FEATURE_STD` (empirical per-feature std over
  ordinary random-walk data, pooled across both object slots since slot 0
  vs. 1 has no structural meaning outside the probes) before concatenating
  the unnormalized `SHAPE_CODES` — this fixes the loss weighting *and* the
  evaluation-metric weighting for the perceptual part in one place, rather
  than needing separate reweighting logic elsewhere. Rerun
  `compute_olm_feature_variance.py` and update `FEATURE_STD` if anything the
  perceptual feature distribution depends on changes (`APERTURE_DEG`,
  `SIZE_AREA`, `POS_BOUND`, `PROBE_*` layout, heading dynamics, etc) -- the
  script corrects for `generate_olm_sequences`'s output already being
  normalized by whatever `FEATURE_STD` currently happens to be set (by
  multiplying it back out before measuring), so it always measures the true
  raw distribution even when the existing constant is stale. Current values:
  `FEATURE_STD = [0.698915, 0.702626, 0.220102]`
  (`cos(rel_angle)`, `sin(rel_angle)`, `visual_size)`.
- **State vector** = `[object_0_feature, object_1_feature]` (22 dims with the
  defaults above — no agent position, see above). Objects are always "in the
  environment" — there is no occlusion/delay phase or presence flag;
  visibility is entirely a function of the (always-recomputed) FOV feature.

## Observation noise (infrastructure for a genuine POMDP, off by default)

`OBSERVATION_NOISE_STD` (default `0.0`, i.e. off) would add i.i.d. per-timestep
Gaussian noise to every object's PERCEPTUAL columns only (`cos(rel_angle)`,
`sin(rel_angle)`, `visual_size` — post-`FEATURE_STD` normalization, so this
value is directly a signal-to-noise ratio against the already-unit-variance
signal) — `SHAPE_CODES` stays exact either way, since identity remains a clean
given signal throughout this representation (see "Representation" above). Not
to be confused with the unrelated "Causal noise manipulation" below, which
perturbs a *trained* EC/CB's internal `z`/GRU state at evaluation time for a
causal-ablation analysis, not the environment's observations themselves.

**Why this was built, and why it's off.** Before this existed, every
observation was an exact (if lossy) deterministic function of the true world
state — nothing was ever noisy, only some things (chiefly the agent's own
position) were never revealed at all (see "Representation"'s discussion of
what EC's `z` needs to infer). That meant belief correction after a
surprising event was instantaneous: the very next observation, taken alone,
was always fully conclusive, so there was never a reason for the network to
keep being "wrong" once a fresh true reading arrived. Measured directly on a
trained model: the excess prediction error from a location-change
manipulation was 0.122 at `t_probe` and already down to 0.0045 (a ~27x drop)
one step later — a genuinely ~1-timestep-wide spike, confirmed to dilute
`evaluate_preference_indices`' reported effect size by ~7x when averaged over
a wide window (see below, and why `_fractional_preference_index`'s `window`
default changed to `3`).

The hypothesis was that this is a structurally expected consequence of never
having any actual observation *uncertainty* to resolve, and that adding real
noise would force EC's GRU to integrate evidence over multiple timesteps
(filtering, not just "remember the last action"), making a real displacement
overcome an accumulated prior gradually rather than snap instantly. Tested at
`std=0.5` (noise variance 0.25, signal still dominates a single reading) and
`std=1.5` (noise variance 2.25, noise-dominated) on trained models: **neither
produced any sustained decay** — still a one-timestep spike either way — and
the *higher* noise made the effect **weaker**, not stronger (the
location-change preference-index gap shrank monotonically with added noise:
0.059 at `std=0` → 0.041 at `std=0.5` → 0.010 at `std=1.5`, all at `window=1`).
That rules out "wrong noise magnitude" as the fix.

The likely underlying reason: `PROBE_LEFT`-to-`PROBE_CENTER` is a large jump
(2.5 units) relative to any per-reading noise tried here, and a
(near-)optimal Bayesian filter has little reason to show *sustained* inertia
against a single strongly-inconsistent reading — passive per-step
observation noise mainly creates ambiguity for *small* discrepancies (is this
noise, or a real change?), not for an unambiguous, large relocation like this
one. A momentary spike was accepted as the correct, expected model of an
instantaneous orienting response, rather than continuing to chase a
sustained one. If a sustained signal is wanted later, **intermittent
sampling/occlusion** (not yet implemented) is the more likely candidate
mechanism — it directly limits how much evidence can arrive per unit time
(confirming "did this change" needs multiple discrete looks spread over
time), which per-step noise on a fully-and-continuously-sampled signal does
not, regardless of magnitude.

The infrastructure itself (below) is fully wired, tested, and left available
for whoever wants to revisit this — just defaulted off rather than removed.

**Reproducibility across matched probe pairs.** `_sample_observation_noise`
draws one noise value per absolute timestep (not per row, not per
control/manipulated variant), precomputed once per episode from the same
`rng` used for everything else in that episode. Two consequences:

- A given true moment gets exactly one noisy reading, regardless of whether a
  row references it as that row's own `state_t` or the *previous* row's
  `state_next` (both are the same real timestep, just packed into adjacent
  rows of the transition tensor).
- `_build_probe_transitions` draws this noise array *once*, before building
  either the control or the manipulated variant, and both share it — this is
  what keeps the "control and manipulated transitions are byte-identical
  through the row before `t_probe`" invariant (`test_olm_probes.py`) true
  despite noisy observations, and means any post-`t_probe` difference between
  the pair reflects exactly the manipulation itself, not extra
  noise-driven variance between two independently-drawn noise realizations
  (a standard "common random numbers" variance-reduction technique for
  matched comparisons).

**API.** `generate_olm_sequences(..., observation_noise_std=OBSERVATION_NOISE_STD)`
and all three probe generators (`generate_probe_location_change_data` etc.,
via `_generate_probe_data`) accept `observation_noise_std`, defaulting to
`OBSERVATION_NOISE_STD` (currently `0.0`) so probe episodes always match
whatever ordinary training data is currently using, whether that's noise-free
(the default) or not. `compute_olm_feature_variance.py` passes `0.0`
explicitly regardless of the module default, since `FEATURE_STD` describes
the environment's own raw signal scale — a property that shouldn't shift
depending on how much observation noise happens to be layered on top of it.

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

## Closed-loop training (alternate architecture)

`train_olm_closed_loop` / `_closed_loop_forward` are a **different, incompatible
architecture** — not a variant of `train_olm` — built specifically so the
causal-noise manipulation could target "noise on the prediction-error signal
fed back to EC," which doesn't exist as a pathway in the main `train_olm`
model (there, `ec(trans)` only ever sees the raw transition; CB's output
never feeds back into EC's input). This mirrors the one existing precedent
for that pattern, `ECCBTrainer.replay_ec_sequence` in
`core/trainers/ec_cb.py`, but as full-sequence BPTT rather than that
trainer's detached single-step replay (OLM's episodes are short enough,
`SEQ_LEN` ~60, that unrolling and backpropagating through the whole episode
is tractable, and doesn't need a replay buffer or stored hidden states).

Per step `t`:

```
z_prev       = ec.readout(h)                  # stale context, from before this step
cb_pred_prev = cb(s_t, a_t, z_prev)            # prediction using stale context
pred_err_t   = cb_pred_prev - s_{t+1}          # fed-back error signal

h            = ec.gru([s_t, a_t, pred_err_t], h)   # EC's input this step (NOT [s_t, a_t, s_{t+1}])
z_t          = ec.readout(h)                  # fresh context, after seeing pred_err_t

s_pred       = cb(s_t, a_t, z_t)               # this is what the loss uses
loss        += MSE(s_pred, s_{t+1})
```

Two things worth calling out:

- EC's per-step input is `[s_t, a_t, pred_err_t]` — `pred_err_t` *replaces*
  the raw `s_{t+1}` block used by the non-closed-loop model, it isn't
  appended alongside it. This keeps the same total input width
  (`TRANS_DIM`) as `train_olm`'s EC, but EC now has to rely on its own
  prediction error as the informative signal about what happened, rather
  than being handed the ground truth directly. (The original
  `ECCBTrainer` design includes both the raw `s_{t+1}` *and* `pred_err_t`;
  we deliberately dropped the raw block here.)
- No time-shift: the stale-context mechanism is itself what prevents EC
  from trivially copying the answer (the prediction that generates
  `pred_err_t` is made *before* EC has seen the outcome), so this predicts
  `state_{t+1}` immediately rather than `state_{t+2}` like `train_olm` does.

Saved to its own checkpoint, `checkpoints/olm_task_closed_loop.pt` — doesn't
touch `train_olm`/`CKPT_PATH`. Considerably slower per epoch than `train_olm`:
`_closed_loop_forward` is a Python-level loop over `seq_len` steps (two CB
calls per step) rather than a single batched GRU call over the whole
sequence, so budget for that when choosing `n_train`/`seq_len`/`epochs`.

**Not yet done**: none of `evaluate_displacement`, `evaluate_preference_indices`,
or `evaluate_causal_noise` have been ported to this architecture — they all
call `_shifted_forward` and assume its shifted-by-one `(B, T-1, STATE_DIM)`
output shape, which doesn't apply here (`_closed_loop_forward` returns
`(B, T, STATE_DIM)`, no shift). Porting the probes/preference-index/causal-noise
evaluations to the closed-loop model, including deciding what `"cb_input"`-site
noise even means when there are now two CB calls per step (`z_prev` vs.
`z_t`), is a deliberate follow-up, not yet implemented.

## Probes: deterministic conditions, randomized trajectory

Unlike the base training data (random trajectories, random object placement),
the probes described in this section and the next two are **deterministic
over conditions** — shape/size pair, manipulated slot, and manipulation type
are all exhaustively enumerated, not sampled. The one thing that *is*
randomized is the agent's own trajectory during a probe episode, and
deliberately so — see below.

**Prescribed object slots** (`PROBE_LEFT = (-2.5, 0)`, `PROBE_RIGHT = (2.5,
0)`, `PROBE_CENTER = (0, 0)`, the displacement target) — chosen with headroom
for the largest possible object (triangle/large, r≈1.86) regardless of what
ends up there, so a shape/size swap never has to worry about whether the
replacement still fits (see "Representation" above).

**Trajectory: an ordinary, freshly-randomized random walk** (`_random_walk`,
same `_sample_actions` mechanism as `generate_olm_sequences`) — every probe
episode gets its own independent walk, not a shared/fixed one. This replaced
an earlier scripted "circle around a fixed vantage point, then sweep gaze
across LEFT/CENTER/RIGHT" trajectory that was built to guarantee visibility
under a narrow field of view. Once `APERTURE_DEG` widened to 360 (see
"Representation"), visibility became unconditional — every vertex
contributes to the feature regardless of heading — so that guarantee came for
free and the scripted trajectory's only remaining effect turned out to be a
**confound**: `CENTER` sits exactly on the scripted path's vantage-point
symmetry axis, so objects there were measurably easier to predict than
objects at `LEFT`/`RIGHT` regardless of any real detection (confirmed
empirically: an object fixed at `CENTER` had ~42% smaller mean squared
feature magnitude than one at `LEFT`/`RIGHT`, averaged over the scripted
post-`t_probe` sweep) — an object relocated *to* `CENTER` looked like an
improvement almost no matter what. Switching to a genuine, freshly-randomized
random walk removes this: heading is uncorrelated with any particular
object's true position, so no location is systematically favored. The walk
must be freshly randomized *per episode* rather than a single fixed instance
reused everywhere — a fixed walk would just bake in whatever asymmetries that
one particular walk happens to have, rather than averaging them out across
episodes. (Confirmed: re-running `evaluate_displacement` after this change
removed the spurious pre-existing dip in the relocated object's error at
`t_probe` entirely.)

**Deterministic condition enumeration** (`_enumerate_probe_pairs`): every
ordered assignment of 2 distinct `POLYGON_TYPES` to `(left, right)` (9×8=72),
crossed with which slot is manipulated (×2) = 144 combinations. Rotation
can't be enumerated (continuous), so it's swept over 3 fixed, evenly-spaced
values (`ROTATION_SWEEP = (0, 2π/3, 4π/3)`), shared by both objects within a
given replicate rather than varied independently.

**Three probe types**, all built on the same `_build_probe_transitions`
(paired control/manipulated episodes sharing one random walk, identical up to
`t_probe`) and returning `(control_trans, manipulated_trans,
manipulated_idxs, t_probe)` — `manipulated_idxs` is `0` (left) or `1` (right):

- `generate_probe_location_change_data` (OLM): the manipulated slot's object
  relocates to `PROBE_CENTER` at `t_probe`; shape, size, and rotation
  unchanged. 144 combos × 3 rotations = **432 episodes**.
- `generate_probe_shape_change_data` (ORM-shape): the manipulated slot's
  *shape* changes to each of the other 2 shapes, holding size/center/rotation
  fixed. 144 × 2 × 3 = **864 episodes**.
- `generate_probe_size_change_data` (ORM-size): as above but *size* changes
  (2 other sizes), holding shape/center/rotation fixed. **864 episodes**.

Each generator (and `_generate_probe_data`, their shared driver) takes a
`seed` — one `rng` is created from it and threaded through every episode
*without* being reset per episode, so consecutive episodes get different
walks (same seed across a full call ⇒ reproducible overall result; different
seed ⇒ a genuinely different set of walks). `t_probe` defaults to
`seq_len // 2` (a plain fixed fraction now, not derived from any scripted
phase structure) and can be overridden explicitly.

In every case only the one named factor differs between the pre- and
post-`t_probe` object — verified (`test_olm_probes.py`) that control and
manipulated transitions are byte-identical through the row before `t_probe`
and diverge exactly starting there, that different episodes get different
trajectories, and that a given `seed` reproduces exactly while a different
one doesn't.

`evaluate_displacement(ec, cb, seq_len=SEQ_LEN, device=None, fig_path=FIG_PATH)`
runs `generate_probe_location_change_data`, then the trained EC/CB (no
gradient, `.eval()`) via the same time-shifted procedure as training, and
computes per-timestep squared error restricted to each object's feature slot
(`_slot_error`). Plots three curves — (moved slot, manipulated), (same slot,
control), (other slot, manipulated) — vs. timestep, with a vertical line at
`t_probe`, saved to `fig_path` (default `figs/olm_displacement_probe.png`).

Note on indexing: the array at position `j` (0-indexed) holds the error for
predicting `state(j+2)`, because of the time-shift in `_shifted_forward` — the
plotting code accounts for this (`t_axis = np.arange(2, seq_len + 1)`) so the
x-axis correctly reflects which timestep's state is being predicted. Get this
wrong and the plotted spike appears one step before `t_probe` instead of on it
(this was caught and fixed during development, back when the base task's
representation was still `[agent_xy, object features]`).

## Preference index across the three probes

Location-change (OLM) is one half of the classic object-memory dissociation;
shape-change and size-change together are the other half (identity novelty —
new appearance, familiar location), deliberately split into two rather than
one combined "identity changed" condition, since shape and size are now
independent factors (see "Representation") and it's worth being able to ask
whether the network discriminates each independently.

`evaluate_preference_indices(ec, cb, seq_len=SEQ_LEN, window=3,
device=None, fig_paths=None)` runs all three probes
(`generate_probe_location_change_data`/`_shape_change_data`/`_size_change_data`)
through the trained model and computes a **fractional preference index** per
episode:
```
e_slot = mean squared error for that slot, averaged over predicted-timesteps
         [t_probe, t_probe + window)   (window=None -> average to the end)
preference_index = e_changed / (e_obj0 + e_obj1)
```
`window` defaults to `3`, not `None` — measured directly on a trained model
that the excess prediction error from a manipulation is a genuinely
~1-timestep-wide spike (CB re-observes the fully truthful current state
every step, with no occlusion/delay, so there's nothing left to keep
predicting wrong once the new state is directly visible again): excess error
dropped from 0.122 at `t_probe` to 0.0045 one step later, indistinguishable
from noise by `t_probe + 2`. Averaging over the old `window=None` default
(end of episode, ~30 steps) diluted that one real timestep of signal with
~29 timesteps of noise — measured to weaken the location-change
control-vs-changed gap by ~7x (0.059 at `window=1` down to 0.008 at
`window=None`). Pass `window=None` explicitly to restore the old
end-of-episode averaging (e.g. to check whether a probe's effect turns out
to be more sustained than location-change's was).
i.e. the fraction of total error attributable to whichever slot was actually
manipulated (`manipulated_idxs`, `0`=left/`1`=right) — the same style as a
behavioral discrimination index (time at novel / total time). ~0.5 = no
preference (chance), closer to 1 = more error attributed to the changed/novel
slot ("preference for novel"), closer to 0 = the opposite ("preference for
familiar").

This was originally a softmax over the raw errors, but plain softmax is
sensitive to the *absolute* gap between the two errors, not their ratio, so as
training shrinks the overall error scale the index collapsed toward 0.5 even
when the proportional difference between slots stayed large (e.g.
`softmax([0.001, 0.002])` ≈ `[0.4997, 0.5003]` despite a 2x ratio). The
fraction is automatically scale-invariant — dividing by the pair's own total
already rescales away the model's absolute error magnitude, no temperature
hyperparameter needed.

`_slot_error` is a plain full-feature squared error (mean over a slot's
`OBJ_FEATURE_DIM` columns) — there used to be a `feature_group` parameter
to restrict to an "identity" or "location" sub-group (back when a slot's
columns were `[identity (8), distance (1)]`), needed because mixing both
groups diluted whichever one a given manipulation didn't touch. A clean
identity/perceptual partition exists again now that shape identity lives in
its own dedicated `SHAPE_CODES` columns (unlike the intervening
polygon/FOV-with-geometric-shape-features design, where `[cos(rel_angle),
sin(rel_angle), compactness, vertical_asymmetry]` was a single ambiguous
perceptual readout with no clean identity/location partition at all) —
`_slot_error` simply hasn't been changed to exploit it. Worth being aware of
if comparing preference indices *across* probe types: a shape-change
manipulation now flips the entire, large, unambiguous `SHAPE_CODES` sub-vector
(expected squared jump ~2 x `SHAPE_CODE_DIM` between two ~independent
standard-normal draws), while location/size changes only perturb the much
smaller-scale 3-dim perceptual sub-vector — so a full-feature error may not
put the three probes' signals on comparable footing anymore the way it did
under the old, uniformly-ambiguous representation. Within a single probe
type (control vs. changed), the comparison is still apples-to-apples — every
probe (location/shape/size-change) still perturbs the same columns for that
probe (just via different underlying causes) — so a plain full-feature error
remains the right, unconfounded control-vs-changed comparison *within* a
probe type, even though the cross-probe-type magnitude comparison above now
warrants more care than it did under the old, uniformly-ambiguous
representation.

The **control condition is exactly 0.5 by construction**, not just
approximately by chance: `_enumerate_probe_pairs` includes both
`manipulated_slot="left"` and `="right"` for every `(left_type, right_type,
rotation)` triple, and a *control* episode's trans data doesn't depend on
which slot is nominally "manipulated" — so those two entries share identical
`(e0, e1)` with `manipulated_idxs` flipped (`0` vs `1`), and their fractions
(`e0/(e0+e1)` vs `e1/(e0+e1) = 1 - e0/(e0+e1)`) average to exactly `0.5`. A
nice built-in sanity check that requires no statistical luck.

Produces, per probe (`"location"`/`"shape"`/`"size"`), a `pref_{name}_control`
and `pref_{name}_change` array plus `{name}_t_probe`, plotted as its own box
comparison (chance-level `0.5` reference line, y-axis fixed to `[0, 1]`) to
`fig_paths[name]` if given, else the module default
(`LOCATION_PREF_FIG_PATH`/`SHAPE_PREF_FIG_PATH`/`SIZE_PREF_FIG_PATH`). Only
valid for `N_OBJECTS == 2` (hardcodes slots 0/1), same constraint as
`evaluate_displacement`'s `other_idxs = 1 - moved_idxs`. Standalone — like
`evaluate_displacement`, **not** wired into `train_olm_multi_seed`; call it
directly (and `np.savez` the result yourself) from whatever aggregator you
build across seeds/conditions (see `_olm_aggregate_eval_impl.py`, which does
exactly this across the three probes).

## Causal noise manipulation

A causal complement to the (purely observational) preference-index probe:
instead of just reading out prediction error from a trained, unperturbed
model, inject noise into the circuit and see how much that degrades the
novelty-detection effect. Two distinct injection sites, testing two different
hypotheses about where the identity-location binding lives:

- **`"gru"`**: noise added to EC's GRU hidden state at every timestep during
  an explicit **stimulation period** `[stim_start, stim_end)` — compounding
  forward through the recurrence, i.e. corrupting the memory trace itself, so
  it stays degraded for all subsequent timesteps (not just the ones being
  stimulated). Tests whether an intact recurrent memory *substrate* is
  necessary for the effect.
- **`"cb_input"`**: noise added only to `z` (EC's context readout) at the
  timesteps CB actually consumes, during the same stimulation period — EC's
  own hidden state and recurrence are left untouched. Tests whether a clean
  *communication channel* to CB is necessary, independent of whether EC's
  internal memory is intact.

Both use `alpha` as a **literal noise magnitude — the expected L2 norm of the
injected perturbation vector**, matched across sites regardless of
dimensionality: per-element std = `alpha / sqrt(dim)`, where `dim` is 256 for
EC's GRU hidden state and 32 for its `z = tanh(...)` readout (by default).
This went through two fixes, both worth understanding since either one alone
would still leave the comparison confounded:

1. *Not* scaled by the perturbed tensor's own std (an earlier version did
   this) — that meant the same `alpha` didn't inject the same amount of
   noise at each site to begin with, before even accounting for the
   `"gru"`-site noise additionally passing through EC's readout (a
   dimensionality-reducing, nonlinear transform) and compounding through the
   recurrence before ever reaching CB.
2. *Not* just a fixed per-element std either — a fixed std still isn't
   matched across sites, because the expected norm of an isotropic Gaussian
   vector scales as `sqrt(dim)`: the same per-element std over 256 dims (the
   hidden state) has `sqrt(256/32) = sqrt(8) ≈ 2.83x` more total magnitude
   than over 32 dims (the readout). Dividing by `sqrt(dim)` fixes this —
   expected total norm is `alpha` regardless of which tensor is perturbed.

With both fixed, a given `alpha` injects the same overall amount of noise at
both sites, so any difference between the two dose-response curves reflects
a real circuit difference (readout transform + recurrent compounding for
`"gru"` vs. one-shot direct injection for `"cb_input"`), not an accidental
mismatch in how much noise was actually added.

There is no default phase structure to anchor the stimulation period to
(objects are always visible, there's no encode/delay/probe split) — you pick
`stim_start`/`stim_end` explicitly for whatever window you want to test (e.g.
the interval before `t_probe` during which the identity-location binding must
be held).

- `_shifted_forward_noisy(ec, cb, trans, noise_site, alpha, stim_start,
  stim_end)` is the noisy analog of `_shifted_forward`. `noise_site="gru"`
  requires manually unrolling EC's GRU step by step (a single `ec(trans)`
  call can't be perturbed mid-unroll); `alpha=0` is exactly equivalent to the
  clean `_shifted_forward` regardless of `noise_site` (verified: bit-for-bit
  identical predictions in testing).
- `evaluate_causal_noise(ec, cb, alphas, stim_start, stim_end, seq_len=SEQ_LEN,
  window=3, device=None, fig_paths=None)` sweeps both noise sites across
  the given `alphas`, reusing the same three deterministic probes as
  `evaluate_preference_indices` (location/shape/size-change) — for each
  `(site, alpha)` it recomputes the fractional preference index for every
  probe's control/change pair. The control condition is included as a sanity
  check: it should stay near chance (0.5) even at high noise, since nothing
  was actually manipulated in those episodes — if it drifts, the noise itself
  is introducing a spurious bias rather than selectively disrupting the real
  novelty signal.
- Produces one dose-response line plot per probe (y-axis fixed to `[0, 1]`),
  each with 4 lines (`gru`/`cb_input` × control/changed) plus the chance-level
  reference, saved to `fig_paths[name]` if given, else the module default
  (`LOCATION_CAUSAL_FIG_PATH`/`SHAPE_CAUSAL_FIG_PATH`/`SIZE_CAUSAL_FIG_PATH`).
  Returns a flat results dict (`{site}_{name}_{control,change}_mean`/`_sem`
  for each of the 3 probes, plus `alphas`/`stim_start`/`stim_end`/`window` and
  each probe's own `{name}_t_probe`) ready for `np.savez`.
- Standalone, same as `evaluate_preference_indices` — not wired into
  `train_olm_multi_seed`.

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
evaluate_displacement(ec, cb, seq_len=80, device=device)
```

Key constants (top of `_olm_task_impl.py`): `POS_BOUND` (position bound),
`MAX_STEP` (max action magnitude), `SEQ_LEN`, `N_TRAIN`, `N_VAL`,
`BATCH_SIZE`, `EPOCHS`, `LR`, `LATENT_DIM`, `CB_HIDDEN`, `N_OBJECTS`,
`SHAPE_VERTICES`/`SIZE_AREA` (polygon construction), `APERTURE_DEG` (FOV),
`PROBE_LEFT`/`PROBE_RIGHT`/`PROBE_CENTER` (probe object slots),
`ROTATION_SWEEP` (probe rotation values).

### Training multiple seeds

`train_olm_multi_seed(seeds, out_dir=MULTI_SEED_DIR, train_kwargs=None,
eval_kwargs=None)` trains and evaluates one model per seed, useful for
checking whether the displacement-detection effect is robust across training
instances rather than a fluke of one seed. For each seed it calls
`train_olm(seed=seed, ...)` and `evaluate_displacement(...)` (the probe itself
is fully deterministic now, so only the trained model varies by seed, not the
probe episodes), and writes everything under `out_dir/seed_{seed}/`:

- `model.pt` — `{"ec", "cb", "history"}`
- `training_history.npz` — `train_mse`/`val_mse` per epoch (same arrays as
  `history`, saved separately so loss curves can be loaded without
  deserializing model weights) — and `training_curve.png` for a quick look.
- `trial_diagnostic.png` — `plot_trial_prediction` on one training trial with
  the final trained model (predicted vs. actual per-object feature
  trajectory); `train_olm_multi_seed` always points `train_olm`'s
  `trial_fig_path` here per seed, overriding whatever's in `train_kwargs`, so
  each seed gets its own file instead of all seeds overwriting the same
  shared default path.
- `displacement_probe.png`
- `displacement_probe_results.npz` — the dict `evaluate_displacement` returns
  (`mean_moved_manipulated`, `mean_moved_control`, `mean_other_manipulated`,
  the corresponding `std_*` arrays, plus `t_probe`/`n_conditions`), so
  per-seed results can be reloaded and aggregated without retraining.

```python
from motor_structure_learning.experiments._olm_task_impl import train_olm_multi_seed

results = train_olm_multi_seed(
    seeds=[0, 1, 2, 3, 4],
    out_dir="checkpoints/olm_multi_seed",
    train_kwargs=dict(epochs=100),   # applied to every seed
    eval_kwargs=dict(seq_len=80),    # applied to every seed
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
- Exactly 2 objects shown simultaneously (`N_OBJECTS`, architecturally fixed —
  see the note in `evaluate_displacement` about `other_idxs = 1 - moved_idxs`
  only being valid for 2) and a `POS_BOUND = 5.0` bounded region; neither is
  derived from first principles — untested at other scales.
- No slip/force-field context perturbation (unlike the base task's
  `simulate_state_action_and_next_state`) — dropped deliberately since OLM
  isn't testing context/force-field inference.
- `main()` has no CLI flags; see the Python-API usage above for
  customization.
- The probes are deterministic over shape/size/manipulated-slot conditions
  (see "Probes: deterministic conditions, randomized trajectory"), with the
  agent's trajectory itself a freshly-randomized random walk per episode —
  PROBE_LEFT/RIGHT/CENTER are still fixed constants tuned so the largest
  possible object fits regardless of what's placed there, but aren't derived
  from first principles beyond that. Rerun `compute_olm_feature_variance.py`
  and update `FEATURE_STD` whenever `APERTURE_DEG` (or `SIZE_AREA`/
  `POS_BOUND`/`PROBE_*` layout) changes — the feature distribution, and hence
  the correct normalization, depends on it directly (e.g. a much wider
  aperture brings far more vertices into view at once, which shifts the
  feature's spread dramatically). The script corrects for
  `generate_olm_sequences`'s output already being normalized by whatever
  `FEATURE_STD` currently happens to be set (by multiplying it back out
  before measuring), so it always measures the true raw distribution even
  when the existing constant is stale.
- 432+864+864 = 2160 probe episodes are generated fresh (not cached) on every
  call to `evaluate_displacement`/`evaluate_preference_indices`/
  `evaluate_causal_noise` — cheap relative to training, but `evaluate_causal_noise`
  repeats this per `(site, alpha)` combination, so a large `alphas` sweep
  costs proportionally more forward passes (no training, just inference).
