"""
Object Location Memory (OLM) task.

Extends the EC -> CB prediction-error-minimization task (see ECtoCB_MSE.py,
train_shifted()) with two static objects in the environment. EC and CB are
trained exactly the same way as the base task -- minimize MSE prediction
error of the next state under a random-action trajectory -- except the state
now includes egocentric perceptual features of both objects, not just agent
position.

Objects never move during training: the network only ever learns "objects
are static." Displacement is introduced solely at evaluation time
(evaluate_displacement), as a held-out probe -- so a spike in CB's
prediction error for the relocated object's features is a genuine,
untrained novelty-detection signal, analogous to preferential exploration in
real object-location-memory experiments.

Actions are continuous: a random direction (angle) and a random magnitude in
[0, MAX_STEP], applied to agent position bounded within [-POS_BOUND, POS_BOUND]
per axis. Heading is the sampled direction itself (not the resulting
displacement), so it stays well-defined even when the sampled magnitude is
near zero.

Objects are regular polygons (shape = vertex count, size = edge length; see
sample_polygon_object) observed through a field of view (aperture APERTURE_DEG,
centered on the agent's heading -- see _visual_feature). Each object's feature
has two parts, reflecting an assumption about the animal's visual system: SHAPE
identity is assumed already resolved upstream (analogous to ventral-stream
object recognition) and handed to the network as a fixed, arbitrary per-shape
code (SHAPE_CODES) rather than something CB has to infer -- this is motivated
by the ORM (shape/size-change) preference-index results being largely
independent of CB (see OLM_TASK.md), suggesting identity resolution doesn't
have to run through CB's forward-model pathway at all. SIZE is deliberately
NOT part of that given code -- apparent size remains an ambiguous, ecologically
plausible perceptual cue (visual_size, confounded with distance -- a big object
far away and a small one close up look the same) that must still be inferred,
the same way real vision doesn't hand you true size for free. Concretely:
  perceptual = [cos(rel_angle), sin(rel_angle), visual_size]   # circular-statistics view/distance cues
  feature    = concat(perceptual / FEATURE_STD, SHAPE_CODES[shape])
(all zero if no vertices are visible); see _visual_feature and OLM_TASK.md for
the derivation and the history of this design (an earlier version instead
derived shape identity geometrically from compactness/vertical_asymmetry, both
computed purely from the polygon's own vertices). (count -- how many vertices
are visible -- was dropped: at a wide/omnidirectional aperture it's mostly
just each object's own vertex count, near heading-independent, so it added
little beyond noise; see OLM_TASK.md.)

The agent's own position is NOT part of the state -- unlike object features,
it is never observed directly, only inferrable (if at all) from integrating
the action history.

Flow (per timestep):
  state          = [object_0 feature, object_1 feature]   # no agent position
  [state_t, action_t, state_{t+1}] -> EC (GRU) -> z_t
  [state_{t+1}, action_{t+1}, z_t] -> CB (MLP) -> predicted state_{t+2}
                                            |
                          MSE(predicted, actual state_{t+2})
                                            |
                          gradient flows back through CB -> EC
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from torch.utils.data import DataLoader, TensorDataset

from motor_structure_learning.core import EntorhinalCortex
from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params

# Hyperparameters
POS_BOUND = 5.0   # agent/object positions are bounded in [-POS_BOUND, POS_BOUND] per axis
MAX_STEP = 1.0    # max action magnitude (direction and magnitude are both drawn continuously)
SEQ_LEN = 60

# Heading dynamics: the turning rate (angular velocity) follows a discrete-time
# Ornstein-Uhlenbeck process -- mean-reverting toward zero net turning bias, but locally
# persistent/correlated -- and heading is the running integral of that turning rate over
# the episode (see _sample_actions). This replaces an earlier fully independent,
# uniform-random heading redrawn every single timestep (see OLM_TASK.md's "teleporting
# gaze" discussion) with a "correlated random walk" style of movement -- closer to how a
# real animal's gaze/heading actually moves (smooth turning with inertia, not snapping to
# an unrelated random direction every frame), while still exploring every direction over
# the course of an episode (there's no absolute preferred heading, only inertia in how
# fast it turns).
HEADING_OU_THETA = 1.0   # mean-reversion rate of the turning rate
HEADING_OU_SIGMA = 1.5   # diffusion strength of the turning rate
HEADING_DT = 1.0         # integration timestep (one environment step)
N_TRAIN = 2000
N_VAL = 200
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 32
CB_HIDDEN = 256

N_OBJECTS = 2       # objects shown simultaneously in any one episode

# Structured object construction: each object is a regular polygon, one of 3 shapes
# (vertex count) x 3 sizes (area, same across shapes so e.g. "small" means the same
# overall size for every shape -- a fixed radius or edge length both leave area growing
# with vertex count) = 9 possible (shape, size) combinations. Areas below match what a
# square's area was under the earlier fixed-radius scheme (r=0.5/1.0/1.5), so the overall
# scale is unchanged, just now shared exactly across shapes.
SHAPE_VERTICES = {"triangle": 3, "square": 4, "pentagon": 5}
SIZE_AREA = {"small": 0.5, "medium": 1.0, "large": 2.0}
SHAPE_NAMES = tuple(SHAPE_VERTICES)
SIZE_NAMES = tuple(SIZE_AREA)
POLYGON_TYPES = tuple((shape, size) for shape in SHAPE_NAMES for size in SIZE_NAMES)  # 9 canonical types

# Field of view: object features are derived only from vertices within this aperture,
# centered on the agent's heading -- see _visual_feature.
APERTURE_DEG = 360.0
APERTURE_RAD = np.deg2rad(APERTURE_DEG)

# Probe episodes (displacement/shape-change/size-change): objects sit at two prescribed
# slots (not randomly placed) so a novelty manipulation never has to worry about whether a
# differently-shaped/sized replacement still fits at the same spot -- these are chosen
# with headroom for the largest possible object (triangle/large, r~1.86) regardless of
# what ends up there. CENTER is the displacement probe's relocation target.
#
# The agent's trajectory itself is an ordinary, freshly-randomized random walk (same
# _sample_actions/generate_olm_sequences mechanism as regular training data) -- NOT a
# scripted path. An earlier version used a deterministic scripted "circle around a fixed
# vantage point, then sweep gaze across LEFT/CENTER/RIGHT" trajectory, built to guarantee
# visibility under a narrow field of view. Once APERTURE_DEG widened to 360 (see below),
# visibility became unconditional (every vertex contributes regardless of heading), so
# that guarantee came for free and the scripted trajectory's only remaining effect was a
# confound: CENTER sits exactly on the scripted path's vantage-point symmetry axis, so
# objects there were *inherently* easier to predict (smaller-magnitude, near-zero-mean
# deviation) than objects at LEFT/RIGHT, regardless of any real detection -- an object
# relocated to CENTER looked like an improvement almost no matter what. A random walk
# removes this: with heading uncorrelated with any particular object's true position,
# no location is systematically favored, and it must be freshly randomized per episode
# (not a fixed shared instance) -- a fixed walk would just bake in whatever asymmetries
# that one walk happens to have, rather than averaging them out across episodes.
PROBE_LEFT = (-2.5, 0.0)
PROBE_RIGHT = (2.5, 0.0)
PROBE_CENTER = (0.0, 0.0)

ACTION_DIM = 2

# Object feature = [perceptual (ambiguous, inferred) || shape code (given, fixed)] -- see
# module docstring and _visual_feature. Only the perceptual part is empirically
# normalized (FEATURE_STD); the shape code is a fixed draw from a standard normal, already
# ~unit variance per element by construction, so it needs no additional scaling.
PERCEPTUAL_FEATURE_DIM = 3                   # cos(rel_angle), sin(rel_angle), visual_size
SHAPE_CODE_DIM = 8                           # width of each shape's fixed random identity code
OBJ_FEATURE_DIM = PERCEPTUAL_FEATURE_DIM + SHAPE_CODE_DIM

STATE_DIM = N_OBJECTS * OBJ_FEATURE_DIM      # no agent position -- see module docstring
TRANS_DIM = 2 * STATE_DIM + ACTION_DIM       # [state_t, action_t, state_{t+1}]

# Empirical per-feature std of the 3-dim perceptual sub-feature over ordinary random-walk
# data (see compute_olm_feature_variance.py, N_SEQS=500 @ SEQ_LEN=60, seed=0), used to
# normalize that part of the representation to ~unit variance per feature. Without this,
# one feature can dominate an unweighted MSE loss (and _slot_error/preference-index
# computation) almost regardless of how well the other is learned -- e.g. `count`
# (dropped -- see _visual_feature) had ~78x mean(dev)'s raw variance and ~450x
# std(dev)'s at APERTURE_DEG=60 back when the representation was still
# [mean(dev), std(dev)]. This is specific to the current APERTURE_DEG (currently 360) as
# well as SIZE_AREA/POS_BOUND/PROBE_* layout -- rerun that script and update this if any
# of them change. (compute_olm_feature_variance.py corrects for the fact that
# generate_olm_sequences's output is already normalized by whatever FEATURE_STD is
# currently set, by multiplying it back out before measuring -- so it always measures the
# true raw distribution, even when this constant is stale.) These 3 values are carried
# over unchanged from the earlier 5-dim representation's cos(rel_angle)/sin(rel_angle)/
# visual_size entries -- each feature's own marginal std doesn't depend on which other
# features are also present in the vector, so dropping compactness/vertical_asymmetry
# in favor of the shape code below doesn't invalidate them.
FEATURE_STD = np.array([0.698915, 0.702626, 0.220102], dtype=np.float32)
# cos(rel_angle), sin(rel_angle), visual_size

# Fixed, arbitrary per-shape identity code -- drawn once from a standard normal with its
# own dedicated seed (independent of any training/episode seed, so it's constant across
# runs) and never re-derived from geometry, standing in for identity already resolved by
# an upstream visual system (see module docstring). Deliberately keyed by SHAPE_NAMES only
# (not POLYGON_TYPES): a given shape's code is identical across all 3 sizes, so this
# channel carries no size information -- size stays purely in the ambiguous, distance-
# confounded visual_size perceptual cue above.
_SHAPE_CODE_RNG = np.random.default_rng(12345)
SHAPE_CODES = {shape: _SHAPE_CODE_RNG.standard_normal(SHAPE_CODE_DIM).astype(np.float32)
               for shape in SHAPE_NAMES}

# Observation noise: i.i.d. per-timestep Gaussian noise added to the PERCEPTUAL columns
# only (never SHAPE_CODES -- identity stays a clean, exact given signal, consistent with
# everywhere else in this module), applied post-FEATURE_STD normalization (i.e. in
# already-unit-variance-normalized units, so this value is directly a signal-to-noise
# ratio against the clean signal). Infrastructure is fully wired up and tested
# (generate_olm_sequences and all three probe generators accept observation_noise_std,
# with matched noise shared between probe control/manipulated pairs -- see
# _sample_observation_noise) but DEFAULTS TO 0 (off) -- kept available for future use,
# not because it's unused.
#
# Motivation it was built for, and why it's off by default: the goal was to make this a
# genuine POMDP (previously every observation was a deterministic, fully trustworthy
# function of the true state, so belief correction after a displacement was instantaneous
# -- a real but only ~1-timestep-wide prediction-error spike, see OLM_TASK.md) by forcing
# EC's GRU to integrate noisy evidence over time rather than trust a single reading,
# hoping this would produce a more sustained, behaviorally-realistic multi-step decay.
# Tested at std=0.5 (noise variance 0.25, signal still dominates a single reading) and
# std=1.5 (noise variance 2.25, noise-dominated): neither produced any sustained decay --
# still a one-timestep spike either way -- and the higher value made the effect *weaker*
# (preference-index gap shrank monotonically with noise, not the reverse), ruling out
# "just needs more/less noise" as the fix. The likely reason: PROBE_LEFT-to-PROBE_CENTER
# is a large jump relative to any per-reading noise tried, and an (near-)optimal Bayesian
# filter has little reason to show *sustained* inertia against a single strongly
# inconsistent reading -- passive per-step noise mainly creates ambiguity for small
# discrepancies, not this task's large, unambiguous relocation. A momentary spike was
# accepted as the correct, expected model of an instantaneous orienting response rather
# than continuing to chase a sustained one -- see OLM_TASK.md's "Observation noise"
# section for the full writeup, including why intermittent-sampling/occlusion (not yet
# implemented) is a more likely candidate if a sustained signal is wanted later.
OBSERVATION_NOISE_STD = 0.0

CKPT_PATH = "checkpoints/olm_task.pt"
CLOSED_LOOP_CKPT_PATH = "checkpoints/olm_task_closed_loop.pt"
FIG_PATH = "figs/olm_displacement_probe.png"
LOCATION_PREF_FIG_PATH = "figs/olm_preference_index_location.png"
SHAPE_PREF_FIG_PATH = "figs/olm_preference_index_shape.png"
SIZE_PREF_FIG_PATH = "figs/olm_preference_index_size.png"
LOCATION_CAUSAL_FIG_PATH = "figs/olm_causal_noise_location.png"
SHAPE_CAUSAL_FIG_PATH = "figs/olm_causal_noise_shape.png"
SIZE_CAUSAL_FIG_PATH = "figs/olm_causal_noise_size.png"
TRIAL_DIAGNOSTIC_FIG_PATH = "figs/olm_trial_diagnostic.png"
FEATURE_SPACE_TRAJ_FIG_PATH = "figs/olm_feature_space_trajectories.png"
FEATURE_SPACE_STATE_FIG_PATH = "figs/olm_feature_space_state_space.png"
MULTI_SEED_DIR = "checkpoints/olm_multi_seed"

_FEATURE_NAMES = ("cos(rel_angle)", "sin(rel_angle)", "visual_size") + tuple(
    f"shape_code_{i}" for i in range(SHAPE_CODE_DIM)
)  # OBJ_FEATURE_DIM columns, see _visual_feature

CAUSAL_NOISE_SITES = ("gru", "cb_input")


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _visible_deviation(agent_pos, heading, vertices, aperture=APERTURE_RAD):
    """(dev, visible): each of `vertices`' egocentric deviation angle (bearing relative
    to the agent's heading, i.e. camera direction, wrapped to (-pi, pi]) and whether it's
    within the agent's field of view (aperture, centered on heading). Factored out of
    _visual_feature so anything that needs to check visibility directly (e.g. tests)
    doesn't have to depend on a `count` feature that may not exist in the state."""
    rel = vertices - agent_pos
    phi = np.arctan2(rel[:, 1], rel[:, 0])
    theta_h = np.arctan2(heading[1], heading[0])
    dev = _wrap_to_pi(phi - theta_h)
    return dev, np.abs(dev) < (aperture / 2.0)


def _visual_feature(agent_pos, heading, vertices, shape, aperture=APERTURE_RAD):
    """FOV-based feature for one object, given its (num_vertices, 2) world-frame vertex
    coordinates and its shape name (SHAPE_NAMES).

    Two parts, computed by entirely different mechanisms (see module docstring for the
    motivation):

      C1, S1 = mean(cos(dev_i)), mean(sin(dev_i))  for visible dev_i (see _visible_deviation)
      R1          = |C1, S1|                        # resultant length ("concentration")
      visual_size = sqrt(-2 ln R1)                  # DISTANCE: circular std -- spread/apparent size
      [C1/R1, S1/R1]                                 # VIEW: unit vector along the bearing direction
      SHAPE_CODES[shape]                             # IDENTITY: fixed, arbitrary per-shape code --
                                                      # NOT computed from vertices/geometry at all

      feature = concat([C1/R1, S1/R1, visual_size] / FEATURE_STD, SHAPE_CODES[shape])
                (all zero if no vertices visible)

    An earlier version derived shape identity geometrically instead -- `compactness`
    (isoperimetric quotient) and `vertical_asymmetry` (lowest nonvanishing odd
    trigonometric moment of the vertex angles), both pure per-shape geometric constants,
    computed directly from `vertices` rather than looked up. That version modeled shape
    recognition as something the network had to earn from raw geometry; the current
    version instead assumes an upstream visual system already resolves shape identity
    (analogous to ventral-stream object recognition) and hands it to the network as a
    given, fixed code -- motivated by the ORM (shape/size-change) preference-index probes
    being largely independent of CB, suggesting identity resolution doesn't have to run
    through CB's forward-model pathway at all (see OLM_TASK.md). SHAPE_CODES is
    deliberately keyed by shape only, not (shape, size) -- size is NOT given for free this
    way, and stays purely in the ambiguous, distance-confounded `visual_size` cue below,
    the same way real vision doesn't resolve true size independent of distance.

    `[C1/R1, S1/R1]` (the view/direction pair) uses proper circular statistics (mean
    resultant vector), NOT a plain mean over the individually (-pi, pi]-wrapped `dev_i`
    values, and is represented as a 2D unit vector rather than a raw angle specifically to
    avoid the wraparound discontinuity a scalar bearing would have. This distinction
    mattered a lot in practice for an earlier version of `visual_size` (computed from
    wrapped per-vertex bearings directly): a plain std over already-wrapped angles was
    contaminated by a heading-dependent artifact whenever the object sat near directly
    behind the agent (measured: only 19-43% of its variance was attributable to object
    identity, at every distance tested, vs. 100% after switching to the circular-statistics
    version above -- see OLM_TASK.md).

    Deliberately ambiguous/lossy where it still matters, mirroring real vision rather than
    an unambiguous metric readout: `visual_size` confounds true size with distance (a big
    object far away and a small one close up can produce the same angular spread) --
    there's no separate, unambiguous distance feature, just this one size/distance-
    confounded channel. A `count` (number of visible vertices) feature was dropped -- at a
    wide/omnidirectional aperture it's mostly just each object's own vertex count, near
    heading-independent, so it added little beyond noise.
    """
    dev, visible = _visible_deviation(agent_pos, heading, vertices, aperture)
    if not visible.any():
        return np.zeros(OBJ_FEATURE_DIM, dtype=np.float32)
    v = dev[visible]
    C1, S1 = np.cos(v).mean(), np.sin(v).mean()
    R1 = np.hypot(C1, S1)
    visual_size = np.sqrt(np.clip(-2.0 * np.log(np.clip(R1, 1e-6, 1.0)), 0.0, None))

    r1_safe = max(R1, 1e-6)
    perceptual = np.array([C1 / r1_safe, S1 / r1_safe, visual_size], dtype=np.float32) / FEATURE_STD
    return np.concatenate([perceptual, SHAPE_CODES[shape]])


def _build_state(agent_pos, heading, vertices_list, shapes_list, aperture=APERTURE_RAD):
    """State = concatenated visual feature of each object -- no agent position (see
    module docstring: the agent's own position is never observed directly)."""
    obj_feats = [_visual_feature(agent_pos, heading, vertices, shape, aperture)
                 for vertices, shape in zip(vertices_list, shapes_list)]
    return np.concatenate(obj_feats)


def _sample_observation_noise(rng, n_steps, std=OBSERVATION_NOISE_STD):
    """(n_steps, N_OBJECTS, PERCEPTUAL_FEATURE_DIM) i.i.d. Gaussian observation noise --
    see OBSERVATION_NOISE_STD. One draw per ABSOLUTE timestep (0..n_steps-1) per object's
    perceptual sub-vector, precomputed once per episode rather than redrawn per call, for
    two reasons:
      1. A given true moment gets exactly one noisy reading, regardless of whether a row
         references it as that row's "state_t" or the PREVIOUS row's "state_next" (both
         are the same real timestep, just packed into adjacent rows of the transition
         tensor -- see generate_olm_sequences).
      2. Paired probe episodes (control vs. manipulated -- see _build_probe_transitions)
         can share the exact same noise realization at shared timesteps, isolating the
         manipulation's effect rather than adding spurious extra noise-driven variance
         between the pair ("common random numbers", a standard variance-reduction
         technique for matched comparisons).
    std=0 gives an all-zero array (i.e. the earlier noise-free behavior) without any
    special-casing by callers.
    """
    return (std * rng.standard_normal((n_steps, N_OBJECTS, PERCEPTUAL_FEATURE_DIM))).astype(np.float32)


def _apply_observation_noise(state, noise_t):
    """Add this timestep's per-object perceptual-only noise (noise_t, shape (N_OBJECTS,
    PERCEPTUAL_FEATURE_DIM) -- one row of _sample_observation_noise's output) to a
    (STATE_DIM,) state vector, leaving every object's SHAPE_CODES columns exact (identity
    is never noisy -- see OBSERVATION_NOISE_STD)."""
    noisy = state.copy()
    for obj in range(N_OBJECTS):
        start = obj * OBJ_FEATURE_DIM
        noisy[start:start + PERCEPTUAL_FEATURE_DIM] += noise_t[obj]
    return noisy


def _polygon_radius_for_area(num_vertices, area):
    """Circumradius a regular polygon needs for the given area (inverse of
    area = 0.5 * num_vertices * radius**2 * sin(2*pi/num_vertices))."""
    return np.sqrt(2.0 * area / (num_vertices * np.sin(2 * np.pi / num_vertices)))


def _polygon_vertices(center, num_vertices, radius, rotation):
    """(num_vertices, 2) vertex coordinates of a regular polygon: evenly spaced on a
    circle of the given radius (center-to-vertex length), starting at angle `rotation`."""
    angles = rotation + 2 * np.pi * np.arange(num_vertices) / num_vertices
    return center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def sample_polygon_object(rng, pos_bound=POS_BOUND, shape=None, size=None):
    """Sample one structured polygon object: a random (shape, size) pair (or use the ones
    given), then a random center + rotation such that it fits within
    [-pos_bound, pos_bound]^2 *for every rotation* -- since every vertex lies on a circle
    of the shape's radius (independent of rotation), it's enough to sample the center from
    a shrunk [-(pos_bound - radius), pos_bound - radius]^2 box; no rejection sampling
    needed.

    `size` sets the target *area* directly (SIZE_AREA), the same value for every shape --
    so e.g. "small" means the same overall size for a triangle, square, or pentagon.
    Radius is then derived per-shape (`_polygon_radius_for_area`) so that area, not
    radius or edge length, is what's held constant across shapes (both of those would
    leave area growing with vertex count at a fixed size).

    Returns a dict: shape, size, num_vertices, area, radius, center (2,), rotation
    (radians), vertices (num_vertices, 2).
    """
    shape = shape if shape is not None else rng.choice(SHAPE_NAMES)
    size = size if size is not None else rng.choice(SIZE_NAMES)
    num_vertices = SHAPE_VERTICES[shape]
    area = SIZE_AREA[size]
    radius = float(_polygon_radius_for_area(num_vertices, area))

    if radius >= pos_bound:
        raise ValueError(
            f"shape={shape!r} size={size!r} has radius {radius:.3f} (area={area}), which "
            f"doesn't fit within pos_bound={pos_bound}"
        )
    center = rng.uniform(-(pos_bound - radius), pos_bound - radius, size=2).astype(np.float32)
    rotation = float(rng.uniform(0.0, 2 * np.pi))
    vertices = _polygon_vertices(center, num_vertices, radius, rotation).astype(np.float32)

    return {
        "shape": shape,
        "size": size,
        "num_vertices": num_vertices,
        "area": area,
        "radius": radius,
        "center": center,
        "rotation": rotation,
        "vertices": vertices,
    }


def visualize_object_samples(n_samples=12, pos_bound=POS_BOUND, seed=0,
                              fig_path="figs/olm_object_samples.png"):
    """Sanity-check plot: draw n_samples randomly-constructed polygon objects (random
    shape, size, center, rotation) inside the arena, so the geometry can be checked by eye
    -- triangle/square/pentagon have the right vertex count, "large" is visibly bigger
    than "small", rotations look random, and every shape stays within the dashed arena
    boundary [-pos_bound, pos_bound]^2.
    """
    rng = np.random.default_rng(seed)
    objects = [sample_polygon_object(rng, pos_bound=pos_bound) for _ in range(n_samples)]

    colors = {"triangle": "firebrick", "square": "steelblue", "pentagon": "seagreen"}
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                            fill=False, edgecolor="black", linestyle="--"))

    for obj in objects:
        ax.add_patch(MplPolygon(obj["vertices"], closed=True, facecolor=colors[obj["shape"]],
                                 alpha=0.5, edgecolor="black"))
        ax.annotate(f"{obj['shape'].title()[:3]} ({obj['size'].title()[0]})", obj["center"], fontsize=6,
                    ha="center", va="center")

    pad = pos_bound * 0.2
    ax.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax.set_aspect("equal")
    ax.set_title("Random object placements")
    make_axis_nice(ax)

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return objects


def plot_feature_space_trajectories(seq_len=SEQ_LEN, center=PROBE_CENTER, rotation=0.4,
                                     object_types=POLYGON_TYPES, seed=0,
                                     fig_path=FEATURE_SPACE_TRAJ_FIG_PATH,
                                     state_fig_path=FEATURE_SPACE_STATE_FIG_PATH,
                                     state_dims=(3, 4)):
    """Illustrative (not evaluative) plot: for ONE shared random-walk agent trajectory,
    place each of `object_types` in turn at the identical `center`/`rotation`, and plot the
    resulting `OBJ_FEATURE_DIM`-dim feature trajectories (see _visual_feature) -- one curve
    per object type -- two ways: (1) each feature vs. time (`fig_path`) and (2) as a path
    through a 2D slice of feature space, `state_dims` (default `(3, 4)` = the first two
    dims of `SHAPE_CODES[shape]` -- see `_FEATURE_NAMES`), directly (`state_fig_path`).
    Since the agent path and object location are held fixed across every curve, any
    difference between curves in the perceptual dims (0-2) is attributable entirely to
    viewing geometry (distance/direction to a same-sized object at different rotations),
    while the shape-code dims are, by construction, identical for every object sharing a
    shape regardless of anything about the trajectory -- a direct illustration that
    identity in this representation is a given, not something perceptually earned (see
    module docstring; the location-change probe covers *location* discrimination
    separately).

    `rotation` defaults to a non-special value (0.4 rad) for consistency with the earlier
    (geometric shape-feature) version of this plot, though it no longer has any special
    effect on the current SHAPE_CODES-based representation (SHAPE_CODES doesn't depend on
    rotation at all).

    A top panel in the vs.-time plot shows agent-to-`center` distance over time (shared by
    every object type, since they're all centered at the same point) purely for context:
    the direction components (`cos`/`sin(rel_angle)`) change fast -- not because of
    proximity to the object, but because heading itself is redrawn independently and
    uniformly every timestep (see OLM_TASK.md's note on _sample_actions) -- so without the
    distance panel those jumps would be indistinguishable from genuine distance-driven
    differences. The `shape_code_*` dims, by contrast, are exact per-shape constants (see
    _visual_feature, SHAPE_CODES) -- constant over the whole trajectory and across every
    size, since the code is fixed and keyed by shape only.

    The state-space plot is the same information laid out as a phase portrait instead --
    lines connect consecutive timesteps in the `state_dims` slice directly, with
    circle/star markers for each curve's start/end point. With the default `state_dims`
    (both shape-code dims), this is a static scatter rather than a moving trajectory: each
    shape sits at exactly one fixed point (start/end markers coincide), with all 3 sizes
    landing on that same point (SHAPE_CODES doesn't vary with size). Pick a `state_dims`
    that includes `visual_size` (index 2) or a direction dim (0 or 1) instead to see
    genuine time-varying structure.

    Color encodes shape, linestyle encodes size (though with the default `state_dims`,
    same-shape points at different sizes overlap exactly, since size has no effect on
    SHAPE_CODES at all).

    Returns {(shape, size): (seq_len+1, OBJ_FEATURE_DIM) feature array} for programmatic
    use (e.g. testing) in addition to saving both plots.
    """
    rng = np.random.default_rng(seed)
    positions, headings = _random_walk(rng, seq_len)
    center = np.asarray(center, dtype=np.float32)
    center_tuple = tuple(float(c) for c in center)
    t_axis = np.arange(seq_len + 1)
    distance = np.linalg.norm(positions - center, axis=-1)

    shape_color = {"triangle": "firebrick", "square": "steelblue", "pentagon": "seagreen"}
    size_ls = {"small": ":", "medium": "--", "large": "-"}

    feats_by_type = {}
    for shape, size in object_types:
        radius = _polygon_radius_for_area(SHAPE_VERTICES[shape], SIZE_AREA[size])
        vertices = _polygon_vertices(center, SHAPE_VERTICES[shape], radius, rotation)
        feats_by_type[(shape, size)] = np.stack([
            _visual_feature(positions[t], headings[t], vertices, shape) for t in range(seq_len + 1)
        ])

    set_plt_params(style="fast")

    # 1) Each feature vs. time, plus a shared distance-to-object context panel.
    fig, axes = plt.subplots(1 + OBJ_FEATURE_DIM, 1, figsize=(8, 3 * (1 + OBJ_FEATURE_DIM)), sharex=True)

    axes[0].plot(t_axis, distance, color="black")
    axes[0].set_ylabel("distance to object")
    axes[0].set_title(f"Feature-space trajectories -- shared agent path, object at {center_tuple}")
    make_axis_nice(axes[0])

    for shape, size in object_types:
        feats = feats_by_type[(shape, size)]
        for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
            axes[1 + feat_idx].plot(t_axis, feats[:, feat_idx], color=shape_color[shape],
                                     ls=size_ls[size], alpha=0.8, label=f"{shape}/{size}")

    for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
        axes[1 + feat_idx].set_ylabel(f"{feat_name} (normalized)")
        make_axis_nice(axes[1 + feat_idx])
    axes[-1].set_xlabel("Timestep")
    axes[1].legend(fontsize=6, ncol=3, loc="upper right")

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    # 2) State-space (phase-plane) view: a 2D slice of the feature vector directly.
    dim0, dim1 = state_dims
    fig, ax = plt.subplots(figsize=(6, 6))
    for shape, size in object_types:
        feats = feats_by_type[(shape, size)]
        ax.plot(feats[:, dim0], feats[:, dim1], color=shape_color[shape], ls=size_ls[size],
                alpha=0.8, label=f"{shape}/{size}")
        ax.scatter(feats[0, dim0], feats[0, dim1], color=shape_color[shape], marker="o", s=35, zorder=3)
        ax.scatter(feats[-1, dim0], feats[-1, dim1], color=shape_color[shape], marker="*", s=70, zorder=3)

    ax.set_xlabel(f"{_FEATURE_NAMES[dim0]} (normalized)")
    ax.set_ylabel(f"{_FEATURE_NAMES[dim1]} (normalized)")
    ax.set_title(f"Feature-space state-space trajectories -- shared agent path, object at {center_tuple}\n"
                 f"(o = start, * = end)")
    ax.legend(fontsize=6, ncol=3, loc="best")
    make_axis_nice(ax)

    os.makedirs(os.path.dirname(state_fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(state_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {state_fig_path}")

    return feats_by_type


def _sample_actions(rng, n, max_step=MAX_STEP):
    """n continuous random actions: a correlated turning direction (angle) and an
    independent random magnitude in [0, max_step].

    Direction is no longer redrawn independently every step -- instead, an angular
    velocity (turning rate) follows a discrete-time Ornstein-Uhlenbeck process (exact
    discretization, see HEADING_OU_THETA/HEADING_OU_SIGMA/HEADING_DT), and heading angle
    is that turning rate's running integral over the episode. This is what makes the
    action meaningfully "which way the animal turns and how strongly it moves in that
    direction," rather than an unrelated fresh direction each step. Magnitude is still
    drawn i.i.d. uniform per step (unchanged) -- only the direction component is now
    temporally correlated.

    Also returns the resulting headings as unit vectors (camera directions for the
    FOV-based object features) -- the direction the agent was facing while taking each
    action, not the resulting displacement, so heading stays well-defined even for
    near-zero-magnitude actions.
    """
    magnitudes = rng.uniform(0.0, max_step, size=n).astype(np.float32)

    ou_decay = np.exp(-HEADING_OU_THETA * HEADING_DT)
    ou_noise_std = HEADING_OU_SIGMA * np.sqrt((1.0 - ou_decay ** 2) / (2.0 * HEADING_OU_THETA))
    omega = 0.0  # turning rate (angular velocity), OU-process state, starts at rest
    angle = rng.uniform(0.0, 2 * np.pi)  # initial heading angle -- uniform, fresh per episode

    angles = np.empty(n, dtype=np.float32)
    for t in range(n):
        omega = omega * ou_decay + ou_noise_std * rng.standard_normal()
        angle = (angle + omega * HEADING_DT) % (2 * np.pi)
        angles[t] = angle

    headings = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    actions = headings * magnitudes[:, None]
    return actions, headings


def _sample_episode_objects(rng, n_objects=N_OBJECTS, pos_bound=POS_BOUND):
    """Sample n_objects distinct polygon objects (shape, size drawn without replacement
    from the 9 canonical POLYGON_TYPES), each freshly placed via sample_polygon_object."""
    idxs = rng.choice(len(POLYGON_TYPES), size=n_objects, replace=False)
    return [
        sample_polygon_object(rng, pos_bound=pos_bound, shape=POLYGON_TYPES[i][0], size=POLYGON_TYPES[i][1])
        for i in idxs
    ]


def _random_walk(rng, seq_len, pos_bound=POS_BOUND, max_step=MAX_STEP):
    """Ordinary random-walk agent trajectory -- same mechanism as generate_olm_sequences
    -- for use by the probes (see the module comment above PROBE_LEFT for why this
    replaced an earlier scripted trajectory). Freshly randomized every call (draws from
    `rng`, which callers should NOT reseed per episode -- see _generate_probe_data).

    Returns (positions, headings), each (seq_len+1, 2).
    """
    actions, action_dirs = _sample_actions(rng, seq_len, max_step)
    positions = np.zeros((seq_len + 1, 2), dtype=np.float32)
    headings = np.zeros((seq_len + 1, 2), dtype=np.float32)
    positions[0] = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
    headings[0] = (1.0, 0.0)
    for t in range(seq_len):
        positions[t + 1] = np.clip(positions[t] + actions[t], -pos_bound, pos_bound)
        headings[t + 1] = action_dirs[t]
    return positions, headings


# Rotation can't be enumerated (continuous), so it's swept over 3 fixed, evenly-spaced
# values instead, shared by both objects within a given replicate (not varied
# independently per object).
ROTATION_SWEEP = (0.0, 2 * np.pi / 3, 4 * np.pi / 3)


def _enumerate_probe_pairs():
    """Every ordered assignment of 2 distinct POLYGON_TYPES to the (left, right) slots,
    crossed with which slot is manipulated -- the deterministic condition set shared by
    all three probe types. 9*8 ordered assignments * 2 manipulated-slot choices = 144
    combinations."""
    combos = []
    for left_type in POLYGON_TYPES:
        for right_type in POLYGON_TYPES:
            if left_type == right_type:
                continue
            for manipulated_slot in ("left", "right"):
                combos.append((left_type, right_type, manipulated_slot))
    return combos


def _probe_vertices(shape, size, center, rotation):
    num_vertices = SHAPE_VERTICES[shape]
    radius = float(_polygon_radius_for_area(num_vertices, SIZE_AREA[size]))
    return _polygon_vertices(np.asarray(center, dtype=np.float32), num_vertices, radius, rotation).astype(np.float32)


def _build_probe_transitions(left_type, right_type, rotation, manipulated_slot, manipulation,
                              rng, seq_len, t_probe, observation_noise_std=OBSERVATION_NOISE_STD):
    """(control_trans, manipulated_trans), each (seq_len, TRANS_DIM): paired episodes
    sharing one freshly-randomized agent trajectory (_random_walk, drawn from `rng` --
    see the module comment above PROBE_LEFT for why this is a random walk rather than a
    scripted path) and identical objects up to t_probe. At t_probe, the manipulated
    variant changes the manipulated_slot's object per `manipulation`; the control
    variant never changes. Both objects share `rotation` -- rotation is deterministically
    swept (ROTATION_SWEEP), not independently randomized per object.

    Both variants also share the exact same observation-noise realization (see
    _sample_observation_noise) -- drawn once, before either `build()` call below, and
    reused by both -- rather than each variant drawing its own independent noise. This
    preserves the "control and manipulated transitions are byte-identical through the row
    before t_probe" invariant (test_olm_probes.py) despite noisy observations, and means
    any difference after t_probe reflects exactly the manipulation itself, not extra
    noise-driven variance between the pair (common random numbers).

    `manipulation` is one of:
      {"kind": "location"}                  -- center becomes PROBE_CENTER
      {"kind": "shape", "new_shape": ...}    -- shape changes, size/center unchanged
      {"kind": "size", "new_size": ...}      -- size changes, shape/center unchanged
    In every case rotation is unchanged -- only the one named factor differs between the
    pre- and post-manipulation object.
    """
    types = {"left": left_type, "right": right_type}
    slot_center = {"left": PROBE_LEFT, "right": PROBE_RIGHT}

    def make_object(slot, changed):
        """(shape, vertices) for this slot -- shape is tracked alongside vertices since
        _build_state now needs it directly (SHAPE_CODES lookup), not derived from
        geometry (see _visual_feature)."""
        shape, size = types[slot]
        center = slot_center[slot]
        if changed:
            if manipulation["kind"] == "location":
                center = PROBE_CENTER
            elif manipulation["kind"] == "shape":
                shape = manipulation["new_shape"]
            elif manipulation["kind"] == "size":
                size = manipulation["new_size"]
        return shape, _probe_vertices(shape, size, center, rotation)

    other_slot = "right" if manipulated_slot == "left" else "left"
    before = {manipulated_slot: make_object(manipulated_slot, False),
              other_slot: make_object(other_slot, False)}
    after = {manipulated_slot: make_object(manipulated_slot, True),
             other_slot: before[other_slot]}

    positions, headings = _random_walk(rng, seq_len)
    noise = _sample_observation_noise(rng, seq_len + 1, std=observation_noise_std)

    def build(objects_at):
        rows = []
        for t in range(seq_len):
            shapes_t, verts_t = zip(*objects_at(t))
            state_t = _apply_observation_noise(
                _build_state(positions[t], headings[t], verts_t, shapes_t), noise[t])
            action = positions[t + 1] - positions[t]
            shapes_next, verts_next = zip(*objects_at(t + 1))
            state_next = _apply_observation_noise(
                _build_state(positions[t + 1], headings[t + 1], verts_next, shapes_next), noise[t + 1])
            rows.append(np.concatenate([state_t, action, state_next]))
        return torch.tensor(np.stack(rows), dtype=torch.float32)

    control_trans = build(lambda i: [before["left"], before["right"]])
    manipulated_trans = build(
        lambda i: [after["left"], after["right"]] if i >= t_probe
        else [before["left"], before["right"]]
    )
    return control_trans, manipulated_trans


def _generate_probe_data(manipulations, seq_len=SEQ_LEN, t_probe=None, rotations=ROTATION_SWEEP, seed=0,
                          observation_noise_std=OBSERVATION_NOISE_STD):
    """Shared driver for the three probe generators below: builds (control_trans,
    manipulated_trans) for every (pair/manipulated-slot combination) x manipulation x
    rotation, where `manipulations` is a function
    (left_type, right_type, manipulated_slot) -> list of manipulation dicts to test for
    that combination (see _build_probe_transitions).

    Each episode gets its own freshly-randomized agent trajectory, drawn from a single
    `rng` seeded from `seed` that is NOT reset between episodes -- so every episode's
    walk differs (rather than all episodes sharing one fixed walk, which would just bake
    in whatever location-dependent asymmetries that one walk happens to have instead of
    averaging them out across episodes; see the module comment above PROBE_LEFT). The
    same `rng` also drives each episode's observation noise (see
    _sample_observation_noise) -- `observation_noise_std` defaults to matching ordinary
    training data (OBSERVATION_NOISE_STD) so probe episodes look like "more of the same"
    environment rather than an unrealistically clean test condition; pass 0.0 to disable.
    """
    if t_probe is None:
        t_probe = seq_len // 2
    rng = np.random.default_rng(seed)

    control_list, manipulated_list, idx_list = [], [], []
    for left_type, right_type, manipulated_slot in _enumerate_probe_pairs():
        for manipulation in manipulations(left_type, right_type, manipulated_slot):
            for rotation in rotations:
                c, m = _build_probe_transitions(left_type, right_type, rotation, manipulated_slot,
                                                 manipulation, rng, seq_len, t_probe,
                                                 observation_noise_std=observation_noise_std)
                control_list.append(c)
                manipulated_list.append(m)
                idx_list.append(0 if manipulated_slot == "left" else 1)

    return torch.stack(control_list), torch.stack(manipulated_list), np.array(idx_list), t_probe


def generate_probe_location_change_data(seq_len=SEQ_LEN, t_probe=None, rotations=ROTATION_SWEEP, seed=0,
                                         observation_noise_std=OBSERVATION_NOISE_STD):
    """Deterministic, exhaustive location-change (OLM) probe: every _enumerate_probe_pairs
    combination x ROTATION_SWEEP -- the manipulated slot's object relocates from its
    prescribed slot to PROBE_CENTER at t_probe, shape/size/rotation unchanged. The agent's
    trajectory (not the objects/manipulation) is randomized per episode -- see
    _generate_probe_data.

    Returns (control_trans, manipulated_trans) each (N, seq_len, TRANS_DIM),
    manipulated_idxs (N,) in {0=left, 1=right}, and t_probe.
    """
    return _generate_probe_data(lambda l, r, s: [{"kind": "location"}],
                                 seq_len=seq_len, t_probe=t_probe, rotations=rotations, seed=seed,
                                 observation_noise_std=observation_noise_std)


def generate_probe_shape_change_data(seq_len=SEQ_LEN, t_probe=None, rotations=ROTATION_SWEEP, seed=0,
                                      observation_noise_std=OBSERVATION_NOISE_STD):
    """Deterministic, exhaustive shape-change (ORM) probe: as
    generate_probe_location_change_data, but the manipulated slot's SHAPE changes (to
    each of the other 2 shapes) at t_probe, holding size/center/rotation fixed.
    """
    def manipulations(left_type, right_type, manipulated_slot):
        current_shape, _ = left_type if manipulated_slot == "left" else right_type
        return [{"kind": "shape", "new_shape": s} for s in SHAPE_NAMES if s != current_shape]

    return _generate_probe_data(manipulations, seq_len=seq_len, t_probe=t_probe, rotations=rotations, seed=seed,
                                 observation_noise_std=observation_noise_std)


def generate_probe_size_change_data(seq_len=SEQ_LEN, t_probe=None, rotations=ROTATION_SWEEP, seed=0,
                                     observation_noise_std=OBSERVATION_NOISE_STD):
    """Deterministic, exhaustive size-change (ORM) probe: as
    generate_probe_shape_change_data, but the manipulated slot's SIZE changes (to each
    of the other 2 sizes) at t_probe, holding shape/center/rotation fixed.
    """
    def manipulations(left_type, right_type, manipulated_slot):
        _, current_size = left_type if manipulated_slot == "left" else right_type
        return [{"kind": "size", "new_size": s} for s in SIZE_NAMES if s != current_size]

    return _generate_probe_data(manipulations, seq_len=seq_len, t_probe=t_probe, rotations=rotations, seed=seed,
                                 observation_noise_std=observation_noise_std)


def generate_olm_sequences(n_seqs, seq_len=SEQ_LEN, pos_bound=POS_BOUND, max_step=MAX_STEP,
                            n_objects=N_OBJECTS, seed=0, observation_noise_std=OBSERVATION_NOISE_STD):
    """Random-action trajectories with two static polygon objects per episode (shape/size
    drawn without replacement from the 9 canonical POLYGON_TYPES, freshly placed via
    sample_polygon_object). State is the FOV-based visual feature of each object only --
    the agent's own position is never part of the state (see module docstring).

    `observation_noise_std` (default OBSERVATION_NOISE_STD) makes this a genuine POMDP --
    see its module-level comment. Pass 0.0 for the earlier noise-free behavior (e.g. when
    measuring the clean signal's own statistics, as compute_olm_feature_variance.py does).
    """
    rng = np.random.default_rng(seed)

    all_trans = []
    for _ in range(n_seqs):
        agent_pos = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
        heading = np.array([1.0, 0.0], dtype=np.float32)
        episode_objects = _sample_episode_objects(rng, n_objects, pos_bound)
        vertices_list = [obj["vertices"] for obj in episode_objects]
        shapes_list = [obj["shape"] for obj in episode_objects]

        actions, headings = _sample_actions(rng, seq_len, max_step)
        noise = _sample_observation_noise(rng, seq_len + 1, std=observation_noise_std)

        rows = []
        for t in range(seq_len):
            state_t = _apply_observation_noise(
                _build_state(agent_pos, heading, vertices_list, shapes_list), noise[t])

            action = actions[t]
            next_pos = np.clip(agent_pos + action, -pos_bound, pos_bound)
            next_heading = headings[t]

            state_next = _apply_observation_noise(
                _build_state(next_pos, next_heading, vertices_list, shapes_list), noise[t + 1])
            rows.append(np.concatenate([state_t, action, state_next]))

            agent_pos, heading = next_pos, next_heading

        all_trans.append(np.stack(rows))

    return torch.tensor(np.stack(all_trans), dtype=torch.float32)  # (N, T, TRANS_DIM)


class OLMForwardModel(nn.Module):
    """MLP forward model: [state, action, z] -> predicted next state. Mirrors CerebellumCB
    (ECtoCB_MSE.py) but with a configurable output dim, since the shared
    CerebellumForwardModelContinuous hardcodes a 2-dim output."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


def _shifted_forward(ec, cb, trans):
    """z_t predicts one step ahead, mirroring train_shifted() in ECtoCB_MSE.py."""
    z, _ = ec(trans)
    z_prev = z[:, :-1, :]
    s_curr = trans[:, 1:, :STATE_DIM]
    a_curr = trans[:, 1:, STATE_DIM:STATE_DIM + ACTION_DIM]
    s_target = trans[:, 1:, STATE_DIM + ACTION_DIM:]
    s_pred = cb(s_curr, a_curr, z_prev)
    return s_pred, s_target


def _shifted_forward_noisy(ec, cb, trans, noise_site, alpha, stim_start, stim_end):
    """Like _shifted_forward, but injects isotropic Gaussian noise with expected L2 norm
    `alpha` per perturbed timestep, active only for timesteps in [stim_start, stim_end)
    -- an explicit "stimulation period" you define, not tied to any default phase.

    alpha is a literal noise magnitude (expected total perturbation norm), deliberately
    NOT tensor-relative -- but also deliberately NOT just a per-element std, because EC's
    GRU hidden state (256-dim by default) and its context readout z = tanh(...) (32-dim)
    have different dimensionality: for a fixed per-element std, the expected norm of an
    isotropic Gaussian vector scales as sqrt(dim), so the same per-element std would
    inject sqrt(256/32) = sqrt(8) ~= 2.83x more total noise into the hidden state than
    into the readout. Dividing by sqrt(dim) (per-element std = alpha / sqrt(dim)) fixes
    this: expected total norm is alpha regardless of which tensor is perturbed, so a
    given alpha injects the same overall amount of noise at both sites, and any
    difference between the two sites' dose-response curves reflects a real circuit
    difference (readout transform + recurrent compounding for "gru" vs. one-shot direct
    injection for "cb_input"), not an accidental mismatch in how much noise was added.

    noise_site:
      - "gru": noise added to EC's GRU hidden state at every step in [stim_start, stim_end),
        compounding forward through the recurrence -- a persistently noisy memory trace.
        This requires manually unrolling the GRU step by step (a single ec(trans) call
        can't be perturbed mid-unroll), unlike the clean/cb_input paths below.
      - "cb_input": noise added to z (EC's context readout) only at the timesteps in
        [stim_start, stim_end) that CB actually consumes -- EC's own hidden state and
        recurrence are untouched; only the readout CB sees is corrupted.
      - alpha == 0: identical to _shifted_forward, regardless of noise_site.
    """
    B, T, _ = trans.shape

    if noise_site == "gru" and alpha > 0:
        h = torch.zeros(1, B, ec.gru.hidden_size, device=trans.device, dtype=trans.dtype)
        h_noise_std = alpha / (h.shape[-1] ** 0.5)
        z_steps = []
        for t in range(T):
            _, h = ec.gru(trans[:, t:t + 1, :], h)
            if stim_start <= t < stim_end:
                h = h + h_noise_std * torch.randn_like(h)
            z_steps.append(ec.readout(h.squeeze(0)))
        z = torch.stack(z_steps, dim=1)  # (B, T, latent_dim)
    else:
        z, _ = ec(trans)

    if noise_site == "cb_input" and alpha > 0:
        idx0, idx1 = max(stim_start, 0), min(stim_end, T)
        if idx1 > idx0:
            z = z.clone()
            seg = z[:, idx0:idx1, :]
            z_noise_std = alpha / (seg.shape[-1] ** 0.5)
            z[:, idx0:idx1, :] = seg + z_noise_std * torch.randn_like(seg)

    z_prev = z[:, :-1, :]
    s_curr = trans[:, 1:, :STATE_DIM]
    a_curr = trans[:, 1:, STATE_DIM:STATE_DIM + ACTION_DIM]
    s_target = trans[:, 1:, STATE_DIM + ACTION_DIM:]
    s_pred = cb(s_curr, a_curr, z_prev)
    return s_pred, s_target


def plot_trial_prediction(ec, cb, trans, device=None, fig_path=TRIAL_DIAGNOSTIC_FIG_PATH):
    """Diagnostic plot for a single trial: predicted vs. actual per-object feature
    trajectories (N_OBJECTS x OBJ_FEATURE_DIM subplots by default), so prediction
    quality can be checked feature-by-feature and object-by-object, beyond what the
    aggregate MSE curve shows.

    `trans` is one trial's (seq_len, TRANS_DIM) transition tensor (e.g. a row of
    train_olm's tr_trans) -- a (1, seq_len, TRANS_DIM) batch of size 1 also works.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()

    if trans.dim() == 2:
        trans = trans.unsqueeze(0)

    with torch.no_grad():
        s_pred, s_target = _shifted_forward(ec, cb, trans.to(device))

    s_pred = s_pred[0].cpu().numpy()    # (T-1, STATE_DIM)
    s_target = s_target[0].cpu().numpy()
    t_axis = np.arange(2, s_pred.shape[0] + 2)  # see evaluate_displacement's indexing note

    set_plt_params(style="fast")
    fig, axes = plt.subplots(N_OBJECTS, OBJ_FEATURE_DIM,
                              figsize=(4 * OBJ_FEATURE_DIM, 3 * N_OBJECTS), squeeze=False)

    for obj_idx in range(N_OBJECTS):
        start = obj_idx * OBJ_FEATURE_DIM
        for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
            ax = axes[obj_idx][feat_idx]
            col = start + feat_idx
            ax.plot(t_axis, s_target[:, col], label="actual", color="black")
            ax.plot(t_axis, s_pred[:, col], label="predicted", color="firebrick", ls="--")
            ax.set_title(f"object {obj_idx} -- {feat_name}", fontsize=9)
            ax.set_xlabel("Timestep")
            if feat_idx == 0:
                ax.set_ylabel("Normalized feature value")
            make_axis_nice(ax)

    axes[0][0].legend(fontsize=7)
    fig.suptitle("Predicted vs. actual per-object feature trajectory")
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def train_olm(n_train=N_TRAIN, n_val=N_VAL, seq_len=SEQ_LEN, epochs=EPOCHS,
              batch_size=BATCH_SIZE, lr=LR, seed=0, device=None, ckpt_path=CKPT_PATH,
              trial_fig_path=TRIAL_DIAGNOSTIC_FIG_PATH):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training sequences: {n_train} | Val: {n_val} | Epochs: {epochs}")

    tr_trans = generate_olm_sequences(n_train, seq_len=seq_len, seed=seed)
    va_trans = generate_olm_sequences(n_val, seq_len=seq_len, seed=seed + 1000)

    train_loader = DataLoader(TensorDataset(tr_trans), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_trans), batch_size=batch_size)

    ec = EntorhinalCortex(input_dim=TRANS_DIM, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb = OLMForwardModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=lr)
    history = {"train_mse": [], "val_mse": []}

    print("\nTraining EC -> CB on the OLM task (MSE prediction loss)...\n")

    for epoch in range(1, epochs + 1):
        ec.train(); cb.train()
        total_mse = 0.0
        for (trans,) in train_loader:
            trans = trans.to(device)
            s_pred, s_target = _shifted_forward(ec, cb, trans)
            loss = F.mse_loss(s_pred, s_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item() * trans.shape[0]
        total_mse /= n_train
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval()
        val_mse = 0.0
        with torch.no_grad():
            for (trans,) in val_loader:
                trans = trans.to(device)
                s_pred, s_target = _shifted_forward(ec, cb, trans)
                val_mse += F.mse_loss(s_pred, s_target).item() * trans.shape[0]
        val_mse /= n_val
        history["val_mse"].append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history}, ckpt_path)
    print(f"\nSaved {ckpt_path}")

    plot_trial_prediction(ec, cb, tr_trans[0], device=device, fig_path=trial_fig_path)

    return ec, cb, history, device


def _closed_loop_forward(ec, cb, trans):
    """Autoregressive closed-loop forward pass (see core/trainers/ec_cb.py's
    ECCBTrainer.replay_ec_sequence for the only other precedent of this pattern).

    At each step t: CB predicts s_{t+1} using the STALE context z_{t-1} (from before this
    step's GRU update); the resulting error pred_err_t = cb_pred_prev - s_{t+1} becomes
    part of EC's input for updating to z_t -- replacing the raw next-state block used by
    the non-closed-loop _shifted_forward, not appended alongside it, so the per-step
    input is [s_t, a_t, pred_err_t] (same total width as [s_t, a_t, s_{t+1}], i.e. same
    TRANS_DIM as the non-closed-loop model). CB then predicts s_{t+1} again using the
    fresh z_t, and it's this second prediction the loss is computed on.

    No time-shift: the stale-context mechanism is itself what prevents EC from trivially
    copying the answer, so (unlike _shifted_forward) this predicts s_{t+1} immediately
    rather than s_{t+2}.

    Full-sequence BPTT: differentiates through the entire per-episode unroll (no
    detaching / replay buffer, unlike ECCBTrainer.replay_ec_sequence, which only
    backprops one step at a time against a stored hidden state) -- OLM's episodes are
    short enough (SEQ_LEN ~60) for this to be tractable.

    Returns (s_pred, s_target), each (B, T, STATE_DIM) -- no shift, so T long, not T-1.
    """
    B, T, _ = trans.shape
    h = torch.zeros(1, B, ec.gru.hidden_size, device=trans.device, dtype=trans.dtype)

    s_preds, s_targets = [], []
    for t in range(T):
        s_t = trans[:, t, :STATE_DIM]
        a_t = trans[:, t, STATE_DIM:STATE_DIM + ACTION_DIM]
        s_next = trans[:, t, STATE_DIM + ACTION_DIM:]

        z_prev = ec.readout(h.squeeze(0))          # stale context, from before this step
        cb_pred_prev = cb(s_t, a_t, z_prev)
        pred_err_t = cb_pred_prev - s_next

        step_input = torch.cat([s_t, a_t, pred_err_t], dim=-1).unsqueeze(1)  # (B, 1, TRANS_DIM)
        _, h = ec.gru(step_input, h)
        z_t = ec.readout(h.squeeze(0))              # fresh context, after seeing pred_err_t

        s_pred = cb(s_t, a_t, z_t)                  # this is the prediction the loss uses
        s_preds.append(s_pred)
        s_targets.append(s_next)

    return torch.stack(s_preds, dim=1), torch.stack(s_targets, dim=1)


def train_olm_closed_loop(n_train=N_TRAIN, n_val=N_VAL, seq_len=SEQ_LEN, epochs=EPOCHS,
                           batch_size=BATCH_SIZE, lr=LR, seed=0, device=None,
                           ckpt_path=CLOSED_LOOP_CKPT_PATH):
    """Same task/data as train_olm, but EC/CB are trained with the closed-loop
    _closed_loop_forward instead of _shifted_forward -- a different, incompatible
    architecture (EC's third input block is pred_err_t, not raw s_{t+1}), so this saves
    to its own checkpoint path and does not touch train_olm/CKPT_PATH.

    Note: this is considerably slower per epoch than train_olm -- _closed_loop_forward
    is a Python-level loop over seq_len steps (two CB calls per step) rather than a
    single batched GRU call, so plan for smaller n_train/seq_len when just smoke-testing.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training sequences: {n_train} | Val: {n_val} | Epochs: {epochs}")

    tr_trans = generate_olm_sequences(n_train, seq_len=seq_len, seed=seed)
    va_trans = generate_olm_sequences(n_val, seq_len=seq_len, seed=seed + 1000)

    train_loader = DataLoader(TensorDataset(tr_trans), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_trans), batch_size=batch_size)

    ec = EntorhinalCortex(input_dim=TRANS_DIM, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb = OLMForwardModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=lr)
    history = {"train_mse": [], "val_mse": []}

    print("\nTraining EC -> CB on the OLM task (closed-loop, MSE prediction loss)...\n")

    for epoch in range(1, epochs + 1):
        ec.train(); cb.train()
        total_mse = 0.0
        for (trans,) in train_loader:
            trans = trans.to(device)
            s_pred, s_target = _closed_loop_forward(ec, cb, trans)
            loss = F.mse_loss(s_pred, s_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item() * trans.shape[0]
        total_mse /= n_train
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval()
        val_mse = 0.0
        with torch.no_grad():
            for (trans,) in val_loader:
                trans = trans.to(device)
                s_pred, s_target = _closed_loop_forward(ec, cb, trans)
                val_mse += F.mse_loss(s_pred, s_target).item() * trans.shape[0]
        val_mse /= n_val
        history["val_mse"].append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history}, ckpt_path)
    print(f"\nSaved {ckpt_path}")

    return ec, cb, history, device


def _slot_error(s_pred, s_target, obj_idx):
    """Per-slot squared error, averaged over that slot's OBJ_FEATURE_DIM columns
    (see _visual_feature).

    Unlike the earlier [identity, distance] representation, there's no identity/location
    sub-group to split here: all OBJ_FEATURE_DIM columns are a single ambiguous
    perceptual readout, and every probe (location/shape/size-change) perturbs the same
    columns (just different underlying causes), so a plain full-feature error is the
    right comparison across all three probe types.
    """
    start = obj_idx * OBJ_FEATURE_DIM
    end = start + OBJ_FEATURE_DIM
    return ((s_pred[..., start:end] - s_target[..., start:end]) ** 2).mean(dim=-1)  # (B, T-1)


def evaluate_displacement(ec, cb, seq_len=SEQ_LEN, t_probe=None, seed=0, device=None, fig_path=FIG_PATH):
    """Deterministic-over-conditions, exhaustive location-change probe
    (generate_probe_location_change_data): compare per-slot prediction error between
    control (never moves) and manipulated (relocated to PROBE_CENTER at t_probe)
    episodes, across every enumerated condition (shape/size pair x manipulated slot x
    rotation -- see _enumerate_probe_pairs). Each condition's agent trajectory is its
    own fresh random walk, seeded from `seed`."""
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()

    control_trans, manipulated_trans, moved_idxs, t_probe = generate_probe_location_change_data(
        seq_len=seq_len, t_probe=t_probe, seed=seed,
    )

    with torch.no_grad():
        s_pred_c, s_target_c = _shifted_forward(ec, cb, control_trans.to(device))
        s_pred_m, s_target_m = _shifted_forward(ec, cb, manipulated_trans.to(device))

    other_idxs = 1 - moved_idxs  # only valid for N_OBJECTS == 2

    def gather(s_pred, s_target, idxs):
        return torch.stack([_slot_error(s_pred[i], s_target[i], int(idxs[i])) for i in range(len(idxs))])

    err_moved_manip = gather(s_pred_m, s_target_m, moved_idxs)
    err_moved_ctrl = gather(s_pred_c, s_target_c, moved_idxs)
    err_other_manip = gather(s_pred_m, s_target_m, other_idxs)

    mean_moved_manip = err_moved_manip.mean(dim=0).cpu().numpy()
    mean_moved_ctrl = err_moved_ctrl.mean(dim=0).cpu().numpy()
    mean_other_manip = err_other_manip.mean(dim=0).cpu().numpy()

    std_moved_manip = err_moved_manip.std(dim=0).cpu().numpy()
    std_moved_ctrl = err_moved_ctrl.std(dim=0).cpu().numpy()
    std_other_manip = err_other_manip.std(dim=0).cpu().numpy()

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    # Selection index j (0-indexed into s_pred/s_target) predicts state(j+2):
    # z_prev[j] has seen transitions through row j (i.e. through state(j+1)),
    # and s_target[j] is state_next of row (j+1), i.e. state(j+2).
    t_axis = np.arange(2, seq_len + 1)
    n = len(err_moved_manip)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("OLM (location-change)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean squared prediction error")
    ax.plot(t_axis, mean_moved_manip, label="Relocated object", color="firebrick")
    ax.fill_between(t_axis, mean_moved_manip - std_moved_manip / np.sqrt(n),
                     mean_moved_manip + std_moved_manip / np.sqrt(n), color="firebrick", alpha=0.25)

    ax.plot(t_axis, mean_moved_ctrl, label="Same object, control (never moves)", color="black", ls="--")
    ax.fill_between(t_axis, mean_moved_ctrl - std_moved_ctrl / np.sqrt(n),
                     mean_moved_ctrl + std_moved_ctrl / np.sqrt(n), color="black", alpha=0.25)

    ax.plot(t_axis, mean_other_manip, label="Other (unmanipulated) object", color="steelblue")
    ax.fill_between(t_axis, mean_other_manip - std_other_manip / np.sqrt(n),
                     mean_other_manip + std_other_manip / np.sqrt(n), color="steelblue", alpha=0.25)

    ax.axvline(t_probe, color="gray", linestyle="--", label=f"relocation (t={t_probe})")
    ax.legend(fontsize=7)
    ax.set_xlim(2, seq_len)
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    return {
        "mean_moved_manipulated": mean_moved_manip,
        "mean_moved_control": mean_moved_ctrl,
        "mean_other_manipulated": mean_other_manip,
        "std_moved_manipulated": std_moved_manip,
        "std_moved_control": std_moved_ctrl,
        "std_other_manipulated": std_other_manip,
        "t_probe": t_probe,
        "n_conditions": n,
    }


def _fractional_preference_index(s_pred, s_target, changed_idxs, t_probe, window=3):
    """Per-episode fractional preference index: err_changed / (err_obj0 + err_obj1), where
    each slot's error is averaged over predicted-timesteps [t_probe, t_probe+window)
    (window=None -> average to the end of the sequence), and `changed_idxs` is the
    manipulated slot (0=left, 1=right). ~0.5 = no preference; closer to 1 = more error
    attributed to the changed/novel object ("preference for novel"); closer to 0 = the
    opposite ("preference for familiar"). Only valid for N_OBJECTS == 2 (hardcodes slots
    0/1, same constraint as evaluate_displacement's other_idxs).

    `window` defaults to 3, not None (average to end of episode) -- confirmed empirically
    that the excess prediction error at a manipulation is extremely short-lived (CB
    re-observes the fully-truthful current state every single step, with no occlusion or
    delay, so there's nothing to keep predicting wrong once the new state is directly
    visible again): on a trained model, excess error dropped from 0.122 at t_probe to
    0.0045 one step later, indistinguishable from noise from t_probe+2 on. Averaging over
    a wide window (e.g. the old None default, ~30 steps to end of episode) dilutes that
    one real timestep of signal with ~29 timesteps of noise -- measured to weaken the
    location-change control-vs-changed gap by ~7x (0.059 at window=1 down to 0.008 at
    window=None) for no benefit. window=3 keeps a little slack for off-by-one timing
    without averaging in more than one or two timesteps of pure noise.

    Plain fraction rather than softmax: automatically scale-invariant (dividing by the
    pair's own total already rescales away the model's absolute error magnitude, so no
    temperature hyperparameter is needed), and directly interpretable as "the fraction of
    total prediction error attributable to the changed object" -- the same style as a
    behavioral discrimination index (time at novel / total time).
    """
    err0 = _slot_error(s_pred, s_target, 0)  # (B, T-1)
    err1 = _slot_error(s_pred, s_target, 1)  # (B, T-1)

    start = max(t_probe - 2, 0)  # index j predicts state(j+2); j=t_probe-2 predicts state(t_probe)
    end = err0.shape[1] if window is None else min(start + window, err0.shape[1])

    e0 = err0[:, start:end].mean(dim=1)
    e1 = err1[:, start:end].mean(dim=1)

    total = torch.clamp(e0 + e1, min=1e-12)
    fractions = torch.stack([e0, e1], dim=1) / total.unsqueeze(1)  # (B, 2)
    changed = torch.as_tensor(changed_idxs, dtype=torch.long, device=fractions.device)
    return fractions.gather(1, changed.unsqueeze(1)).squeeze(1)  # (B,)


def _plot_preference_group(data, labels, title, fig_path):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0.5, color="gray", linestyle="--", label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Preference index\n(0 = familiar, 1 = novel)")
    ax.set_title(f"{title} preference index")
    ax.legend()
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


_PROBE_GENERATORS = {
    "location": (generate_probe_location_change_data, "OLM (location-change)", LOCATION_PREF_FIG_PATH, LOCATION_CAUSAL_FIG_PATH),
    "shape": (generate_probe_shape_change_data, "ORM (shape-change)", SHAPE_PREF_FIG_PATH, SHAPE_CAUSAL_FIG_PATH),
    "size": (generate_probe_size_change_data, "ORM (size-change)", SIZE_PREF_FIG_PATH, SIZE_CAUSAL_FIG_PATH),
}


def evaluate_preference_indices(ec, cb, seq_len=SEQ_LEN, t_probe=None, window=3, seed=0,
                                 device=None, fig_paths=None):
    """Compare fractional preference indices across the three deterministic, exhaustive
    probes: location-change (OLM), shape-change (ORM-shape), size-change (ORM-size) --
    each against its own control baseline (should hover around chance = 0.5). Each
    probe's conditions get their own fresh random-walk trajectories, seeded from `seed`
    (shape-change and size-change reuse the same `seed` as location-change -- they're
    independent probes, not paired episodes, so this doesn't create any coupling).

    `window` defaults to 3 -- see _fractional_preference_index for why: the excess
    prediction error at a manipulation is a ~1-timestep spike (CB re-observes the fully
    truthful current state every step, so nothing keeps it wrong once the new state is
    visible again), and the previous window=None default (average to end of episode)
    was measured to dilute that real signal by ~7x. Pass window=None explicitly to
    restore the old end-of-episode averaging if you need it (e.g. to check whether a
    probe's effect is unexpectedly more sustained than location-change's was).

    fig_paths, if given, overrides the default path for one or more of {"location",
    "shape", "size"} (dict); unspecified probes fall back to their module-level default
    (LOCATION_PREF_FIG_PATH etc).

    Returns per-episode preference-index arrays for all three probes' control/change
    conditions plus their t_probe, and saves one comparison box plot per probe.
    Standalone -- not wired into train_olm_multi_seed; call it (and np.savez the result)
    from your own aggregator.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    fig_paths = fig_paths or {}

    results = {"window": -1 if window is None else window}
    for name, (generator, title, default_fig_path, _) in _PROBE_GENERATORS.items():
        ctrl, manip, idxs, t_probe = generator(seq_len=seq_len, t_probe=t_probe, seed=seed)

        with torch.no_grad():
            sp_ctrl, st_ctrl = _shifted_forward(ec, cb, ctrl.to(device))
            sp_manip, st_manip = _shifted_forward(ec, cb, manip.to(device))

        pref_control = _fractional_preference_index(sp_ctrl, st_ctrl, idxs, t_probe, window).cpu().numpy()
        pref_change = _fractional_preference_index(sp_manip, st_manip, idxs, t_probe, window).cpu().numpy()

        _plot_preference_group([pref_control, pref_change], ["control", "changed"], title,
                                fig_paths.get(name, default_fig_path))

        results[f"pref_{name}_control"] = pref_control
        results[f"pref_{name}_change"] = pref_change
        results[f"{name}_t_probe"] = t_probe

    return results


def _plot_causal_noise_group(sweep, change_key, control_key, title, fig_path):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"gru": "firebrick", "cb_input": "steelblue"}

    for site in CAUSAL_NOISE_SITES:
        alphas = sweep[site]["alphas"]
        color = colors.get(site)

        mean_c, sem_c = sweep[site][change_key]["mean"], sweep[site][change_key]["sem"]
        ax.plot(alphas, mean_c, color=color, marker="o", label=f"{site} (changed)")
        ax.fill_between(alphas, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.2)

        mean_ctrl, sem_ctrl = sweep[site][control_key]["mean"], sweep[site][control_key]["sem"]
        ax.plot(alphas, mean_ctrl, color=color, marker="o", linestyle="--", alpha=0.6,
                label=f"{site} (control)")
        ax.fill_between(alphas, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color=color, alpha=0.1)

    ax.axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=sweep[CAUSAL_NOISE_SITES[0]]["alphas"][0],
                 right=sweep[CAUSAL_NOISE_SITES[0]]["alphas"][-1])
    ax.set_xlabel("Noise magnitude (alpha)")
    ax.set_ylabel("Preference index\n(0 = familiar, 1 = novel)")
    ax.set_title(f"{title} causal noise dose-response")
    ax.legend(fontsize=7)
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def evaluate_causal_noise(ec, cb, alphas, stim_start, stim_end, seq_len=SEQ_LEN, t_probe=None,
                           window=3, seed=0, device=None, fig_paths=None):
    """Dose-response sweep: how much does injecting noise into EC's GRU hidden state
    ("gru") vs. into CB's context input ("cb_input") degrade the fractional preference
    index, across a range of noise magnitudes (alphas)?

    `window` defaults to 3, same reasoning and measurement as evaluate_preference_indices
    -- the manipulation's excess prediction error is a ~1-timestep spike, and averaging
    over the old window=None (end-of-episode) default was measured to dilute a real
    ~7x-larger effect down to near-chance. Pass window=None to restore the old behavior.

    Noise is active only during [stim_start, stim_end) -- an explicit "stimulation
    period" you define (e.g. a memory-maintenance window before t_probe) -- with alpha
    itself the literal noise std (see _shifted_forward_noisy), the same absolute
    magnitude at both sites, so any difference between the two curves reflects a real
    circuit difference rather than an accidental mismatch in how much noise was injected.

    Reuses the same three deterministic probes as evaluate_preference_indices
    (location/shape/size-change), but passes them through _shifted_forward_noisy instead
    of _shifted_forward for each (site, alpha) combination -- including the control
    condition, as a sanity check that noise alone (with nothing actually changed) doesn't
    spuriously pull the index away from chance (0.5).

    fig_paths, if given, overrides the default path for one or more of {"location",
    "shape", "size"} (dict); unspecified probes fall back to their module-level default.

    Returns per-(site, alpha) mean/SEM fractional preference index for all three probes'
    control/change conditions, and saves one dose-response line plot per probe.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    fig_paths = fig_paths or {}

    probe_data = {}
    conditions = []
    for name, (generator, _, _, _) in _PROBE_GENERATORS.items():
        ctrl, manip, idxs, probe_t = generator(seq_len=seq_len, t_probe=t_probe, seed=seed)
        probe_data[name] = (ctrl.to(device), manip.to(device), idxs, probe_t)
        conditions += [f"{name}_control", f"{name}_change"]

    sweep = {site: {"alphas": np.array(alphas), **{c: {"mean": [], "sem": []} for c in conditions}}
             for site in CAUSAL_NOISE_SITES}

    with torch.no_grad():
        for site in CAUSAL_NOISE_SITES:
            for alpha in alphas:
                pref = {}
                for name, (ctrl, manip, idxs, t_probe) in probe_data.items():
                    sp_ctrl, st_ctrl = _shifted_forward_noisy(ec, cb, ctrl, site, alpha, stim_start, stim_end)
                    sp_manip, st_manip = _shifted_forward_noisy(ec, cb, manip, site, alpha, stim_start, stim_end)
                    pref[f"{name}_control"] = _fractional_preference_index(sp_ctrl, st_ctrl, idxs, t_probe, window)
                    pref[f"{name}_change"] = _fractional_preference_index(sp_manip, st_manip, idxs, t_probe, window)

                for c, vals in pref.items():
                    vals = vals.cpu().numpy()
                    sweep[site][c]["mean"].append(float(vals.mean()))
                    sweep[site][c]["sem"].append(
                        float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    )

    for site in CAUSAL_NOISE_SITES:
        for c in conditions:
            sweep[site][c]["mean"] = np.array(sweep[site][c]["mean"])
            sweep[site][c]["sem"] = np.array(sweep[site][c]["sem"])

    results = {
        "alphas": np.array(alphas),
        "stim_start": stim_start,
        "stim_end": stim_end,
        "window": -1 if window is None else window,
    }
    for name, (_, title, _, default_fig_path) in _PROBE_GENERATORS.items():
        _plot_causal_noise_group(sweep, f"{name}_change", f"{name}_control", title,
                                  fig_paths.get(name, default_fig_path))
        results[f"{name}_t_probe"] = probe_data[name][3]

    for site in CAUSAL_NOISE_SITES:
        for c in conditions:
            results[f"{site}_{c}_mean"] = sweep[site][c]["mean"]
            results[f"{site}_{c}_sem"] = sweep[site][c]["sem"]

    return results


def _save_training_history(seed_dir, history):
    """Save per-epoch train/val MSE separately from the checkpoint (history.npz + a quick plot),
    so loss curves can be inspected/aggregated without loading model weights."""
    np.savez(
        os.path.join(seed_dir, "training_history.npz"),
        train_mse=np.array(history["train_mse"]),
        val_mse=np.array(history["val_mse"]),
    )

    epochs = np.arange(1, len(history["train_mse"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(epochs, history["val_mse"], label="val MSE", color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("OLM training curve")
    ax.legend()
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(seed_dir, "training_curve.png"), bbox_inches="tight")
    plt.close(fig)


def train_olm_multi_seed(seeds, out_dir=MULTI_SEED_DIR, train_kwargs=None, eval_kwargs=None):
    """Train + evaluate one OLM model per seed, saving each under out_dir/seed_{seed}/:
    model.pt (ec/cb/history), training_history.npz + training_curve.png,
    trial_diagnostic.png (predicted-vs-actual feature trajectory, see
    plot_trial_prediction), displacement_probe.png, displacement_probe_results.npz.

    Both train_olm and evaluate_displacement are called with the same `seed`, so a
    given seed_dir is fully determined by that one seed. Pass shared overrides (e.g.
    seq_len, epochs) via train_kwargs/eval_kwargs -- these are applied to every seed.
    `trial_fig_path` is always overridden to seed_dir/trial_diagnostic.png regardless
    of train_kwargs, so each seed gets its own diagnostic instead of all seeds
    overwriting the same shared default path.

    Returns a list of {"seed", "seed_dir", "history", "probe"} dicts, one per seed.
    """
    train_kwargs = dict(train_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    results = []
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'=' * 60}\nOLM multi-seed: seed={seed}\n{'=' * 60}")
        ec, cb, history, device = train_olm(
            seed=seed, ckpt_path=os.path.join(seed_dir, "model.pt"),
            trial_fig_path=os.path.join(seed_dir, "trial_diagnostic.png"), **train_kwargs,
        )
        _save_training_history(seed_dir, history)

        probe = evaluate_displacement(
            ec, cb, device=device,
            fig_path=os.path.join(seed_dir, "displacement_probe.png"), **eval_kwargs,
        )
        np.savez(os.path.join(seed_dir, "displacement_probe_results.npz"), **probe)

        results.append({"seed": seed, "seed_dir": seed_dir, "history": history, "probe": probe})

    return results


def main(seeds=None, out_dir=MULTI_SEED_DIR, train_kwargs=None, eval_kwargs=None):
    """Default: single-seed train_olm() + evaluate_displacement(), as before.

    Pass `seeds` (e.g. a list of ints) to switch to train_olm_multi_seed instead --
    trains + evaluates one model per seed, saved under out_dir/seed_{seed}/.
    """
    set_plt_params(style="fast", file_format="png")

    if seeds is not None:
        if isinstance(seeds, int):
            seeds = np.arange(seeds)

        return train_olm_multi_seed(seeds, out_dir=out_dir, train_kwargs=train_kwargs, eval_kwargs=eval_kwargs)

    ec, cb, history, device = train_olm()
    evaluate_displacement(ec, cb, device=device)


if __name__ == "__main__":
    main()
