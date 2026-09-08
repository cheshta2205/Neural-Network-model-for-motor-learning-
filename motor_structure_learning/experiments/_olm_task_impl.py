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

Object representation -- three true latents through fixed nonlinear embeddings. An
object is nothing but a point in the arena (`center`) plus one hidden identity scalar,
`curvature`, drawn continuously from [-1, 1] once per object (there is no fixed shape
catalogue any more -- curvature varies continuously, so this is effectively an infinite
family of "shapes" along a single axis, not a discrete set). Per timestep, the agent's
perception of an object is built from exactly three continuous hidden variables:
  theta     -- the object's egocentric bearing angle relative to the agent's heading
  distance  -- Euclidean agent-object distance
  curvature -- the object's own hidden identity value
theta and distance depend on the agent's position/heading relative to the object;
curvature depends only on the object's own identity -- see _raw_object_signal for the
exact construction.

Each of these three TRUE latents is passed through its OWN fixed random nonlinear
embedding (a small one-hidden-layer MLP, tanh nonlinearity, weights drawn once from a
dedicated seed and frozen forever -- see _init_embedding_mlp/THETA_EMBED/DISTANCE_EMBED/
CURVATURE_EMBED) down to a single feature, rather than exposing hand-designed features of
each latent directly (an earlier representation used cos/sin(bearing), raw distance, and
a hand-derived distance-scaled noise/gate on curvature). theta's embedding takes (cos
theta, sin theta) as input -- a seam-free 2D encoding of the angle, since a continuous
SCALAR function of theta alone cannot avoid a discontinuity somewhere (theta wraps at
+-pi; no continuous injective map from a circle onto an interval exists) -- so cos/sin
still appear, but only as an internal representation the nonlinear embedding consumes,
never as a final observed feature themselves. Each embedding's raw input (the (cos,sin)
pair, the rescaled distance, curvature itself) is standardized to zero mean/unit variance
first (THETA_INPUT_MEAN/STD, DISTANCE_INPUT_MEAN/STD, CURVATURE_INPUT_MEAN/STD -- see
_raw_object_signal), since none of them are actually zero-mean/unit-variance under the
true generative process. Each embedding's output then gets
SIGNAL-DEPENDENT additive noise (feature_observed = feature + z * SIGNAL_NOISE_SCALE *
abs(feature), z standard-normal) -- the SAME noise rule for all three latents, rather
than a separate hand-tuned distance-scaled visibility radius per channel.

These RAW_DIM=3 raw numbers ([theta_feature, distance_feature, curvature_feature]) are
standardized to unit variance (RAW_STD, see compute_olm_feature_variance.py) and then
passed through ONE fixed random linear map (MIXING_MATRIX) to produce the observed
OBJ_FEATURE_DIM-dim feature -- see _visual_feature. No single observed dimension is "the"
distance or "the" curvature; every dimension is a mixture of all three hidden variables,
and the network has to unmix them, the same way real sensory channels rarely hand a
hidden variable over in isolation. This replaces an earlier representation built from
hand-designed per-latent features (cos/sin(bearing), raw distance, a hand-derived
curvature visibility gate) with a genuinely nonlinear, non-hand-designed embedding per
latent -- see OLM_TASK.md for the fuller history (including the even earlier
circular-statistics-over-a-polygon representation).

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
from matplotlib.patches import Rectangle
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
# the episode (see _sample_actions). This produces a "correlated random walk" style of
# movement -- closer to how a real animal's gaze/heading actually moves (smooth turning
# with inertia, not snapping to an unrelated random direction every frame), while still
# exploring every direction over the course of an episode.
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

# Field of view: object features are derived only when the object's bearing falls within
# this aperture, centered on the agent's heading -- see _raw_object_signal. Kept as a
# general mechanism even though the default is fully omnidirectional (360deg, i.e. a
# no-op) -- narrowing it makes visibility itself, not just curvature, distance-gated.
APERTURE_DEG = 360.0
APERTURE_RAD = np.deg2rad(APERTURE_DEG)

# Probe episodes (displacement/curvature-change): objects sit at two prescribed slots (not
# randomly placed), with headroom that has no shape/size to worry about any more (objects
# are points). CENTER is the displacement probe's relocation target.
#
# The agent's trajectory itself is an ordinary, freshly-randomized random walk (same
# _sample_actions/generate_olm_sequences mechanism as regular training data) -- NOT a
# scripted path, so that no location is systematically favored (see OLM_TASK.md's
# "teleporting gaze" discussion for the history of this choice) -- it must be freshly
# randomized per episode (not a fixed shared instance), or a fixed walk would just bake in
# whatever asymmetries that one walk happens to have, rather than averaging them out.
#
# A SCRIPTED, guaranteed-visitation trajectory (_controlled_trajectory: center->left->
# center->right->center, guaranteeing the agent gets close to both objects every episode
# -- still defined below, and used by olm_sample_visualization.py's design/prototype) was
# tried here and reverted: it produces a much lower "cruising" baseline prediction error
# than genuine random-walk episodes ever see (straight-line, constant-velocity motion is
# far easier to predict than the correlated-turning OU-process walk), which made an
# otherwise-modest, real proximity-to-object effect (confirmed present in ordinary
# random-walk data too, at roughly 1.3x baseline) look like a ~15x artificial spike at
# every turnaround -- confounding any diagnostic reading a prediction-error spike as
# evidence of "surprise". Free random walks don't guarantee the agent ever approaches
# either object, or approaches both equally often/closely, but don't introduce this
# confound either.
PROBE_LEFT = (-2.5, 0.0)
PROBE_RIGHT = (2.5, 0.0)
PROBE_CENTER = (0.0, 0.0)

# Location-change relocation target (see _probe_object_at): each slot's object teleports
# to the FAR extreme on the OPPOSITE side -- (POS_BOUND, 0) for "left", (-POS_BOUND, 0)
# for "right" -- a 7.5-unit relocation each way, not to PROBE_CENTER (only a 2.5-unit
# move). Confirmed empirically (a manipulation-magnitude audit comparing the resulting
# standardized state-change to a typical curvature-change swap) that the earlier
# PROBE_CENTER-based relocation produced a state-change roughly 3x SMALLER than
# curvature's, since 2.5 units is a small fraction of the position range ordinary
# training data explores (unlike a curvature swap, which can span most of curvature's
# whole [-1, 1] range) -- 7.5 units brings the two manipulations back to a comparable
# scale. Both slots move the SAME distance for symmetry, landing exactly at POS_BOUND,
# still within the range ordinary objects/positions occupy.
PROBE_RELOCATION_TARGETS = {"left": (POS_BOUND, 0.0), "right": (-POS_BOUND, 0.0)}

ACTION_DIM = 2

# --- Latent generative model (see module docstring) --------------------------
RAW_DIM = 3                    # theta_feature, distance_feature, curvature_feature
OBJ_FEATURE_DIM = RAW_DIM      # a square mixing -- see MIXING_MATRIX -- so the observed
                                # feature has the same width as the raw signal it mixes
_RAW_NAMES = ("theta_feature", "distance_feature", "curvature_feature")

# Fixed random nonlinear embedding per latent (see module docstring): a small one-
# hidden-layer MLP, tanh hidden, linear output, weights drawn once from a dedicated seed
# and frozen forever -- same "fixed random, seeded once" philosophy as MIXING_MATRIX.
EMBED_HIDDEN = 8               # hidden units in each latent's embedding MLP
_EMBED_RNG = np.random.default_rng(20260907)  # dedicated seed, independent of MIXING_MATRIX's


def _init_embedding_mlp(rng, in_dim, hidden_dim=EMBED_HIDDEN):
    """Fixed random one-hidden-layer MLP (in_dim -> hidden_dim -> 1, tanh hidden, linear
    output) -- see _apply_embedding. Weights scaled ~1/sqrt(fan_in) (Xavier-ish) so
    typical inputs land where tanh is meaningfully nonlinear rather than immediately
    saturating (which would make the "embedding" collapse to a near-constant)."""
    W1 = (rng.standard_normal((hidden_dim, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
    b1 = (rng.standard_normal(hidden_dim) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((1, hidden_dim)) / np.sqrt(hidden_dim)).astype(np.float32)
    b2 = (rng.standard_normal(1) * 0.1).astype(np.float32)
    return W1, b1, W2, b2


def _apply_embedding(x, params):
    """Apply a fixed embedding MLP (see _init_embedding_mlp) to input vector `x`,
    returning a single scalar feature."""
    W1, b1, W2, b2 = params
    h = np.tanh(W1 @ x + b1)
    return float((W2 @ h + b2)[0])


THETA_EMBED = _init_embedding_mlp(_EMBED_RNG, in_dim=2)      # input: (cos theta, sin theta)
DISTANCE_EMBED = _init_embedding_mlp(_EMBED_RNG, in_dim=1)   # input: rescaled distance
CURVATURE_EMBED = _init_embedding_mlp(_EMBED_RNG, in_dim=1)  # input: curvature (already in [-1, 1])
DISTANCE_EMBED_SCALE = 1.0 / (2.0 * POS_BOUND)  # rescales raw distance (up to
                                                  # ~2*POS_BOUND*sqrt(2)) before embedding,
                                                  # so typical distances land roughly in
                                                  # [0, 1] rather than saturating tanh

# Standardize each embedding's raw input to zero mean / unit variance BEFORE the fixed
# random MLP consumes it (see _raw_object_signal) -- same "well-conditioned input" premise
# _init_embedding_mlp's Xavier-ish weight scaling already assumes. Neither (cos theta, sin
# theta) nor distance*DISTANCE_EMBED_SCALE are actually zero-mean/unit-variance under the
# true generative process: the OU-driven correlated-turning random walk (HEADING_OU_*)
# makes objects land slightly more often behind the agent than ahead, biasing cos(theta)'s
# mean away from zero; distance*DISTANCE_EMBED_SCALE is necessarily >=0, so its mean sits
# well above zero with a fairly narrow spread. Measured empirically over ordinary training
# data (see compute_olm_embedding_input_stats.py, N_SEQS=500 @ SEQ_LEN=60, seed=0; rerun
# and update these if POS_BOUND/MAX_STEP/HEADING_OU_*/APERTURE_DEG change). curvature is
# drawn EXACTLY Uniform(-1, 1) (see sample_object), so its mean/std are the analytic values
# (0, 1/sqrt(3)), not an empirical estimate.
THETA_INPUT_MEAN = np.array([-0.115727, -0.001547], dtype=np.float32)   # cos(theta), sin(theta)
THETA_INPUT_STD = np.array([0.699721, 0.704979], dtype=np.float32)
DISTANCE_INPUT_MEAN = np.array([0.548662], dtype=np.float32)
DISTANCE_INPUT_STD = np.array([0.262871], dtype=np.float32)
CURVATURE_INPUT_MEAN = np.array([0.0], dtype=np.float32)
CURVATURE_INPUT_STD = np.array([1.0 / np.sqrt(3.0)], dtype=np.float32)

# Signal-dependent noise: feature_observed = feature + noise_z * SIGNAL_NOISE_SCALE *
# abs(feature) (noise_z a standard-normal draw) -- classic signal-dependent noise (as in
# motor control's Harris & Wolpert-style noise models), generalized here to sensory
# feature encoding: a feature near zero is read out essentially noise-free, one far from
# zero is proportionally noisier. Applied identically to all three nonlinear features
# (theta/distance/curvature) -- ONE shared constant, deliberately simpler than the
# earlier representation's six separate distance-dependent visibility-radius constants
# (a hand-tuned "how far can you still see this precisely" curve per channel) -- here
# there's no separate notion of physical distance-to-object at all, just "bigger signals
# are noisier," applied uniformly.
SIGNAL_NOISE_SCALE = 0.3

STATE_DIM = N_OBJECTS * OBJ_FEATURE_DIM      # no agent position -- see module docstring
TRANS_DIM = 2 * STATE_DIM + ACTION_DIM       # [state_t, action_t, state_{t+1}]

# Empirical per-raw-feature std over ordinary random-walk data (see
# compute_olm_feature_variance.py, N_SEQS=500 @ SEQ_LEN=60, seed=0), used to standardize
# the RAW_DIM raw variables to unit variance BEFORE the fixed linear mixing
# (MIXING_MATRIX) is applied. Without this, whichever raw feature happens to have larger
# natural scale would dominate an unweighted MSE loss (and _slot_error/preference-index
# computation) almost regardless of how well the others are learned -- the same failure
# mode that motivated normalizing the earlier representation (see OLM_TASK.md). Each
# embedding's own signal-dependent noise draw is INCLUDED in this measurement, since
# that's the actual signal CB has to predict. Rerun compute_olm_feature_variance.py and
# update this if POS_BOUND/EMBED_HIDDEN/the embedding seed/SIGNAL_NOISE_SCALE/
# APERTURE_DEG/PROBE_* change.
RAW_STD = np.array([0.811228, 0.069305, 0.300592], dtype=np.float32)
# theta_feature, distance_feature, curvature_feature

# Fixed random linear mixing: an orthogonal RAW_DIM x RAW_DIM matrix (QR of a
# standard-normal draw), drawn once from a dedicated seed (independent of any
# training/episode seed, so it's constant across runs) and never re-derived --
# analogous to the earlier SHAPE_CODES trick, but now applied uniformly to every raw
# variable, not just identity: every observed feature dimension is a genuine mixture of
# theta/distance/curvature. Orthogonal (not just any random matrix) so the mixing is a
# pure rotation of raw space -- it preserves norms/conditioning rather than shrinking or
# amplifying some direction.
_MIXING_RNG = np.random.default_rng(20260903)
MIXING_MATRIX = np.linalg.qr(_MIXING_RNG.standard_normal((RAW_DIM, RAW_DIM)))[0].astype(np.float32)

# Observation noise: i.i.d. per-timestep Gaussian noise added to each object's whole
# OBJ_FEATURE_DIM feature (post-mixing), applied after standardization/mixing (i.e. in
# already-unit-scale-ish units, so this value is directly a signal-to-noise ratio against
# the clean signal). Infrastructure is fully wired up and tested (generate_olm_sequences
# and all probe generators accept observation_noise_std, with matched noise shared between
# probe control/manipulated pairs -- see _sample_observation_noise), and now ON by
# default (a modest 0.1 std) so every observation carries some generic sensor noise, not
# just the curvature/bearing channels' own distance-scaled noise terms -- pass 0.0 to
# disable. Applied only to state_t, never to the clean state_next target (see
# _build_probe_trans_from_trajectory). See OLM_TASK.md's "Observation noise" section
# (from the earlier representation) for why a sustained decay wasn't achievable this way
# and intermittent-sampling/occlusion is a more likely candidate if a sustained signal is
# wanted later.
OBSERVATION_NOISE_STD = 0.1

CKPT_PATH = "checkpoints/olm_task.pt"
CLOSED_LOOP_CKPT_PATH = "checkpoints/olm_task_closed_loop.pt"
FIG_PATH = "figs/olm_displacement_probe.png"
LOCATION_PREF_FIG_PATH = "figs/olm_preference_index_location.png"
CURVATURE_PREF_FIG_PATH = "figs/olm_preference_index_curvature.png"
LOCATION_CAUSAL_FIG_PATH = "figs/olm_causal_noise_location.png"
CURVATURE_CAUSAL_FIG_PATH = "figs/olm_causal_noise_curvature.png"
TRIAL_DIAGNOSTIC_FIG_PATH = "figs/olm_trial_diagnostic.png"
FEATURE_SPACE_TRAJ_FIG_PATH = "figs/olm_feature_space_trajectories.png"
FEATURE_SPACE_STATE_FIG_PATH = "figs/olm_feature_space_state_space.png"
MULTI_SEED_DIR = "checkpoints/olm_multi_seed"
CLOSED_LOOP_MULTI_SEED_DIR = "checkpoints/olm_closed_loop_multi_seed"

_FEATURE_NAMES = tuple(f"mixed_feature_{i}" for i in range(OBJ_FEATURE_DIM))  # OBJ_FEATURE_DIM
                                                                                # columns, see
                                                                                # _visual_feature --
                                                                                # not individually
                                                                                # interpretable, since
                                                                                # every dim mixes all
                                                                                # of bearing/distance/
                                                                                # curvature

CAUSAL_NOISE_SITES = ("gru", "cb_input")


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _bearing_and_distance(agent_pos, heading, center):
    """(dev, distance) for one object: dev is its egocentric bearing deviation (bearing
    relative to the agent's heading, i.e. camera direction, wrapped to (-pi, pi]);
    distance is the Euclidean agent-object distance. Both are the TRUE, clean values --
    see _raw_object_signal for where noise actually enters (signal-dependent, applied
    AFTER each latent's nonlinear embedding, not here).
    """
    rel = center - agent_pos
    distance = float(np.hypot(rel[0], rel[1]))
    phi = np.arctan2(rel[1], rel[0])
    theta_h = np.arctan2(heading[1], heading[0])
    dev = _wrap_to_pi(phi - theta_h)
    return dev, distance


def _raw_object_signal(agent_pos, heading, center, curvature, aperture=APERTURE_RAD,
                        curvature_noise_z=0.0, bearing_noise_z=0.0, distance_noise_z=0.0):
    """RAW_DIM-dim [theta_feature, distance_feature, curvature_feature] for one object --
    the raw hidden-variable vector BEFORE standardization/mixing (see module docstring
    and _visual_feature). All zero if the object's bearing falls outside the agent's
    field of view (aperture, centered on heading; APERTURE_DEG=360 by default, so this
    check is normally a no-op).

    Each TRUE latent (theta, distance, curvature) is passed through its own fixed random
    nonlinear embedding (THETA_EMBED/DISTANCE_EMBED/CURVATURE_EMBED, see
    _apply_embedding): theta's embedding consumes (cos(dev), sin(dev)) -- a seam-free
    encoding of the angle, see module docstring -- distance is rescaled by
    DISTANCE_EMBED_SCALE first, curvature is used as-is (already in [-1, 1]). Each of
    these three raw inputs is then standardized to zero mean / unit variance
    (THETA_INPUT_MEAN/STD, DISTANCE_INPUT_MEAN/STD, CURVATURE_INPUT_MEAN/STD -- see their
    definitions) before being handed to the embedding MLP, since none of them are actually
    zero-mean/unit-variance under the true generative process.

    Noise is SIGNAL-DEPENDENT and applied AFTER each embedding, not on the true latent:
    feature_observed = feature + noise_z * SIGNAL_NOISE_SCALE * abs(feature).
    `curvature_noise_z`/`bearing_noise_z`/`distance_noise_z` are pre-drawn standard-normal
    values (the caller's responsibility -- see _build_state and _sample_observation_noise
    for the same "common random numbers" pattern shared between paired control/manipulated
    episodes) -- named for the latent whose embedding they perturb (bearing_noise_z ->
    the theta_feature).
    """
    dev, distance = _bearing_and_distance(agent_pos, heading, center)
    if np.abs(dev) >= (aperture / 2.0):
        return np.zeros(RAW_DIM, dtype=np.float32)

    theta_input = (np.array([np.cos(dev), np.sin(dev)], dtype=np.float32) - THETA_INPUT_MEAN) / THETA_INPUT_STD
    distance_input = (np.array([distance * DISTANCE_EMBED_SCALE], dtype=np.float32) - DISTANCE_INPUT_MEAN) / DISTANCE_INPUT_STD
    curvature_input = (np.array([curvature], dtype=np.float32) - CURVATURE_INPUT_MEAN) / CURVATURE_INPUT_STD

    theta_feature = _apply_embedding(theta_input, THETA_EMBED)
    distance_feature = _apply_embedding(distance_input, DISTANCE_EMBED)
    curvature_feature = _apply_embedding(curvature_input, CURVATURE_EMBED)

    theta_feature += bearing_noise_z * SIGNAL_NOISE_SCALE * abs(theta_feature)
    distance_feature += distance_noise_z * SIGNAL_NOISE_SCALE * abs(distance_feature)
    curvature_feature += curvature_noise_z * SIGNAL_NOISE_SCALE * abs(curvature_feature)

    return np.array([theta_feature, distance_feature, curvature_feature], dtype=np.float32)


def _visual_feature(agent_pos, heading, center, curvature, aperture=APERTURE_RAD,
                     curvature_noise_z=0.0, bearing_noise_z=0.0, distance_noise_z=0.0):
    """Observed OBJ_FEATURE_DIM-dim feature for one object: standardize the raw
    [theta_feature, distance_feature, curvature_feature] signal (_raw_object_signal,
    RAW_STD) to unit variance per component, then apply the fixed random linear mixing
    MIXING_MATRIX (see module docstring). All zero if the object isn't visible (see
    _raw_object_signal).
    """
    raw = _raw_object_signal(agent_pos, heading, center, curvature, aperture,
                              curvature_noise_z, bearing_noise_z, distance_noise_z)
    return (MIXING_MATRIX @ (raw / RAW_STD)).astype(np.float32)


def _build_state(agent_pos, heading, centers, curvatures, aperture=APERTURE_RAD,
                  curvature_noise_z=None, bearing_noise_z=None, distance_noise_z=None):
    """State = concatenated visual feature of each object -- no agent position (see
    module docstring: the agent's own position is never observed directly).

    curvature_noise_z/bearing_noise_z/distance_noise_z: pre-drawn standard-normal values,
    one per object (matching `centers`/`curvatures`), for _raw_object_signal's
    distance-scaled curvature/bearing/distance noise -- each defaults to all-zero (no
    noise) if omitted, so callers that don't care about this can ignore it entirely.
    """
    if curvature_noise_z is None:
        curvature_noise_z = [0.0] * len(centers)
    if bearing_noise_z is None:
        bearing_noise_z = [0.0] * len(centers)
    if distance_noise_z is None:
        distance_noise_z = [0.0] * len(centers)
    obj_feats = [_visual_feature(agent_pos, heading, center, curvature, aperture, cz, bz, dz)
                 for center, curvature, cz, bz, dz in
                 zip(centers, curvatures, curvature_noise_z, bearing_noise_z, distance_noise_z)]
    return np.concatenate(obj_feats)


def _sample_observation_noise(rng, n_steps, std=OBSERVATION_NOISE_STD):
    """(n_steps, N_OBJECTS, OBJ_FEATURE_DIM) i.i.d. Gaussian observation noise -- see
    OBSERVATION_NOISE_STD. One draw per ABSOLUTE timestep (0..n_steps-1) per object's
    feature vector, precomputed once per episode rather than redrawn per call, for two
    reasons:
      1. A given true moment gets exactly one noisy reading, regardless of whether a row
         references it as that row's "state_t" or the PREVIOUS row's "state_next" (both
         are the same real timestep, just packed into adjacent rows of the transition
         tensor -- see generate_olm_sequences).
      2. Paired probe episodes (control vs. manipulated -- see _build_probe_transitions)
         can share the exact same noise realization at shared timesteps, isolating the
         manipulation's effect rather than adding spurious extra noise-driven variance
         between the pair ("common random numbers", a standard variance-reduction
         technique for matched comparisons).
    std=0 gives an all-zero array (i.e. the noise-free behavior) without any special-casing
    by callers.
    """
    return (std * rng.standard_normal((n_steps, N_OBJECTS, OBJ_FEATURE_DIM))).astype(np.float32)


def _apply_observation_noise(state, noise_t):
    """Add this timestep's per-object noise (noise_t, shape (N_OBJECTS, OBJ_FEATURE_DIM)
    -- one row of _sample_observation_noise's output) to a (STATE_DIM,) state vector."""
    noisy = state.copy()
    for obj in range(N_OBJECTS):
        start = obj * OBJ_FEATURE_DIM
        noisy[start:start + OBJ_FEATURE_DIM] += noise_t[obj]
    return noisy


def _sample_raw_noise_z(rng, n_steps, n_objects=N_OBJECTS):
    """(n_steps, n_objects) i.i.d. standard-normal draws for _raw_object_signal's
    signal-dependent noise on one of the three nonlinear embeddings (theta/distance/
    curvature -- call this once per latent, see SIGNAL_NOISE_SCALE) -- UNSCALED (std=1):
    the actual per-call noise std depends on that call's own embedded feature value,
    computed inside _raw_object_signal itself, not here. Precomputed once per episode for
    the same "common random numbers" reasons as _sample_observation_noise (a given true
    moment gets exactly one noise draw regardless of which row references it, and paired
    probe episodes can share the same realization)."""
    return rng.standard_normal((n_steps, n_objects)).astype(np.float32)


def sample_object(rng, pos_bound=POS_BOUND):
    """Sample one object: a random center in [-pos_bound, pos_bound]^2 and a random
    curvature in [-1, 1] (see module docstring) -- no shape/size catalogue and no
    rejection sampling needed, since a point object always fits in bounds.

    Returns a dict: center (2,), curvature (scalar).
    """
    center = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
    curvature = float(rng.uniform(-1.0, 1.0))
    return {"center": center, "curvature": curvature}


def visualize_object_samples(n_samples=12, pos_bound=POS_BOUND, seed=0,
                              fig_path="figs/olm_object_samples.png"):
    """Sanity-check plot: draw n_samples randomly-constructed objects (random center,
    random curvature) inside the arena, colored by their hidden curvature value, so the
    distribution can be checked by eye -- centers should cover the arena roughly
    uniformly and curvature values should span [-1, 1].
    """
    rng = np.random.default_rng(seed)
    objects = [sample_object(rng, pos_bound=pos_bound) for _ in range(n_samples)]

    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                            fill=False, edgecolor="black", linestyle="--"))

    centers = np.stack([obj["center"] for obj in objects])
    curvatures = np.array([obj["curvature"] for obj in objects])
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=curvatures, cmap="coolwarm",
                     vmin=-1.0, vmax=1.0, s=250, edgecolor="black", zorder=3)
    for obj in objects:
        ax.annotate(f"{obj['curvature']:.2f}", obj["center"], fontsize=6,
                    ha="center", va="center")
    fig.colorbar(sc, ax=ax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax.set_aspect("equal")
    ax.set_title("Random object placements (color = hidden curvature)")
    make_axis_nice(ax)

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return objects


# Representative curvature values used to build the enumerable, deterministic probes
# below (see _enumerate_probe_pairs) -- ordinary training data instead draws curvature
# continuously per object (see sample_object); this fixed 9-value grid just gives the
# probe machinery something enumerable to pair up, the same structural role
# POLYGON_TYPES's 9 (shape, size) combinations used to play.
CURVATURE_GRID = tuple(float(v) for v in np.linspace(-1.0, 1.0, 9))

# Curvature-change (ORM) probe: how far the manipulated slot's curvature shifts at
# t_probe (direction chosen to keep the result within [-1, 1] -- see
# _build_probe_transitions). Swept over a fixed set of magnitudes so the probe can also
# show how the novelty response scales with the size of the curvature change.
CURVATURE_DELTA_SWEEP = (0.5, 1.0, 1.5)


def plot_feature_space_trajectories(seq_len=SEQ_LEN, center=PROBE_CENTER,
                                     curvature_values=CURVATURE_GRID, seed=0,
                                     fig_path=FEATURE_SPACE_TRAJ_FIG_PATH,
                                     state_fig_path=FEATURE_SPACE_STATE_FIG_PATH,
                                     state_dims=(0, 1)):
    """Illustrative (not evaluative) plot: for ONE shared random-walk agent trajectory,
    place a single object of each curvature in `curvature_values` in turn at the identical
    `center`, and plot the resulting OBJ_FEATURE_DIM-dim feature trajectories (see
    _visual_feature) -- one curve per curvature value -- two ways: (1) each feature vs.
    time (`fig_path`) and (2) as a path through a 2D slice of feature space, `state_dims`
    (default (0, 1)), directly (`state_fig_path`).

    Because every observed dimension is a mixture of ALL three hidden variables (see
    MIXING_MATRIX), there is no dimension that is "purely" curvature the way SHAPE_CODES
    columns once were purely identity -- every curve differs from every other curve in
    every dimension, by an amount set by curvature's own mixing weight into that
    dimension, entangled with whatever bearing/distance the shared trajectory produces.

    A top panel in the vs.-time plot shows agent-to-`center` distance over time (shared by
    every curvature value, since they're all centered at the same point) purely for
    context.

    The state-space plot is the same information laid out as a phase portrait instead --
    lines connect consecutive timesteps in the `state_dims` slice directly, with
    circle/star markers for each curve's start/end point. Color runs from blue (most
    negative curvature) to red (most positive), via the "coolwarm" colormap.

    Returns {curvature_value: (seq_len+1, OBJ_FEATURE_DIM) feature array} for programmatic
    use (e.g. testing) in addition to saving both plots.
    """
    rng = np.random.default_rng(seed)
    positions, headings = _random_walk(rng, seq_len)
    center = np.asarray(center, dtype=np.float32)
    center_tuple = tuple(float(c) for c in center)
    t_axis = np.arange(seq_len + 1)
    distance = np.linalg.norm(positions - center, axis=-1)

    cmap = plt.get_cmap("coolwarm")
    colors = {k: cmap((k + 1.0) / 2.0) for k in curvature_values}

    feats_by_curvature = {
        k: np.stack([_visual_feature(positions[t], headings[t], center, k) for t in range(seq_len + 1)])
        for k in curvature_values
    }

    set_plt_params(style="fast")

    # 1) Each feature vs. time, plus a shared distance-to-object context panel.
    fig, axes = plt.subplots(1 + OBJ_FEATURE_DIM, 1, figsize=(8, 3 * (1 + OBJ_FEATURE_DIM)), sharex=True)

    axes[0].plot(t_axis, distance, color="black")
    axes[0].set_ylabel("distance to object")
    axes[0].set_title(f"Feature-space trajectories -- shared agent path, object at {center_tuple}")
    make_axis_nice(axes[0])

    for k in curvature_values:
        feats = feats_by_curvature[k]
        for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
            axes[1 + feat_idx].plot(t_axis, feats[:, feat_idx], color=colors[k], alpha=0.8,
                                     label=f"curvature={k:.2f}")

    for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
        axes[1 + feat_idx].set_ylabel(feat_name)
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
    for k in curvature_values:
        feats = feats_by_curvature[k]
        ax.plot(feats[:, dim0], feats[:, dim1], color=colors[k], alpha=0.8, label=f"curvature={k:.2f}")
        ax.scatter(feats[0, dim0], feats[0, dim1], color=colors[k], marker="o", s=35, zorder=3)
        ax.scatter(feats[-1, dim0], feats[-1, dim1], color=colors[k], marker="*", s=70, zorder=3)

    ax.set_xlabel(_FEATURE_NAMES[dim0])
    ax.set_ylabel(_FEATURE_NAMES[dim1])
    ax.set_title(f"Feature-space state-space trajectories -- shared agent path, object at {center_tuple}\n"
                 f"(o = start, * = end)")
    ax.legend(fontsize=6, ncol=3, loc="best")
    make_axis_nice(ax)

    os.makedirs(os.path.dirname(state_fig_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(state_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {state_fig_path}")

    return feats_by_curvature


def _sample_actions(rng, n, max_step=MAX_STEP):
    """n continuous random actions: a correlated turning direction (angle) and an
    independent random magnitude in [0, max_step].

    Direction is not redrawn independently every step -- instead, an angular velocity
    (turning rate) follows a discrete-time Ornstein-Uhlenbeck process (exact
    discretization, see HEADING_OU_THETA/HEADING_OU_SIGMA/HEADING_DT), and heading angle
    is that turning rate's running integral over the episode. This is what makes the
    action meaningfully "which way the animal turns and how strongly it moves in that
    direction," rather than an unrelated fresh direction each step. Magnitude is still
    drawn i.i.d. uniform per step (unchanged) -- only the direction component is
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
    """Sample n_objects objects for one episode: independent random centers and
    curvatures (see sample_object) -- unlike the old fixed 9-type catalogue, there's no
    "distinct type" constraint to enforce, since curvature is continuous."""
    return [sample_object(rng, pos_bound=pos_bound) for _ in range(n_objects)]


def _random_walk(rng, seq_len, pos_bound=POS_BOUND, max_step=MAX_STEP):
    """Ordinary random-walk agent trajectory -- same mechanism as generate_olm_sequences.
    Still used by the single-trial probe DIAGNOSTIC PLOTS (plot_probe_trial_diagnostic and
    friends in visualize_olm_probe_trial.py), but no longer by the probe-generation core
    used for quantitative evaluation (_build_probe_transitions/_generate_probe_data) --
    see _controlled_trajectory and the module comment above PROBE_LEFT for that switch.
    Freshly randomized every call (draws from `rng`, which callers should NOT reseed per
    episode).

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


def _split_steps(total_steps, n_legs=4):
    """n_legs positive integers summing EXACTLY to total_steps, as evenly as possible
    (e.g. 30 -> [8, 8, 7, 7], 60 -> [15, 15, 15, 15]) -- unlike dividing evenly and
    rounding each leg independently (which can drift the total away from what was
    requested), the first `total_steps % n_legs` legs simply get one extra step. Each leg
    is at least 1 (so total_steps < n_legs still returns n_legs steps, slightly
    overshooting the request rather than producing a zero-length leg)."""
    base, remainder = divmod(total_steps, n_legs)
    return [max(1, base + 1) if i < remainder else max(1, base) for i in range(n_legs)]


def _straight_leg(start, end, n_steps):
    """n_steps equal-sized positions from start to end (exclusive of start, inclusive of
    end) -- shared by _controlled_trajectory below and olm_sample_visualization.py's
    design/prototype script for this same controlled-trajectory idea."""
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    return [start + (end - start) * (i + 1) / n_steps for i in range(n_steps)]


def _moving_turn(waypoint, dir_in, n_steps, radius):
    """n_steps (position, heading) pairs tracing a semicircular arc of `radius`, starting
    AT `waypoint` tangent to `dir_in` and sweeping a continuous ~180-degree turn to end
    tangent to `-dir_in`, offset by `2*radius` perpendicular to `dir_in` -- a small MOVING
    turn (see _controlled_trajectory), not a stationary one: heading rotates smoothly
    (matching the continuous, bounded turning rate of training data) while the agent
    keeps taking normal-magnitude steps (avoiding the OPPOSITE problem a stationary
    turn-in-place has -- several consecutive near-zero-magnitude actions, which is itself
    just as absent from training as an instant reversal is).

    Choosing a small `radius` keeps the resulting positional drift (up to `2*radius` per
    turn, since the path doesn't return to `waypoint` itself) small; unlike a
    reversal smoothed by directly interpolating an already-moving step's heading over a
    FIXED NUMBER of steps at the ORIGINAL leg speed (which was tried first and produced
    a large, uncontrolled drift -- radius there was an accidental side effect of
    (speed, step count), not a deliberately chosen small quantity), this explicitly fixes
    the radius and derives the per-step speed from it (speed = pi*radius/n_steps), so
    drift stays exactly `2*radius` regardless of `n_steps`.
    """
    perp = np.array([-dir_in[1], dir_in[0]], dtype=np.float32)  # 90-degree CCW rotation of dir_in
    center = waypoint + radius * perp
    thetas = (np.arange(1, n_steps + 1) / n_steps) * np.pi
    cos_t, sin_t = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    positions = center + radius * (sin_t * dir_in - cos_t * perp)
    headings = cos_t * dir_in + sin_t * perp  # already unit-norm (dir_in, perp orthonormal)
    return positions.astype(np.float32), headings.astype(np.float32)


def _controlled_trajectory(center, obj1_center, obj2_center, seq_len, t_probe=None,
                            turn_steps=3, turn_radius=0.2):
    """(positions, headings): a SCRIPTED center->obj1->center->obj2->center ("before")
    then center->obj1->center->obj2->center ("after") trajectory -- guarantees the agent
    visits each object exactly once before AND once after t_probe, at a controlled
    distance, unlike _random_walk's free walk (see the module comment above PROBE_LEFT,
    and olm_sample_visualization.py for the original design/prototype of this).

    Of the 4 leg-to-leg transitions within each half, exactly 2 are genuine ~180-degree
    reversals (arriving at obj1, and arriving at obj2 -- passing back through `center` is
    NOT a reversal, since the legs on either side of it continue in the same direction).
    At each of those 2 reversals, the agent executes a small semicircular MOVING turn
    (`turn_steps` steps, radius `turn_radius` -- see _moving_turn) instead of instantly
    reversing.

    Why: heading here is always the direction of travel (heading[t] = normalized
    displacement, see _random_walk), and _sample_actions's correlated-turning OU process
    never produces an instant reversal during training -- turning rate is continuous and
    bounded. An instantaneous reversal is a discontinuity the model never saw, and was
    found empirically to produce large, PURELY GEOMETRIC prediction-error spikes at
    every turnaround regardless of whether anything was actually manipulated there (the
    same spike appears at every waypoint turnaround even in a CONTROL episode where
    nothing changes) -- a real confound for any diagnostic reading a prediction-error
    spike as evidence of "surprise". A small moving turn (rather than turning in place,
    which was tried first) keeps action magnitude normal throughout -- turning in place
    requires several consecutive near-zero-magnitude actions, which occurs at a rate of
    about 0.0075% in training data (vs. ~5% for any single near-zero step), i.e. is
    itself about as absent from training as an instant reversal.

    `turn_steps` steps are borrowed from the leg immediately AFTER each reversal (which
    then covers its remaining distance slightly faster) to keep the total step count
    exactly seq_len; pass turn_steps=0 to disable and get the original instantly-
    reversing polyline. Because the turn doesn't return exactly to the waypoint (see
    _moving_turn), a small positional drift (up to `2*turn_radius` per turn) accumulates
    onward from each reversal -- so with turning enabled, `center` is reached only
    approximately (not pixel-exactly) at index t_probe, and each waypoint is approached
    closely but not touched exactly.

    t_probe defaults to seq_len // 2 (split evenly), matching _generate_probe_data's
    existing default. Entirely deterministic given the inputs -- no rng argument, unlike
    _random_walk.

    positions/headings: (seq_len+1, 2) each -- same convention as _random_walk.
    """
    if t_probe is None:
        t_probe = seq_len // 2
    center = np.asarray(center, dtype=np.float32)
    obj1_center = np.asarray(obj1_center, dtype=np.float32)
    obj2_center = np.asarray(obj2_center, dtype=np.float32)

    positions = [center.copy()]
    headings = [None]

    def walk(end_wp, n):
        start = positions[-1]
        delta = np.asarray(end_wp, dtype=np.float32) - start
        norm = np.linalg.norm(delta)
        unit_dir = (delta / norm) if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)
        for p in _straight_leg(start, end_wp, n):
            positions.append(p)
            headings.append(unit_dir)

    def turn(n):
        if n <= 0:
            return
        dir_in = headings[-1]
        waypoint = positions[-1]
        turn_positions, turn_headings = _moving_turn(waypoint, dir_in, n, turn_radius)
        for p, h in zip(turn_positions, turn_headings):
            positions.append(p)
            headings.append(h)

    for total_steps in (t_probe, seq_len - t_probe):
        n1, n2, n3, n4 = _split_steps(total_steps)
        p1 = min(turn_steps, n2 - 1) if turn_steps > 0 else 0
        p2 = min(turn_steps, n4 - 1) if turn_steps > 0 else 0
        walk(obj1_center, n1)
        turn(p1)
        walk(center, n2 - p1)
        walk(obj2_center, n3)
        turn(p2)
        walk(center, n4 - p2)

    positions = np.stack(positions).astype(np.float32)
    headings[0] = headings[1] if len(headings) > 1 else np.array([1.0, 0.0], dtype=np.float32)
    headings = np.stack(headings).astype(np.float32)
    return positions, headings


def _enumerate_probe_pairs():
    """Every ordered assignment of 2 distinct CURVATURE_GRID values to the (left, right)
    slots, crossed with which slot is manipulated -- the deterministic condition set
    shared by both probe types. 9*8 ordered assignments * 2 manipulated-slot choices = 144
    combinations (same count as the earlier POLYGON_TYPES-based version, now keyed by
    curvature value instead of (shape, size))."""
    combos = []
    for left_k in CURVATURE_GRID:
        for right_k in CURVATURE_GRID:
            if left_k == right_k:
                continue
            for manipulated_slot in ("left", "right"):
                combos.append((left_k, right_k, manipulated_slot))
    return combos


def _probe_object_at(slot, changed, curvatures, manipulation):
    """(center, curvature) for `slot` ("left"/"right"): its baseline configuration
    (changed=False), or its post-manipulation configuration (changed=True) --

      {"kind": "location"}                 -- center becomes PROBE_RELOCATION_TARGETS[slot]
      {"kind": "curvature", "delta": d}    -- curvature shifts by +-d (whichever direction
                                              keeps it within [-1, 1]), center unchanged

    `curvatures` is {"left": ..., "right": ...}. Factored out of _build_probe_transitions
    so plot_probe_trial_diagnostic can reconstruct the exact same before/after object
    configuration for its spatial panel, without re-deriving it from a transition tensor.
    """
    curvature = curvatures[slot]
    center = PROBE_LEFT if slot == "left" else PROBE_RIGHT
    if changed:
        if manipulation["kind"] == "location":
            center = PROBE_RELOCATION_TARGETS[slot]
        elif manipulation["kind"] == "curvature":
            delta = manipulation["delta"]
            curvature = curvature + delta if curvature + delta <= 1.0 else curvature - delta
    return np.asarray(center, dtype=np.float32), curvature


def _probe_before_after(left_curvature, right_curvature, manipulated_slot, manipulation):
    """({"left": (center, curvature), "right": (...)}, ...) BEFORE and AFTER dicts for one
    probe condition -- only the manipulated_slot's object differs between them, and only
    in the one factor `manipulation` names (see _probe_object_at). Shared by
    _build_probe_transitions and plot_probe_trial_diagnostic."""
    curvatures = {"left": left_curvature, "right": right_curvature}
    other_slot = "right" if manipulated_slot == "left" else "left"
    before = {manipulated_slot: _probe_object_at(manipulated_slot, False, curvatures, manipulation),
              other_slot: _probe_object_at(other_slot, False, curvatures, manipulation)}
    after = {manipulated_slot: _probe_object_at(manipulated_slot, True, curvatures, manipulation),
             other_slot: before[other_slot]}
    return before, after


def _build_probe_trans_from_trajectory(positions, headings, noise, objects_at,
                                        curvature_noise_z=None, bearing_noise_z=None,
                                        distance_noise_z=None):
    """(seq_len, TRANS_DIM) transition tensor built from an ALREADY-GENERATED agent
    trajectory/noise (positions, headings: (seq_len+1, 2); noise: see
    _sample_observation_noise) and `objects_at(t) -> [left_obj, right_obj]` (each a
    (center, curvature) pair, see _probe_object_at) -- the inner loop
    _build_probe_transitions uses for both its control and manipulated variants, factored
    out so plot_probe_trial_diagnostic can build transitions against a trajectory it
    already has (and is plotting) rather than regenerating one from the same rng and
    silently desyncing from what the spatial panel shows.

    curvature_noise_z/bearing_noise_z/distance_noise_z: (seq_len+1, N_OBJECTS) pre-drawn
    standard-normal values for _build_state's distance-scaled curvature/bearing/distance
    noise (see _sample_raw_noise_z) -- each defaults to all-zero (no noise) if omitted.

    Noise (curvature_noise_z, bearing_noise_z, distance_noise_z, and
    `noise`/OBSERVATION_NOISE_STD) is applied ONLY to state_t (the row's own input
    observation), never to state_next (the training TARGET) -- the target is always the
    clean, noise-free true state, so the model is supervised toward ground truth rather
    than toward reproducing whatever noise happened to land on a given reading. This means
    row t's state_next and row (t+1)'s state_t -- the SAME real timestep, referenced from
    two different rows -- are deliberately NOT identical: state_next is clean, state_t is
    noisy.
    """
    seq_len = positions.shape[0] - 1
    if curvature_noise_z is None:
        curvature_noise_z = np.zeros((seq_len + 1, N_OBJECTS), dtype=np.float32)
    if bearing_noise_z is None:
        bearing_noise_z = np.zeros((seq_len + 1, N_OBJECTS), dtype=np.float32)
    if distance_noise_z is None:
        distance_noise_z = np.zeros((seq_len + 1, N_OBJECTS), dtype=np.float32)
    rows = []
    for t in range(seq_len):
        centers_t, curvs_t = zip(*objects_at(t))
        state_t = _apply_observation_noise(
            _build_state(positions[t], headings[t], centers_t, curvs_t,
                         curvature_noise_z=curvature_noise_z[t], bearing_noise_z=bearing_noise_z[t],
                         distance_noise_z=distance_noise_z[t]),
            noise[t])
        action = positions[t + 1] - positions[t]
        centers_next, curvs_next = zip(*objects_at(t + 1))
        state_next = _build_state(positions[t + 1], headings[t + 1], centers_next, curvs_next)
        rows.append(np.concatenate([state_t, action, state_next]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _build_probe_transitions(left_curvature, right_curvature, manipulated_slot, manipulation,
                              rng, seq_len, t_probe, observation_noise_std=OBSERVATION_NOISE_STD,
                              add_raw_noise=True):
    """(control_trans, manipulated_trans), each (seq_len, TRANS_DIM): paired episodes
    sharing one freshly-randomized agent trajectory (_random_walk, drawn from `rng` --
    see the module comment above PROBE_LEFT for why this is a random walk rather than a
    scripted path) and identical objects up to t_probe. At t_probe, the manipulated
    variant changes the manipulated_slot's object per `manipulation` (see
    _probe_object_at); the control variant never changes.

    Both variants also share the exact same observation-noise realization (see
    _sample_observation_noise) -- drawn once, before either transition tensor is built,
    and reused by both -- rather than each variant drawing its own independent noise. This
    preserves the "control and manipulated transitions are byte-identical through the row
    before t_probe" invariant (test_olm_probes.py) despite noisy observations, and means
    any difference after t_probe reflects exactly the manipulation itself, not extra
    noise-driven variance between the pair (common random numbers).

    `add_raw_noise=False` disables the signal-dependent additive noise on each latent's
    nonlinear embedding (see SIGNAL_NOISE_SCALE/_raw_object_signal) on top of
    `observation_noise_std=0.0` for a genuinely noise-free probe -- the three noise draws
    are still consumed from `rng` (just not passed through), so the underlying
    trajectory/manipulation stream is identical regardless of this flag; only the noise
    itself is toggled.
    """
    before, after = _probe_before_after(left_curvature, right_curvature, manipulated_slot, manipulation)
    positions, headings = _random_walk(rng, seq_len)
    noise = _sample_observation_noise(rng, seq_len + 1, std=observation_noise_std)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    if not add_raw_noise:
        curvature_noise_z = bearing_noise_z = distance_noise_z = None

    control_trans = _build_probe_trans_from_trajectory(
        positions, headings, noise, lambda i: [before["left"], before["right"]],
        curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
        distance_noise_z=distance_noise_z)
    manipulated_trans = _build_probe_trans_from_trajectory(
        positions, headings, noise,
        lambda i: [after["left"], after["right"]] if i >= t_probe else [before["left"], before["right"]],
        curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
        distance_noise_z=distance_noise_z,
    )
    return control_trans, manipulated_trans


def _generate_probe_data(manipulations, seq_len=SEQ_LEN, t_probe=None, seed=0,
                          observation_noise_std=OBSERVATION_NOISE_STD, add_raw_noise=True):
    """Shared driver for the two probe generators below: builds (control_trans,
    manipulated_trans) for every (pair/manipulated-slot combination) x manipulation, where
    `manipulations` is a function (left_curvature, right_curvature, manipulated_slot) ->
    list of manipulation dicts to test for that combination (see
    _build_probe_transitions).

    Each episode gets its own freshly-randomized agent trajectory, drawn from a single
    `rng` seeded from `seed` that is NOT reset between episodes -- so every episode's
    walk differs (rather than all episodes sharing one fixed walk, which would just bake
    in whatever location-dependent asymmetries that one walk happens to have instead of
    averaging them out across episodes; see the module comment above PROBE_LEFT -- a
    scripted, guaranteed-visitation trajectory was tried here and reverted: it produced a
    much lower "cruising" baseline prediction error than genuine random-walk episodes
    ever see, which made an otherwise-modest, real proximity-to-object effect look like a
    huge artificial spike at every turnaround, confounding the causal-ablation
    diagnostics). The same `rng` also drives each episode's observation noise (see
    _sample_observation_noise) -- `observation_noise_std` defaults to matching ordinary
    training data (OBSERVATION_NOISE_STD) so probe episodes look like "more of the same"
    environment rather than an unrealistically clean test condition; pass 0.0 to disable.
    `add_raw_noise=False` also disables the curvature/bearing additive noise -- see
    _build_probe_transitions.
    """
    if t_probe is None:
        t_probe = seq_len // 2
    rng = np.random.default_rng(seed)

    control_list, manipulated_list, idx_list = [], [], []
    for left_k, right_k, manipulated_slot in _enumerate_probe_pairs():
        for manipulation in manipulations(left_k, right_k, manipulated_slot):
            c, m = _build_probe_transitions(left_k, right_k, manipulated_slot,
                                             manipulation, rng, seq_len, t_probe,
                                             observation_noise_std=observation_noise_std,
                                             add_raw_noise=add_raw_noise)
            control_list.append(c)
            manipulated_list.append(m)
            idx_list.append(0 if manipulated_slot == "left" else 1)

    return torch.stack(control_list), torch.stack(manipulated_list), np.array(idx_list), t_probe


def generate_probe_location_change_data(seq_len=SEQ_LEN, t_probe=None, seed=0,
                                         observation_noise_std=OBSERVATION_NOISE_STD,
                                         add_raw_noise=True):
    """Deterministic, exhaustive location-change (OLM) probe: every _enumerate_probe_pairs
    combination -- the manipulated slot's object relocates from its prescribed slot to
    PROBE_RELOCATION_TARGETS[slot] (the far extreme on the opposite side) at t_probe,
    curvature unchanged. The agent's trajectory (not the objects/manipulation) is
    randomized per episode -- see _generate_probe_data.

    Returns (control_trans, manipulated_trans) each (144, seq_len, TRANS_DIM),
    manipulated_idxs (144,) in {0=left, 1=right}, and t_probe.
    """
    return _generate_probe_data(lambda l, r, s: [{"kind": "location"}],
                                 seq_len=seq_len, t_probe=t_probe, seed=seed,
                                 observation_noise_std=observation_noise_std,
                                 add_raw_noise=add_raw_noise)


def generate_probe_curvature_change_data(seq_len=SEQ_LEN, t_probe=None, deltas=CURVATURE_DELTA_SWEEP,
                                          seed=0, observation_noise_std=OBSERVATION_NOISE_STD,
                                          add_raw_noise=True):
    """Deterministic, exhaustive curvature-change (ORM) probe: as
    generate_probe_location_change_data, but the manipulated slot's CURVATURE shifts by
    each magnitude in `deltas` at t_probe (direction chosen to keep the result within
    [-1, 1] -- see _build_probe_transitions), holding center fixed.

    Returns (control_trans, manipulated_trans) each (144 * len(deltas), seq_len,
    TRANS_DIM), manipulated_idxs (144 * len(deltas),) in {0=left, 1=right}, and t_probe.
    """
    def manipulations(left_k, right_k, manipulated_slot):
        return [{"kind": "curvature", "delta": d} for d in deltas]

    return _generate_probe_data(manipulations, seq_len=seq_len, t_probe=t_probe, seed=seed,
                                 observation_noise_std=observation_noise_std,
                                 add_raw_noise=add_raw_noise)


def _generate_episode(rng, seq_len, pos_bound, max_step, n_objects, observation_noise_std):
    """One episode's (trans, positions, headings, objects) -- the shared body behind
    generate_olm_sequences (which loops this many times over one shared `rng`, keeping
    only `trans`) and plot_episode_diagnostic (which needs the full agent trajectory and
    object configuration too, to draw a spatial panel alongside the usual
    predicted-vs-actual feature panels).

    Returns:
      trans     -- (seq_len, TRANS_DIM) [state_t, action_t, state_{t+1}] rows
      positions -- (seq_len+1, 2) agent position at each timestep
      headings  -- (seq_len+1, 2) agent heading (unit vector) at each timestep
      objects   -- list of {"center", "curvature"} dicts, see sample_object

    Noise (both observation_noise_std and the curvature-noise mechanism) is applied ONLY
    to state_t, never to state_next (the training target stays clean/noise-free) -- see
    _build_probe_trans_from_trajectory's docstring for the same rule and its rationale.
    """
    agent_pos = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
    heading = np.array([1.0, 0.0], dtype=np.float32)
    episode_objects = _sample_episode_objects(rng, n_objects, pos_bound)
    centers = [obj["center"] for obj in episode_objects]
    curvatures = [obj["curvature"] for obj in episode_objects]

    actions, action_headings = _sample_actions(rng, seq_len, max_step)
    noise = _sample_observation_noise(rng, seq_len + 1, std=observation_noise_std)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1, n_objects)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1, n_objects)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1, n_objects)

    positions = np.zeros((seq_len + 1, 2), dtype=np.float32)
    all_headings = np.zeros((seq_len + 1, 2), dtype=np.float32)
    positions[0] = agent_pos
    all_headings[0] = heading

    rows = []
    for t in range(seq_len):
        state_t = _apply_observation_noise(
            _build_state(agent_pos, heading, centers, curvatures,
                         curvature_noise_z=curvature_noise_z[t], bearing_noise_z=bearing_noise_z[t],
                         distance_noise_z=distance_noise_z[t]),
            noise[t])

        action = actions[t]
        next_pos = np.clip(agent_pos + action, -pos_bound, pos_bound)
        next_heading = action_headings[t]

        state_next = _build_state(next_pos, next_heading, centers, curvatures)
        rows.append(np.concatenate([state_t, action, state_next]))

        agent_pos, heading = next_pos, next_heading
        positions[t + 1] = agent_pos
        all_headings[t + 1] = heading

    return np.stack(rows), positions, all_headings, episode_objects


def generate_olm_sequences(n_seqs, seq_len=SEQ_LEN, pos_bound=POS_BOUND, max_step=MAX_STEP,
                            n_objects=N_OBJECTS, seed=0, observation_noise_std=OBSERVATION_NOISE_STD):
    """Random-action trajectories with two static objects per episode (independent random
    center + curvature per object, see sample_object). State is the FOV-based visual
    feature of each object only -- the agent's own position is never part of the state
    (see module docstring).

    `observation_noise_std` (default OBSERVATION_NOISE_STD) makes this a genuine POMDP --
    see its module-level comment. Pass 0.0 for the noise-free behavior (e.g. when
    measuring the clean signal's own statistics, as compute_olm_feature_variance.py does).
    """
    rng = np.random.default_rng(seed)

    all_trans = []
    for _ in range(n_seqs):
        trans, _, _, _ = _generate_episode(rng, seq_len, pos_bound, max_step, n_objects, observation_noise_std)
        all_trans.append(trans)

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


EPISODE_DIAGNOSTIC_FIG_PATH = "figs/olm_episode_diagnostic.png"
PROBE_TRIAL_TITLES = {"location": "OLM (location-change)", "curvature": "ORM (curvature-change)"}
PROBE_TRIAL_FIG_PATHS = {
    "location": "figs/olm_probe_trial_diagnostic.png",
    "curvature": "figs/orm_probe_trial_diagnostic.png",
}


def _one_minus_r2(actual, pred):
    """1 - R^2 = SS_res/SS_tot for one 1D (actual, pred) trace: the fraction of `actual`'s
    own variance left unexplained by `pred` -- 0 is a perfect trace, 1 is "no better than
    predicting `actual`'s own mean", and it can exceed 1 if `pred` is worse than that mean.
    NaN if `actual` is exactly constant (SS_tot == 0, so the ratio is undefined) -- this
    can still blow up to a large-but-finite value for a NEAR-constant `actual` (e.g. an
    object's curvature feature staying nearly flat all episode): any nonzero residual,
    however small in absolute terms, is huge relative to
    ~zero true variance. That is not a bug -- it correctly says the target barely varied
    this trial, so there's essentially nothing here to have learned from in the first
    place -- but it means a huge 1-R^2 should be read as "not much to explain," not
    "prediction is catastrophic," and is best sanity-checked against that latent's own
    plotted scale.
    """
    ss_res = ((actual - pred) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return ss_res / ss_tot if ss_tot > 0 else np.nan


def _mark_worst_point(ax, t_axis, actual_col, pred_col, color, idx):
    """Mark one series at a GIVEN timestep `idx` (deliberately NOT that column's own
    argmax-error point -- see _plot_object_feature_panels, which picks one shared "worst
    timestep" per object from the AGGREGATE mixed-feature error, matching the spike
    visible in the MSE panel above, and reuses it for every one of that object's panels;
    letting each individual column pick its own peak independently was measured to often
    land on an unrelated noise wiggle rather than the manipulation itself, since a single
    column's own deviation is noisier than the multi-column aggregate): a thin connector
    line from actual to predicted at that timestep, plus a dot on predicted, in `color`
    (the same color as that series' line). A manipulation's excess error is typically
    confined to exactly one timestep (see _fractional_preference_index) -- a single-point
    spike is easy to lose among ~60 already-noisy points on a small subplot, so this makes
    it unmissable regardless of how large or small it turns out to be in any one column.
    """
    ax.plot([t_axis[idx], t_axis[idx]], [actual_col[idx], pred_col[idx]],
            color=color, lw=1.2, alpha=0.6, zorder=5)
    ax.scatter([t_axis[idx]], [pred_col[idx]], color=color, s=22, zorder=6,
               edgecolor="white", linewidth=0.5)


def _rolling_mean(x, window):
    """Centered rolling mean over 1D array `x` with the given `window` width, trimmed
    (not zero-padded) at the edges -- like pandas' .rolling(window, center=True,
    min_periods=1).mean(), so every output point is a plain average of whatever window
    actually fits at that position instead of being biased toward zero near the edges.
    Used to smooth per-timestep MSE traces before computing a ratio/fraction from them, so
    a single noisy timestep doesn't dominate the result (see
    plot_probe_near_far_causal_diagnostic's fractional-attribution panels)."""
    n = len(x)
    half = window // 2
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        out[t] = x[lo:hi].mean()
    return out


# Purely a DIAGNOSTIC/VISUALIZATION heuristic (not part of the generative model itself --
# unlike the earlier distance-gated representation, curvature's noise is now
# signal-dependent, not distance-dependent, so there's no actual "close enough to see it"
# threshold left to be faithful to). Used only to pick an illustrative single-trial seed
# where the agent happens to visit the manipulated object closely at some point, so the
# resulting plot shows a trial with a real approach in it rather than one where the agent
# happened to stay far away the whole time -- see _episode_min_object_distances and the
# seed-search loops in plot_episode_diagnostic/plot_probe_trial_causal_diagnostic/
# plot_probe_near_far_causal_diagnostic etc.
CLOSE_APPROACH_RADIUS = 2.5


def _episode_min_object_distances(positions, episode_objects):
    """(N_OBJECTS,) minimum agent-to-object-center distance over an episode's `positions`
    (seq_len+1, 2) -- one entry per object, the closest the agent ever got to it. Used by
    plot_episode_diagnostic to check whether an episode ever brought the agent within
    CLOSE_APPROACH_RADIUS of a given object, purely to pick a visually illustrative trial
    (see CLOSE_APPROACH_RADIUS -- this is no longer tied to any actual observability
    threshold in the generative model)."""
    centers = np.stack([obj["center"] for obj in episode_objects])  # (N_OBJECTS, 2)
    dists = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=-1)  # (T+1, N_OBJECTS)
    return dists.min(axis=0)


def _new_diagnostic_figure():
    """(fig, gs, row_for) with the row layout shared by plot_episode_diagnostic and
    plot_probe_trial_diagnostic: row 0 = spatial panel (left 3 of OBJ_FEATURE_DIM columns)
    + a secondary panel (last column), then per object a (spacer, mixed-features,
    demixed-latents) row triple -- the spacer is an empty row (SPACER_HEIGHT tall, no
    subplot placed in it) purely to put visual whitespace between one object's block of
    panels and the next, so the two objects' panel groups don't visually run together.

    row_for(obj_idx) returns that object's mixed-feature row index; its demixed-latent
    row is always row_for(obj_idx) + 1.
    """
    SPACER_HEIGHT = 0.15
    INCH_PER_ROW_UNIT = 3.0
    height_ratios = [1.6]
    for _ in range(N_OBJECTS):
        height_ratios += [SPACER_HEIGHT, 1, 1]
    fig_height = 4.5 + N_OBJECTS * (2 + SPACER_HEIGHT) * INCH_PER_ROW_UNIT

    # constrained (not tight) layout: needed for the spatial panel's inset-axes colorbar to
    # size/position correctly -- tight_layout doesn't account for it and left the panel
    # squashed into a corner of its allotted row (see the inset_axes note in each panel).
    fig = plt.figure(figsize=(4 * OBJ_FEATURE_DIM, fig_height), layout="constrained")
    gs = fig.add_gridspec(len(height_ratios), OBJ_FEATURE_DIM, height_ratios=height_ratios)

    def row_for(obj_idx):
        return 2 + 3 * obj_idx  # 1 (spatial/secondary) + 3*obj_idx (prior objects' spacer+2 rows) + 1 (this spacer)

    return fig, gs, row_for


def _plot_object_feature_panels(fig, gs, obj_idx, mixed_row, s_target, pred_series, t_axis, t_probe=None):
    """Draw one object's two panel rows into gs[mixed_row, :] and gs[mixed_row + 1, :]:
    predicted-vs-actual trajectories for its mixed OBJ_FEATURE_DIM columns (as the network
    actually outputs, and its MSE loss is computed on -- e.g. showing how error in one
    hidden variable can leak into every mixed column at once), then demixed back into the
    raw hidden variables themselves (theta_feature, distance_feature, curvature_feature --
    see _RAW_NAMES/_raw_object_signal), which instead show which of the three real hidden
    variables are being predicted well. Demixing is exact, not approximate: MIXING_MATRIX
    is a fixed, known ORTHOGONAL matrix, so its inverse is its own transpose (feature =
    MIXING_MATRIX @ (raw/RAW_STD), so for a batch of feature ROW vectors, raw/RAW_STD =
    feature @ MIXING_MATRIX -- the same inversion compute_olm_feature_variance.py uses);
    multiplying back by RAW_STD then reports each latent in its own true physical units
    instead of standardized ones. Every panel is titled with that column/latent's own
    1 - R^2 per prediction series (see _one_minus_r2) -- read a demixed panel's huge 1-R^2
    (e.g. a curvature the agent never got close enough to observe) as "nothing to explain
    here," not "badly predicted"; check its own y-axis scale.

    `pred_series` is a list of (label, color, linestyle, s_pred) -- one or more predicted
    STATE_DIM-wide arrays to overlay against the actual trace, each contributing its own
    1-R^2 to the title (e.g. a single ("predicted", "firebrick", "--", s_pred) for an
    ordinary diagnostic, or an ("intact", ...)/("pred_err=0", ...) pair for a causal
    ablation comparison -- see plot_episode_causal_diagnostic).

    `t_probe`, if given (probe trials only), draws a vertical dotted marker on every panel
    at that timestep, and -- since the manipulation's excess error is typically a single
    timestep wide (CB re-observes the true state every step, see
    _fractional_preference_index) and easy to miss as just one more wiggle among ~60
    points -- also marks each series at one shared "worst timestep" for this object (the
    peak of the AGGREGATE squared error across all its mixed columns, for the first/
    primary prediction series -- exactly what the MSE panel above plots) with a connector
    line from actual to predicted plus a dot, via _mark_worst_point. Every panel for this
    object marks the SAME timestep, so it lines up with the spike in the MSE panel, rather
    than each column independently picking its own (often noisier, unrelated) peak.
    """
    demixed_row = mixed_row + 1
    start = obj_idx * OBJ_FEATURE_DIM
    end = start + OBJ_FEATURE_DIM

    peak_idx = None
    if t_probe is not None:
        primary_pred = pred_series[0][3]
        agg_sq_err = ((s_target[:, start:end] - primary_pred[:, start:end]) ** 2).mean(axis=-1)
        peak_idx = int(np.argmax(agg_sq_err))

    for feat_idx, feat_name in enumerate(_FEATURE_NAMES):
        ax = fig.add_subplot(gs[mixed_row, feat_idx])
        col = start + feat_idx
        actual_col = s_target[:, col]
        ax.plot(t_axis, actual_col, label="actual", color="black")

        r2_strs = []
        for label, color, ls, s_pred in pred_series:
            pred_col = s_pred[:, col]
            r2_strs.append(f"1-R²({label})={_one_minus_r2(actual_col, pred_col):.3f}")
            ax.plot(t_axis, pred_col, label=label, color=color, ls=ls)
            if peak_idx is not None:
                _mark_worst_point(ax, t_axis, actual_col, pred_col, color, peak_idx)
        if t_probe is not None:
            ax.axvline(t_probe, color="gray", ls=":", lw=1, zorder=1)
        ax.set_title(f"object {obj_idx} -- {feat_name} (mixed)\n" + ", ".join(r2_strs), fontsize=8)
        ax.set_xlabel("Timestep")
        if feat_idx == 0:
            ax.set_ylabel("Mixed feature value")
        if obj_idx == 0 and feat_idx == 0:
            ax.legend(fontsize=7)
        make_axis_nice(ax)

    raw_actual = (s_target[:, start:end] @ MIXING_MATRIX) * RAW_STD  # (T-1, RAW_DIM), physical units
    raw_pred_series = [(label, color, ls, (s_pred[:, start:end] @ MIXING_MATRIX) * RAW_STD)
                        for label, color, ls, s_pred in pred_series]

    for feat_idx, feat_name in enumerate(_RAW_NAMES):
        ax = fig.add_subplot(gs[demixed_row, feat_idx])
        actual_col = raw_actual[:, feat_idx]
        ax.plot(t_axis, actual_col, color="black")

        r2_strs = []
        for label, color, ls, raw_pred in raw_pred_series:
            pred_col = raw_pred[:, feat_idx]
            r2_strs.append(f"1-R²({label})={_one_minus_r2(actual_col, pred_col):.3f}")
            ax.plot(t_axis, pred_col, color=color, ls=ls)
            if peak_idx is not None:
                _mark_worst_point(ax, t_axis, actual_col, pred_col, color, peak_idx)
        if t_probe is not None:
            ax.axvline(t_probe, color="gray", ls=":", lw=1, zorder=1)
        ax.set_title(f"object {obj_idx} -- {feat_name} (demixed)\n" + ", ".join(r2_strs), fontsize=8)
        ax.set_xlabel("Timestep")
        if feat_idx == 0:
            ax.set_ylabel("Latent value (physical units)")
        make_axis_nice(ax)


LABEL_Y_OFFSET = 0.55  # data units: lifts an object's text label clear of its marker instead of centering on it


def _draw_trajectory(ax_spatial, positions, t_probe=None):
    """Draw the agent's path (flat gray line + periodic direction arrows -- not
    colorbar-mapped by time, see plot_episode_diagnostic) plus start/end markers, shared
    by all four *_diagnostic spatial panels. If `t_probe` is given (probe trials only),
    also marks the agent's position at exactly that timestep, so it's visible WHERE along
    the path the agent was when the manipulation happened, not just when.
    """
    ax_spatial.plot(positions[:, 0], positions[:, 1], color="dimgray", alpha=0.7, lw=1.5, zorder=2)
    arrow_stride = max(1, (len(positions) - 1) // 10)  # ~10 arrows spread across the episode
    for i in range(0, len(positions) - 1, arrow_stride):
        ax_spatial.annotate("", xy=positions[i + 1], xytext=positions[i], zorder=2,
                             arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.5,
                                              shrinkA=0, shrinkB=0))
    ax_spatial.scatter(*positions[0], color="black", marker="o", s=80, zorder=3, label="agent start")
    ax_spatial.scatter(*positions[-1], color="black", marker="*", s=140, zorder=3, label="agent end")
    if t_probe is not None:
        ax_spatial.scatter(*positions[t_probe], color="black", marker="X", s=130, zorder=4,
                            label=f"agent at manipulation (t={t_probe})")


def _draw_episode_objects(ax_spatial, episode_objects):
    """Draw an ordinary episode's objects for plot_episode_diagnostic/
    plot_episode_causal_diagnostic: each as a single curvature-colored circle, labeled
    above it (LABEL_Y_OFFSET) rather than directly on top of it. Returns the scatter
    mappable (for the caller's colorbar)."""
    centers = np.stack([obj["center"] for obj in episode_objects])
    curvatures = np.array([obj["curvature"] for obj in episode_objects])
    obj_scatter = ax_spatial.scatter(centers[:, 0], centers[:, 1], c=curvatures, cmap="coolwarm",
                                      vmin=-1.0, vmax=1.0, s=300, edgecolor="black", zorder=4)
    for i, obj in enumerate(episode_objects):
        label_xy = (obj["center"][0], obj["center"][1] + LABEL_Y_OFFSET)
        ax_spatial.annotate(f"obj {i}\ncurv={obj['curvature']:.2f}", label_xy, fontsize=7,
                             ha="center", va="bottom", zorder=5)
    return obj_scatter


def _draw_probe_objects(ax_spatial, kind, before, after, manipulated_slot, other_slot,
                         manipulated_obj_idx, other_obj_idx):
    """Draw a probe trial's objects for plot_probe_trial_diagnostic/
    plot_probe_trial_causal_diagnostic: the unchanged object as a single curvature-colored
    circle, and the manipulated object as a SQUARE ("before") and a CIRCLE ("after") --
    shape, not transparency, distinguishes before/after: alpha-shading two markers on the
    same coolwarm colormap the colorbar itself uses was easy to misread as if it encoded a
    different curvature value, rather than the same value at a different time. Both
    markers are always drawn at full opacity, colored from the exact same (cmap, vmin,
    vmax) as the colorbar and the unchanged object.

    For a location change, before/after sit at different points (connected by an arrow);
    for a curvature change they sit at the SAME point, so the (larger) before square is
    drawn first and peeks out from behind the (smaller) after circle instead of being
    fully hidden by it. Every label sits above its marker (LABEL_Y_OFFSET), clear of the
    largest marker involved.

    Returns the scatter mappable (for the caller's colorbar).
    """
    other_center, other_curv = before[other_slot]
    obj_scatter = ax_spatial.scatter(*other_center, c=[other_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                                      marker="o", s=300, edgecolor="black", zorder=4)
    other_label_xy = (other_center[0], other_center[1] + LABEL_Y_OFFSET)
    ax_spatial.annotate(f"object {other_obj_idx} (unchanged)\ncurv={other_curv:.2f}", other_label_xy,
                         fontsize=7, ha="center", va="bottom", zorder=5)

    before_center, before_curv = before[manipulated_slot]
    after_center, after_curv = after[manipulated_slot]
    if kind == "location":
        ax_spatial.scatter(*before_center, c=[before_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            marker="s", s=300, edgecolor="black", zorder=3)
        before_label_xy = (before_center[0], before_center[1] + LABEL_Y_OFFSET)
        ax_spatial.annotate(f"object {manipulated_obj_idx} (before)\ncurv={before_curv:.2f}", before_label_xy,
                             fontsize=7, ha="center", va="bottom", zorder=5)
        ax_spatial.scatter(*after_center, c=[after_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            marker="o", s=300, edgecolor="black", zorder=4)
        after_label_xy = (after_center[0], after_center[1] + LABEL_Y_OFFSET)
        ax_spatial.annotate(f"object {manipulated_obj_idx} (after)\ncurv={after_curv:.2f}", after_label_xy,
                             fontsize=7, ha="center", va="bottom", zorder=5)
        ax_spatial.annotate("", xy=after_center, xytext=before_center, zorder=3,
                             arrowprops=dict(arrowstyle="-|>", color="black", lw=2, ls="--"))
    else:
        ax_spatial.scatter(*before_center, c=[before_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            marker="s", s=550, edgecolor="black", zorder=3)
        ax_spatial.scatter(*after_center, c=[after_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            marker="o", s=280, edgecolor="black", zorder=4)
        # Extra clearance above LABEL_Y_OFFSET alone: the larger "before" square (s=550)
        # extends further from center than the usual s=300 marker.
        label_xy = (after_center[0], after_center[1] + LABEL_Y_OFFSET + 0.45)
        ax_spatial.annotate(f"object {manipulated_obj_idx}\ncurv: {before_curv:.2f} -> {after_curv:.2f}",
                             label_xy, fontsize=7, ha="center", va="bottom", zorder=5)

    return obj_scatter


def plot_episode_diagnostic(ec, cb, seq_len=SEQ_LEN, seed=0, pos_bound=POS_BOUND, device=None,
                             ensure_curvature_visible=True, max_seed_attempts=200,
                             fig_path=EPISODE_DIAGNOSTIC_FIG_PATH):
    """Single-figure diagnostic for one held-out episode: a spatial panel showing the
    environment configuration (object centers, colored by hidden curvature -- see
    visualize_object_samples) and the agent's trajectory through it (arrows showing
    direction of travel), stacked above predicted-vs-actual per-object trajectories, both
    mixed (as the network actually outputs) and demixed back into the raw hidden
    variables (as plot_trial_prediction, extended) -- so all views of the same run sit in
    one figure instead of several separate ones.

    Generates a fresh episode from `seed` (via _generate_episode, the SAME random-walk
    mechanism generate_olm_sequences uses for training data -- nothing about the agent's
    actual dynamics is scripted or biased) rather than accepting a pre-built transition
    tensor, since it needs the raw agent positions and object configuration for the
    spatial panel, not just the (state, action, state) rows that tensor holds.

    ensure_curvature_visible (default True): purely a VISUALIZATION choice now (curvature's
    noise is signal-dependent, not distance-gated, so unlike the earlier representation
    there's no longer any real observability threshold this enforces) -- a single
    random-walk episode often never brings the agent close to one or both objects at all,
    which would make for a visually uninteresting single-trial figure (the agent's own
    trajectory panel shows it drifting past without ever approaching). Rather than
    hand-scripting a trajectory (which would test the model on dynamics it never actually
    experiences), this instead searches over successive seeds (seed, seed+1, seed+2, ...)
    -- each still an ordinary, un-nudged random walk -- for the first one where EVERY
    object's minimum agent distance (_episode_min_object_distances) drops below
    CLOSE_APPROACH_RADIUS, so the resulting figure shows at least one real approach per
    object. Gives up after max_seed_attempts and uses the last attempted episode anyway
    (with a printed warning) rather than searching forever -- e.g. if CLOSE_APPROACH_RADIUS
    is small relative to pos_bound/seq_len, no nearby seed may qualify. Pass False to
    disable the search and always use `seed` directly.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()

    n_attempts = max(1, max_seed_attempts) if ensure_curvature_visible else 1
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        trans_np, positions, headings, episode_objects = _generate_episode(
            rng, seq_len, pos_bound, MAX_STEP, N_OBJECTS, OBSERVATION_NOISE_STD)
        if not ensure_curvature_visible:
            break
        if np.all(_episode_min_object_distances(positions, episode_objects) < CLOSE_APPROACH_RADIUS):
            if attempt > 0:
                print(f"plot_episode_diagnostic: seed={seed} wasn't curvature-visible for every "
                      f"object; using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_episode_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] brought the "
              f"agent within CLOSE_APPROACH_RADIUS of every object; using seed={used_seed} anyway "
              f"(some curvature panel(s) below will be flat/uninformative).")
    seed = used_seed
    trans = torch.tensor(trans_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        s_pred, s_target = _shifted_forward(ec, cb, trans.to(device))

    s_pred = s_pred[0].cpu().numpy()    # (T-1, STATE_DIM)
    s_target = s_target[0].cpu().numpy()
    t_axis = np.arange(2, s_pred.shape[0] + 2)  # see evaluate_displacement's indexing note

    set_plt_params(style="fast")
    fig, gs, row_for = _new_diagnostic_figure()

    # Row 0's own 4-column subgridspec, DECOUPLED from the outer gs's OBJ_FEATURE_DIM
    # columns (which the demixed-feature rows below need one-per-raw-feature -- currently
    # 3, not 4): row 0 always needs a fixed 2-panel (spatial + MSE) or 3-panel (+
    # fractional-attribution, see plot_probe_trial_causal_diagnostic) layout regardless of
    # how many raw features exist, so it gets its own subgrid rather than trying to fit
    # into however many columns OBJ_FEATURE_DIM happens to be.
    row0 = gs[0, :].subgridspec(1, 4)

    # Spatial panel: environment configuration + agent trajectory. Left 3 of 4 columns --
    # the 4th is reserved for the per-trial MSE-over-time panel below. Deliberately auto
    # (not equal) aspect: an equal-aspect square self-limits to the row's height regardless
    # of how many columns it's given, leaving large horizontal gaps on either side within
    # its cell -- auto aspect instead stretches to fill the full width, at the cost of
    # circles/the arena boundary no longer being visually true squares/circles.
    ax_spatial = fig.add_subplot(row0[0, :3])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))

    # Trajectory (not a colorbar-mapped color-by-time line) -- frees the panel's one
    # colorbar slot for hidden curvature instead (see below) -- and the objects
    # themselves, both via shared helpers (used identically by
    # plot_episode_causal_diagnostic).
    _draw_trajectory(ax_spatial, positions)
    obj_scatter = _draw_episode_objects(ax_spatial, episode_objects)

    # An inset axes (a child of ax_spatial, positioned in ITS OWN axes-fraction coordinates)
    # rather than fig.colorbar(..., ax=ax_spatial): the latter asks the constrained-layout
    # engine to carve colorbar space out of ax_spatial's gridspec cell, which was measured
    # to badly off-center/shrink the whole spatial panel. An inset axes sits outside that
    # negotiation entirely.
    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title("Environment configuration and agent trajectory")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # Per-trial MSE over time, next to the spatial panel: per-object squared error
    # (averaged over that object's OBJ_FEATURE_DIM columns, see _slot_error) plus the
    # combined mean over both objects, at each predicted timestep -- the same one-episode
    # timeline as the spatial panel and the per-feature panels below, just aggregated
    # instead of split out feature-by-feature.
    ax_mse = fig.add_subplot(row0[0, 3])
    obj_colors = ("firebrick", "steelblue")
    for obj_idx in range(N_OBJECTS):
        start = obj_idx * OBJ_FEATURE_DIM
        end = start + OBJ_FEATURE_DIM
        obj_mse = ((s_pred[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        ax_mse.plot(t_axis, obj_mse, color=obj_colors[obj_idx % len(obj_colors)], label=f"object {obj_idx}")
    total_mse = ((s_pred - s_target) ** 2).mean(axis=-1)
    ax_mse.plot(t_axis, total_mse, color="black", ls="--", label="both objects")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("Mean squared prediction error")
    ax_mse.set_title("Prediction error over time")
    ax_mse.legend(fontsize=7)
    make_axis_nice(ax_mse)

    # Predicted vs. actual per-object trajectories, mixed + demixed -- see
    # _plot_object_feature_panels.
    pred_series = [("predicted", "firebrick", "--", s_pred)]
    for obj_idx in range(N_OBJECTS):
        _plot_object_feature_panels(fig, gs, obj_idx, row_for(obj_idx), s_target, pred_series, t_axis)

    fig.suptitle(f"Episode diagnostic (seed={seed})")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {"s_pred": s_pred, "s_target": s_target, "positions": positions, "headings": headings,
            "objects": episode_objects}


CLOSED_LOOP_EPISODE_DIAGNOSTIC_FIG_PATH = "figs/olm_episode_causal_diagnostic.png"


def plot_episode_causal_diagnostic(ec, cb, seq_len=SEQ_LEN, seed=0, pos_bound=POS_BOUND, device=None,
                                    stim_start=0, stim_end=None,
                                    ensure_curvature_visible=True, max_seed_attempts=200,
                                    fig_path=CLOSED_LOOP_EPISODE_DIAGNOSTIC_FIG_PATH):
    """The causal-ablation counterpart of plot_episode_diagnostic, for the CLOSED-LOOP
    model (train_olm_closed_loop/CLOSED_LOOP_CKPT_PATH) -- `ec`/`cb` must be a closed-loop
    pair, NOT the standard model plot_episode_diagnostic uses; they are architecturally
    incompatible (see _closed_loop_forward_causal vs. _shifted_forward).

    Same spatial panel as plot_episode_diagnostic, but the MSE and per-object feature
    panels compare the closed-loop model's INTACT predictions against predictions from the
    SAME model with its GRU's prediction-error input (pred_err_t = s_pred_t - s_next)
    forced to exact zero for timesteps in [stim_start, stim_end) (default: the whole
    trial) -- a causal ablation, not a perturbation: at each ablated step, EC updates its
    context as if CB's prediction had been perfect, so any resulting change in prediction
    quality is attributable entirely to the missing prediction-error signal, not to any
    change in the network's weights or hidden state contents otherwise. See
    _closed_loop_forward_causal.

    Note the closed-loop model's own time alignment: it predicts s_{t+1} directly (no
    2-step shift the way the standard model's _shifted_forward does), so t_axis here runs
    1..seq_len, not 2..seq_len like plot_episode_diagnostic's.

    ensure_curvature_visible/max_seed_attempts: identical seed-search mechanism as
    plot_episode_diagnostic (see there) -- applied once, before running either the intact
    or ablated forward pass, so both are guaranteed to see the exact same episode.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()

    n_attempts = max(1, max_seed_attempts) if ensure_curvature_visible else 1
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        trans_np, positions, headings, episode_objects = _generate_episode(
            rng, seq_len, pos_bound, MAX_STEP, N_OBJECTS, OBSERVATION_NOISE_STD)
        if not ensure_curvature_visible:
            break
        if np.all(_episode_min_object_distances(positions, episode_objects) < CLOSE_APPROACH_RADIUS):
            if attempt > 0:
                print(f"plot_episode_causal_diagnostic: seed={seed} wasn't curvature-visible for "
                      f"every object; using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_episode_causal_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] brought "
              f"the agent within CLOSE_APPROACH_RADIUS of every object; using seed={used_seed} anyway "
              f"(some curvature panel(s) below will be flat/uninformative).")
    seed = used_seed
    trans = torch.tensor(trans_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        s_pred_intact, s_target = _closed_loop_forward_causal(ec, cb, trans.to(device), zero_pred_err=False)
        s_pred_ablated, _ = _closed_loop_forward_causal(ec, cb, trans.to(device), zero_pred_err=True,
                                                          stim_start=stim_start, stim_end=stim_end)

    s_pred_intact = s_pred_intact[0].cpu().numpy()    # (T, STATE_DIM)
    s_pred_ablated = s_pred_ablated[0].cpu().numpy()
    s_target = s_target[0].cpu().numpy()
    t_axis = np.arange(1, s_pred_intact.shape[0] + 1)  # no shift -- see _closed_loop_forward_causal

    set_plt_params(style="fast")
    fig, gs, row_for = _new_diagnostic_figure()
    row0 = gs[0, :].subgridspec(1, 4)  # see plot_episode_diagnostic's note on why row 0
                                        # gets its own subgrid, decoupled from OBJ_FEATURE_DIM

    # Spatial panel: identical to plot_episode_diagnostic (shared helpers).
    ax_spatial = fig.add_subplot(row0[0, :3])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))
    _draw_trajectory(ax_spatial, positions)
    obj_scatter = _draw_episode_objects(ax_spatial, episode_objects)

    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title("Environment configuration and agent trajectory")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # Prediction error: intact closed-loop model vs. the SAME model with pred_err_t forced
    # to zero (see _closed_loop_forward_causal) -- the causal effect of removing EC's
    # prediction-error feedback, holding everything else (weights, trajectory, objects)
    # fixed. Shaded region marks the ablation window.
    ax_mse = fig.add_subplot(row0[0, 3])
    obj_colors = ("firebrick", "steelblue")
    stim_end_plot = seq_len if stim_end is None else stim_end
    ax_mse.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray", alpha=0.12,
                   label="pred_err=0 window", zorder=0)
    for obj_idx in range(N_OBJECTS):
        start = obj_idx * OBJ_FEATURE_DIM
        end = start + OBJ_FEATURE_DIM
        mse_intact = ((s_pred_intact[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        mse_ablated = ((s_pred_ablated[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        ax_mse.plot(t_axis, mse_intact, color=obj_colors[obj_idx], ls="-", alpha=0.6,
                    label=f"object {obj_idx} (intact)")
        ax_mse.plot(t_axis, mse_ablated, color=obj_colors[obj_idx], ls=":", label=f"object {obj_idx} (pred_err=0)")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("Mean squared prediction error")
    ax_mse.set_title("Prediction error: intact vs. pred_err=0")
    ax_mse.legend(fontsize=6)
    make_axis_nice(ax_mse)

    # Predicted vs. actual per-object trajectories: intact vs. pred_err=0, mixed + demixed
    # -- see _plot_object_feature_panels.
    pred_series = [("intact", "firebrick", "--", s_pred_intact), ("pred_err=0", "steelblue", ":", s_pred_ablated)]
    for obj_idx in range(N_OBJECTS):
        _plot_object_feature_panels(fig, gs, obj_idx, row_for(obj_idx), s_target, pred_series, t_axis)

    fig.suptitle(f"Episode causal diagnostic (closed-loop, seed={seed}, "
                 f"pred_err=0 for t in [{stim_start}, {stim_end_plot}))")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "s_pred_intact": s_pred_intact, "s_pred_ablated": s_pred_ablated, "s_target": s_target,
        "positions": positions, "headings": headings, "objects": episode_objects, "seed": seed,
    }


def plot_probe_trial_diagnostic(ec, cb, kind, seq_len=SEQ_LEN, t_probe=None, seed=0,
                                 pos_bound=POS_BOUND, device=None, condition_idx=0,
                                 ensure_curvature_visible=None, max_seed_attempts=200,
                                 fig_path=None):
    """The probe-trial counterpart of plot_episode_diagnostic: same spatial + MSE +
    per-object mixed/demixed feature panels (see _new_diagnostic_figure/
    _plot_object_feature_panels), but built from ONE condition of the deterministic OLM
    (kind="location") or ORM (kind="curvature") probe -- see _enumerate_probe_pairs --
    instead of an ordinary training-distribution episode, so the manipulation itself
    (a relocation or a curvature shift at t_probe) is visible in every panel:
      - the spatial panel draws the manipulated object's BEFORE (faded) and AFTER (solid)
        state -- connected by an arrow for a location change, or as a labeled
        before/after curvature pair at the same point for a curvature change -- alongside
        the unchanged object and the agent's trajectory;
      - the MSE panel overlays this trial's own manipulated-vs-CONTROL (never changes)
        prediction error, the actual novelty-detection comparison this task is built
        around, with a marker at t_probe;
      - every per-object panel below is built from the MANIPULATED trial (the one with
        the discontinuity) and marked at t_probe too.

    `condition_idx` selects which of _enumerate_probe_pairs()'s 144 (curvature-pair,
    manipulated-slot) combinations to use (default: the first). For kind="curvature",
    the manipulation is CURVATURE_DELTA_SWEEP's LARGEST delta, for the clearest jump.

    ensure_curvature_visible (default: True for kind="curvature", False for
    kind="location"): a location change is visible in bearing/distance regardless of how
    close the agent gets (see _raw_object_signal -- only curvature itself is
    distance-gated), so the search is skipped by default for kind="location". For
    kind="curvature", exactly as plot_episode_diagnostic: searches successive seeds
    (seed, seed+1, ...), each still an ordinary, un-nudged random walk, for the first one
    where the agent comes within CLOSE_APPROACH_RADIUS of the manipulated object at some
    point in the trial -- otherwise the change may sit entirely outside what the agent
    ever actually observes. Pass True/False explicitly to override the kind-based default.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    if t_probe is None:
        t_probe = seq_len // 2
    if kind not in PROBE_TRIAL_TITLES:
        raise ValueError(f"kind must be one of {tuple(PROBE_TRIAL_TITLES)}, got {kind!r}")
    fig_path = fig_path or PROBE_TRIAL_FIG_PATHS[kind]
    if ensure_curvature_visible is None:
        ensure_curvature_visible = (kind == "curvature")

    left_k, right_k, manipulated_slot = _enumerate_probe_pairs()[condition_idx]
    other_slot = "right" if manipulated_slot == "left" else "left"
    manipulation = ({"kind": "location"} if kind == "location"
                     else {"kind": "curvature", "delta": CURVATURE_DELTA_SWEEP[-1]})
    before, after = _probe_before_after(left_k, right_k, manipulated_slot, manipulation)
    manipulated_center = after[manipulated_slot][0]  # fixed across the seed search below

    n_attempts = max(1, max_seed_attempts) if ensure_curvature_visible else 1
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        positions, headings = _random_walk(rng, seq_len, pos_bound=pos_bound)
        if not ensure_curvature_visible:
            break
        if np.linalg.norm(positions - manipulated_center, axis=-1).min() < CLOSE_APPROACH_RADIUS:
            if attempt > 0:
                print(f"plot_probe_trial_diagnostic: seed={seed} wasn't curvature-visible; "
                      f"using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_probe_trial_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] brought the "
              f"agent within CLOSE_APPROACH_RADIUS of the manipulated object; using seed={used_seed} "
              f"anyway (the curvature change below may be flat/uninformative).")
    seed = used_seed

    noise = _sample_observation_noise(rng, seq_len + 1, std=OBSERVATION_NOISE_STD)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    control_trans = _build_probe_trans_from_trajectory(
        positions, headings, noise, lambda i: [before["left"], before["right"]],
        curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
        distance_noise_z=distance_noise_z).unsqueeze(0)
    manipulated_trans = _build_probe_trans_from_trajectory(
        positions, headings, noise,
        lambda i: [after["left"], after["right"]] if i >= t_probe else [before["left"], before["right"]],
        curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
        distance_noise_z=distance_noise_z,
    ).unsqueeze(0)

    with torch.no_grad():
        s_pred_c, s_target_c = _shifted_forward(ec, cb, control_trans.to(device))
        s_pred_m, s_target_m = _shifted_forward(ec, cb, manipulated_trans.to(device))

    s_pred_c = s_pred_c[0].cpu().numpy(); s_target_c = s_target_c[0].cpu().numpy()
    s_pred = s_pred_m[0].cpu().numpy(); s_target = s_target_m[0].cpu().numpy()
    t_axis = np.arange(2, s_pred.shape[0] + 2)  # see evaluate_displacement's indexing note

    manipulated_obj_idx = 0 if manipulated_slot == "left" else 1
    other_obj_idx = 1 - manipulated_obj_idx
    title = PROBE_TRIAL_TITLES[kind]

    set_plt_params(style="fast")
    fig, gs, row_for = _new_diagnostic_figure()
    row0 = gs[0, :].subgridspec(1, 4)  # see plot_episode_diagnostic's note on why row 0
                                        # gets its own subgrid, decoupled from OBJ_FEATURE_DIM

    # Spatial panel: environment configuration (before/after the manipulation) + agent
    # trajectory -- see plot_episode_diagnostic for the trajectory/colorbar rationale.
    ax_spatial = fig.add_subplot(row0[0, :3])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))

    _draw_trajectory(ax_spatial, positions, t_probe=t_probe)
    obj_scatter = _draw_probe_objects(ax_spatial, kind, before, after, manipulated_slot, other_slot,
                                       manipulated_obj_idx, other_obj_idx)

    # See plot_episode_diagnostic for why this is an inset axes rather than
    # fig.colorbar(..., ax=ax_spatial).
    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title(f"{title} -- environment + trajectory")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # Prediction error: manipulated trial vs. its own control (never changes) -- the
    # actual novelty-detection comparison this task is built around (see
    # _fractional_preference_index for the aggregate version across all 144+ conditions;
    # this is just the one condition/trial plotted above).
    ax_mse = fig.add_subplot(row0[0, 3])
    obj_colors = ("firebrick", "steelblue")
    for obj_idx in range(N_OBJECTS):
        start = obj_idx * OBJ_FEATURE_DIM
        end = start + OBJ_FEATURE_DIM
        mse_manip = ((s_pred[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        mse_ctrl = ((s_pred_c[:, start:end] - s_target_c[:, start:end]) ** 2).mean(axis=-1)
        ax_mse.plot(t_axis, mse_manip, color=obj_colors[obj_idx], ls="-", label=f"object {obj_idx} (manipulated)")
        ax_mse.plot(t_axis, mse_ctrl, color=obj_colors[obj_idx], ls="--", alpha=0.6, label=f"object {obj_idx} (control)")
    ax_mse.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("Mean squared prediction error")
    ax_mse.set_title("Prediction error: manipulated vs. control")
    ax_mse.legend(fontsize=6)
    make_axis_nice(ax_mse)

    # Predicted vs. actual per-object trajectories (manipulated trial), mixed + demixed,
    # each marked at t_probe -- see _plot_object_feature_panels.
    pred_series = [("predicted", "firebrick", "--", s_pred)]
    for obj_idx in range(N_OBJECTS):
        _plot_object_feature_panels(fig, gs, obj_idx, row_for(obj_idx), s_target, pred_series, t_axis, t_probe=t_probe)

    fig.suptitle(f"{title} probe trial diagnostic (seed={seed}, condition={condition_idx})")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "s_pred": s_pred, "s_target": s_target,
        "s_pred_control": s_pred_c, "s_target_control": s_target_c,
        "positions": positions, "headings": headings,
        "before": before, "after": after, "manipulated_slot": manipulated_slot,
        "t_probe": t_probe, "seed": seed,
    }


PROBE_TRIAL_CAUSAL_FIG_PATHS = {
    "location": "figs/olm_probe_trial_causal_diagnostic.png",
    "curvature": "figs/orm_probe_trial_causal_diagnostic.png",
}

# Default lead time (timesteps) before t_probe at which the pred_err=0 ablation window
# starts, for plot_probe_trial_causal_diagnostic/plot_probe_near_far_causal_diagnostic's
# stim_start=None default -- ablating from t=0 diluted the comparison with a long stretch
# of pre-manipulation steps where intact and ablated have nothing interesting to differ
# on; starting just before the manipulation isolates the window that actually matters.
CAUSAL_STIM_LEAD = 23

# Default periodic prediction-error-noise perturbation schedule for evaluate_causal_
# ablation (see _closed_loop_forward_periodic_noise) -- on for CAUSAL_PERIODIC_ON_DURATION
# consecutive steps, off for the rest of each CAUSAL_PERIODIC_PERIOD-length cycle,
# repeating; CAUSAL_PERIODIC_ALPHA is the expected L2 norm of the noise injected at EACH
# perturbed timestep -- total across all STATE_DIM dimensions together, not per dimension
# (see _closed_loop_forward_periodic_noise).
CAUSAL_PERIODIC_PERIOD = 10
CAUSAL_PERIODIC_ON_DURATION = 3
CAUSAL_PERIODIC_ALPHA = 0.75


def plot_probe_trial_causal_diagnostic(ec, cb, kind, seq_len=SEQ_LEN, t_probe=None, seed=0,
                                        pos_bound=POS_BOUND, device=None, condition_idx=0,
                                        stim_start=None, stim_end=None, attribution_window=5,
                                        ensure_curvature_visible=None, max_seed_attempts=200,
                                        fig_path=None):
    """The causal-ablation counterpart of plot_probe_trial_diagnostic, for the
    CLOSED-LOOP model (train_olm_closed_loop/CLOSED_LOOP_CKPT_PATH) -- `ec`/`cb` must be a
    closed-loop pair, architecturally incompatible with the standard model
    plot_probe_trial_diagnostic uses (see _closed_loop_forward_causal vs.
    _shifted_forward).

    Same probe condition/spatial panel as plot_probe_trial_diagnostic (the manipulated
    object's before/after state, the unchanged object, the agent's trajectory -- see
    there for the full rationale), but the MSE and per-object feature panels compare the
    MANIPULATED trial's INTACT closed-loop predictions against the SAME trial with the
    GRU's prediction-error input forced to exact zero for timesteps in
    [stim_start, stim_end) -- i.e. does the model's reaction to the manipulation at t_probe
    (its "novelty response") depend causally on actually getting to update its context off
    a genuine prediction-error signal, or does removing that signal leave the post-t_probe
    predictions largely unchanged? See _closed_loop_forward_causal for exactly what
    "forced to zero" means. stim_start defaults to CAUSAL_STIM_LEAD timesteps before
    t_probe, not t=0 -- ablating from the very start of the trial buries the comparison in
    a long stretch of pre-manipulation steps where intact and ablated have nothing
    interesting to differ on; starting just before the manipulation isolates the window
    that actually matters. stim_end still defaults to the whole rest of the trial.

    Note the closed-loop model's own time alignment: it predicts s_{t+1} directly (no
    2-step shift the way the standard model's _shifted_forward does), so t_axis here runs
    1..seq_len, not 2..seq_len like plot_probe_trial_diagnostic's -- t_probe itself is on
    the same absolute-timestep clock either way, so the marker lines up correctly.

    A 5th column holds a fractional-attribution panel (manipulated-object MSE /
    (manipulated-object MSE + other-object MSE), bounded [0, 1], intact vs. pred_err=0) --
    the same per-timestep continuous analog of _fractional_preference_index's windowed
    fraction that plot_probe_near_far_causal_diagnostic uses, here for this single trial
    rather than a near/far pair. Both traces are smoothed with a centered
    `attribution_window`-wide rolling mean (see _rolling_mean) BEFORE the fraction is
    computed, since a single timestep's raw MSE is noisy enough to swing the fraction
    wildly even when neither object's error has really changed.

    See plot_probe_trial_diagnostic for `condition_idx`/`ensure_curvature_visible`.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    if t_probe is None:
        t_probe = seq_len // 2
    if stim_start is None:
        stim_start = max(0, t_probe - CAUSAL_STIM_LEAD)  # ablate from just before the manipulation, not from t=0
    if kind not in PROBE_TRIAL_TITLES:
        raise ValueError(f"kind must be one of {tuple(PROBE_TRIAL_TITLES)}, got {kind!r}")
    fig_path = fig_path or PROBE_TRIAL_CAUSAL_FIG_PATHS[kind]
    if ensure_curvature_visible is None:
        ensure_curvature_visible = (kind == "curvature")

    left_k, right_k, manipulated_slot = _enumerate_probe_pairs()[condition_idx]
    other_slot = "right" if manipulated_slot == "left" else "left"
    manipulation = ({"kind": "location"} if kind == "location"
                     else {"kind": "curvature", "delta": CURVATURE_DELTA_SWEEP[-1]})
    before, after = _probe_before_after(left_k, right_k, manipulated_slot, manipulation)
    manipulated_center = after[manipulated_slot][0]

    n_attempts = max(1, max_seed_attempts) if ensure_curvature_visible else 1
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        positions, headings = _random_walk(rng, seq_len, pos_bound=pos_bound)
        if not ensure_curvature_visible:
            break
        if np.linalg.norm(positions - manipulated_center, axis=-1).min() < CLOSE_APPROACH_RADIUS:
            if attempt > 0:
                print(f"plot_probe_trial_causal_diagnostic: seed={seed} wasn't curvature-visible; "
                      f"using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_probe_trial_causal_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] "
              f"brought the agent within CLOSE_APPROACH_RADIUS of the manipulated object; using "
              f"seed={used_seed} anyway (the curvature change below may be flat/uninformative).")
    seed = used_seed

    noise = _sample_observation_noise(rng, seq_len + 1, std=OBSERVATION_NOISE_STD)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    manipulated_trans = _build_probe_trans_from_trajectory(
        positions, headings, noise,
        lambda i: [after["left"], after["right"]] if i >= t_probe else [before["left"], before["right"]],
        curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
        distance_noise_z=distance_noise_z,
    ).unsqueeze(0)

    with torch.no_grad():
        s_pred_intact, s_target = _closed_loop_forward_causal(ec, cb, manipulated_trans.to(device),
                                                                zero_pred_err=False)
        s_pred_ablated, _ = _closed_loop_forward_causal(ec, cb, manipulated_trans.to(device),
                                                          zero_pred_err=True,
                                                          stim_start=stim_start, stim_end=stim_end)

    s_pred_intact = s_pred_intact[0].cpu().numpy()    # (T, STATE_DIM)
    s_pred_ablated = s_pred_ablated[0].cpu().numpy()
    s_target = s_target[0].cpu().numpy()
    t_axis = np.arange(1, s_pred_intact.shape[0] + 1)  # no shift -- see _closed_loop_forward_causal

    manipulated_obj_idx = 0 if manipulated_slot == "left" else 1
    other_obj_idx = 1 - manipulated_obj_idx
    title = PROBE_TRIAL_TITLES[kind]

    set_plt_params(style="fast")
    fig, gs, row_for = _new_diagnostic_figure()
    # Row 0 needs 3 panels here (spatial + MSE + fractional-attribution) -- its own
    # 4-column subgrid, decoupled from OBJ_FEATURE_DIM (see plot_episode_diagnostic's note).
    row0 = gs[0, :].subgridspec(1, 4)

    # Spatial panel: narrower than plot_probe_trial_diagnostic's (2 columns, not 3) to
    # free a column for the fractional-attribution panel below.
    ax_spatial = fig.add_subplot(row0[0, :2])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))
    _draw_trajectory(ax_spatial, positions, t_probe=t_probe)
    obj_scatter = _draw_probe_objects(ax_spatial, kind, before, after, manipulated_slot, other_slot,
                                       manipulated_obj_idx, other_obj_idx)

    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title(f"{title} -- environment + trajectory")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # Prediction error (manipulated trial only): intact closed-loop model vs. the SAME
    # trial with pred_err_t forced to zero -- does the model's response to the
    # manipulation at t_probe depend causally on genuine prediction-error feedback?
    ax_mse = fig.add_subplot(row0[0, 2])
    obj_colors = ("firebrick", "steelblue")
    stim_end_plot = seq_len if stim_end is None else stim_end
    ax_mse.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray", alpha=0.12,
                   label="pred_err=0 window", zorder=0)
    mse_intact_by_obj, mse_ablated_by_obj = {}, {}
    for obj_idx in range(N_OBJECTS):
        start = obj_idx * OBJ_FEATURE_DIM
        end = start + OBJ_FEATURE_DIM
        mse_intact = ((s_pred_intact[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        mse_ablated = ((s_pred_ablated[:, start:end] - s_target[:, start:end]) ** 2).mean(axis=-1)
        mse_intact_by_obj[obj_idx] = mse_intact
        mse_ablated_by_obj[obj_idx] = mse_ablated
        ax_mse.plot(t_axis, mse_intact, color=obj_colors[obj_idx], ls="-", alpha=0.6,
                    label=f"object {obj_idx} (intact)")
        ax_mse.plot(t_axis, mse_ablated, color=obj_colors[obj_idx], ls=":", label=f"object {obj_idx} (pred_err=0)")
    ax_mse.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("Mean squared prediction error")
    ax_mse.set_title("Prediction error: intact vs. pred_err=0")
    ax_mse.legend(fontsize=6)
    make_axis_nice(ax_mse)

    # Fractional-attribution panel: manipulated-object MSE / (manipulated-object MSE +
    # other-object MSE), intact vs. pred_err=0 -- see plot_probe_near_far_causal_
    # diagnostic's identical construction (there, one such panel per near/far trial; here
    # just the one, single-trial version). Smoothed with a centered rolling mean BEFORE
    # taking the ratio, same reasoning as there.
    eps = 1e-8
    mse_manip_i_sm, mse_manip_a_sm, mse_other_i_sm, mse_other_a_sm = (
        _rolling_mean(x, attribution_window) for x in
        (mse_intact_by_obj[manipulated_obj_idx], mse_ablated_by_obj[manipulated_obj_idx],
         mse_intact_by_obj[other_obj_idx], mse_ablated_by_obj[other_obj_idx])
    )
    frac_intact = mse_manip_i_sm / (mse_manip_i_sm + mse_other_i_sm + eps)
    frac_ablated = mse_manip_a_sm / (mse_manip_a_sm + mse_other_a_sm + eps)

    ax_frac = fig.add_subplot(row0[0, 3])
    ax_frac.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray", alpha=0.12,
                     label="pred_err=0 window", zorder=0)
    ax_frac.axhline(0.5, color="gray", ls="-", lw=1, alpha=0.7, label="equal contribution (0.5)")
    ax_frac.plot(t_axis, frac_intact, color="firebrick", ls="-", alpha=0.6, label="intact")
    ax_frac.plot(t_axis, frac_ablated, color="firebrick", ls=":", label="pred_err=0")
    ax_frac.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_frac.set_ylim(0, 1)
    ax_frac.set_xlabel("Timestep")
    ax_frac.set_ylabel("Fractional error attribution\n(manipulated / (manipulated + other))")
    ax_frac.set_title("Error attribution fraction")
    ax_frac.legend(fontsize=6)
    make_axis_nice(ax_frac)

    # Predicted vs. actual per-object trajectories (manipulated trial): intact vs.
    # pred_err=0, mixed + demixed, each marked at t_probe -- see
    # _plot_object_feature_panels.
    pred_series = [("intact", "firebrick", "--", s_pred_intact), ("pred_err=0", "steelblue", ":", s_pred_ablated)]
    for obj_idx in range(N_OBJECTS):
        _plot_object_feature_panels(fig, gs, obj_idx, row_for(obj_idx), s_target, pred_series, t_axis, t_probe=t_probe)

    fig.suptitle(f"{title} probe trial causal diagnostic (closed-loop, seed={seed}, "
                 f"condition={condition_idx}, pred_err=0 for t in [{stim_start}, {stim_end_plot}))")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "s_pred_intact": s_pred_intact, "s_pred_ablated": s_pred_ablated, "s_target": s_target,
        "positions": positions, "headings": headings,
        "before": before, "after": after, "manipulated_slot": manipulated_slot,
        "t_probe": t_probe, "seed": seed,
    }


PROBE_NEAR_FAR_FIG_PATHS = {
    "location": "figs/olm_near_far_diagnostic.png",
    "curvature": "figs/orm_near_far_diagnostic.png",
}


def plot_probe_near_far_diagnostic(ec, cb, kind, seq_len=SEQ_LEN, t_probe=None, seed=0,
                                    pos_bound=POS_BOUND, device=None, condition_idx=0,
                                    max_seed_attempts=200, fig_path=None):
    """Isolates the effect of PROXIMITY on the OLM/ORM novelty response: the SAME
    manipulation (a relocation for kind="location", a curvature shift for
    kind="curvature") applied, in turn, to whichever of the two probe-slot objects the
    agent happens to get closer to over the trial ("near") vs. the other ("far") -- on the
    SAME shared trajectory, noise, and curvature values, so proximity is the only thing
    that differs between the two runs (everything else -- network weights, agent
    dynamics, the (left_curvature, right_curvature) pair -- is held fixed). This directly
    answers "does it matter whether the changed object was near or far from the agent?"
    rather than plot_probe_trial_diagnostic's single fixed manipulated_slot.

    `condition_idx` selects a (left_curvature, right_curvature) pair from
    _enumerate_probe_pairs() for kind="location" (its manipulated-slot half is ignored
    here, since both slots get their own trial). For kind="curvature", `condition_idx` is
    ignored entirely -- BOTH slots instead start at curvature=+1.0 (identical, so there is
    no pre-existing left/right asymmetry to confound the comparison) and the swapped one
    shifts all the way to curvature=-1.0 (a fixed delta of 2.0, deliberately not
    CURVATURE_DELTA_SWEEP's largest -- this comparison wants the single most detectable
    possible change, to make near-vs-far differences as visible as possible, not to sample
    the sweep).

    Searches successive seeds (seed, seed+1, ...) -- ordinary, un-nudged random walks --
    for the first one where the NEARER slot's minimum agent distance drops below
    CLOSE_APPROACH_RADIUS at some point (so there is at least something for the "near"
    condition to detect); whether the "far" slot ever gets that close is exactly what's
    being compared, so it is deliberately NOT required.

    Figure: one spatial panel (both objects, before/after the manipulation, colored
    firebrick-edged for "near" and steelblue-edged for "far"), one MSE panel comparing the
    near-swap and far-swap trials' prediction error for whichever object was swapped in
    each, then TWO full demixed-latent rows (all 3 of _RAW_NAMES each) -- one for the
    near-swap trial, one for the far-swap trial -- so every hidden variable's
    predicted-vs-actual trace can be compared side by side between the two trials, not
    just curvature/distance.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    if t_probe is None:
        t_probe = seq_len // 2
    if kind not in PROBE_TRIAL_TITLES:
        raise ValueError(f"kind must be one of {tuple(PROBE_TRIAL_TITLES)}, got {kind!r}")
    fig_path = fig_path or PROBE_NEAR_FAR_FIG_PATHS[kind]

    if kind == "curvature":
        left_k = right_k = 1.0  # identical baseline -- see docstring
        manipulation = {"kind": "curvature", "delta": 2.0}  # +1.0 -> -1.0 exactly
    else:
        left_k, right_k, _ = _enumerate_probe_pairs()[condition_idx]
        manipulation = {"kind": "location"}

    n_attempts = max(1, max_seed_attempts)
    dist_left = dist_right = None
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        positions, headings = _random_walk(rng, seq_len, pos_bound=pos_bound)
        dist_left = np.linalg.norm(positions - np.asarray(PROBE_LEFT), axis=-1).min()
        dist_right = np.linalg.norm(positions - np.asarray(PROBE_RIGHT), axis=-1).min()
        if min(dist_left, dist_right) < CLOSE_APPROACH_RADIUS:
            if attempt > 0:
                print(f"plot_probe_near_far_diagnostic: seed={seed} had no near-enough slot; "
                      f"using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_probe_near_far_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] brought "
              f"the agent within CLOSE_APPROACH_RADIUS of either slot; using seed={used_seed} anyway "
              f"(the 'near' condition below may be flat/uninformative too).")
    seed = used_seed

    near_slot = "left" if dist_left <= dist_right else "right"
    far_slot = "right" if near_slot == "left" else "left"
    near_obj_idx = 0 if near_slot == "left" else 1
    far_obj_idx = 0 if far_slot == "left" else 1

    noise = _sample_observation_noise(rng, seq_len + 1, std=OBSERVATION_NOISE_STD)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1)

    def build_swap(manipulated_slot):
        before, after = _probe_before_after(left_k, right_k, manipulated_slot, manipulation)
        trans = _build_probe_trans_from_trajectory(
            positions, headings, noise,
            lambda i: [after["left"], after["right"]] if i >= t_probe else [before["left"], before["right"]],
            curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
            distance_noise_z=distance_noise_z,
        ).unsqueeze(0)
        with torch.no_grad():
            s_pred, s_target = _shifted_forward(ec, cb, trans.to(device))
        return s_pred[0].cpu().numpy(), s_target[0].cpu().numpy(), before, after

    s_pred_near, s_target_near, before_near, after_near = build_swap(near_slot)
    s_pred_far, s_target_far, before_far, after_far = build_swap(far_slot)
    t_axis = np.arange(2, s_pred_near.shape[0] + 2)  # see evaluate_displacement's indexing note

    title = PROBE_TRIAL_TITLES[kind]

    set_plt_params(style="fast")
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.6, 1, 1])

    # Spatial panel: both objects, near edged firebrick / far edged steelblue (the same
    # two colors used for the MSE comparison below), facecolor still mapped from the exact
    # (cmap, vmin, vmax) the colorbar uses -- see _draw_probe_objects for why shape/edge,
    # not alpha, distinguishes conditions here.
    ax_spatial = fig.add_subplot(gs[0, :3])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))
    _draw_trajectory(ax_spatial, positions, t_probe=t_probe)

    near_before_center, near_before_curv = before_near[near_slot]
    far_before_center, far_before_curv = before_far[far_slot]

    if kind == "location":
        near_after_center, _ = after_near[near_slot]
        far_after_center, _ = after_far[far_slot]
        ax_spatial.annotate("", xy=near_after_center, xytext=near_before_center, zorder=3,
                             arrowprops=dict(arrowstyle="-|>", color="firebrick", lw=2, ls="--"))
        ax_spatial.annotate("", xy=far_after_center, xytext=far_before_center, zorder=3,
                             arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2, ls="--"))
        ax_spatial.scatter(*near_after_center, marker="*", s=180, color="black", zorder=4,
                            label="relocation target (both trials)")

    ax_spatial.scatter(*near_before_center, c=[near_before_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        s=320, edgecolor="firebrick", linewidth=2.5, zorder=4)
    ax_spatial.annotate(f"object {near_obj_idx} (near)\ncurv={near_before_curv:.2f}",
                         (near_before_center[0], near_before_center[1] + LABEL_Y_OFFSET),
                         fontsize=7, ha="center", va="bottom", zorder=5)

    obj_scatter = ax_spatial.scatter(*far_before_center, c=[far_before_curv], cmap="coolwarm",
                                      vmin=-1.0, vmax=1.0, s=320, edgecolor="steelblue", linewidth=2.5, zorder=4)
    ax_spatial.annotate(f"object {far_obj_idx} (far)\ncurv={far_before_curv:.2f}",
                         (far_before_center[0], far_before_center[1] + LABEL_Y_OFFSET),
                         fontsize=7, ha="center", va="bottom", zorder=5)

    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title(f"{title} near-vs-far -- same trajectory, one object swapped per trial")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # Prediction error of the SWAPPED object: near-swap trial vs. far-swap trial.
    ax_mse = fig.add_subplot(gs[0, 3])
    s_n, e_n = near_obj_idx * OBJ_FEATURE_DIM, (near_obj_idx + 1) * OBJ_FEATURE_DIM
    s_f, e_f = far_obj_idx * OBJ_FEATURE_DIM, (far_obj_idx + 1) * OBJ_FEATURE_DIM
    mse_near = ((s_pred_near[:, s_n:e_n] - s_target_near[:, s_n:e_n]) ** 2).mean(axis=-1)
    mse_far = ((s_pred_far[:, s_f:e_f] - s_target_far[:, s_f:e_f]) ** 2).mean(axis=-1)
    ax_mse.plot(t_axis, mse_near, color="firebrick", label="near swapped")
    ax_mse.plot(t_axis, mse_far, color="steelblue", label="far swapped")
    ax_mse.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("MSE (swapped object)")
    ax_mse.set_title("Prediction error: near- vs. far-swap")
    ax_mse.legend(fontsize=6)
    make_axis_nice(ax_mse)

    # Two full demixed-latent rows -- near-swap trial, then far-swap trial -- ALL 3 of
    # _RAW_NAMES each (not just curvature/distance), so every hidden variable's
    # predicted-vs-actual trace is directly comparable between the two trials, row by row.
    # Predicted lines are colored firebrick (near row) / steelblue (far row) to match the
    # spatial and MSE panels above (NOT the usual single-series "predicted = firebrick"
    # convention -- that would clash with "far" already meaning steelblue elsewhere in
    # this same figure). Each row's own worst-missed timestep (the same peak already
    # visible in the MSE panel above -- mse_near/mse_far's argmax, not each column's own
    # independent peak, so it's the same event across that whole row) is marked via
    # _mark_worst_point, since it's otherwise a single-point spike easy to lose among ~60
    # points (see _plot_object_feature_panels).
    peak_idx_near = int(np.argmax(mse_near))
    peak_idx_far = int(np.argmax(mse_far))

    def demix(s_target, s_pred, obj_idx):
        s, e = obj_idx * OBJ_FEATURE_DIM, (obj_idx + 1) * OBJ_FEATURE_DIM
        raw_actual = (s_target[:, s:e] @ MIXING_MATRIX) * RAW_STD
        raw_pred = (s_pred[:, s:e] @ MIXING_MATRIX) * RAW_STD
        return raw_actual, raw_pred

    raw_actual_near, raw_pred_near = demix(s_target_near, s_pred_near, near_obj_idx)
    raw_actual_far, raw_pred_far = demix(s_target_far, s_pred_far, far_obj_idx)

    rows = [
        (1, "near", near_obj_idx, raw_actual_near, raw_pred_near, "firebrick", peak_idx_near),
        (2, "far", far_obj_idx, raw_actual_far, raw_pred_far, "steelblue", peak_idx_far),
    ]
    for row_idx, row_label, obj_idx, raw_actual, raw_pred, color, peak_idx in rows:
        for feat_idx, feat_name in enumerate(_RAW_NAMES):
            ax = fig.add_subplot(gs[row_idx, feat_idx])
            actual_col = raw_actual[:, feat_idx]
            pred_col = raw_pred[:, feat_idx]
            r2 = _one_minus_r2(actual_col, pred_col)
            ax.plot(t_axis, actual_col, color="black", label="actual")
            ax.plot(t_axis, pred_col, color=color, ls="--", label="predicted")
            _mark_worst_point(ax, t_axis, actual_col, pred_col, color, peak_idx)
            ax.axvline(t_probe, color="gray", ls=":", lw=1, zorder=1)
            ax.set_title(f"{row_label} (obj {obj_idx}) -- {feat_name} (demixed)\n1-R²={r2:.3f}", fontsize=8)
            ax.set_xlabel("Timestep")
            if feat_idx == 0:
                ax.set_ylabel("Latent value (physical units)")
                ax.legend(fontsize=6)
            make_axis_nice(ax)

    fig.suptitle(f"{title} near-vs-far diagnostic (seed={seed}, condition={condition_idx})")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "s_pred_near": s_pred_near, "s_target_near": s_target_near,
        "s_pred_far": s_pred_far, "s_target_far": s_target_far,
        "near_slot": near_slot, "far_slot": far_slot,
        "positions": positions, "headings": headings, "t_probe": t_probe, "seed": seed,
    }


PROBE_NEAR_FAR_CAUSAL_FIG_PATHS = {
    "location": "figs/olm_near_far_causal_diagnostic.png",
    "curvature": "figs/orm_near_far_causal_diagnostic.png",
}


def plot_probe_near_far_causal_diagnostic(ec, cb, kind, seq_len=SEQ_LEN, t_probe=None, seed=0,
                                           pos_bound=POS_BOUND, device=None, condition_idx=0,
                                           stim_start=None, stim_end=None, attribution_window=5,
                                           max_seed_attempts=200, fig_path=None):
    """Combines plot_probe_near_far_diagnostic's proximity comparison with
    plot_probe_trial_causal_diagnostic's causal ablation. Requires the CLOSED-LOOP model
    (train_olm_closed_loop/CLOSED_LOOP_CKPT_PATH -- `ec`/`cb` must be a closed-loop pair,
    architecturally incompatible with the standard model plot_probe_near_far_diagnostic
    uses; see _closed_loop_forward_causal vs. _shifted_forward).

    The SAME near-swap and far-swap trials as plot_probe_near_far_diagnostic (same shared
    trajectory/noise/curvature values -- see there for `condition_idx`, including the
    kind="curvature" +1.0/-1.0 override, and the seed-search criterion) are each run BOTH
    intact and with the GRU's prediction-error input forced to zero for timesteps in
    [stim_start, stim_end) (see _closed_loop_forward_causal), so FOUR conditions total
    (near-intact, near-ablated, far-intact, far-ablated) are directly comparable: does
    proximity matter more, less, or differently once genuine prediction-error feedback is
    no longer available to the model? stim_start defaults to CAUSAL_STIM_LEAD
    timesteps before t_probe, not t=0 -- see plot_probe_trial_causal_diagnostic for why;
    stim_end still defaults to the whole rest of the trial.

    Figure: spatial + MSE panel as plot_probe_near_far_diagnostic, PLUS a 5th column
    holding two fractional-attribution panels (swapped-object MSE / (swapped-object MSE +
    unmoved-object MSE), bounded [0, 1], intact vs. pred_err=0 -- a per-timestep,
    continuous analog of _fractional_preference_index's windowed fraction, answering
    "what fraction of the two objects' combined error is attributed to the one that
    actually changed, and does that depend on prediction-error feedback?"), one spanning
    the near trial's two rows and one spanning the far trial's two rows. Each trial's four
    underlying per-timestep MSE traces (swapped/unmoved x intact/ablated) are smoothed
    with a centered `attribution_window`-wide rolling mean (see _rolling_mean) BEFORE the
    fraction is computed from them, since a single timestep's raw MSE is noisy enough to
    swing the fraction wildly even when neither object's error has really changed -- this
    trades a little temporal precision for a fraction trace that's actually readable.
    Then FOUR full demixed-latent rows (all 3 of _RAW_NAMES each), not two: each trial
    predicts the whole state regardless of which object moved, so both the near-swap and
    far-swap trials each get two rows -- their SWAPPED object and their OTHER (unmoved
    that trial) object -- to show what happens to the object that DIDN'T change too, not
    just the one that did. Color is by trial (near-swap=firebrick, far-swap=steelblue)
    regardless of which object a row shows. Every panel overlays TWO predictions, intact
    (dashed) and pred_err=0 (dotted), and the MSE/fraction panels shade the ablation
    window. Note the closed-loop model's own time alignment: it predicts s_{t+1} directly
    (no 2-step shift), so t_axis here runs 1..seq_len, not 2..seq_len.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    if t_probe is None:
        t_probe = seq_len // 2
    if stim_start is None:
        stim_start = max(0, t_probe - CAUSAL_STIM_LEAD)  # ablate from just before the manipulation, not from t=0
    if kind not in PROBE_TRIAL_TITLES:
        raise ValueError(f"kind must be one of {tuple(PROBE_TRIAL_TITLES)}, got {kind!r}")
    fig_path = fig_path or PROBE_NEAR_FAR_CAUSAL_FIG_PATHS[kind]

    if kind == "curvature":
        left_k = right_k = 1.0  # identical baseline, swap goes all the way to -1.0
        manipulation = {"kind": "curvature", "delta": 2.0}
    else:
        left_k, right_k, _ = _enumerate_probe_pairs()[condition_idx]
        manipulation = {"kind": "location"}

    n_attempts = max(1, max_seed_attempts)
    dist_left = dist_right = None
    for attempt in range(n_attempts):
        used_seed = seed + attempt
        rng = np.random.default_rng(used_seed)
        positions, headings = _random_walk(rng, seq_len, pos_bound=pos_bound)
        dist_left = np.linalg.norm(positions - np.asarray(PROBE_LEFT), axis=-1).min()
        dist_right = np.linalg.norm(positions - np.asarray(PROBE_RIGHT), axis=-1).min()
        if min(dist_left, dist_right) < CLOSE_APPROACH_RADIUS:
            if attempt > 0:
                print(f"plot_probe_near_far_causal_diagnostic: seed={seed} had no near-enough slot; "
                      f"using seed={used_seed} instead (found after {attempt + 1} tries).")
            break
    else:
        print(f"plot_probe_near_far_causal_diagnostic: no seed in [{seed}, {seed + n_attempts - 1}] "
              f"brought the agent within CLOSE_APPROACH_RADIUS of either slot; using seed={used_seed} "
              f"anyway (the 'near' condition below may be flat/uninformative too).")
    seed = used_seed

    near_slot = "left" if dist_left <= dist_right else "right"
    far_slot = "right" if near_slot == "left" else "left"
    near_obj_idx = 0 if near_slot == "left" else 1
    far_obj_idx = 0 if far_slot == "left" else 1

    noise = _sample_observation_noise(rng, seq_len + 1, std=OBSERVATION_NOISE_STD)
    curvature_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    bearing_noise_z = _sample_raw_noise_z(rng, seq_len + 1)
    distance_noise_z = _sample_raw_noise_z(rng, seq_len + 1)

    def build_swap(manipulated_slot):
        before, after = _probe_before_after(left_k, right_k, manipulated_slot, manipulation)
        trans = _build_probe_trans_from_trajectory(
            positions, headings, noise,
            lambda i: [after["left"], after["right"]] if i >= t_probe else [before["left"], before["right"]],
            curvature_noise_z=curvature_noise_z, bearing_noise_z=bearing_noise_z,
            distance_noise_z=distance_noise_z,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            s_pred_intact, s_target = _closed_loop_forward_causal(ec, cb, trans, zero_pred_err=False)
            s_pred_ablated, _ = _closed_loop_forward_causal(ec, cb, trans, zero_pred_err=True,
                                                              stim_start=stim_start, stim_end=stim_end)
        return (s_pred_intact[0].cpu().numpy(), s_pred_ablated[0].cpu().numpy(),
                s_target[0].cpu().numpy(), before, after)

    s_pred_near_i, s_pred_near_a, s_target_near, before_near, after_near = build_swap(near_slot)
    s_pred_far_i, s_pred_far_a, s_target_far, before_far, after_far = build_swap(far_slot)
    t_axis = np.arange(1, s_pred_near_i.shape[0] + 1)  # no shift -- see _closed_loop_forward_causal

    title = PROBE_TRIAL_TITLES[kind]

    set_plt_params(style="fast")
    fig = plt.figure(figsize=(19, 16.5), layout="constrained")
    gs = fig.add_gridspec(5, 5, height_ratios=[1.6, 1, 1, 1, 1])

    # Spatial panel: identical to plot_probe_near_far_diagnostic.
    ax_spatial = fig.add_subplot(gs[0, :3])
    ax_spatial.add_patch(Rectangle((-pos_bound, -pos_bound), 2 * pos_bound, 2 * pos_bound,
                                    fill=False, edgecolor="black", linestyle="--"))
    _draw_trajectory(ax_spatial, positions, t_probe=t_probe)

    near_before_center, near_before_curv = before_near[near_slot]
    far_before_center, far_before_curv = before_far[far_slot]

    if kind == "location":
        near_after_center, _ = after_near[near_slot]
        far_after_center, _ = after_far[far_slot]
        ax_spatial.annotate("", xy=near_after_center, xytext=near_before_center, zorder=3,
                             arrowprops=dict(arrowstyle="-|>", color="firebrick", lw=2, ls="--"))
        ax_spatial.annotate("", xy=far_after_center, xytext=far_before_center, zorder=3,
                             arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2, ls="--"))
        ax_spatial.scatter(*near_after_center, marker="*", s=180, color="black", zorder=4,
                            label="relocation target (both trials)")

    ax_spatial.scatter(*near_before_center, c=[near_before_curv], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                        s=320, edgecolor="firebrick", linewidth=2.5, zorder=4)
    ax_spatial.annotate(f"object {near_obj_idx} (near)\ncurv={near_before_curv:.2f}",
                         (near_before_center[0], near_before_center[1] + LABEL_Y_OFFSET),
                         fontsize=7, ha="center", va="bottom", zorder=5)

    obj_scatter = ax_spatial.scatter(*far_before_center, c=[far_before_curv], cmap="coolwarm",
                                      vmin=-1.0, vmax=1.0, s=320, edgecolor="steelblue", linewidth=2.5, zorder=4)
    ax_spatial.annotate(f"object {far_obj_idx} (far)\ncurv={far_before_curv:.2f}",
                         (far_before_center[0], far_before_center[1] + LABEL_Y_OFFSET),
                         fontsize=7, ha="center", va="bottom", zorder=5)

    cax = ax_spatial.inset_axes([1.02, 0.0, 0.03, 1.0])
    fig.colorbar(obj_scatter, cax=cax, label="hidden curvature")

    pad = pos_bound * 0.2
    ax_spatial.set_xlim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_ylim(-pos_bound - pad, pos_bound + pad)
    ax_spatial.set_title(f"{title} near-vs-far -- same trajectory, one object swapped per trial")
    ax_spatial.legend(fontsize=7, loc="upper right")
    make_axis_nice(ax_spatial)

    # MSE panel: near/far x intact/pred_err=0 -- 4 lines (solid=intact, dotted=pred_err=0,
    # colored by near/far), shaded region marks the ablation window (see
    # plot_episode_causal_diagnostic). Spans the last 2 columns (not just 1, like the
    # other *_diagnostic functions' MSE panels) since this figure's row 0 has a 5th
    # column reserved for the fractional-attribution panels below, which don't start
    # until row 1 -- without the span, row 0's column 4 would just sit empty.
    ax_mse = fig.add_subplot(gs[0, 3:5])
    s_n, e_n = near_obj_idx * OBJ_FEATURE_DIM, (near_obj_idx + 1) * OBJ_FEATURE_DIM
    s_f, e_f = far_obj_idx * OBJ_FEATURE_DIM, (far_obj_idx + 1) * OBJ_FEATURE_DIM
    mse_near_i = ((s_pred_near_i[:, s_n:e_n] - s_target_near[:, s_n:e_n]) ** 2).mean(axis=-1)
    mse_near_a = ((s_pred_near_a[:, s_n:e_n] - s_target_near[:, s_n:e_n]) ** 2).mean(axis=-1)
    mse_far_i = ((s_pred_far_i[:, s_f:e_f] - s_target_far[:, s_f:e_f]) ** 2).mean(axis=-1)
    mse_far_a = ((s_pred_far_a[:, s_f:e_f] - s_target_far[:, s_f:e_f]) ** 2).mean(axis=-1)

    stim_end_plot = seq_len if stim_end is None else stim_end
    ax_mse.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray", alpha=0.12,
                   label="pred_err=0 window", zorder=0)
    ax_mse.plot(t_axis, mse_near_i, color="firebrick", ls="-", alpha=0.6, label="near (intact)")
    ax_mse.plot(t_axis, mse_near_a, color="firebrick", ls=":", label="near (pred_err=0)")
    ax_mse.plot(t_axis, mse_far_i, color="steelblue", ls="-", alpha=0.6, label="far (intact)")
    ax_mse.plot(t_axis, mse_far_a, color="steelblue", ls=":", label="far (pred_err=0)")
    ax_mse.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_mse.set_xlabel("Timestep")
    ax_mse.set_ylabel("MSE (swapped object)")
    ax_mse.set_title("Prediction error: near/far x intact/pred_err=0")
    ax_mse.legend(fontsize=5.5)
    make_axis_nice(ax_mse)

    # Fractional-attribution column: for each trial, swapped-object MSE / (swapped-object
    # MSE + unmoved-object MSE), intact vs. pred_err=0 -- what FRACTION of the two
    # objects' combined error is attributed to the one that actually changed, and does
    # that attribution itself depend on genuine prediction-error feedback? (a
    # per-timestep, continuous analog of _fractional_preference_index's windowed
    # fraction, and the same [0, 1] quantity: 0.5 = equal contribution, marked with a
    # reference line; toward 1 = error increasingly attributed to the swapped object).
    # Each trial's panel spans both of that trial's rows (it's a property of the trial,
    # not of either individual object row) rather than repeating in both.
    s_um_n, e_um_n = far_obj_idx * OBJ_FEATURE_DIM, (far_obj_idx + 1) * OBJ_FEATURE_DIM  # near trial's unmoved obj
    s_um_f, e_um_f = near_obj_idx * OBJ_FEATURE_DIM, (near_obj_idx + 1) * OBJ_FEATURE_DIM  # far trial's unmoved obj
    mse_near_unmoved_i = ((s_pred_near_i[:, s_um_n:e_um_n] - s_target_near[:, s_um_n:e_um_n]) ** 2).mean(axis=-1)
    mse_near_unmoved_a = ((s_pred_near_a[:, s_um_n:e_um_n] - s_target_near[:, s_um_n:e_um_n]) ** 2).mean(axis=-1)
    mse_far_unmoved_i = ((s_pred_far_i[:, s_um_f:e_um_f] - s_target_far[:, s_um_f:e_um_f]) ** 2).mean(axis=-1)
    mse_far_unmoved_a = ((s_pred_far_a[:, s_um_f:e_um_f] - s_target_far[:, s_um_f:e_um_f]) ** 2).mean(axis=-1)

    # Smooth each raw per-timestep MSE trace with a centered rolling mean BEFORE taking
    # the ratio -- a single timestep's raw MSE is noisy enough to swing the fraction
    # wildly even when neither object's error has genuinely changed (see docstring).
    (mse_near_i_sm, mse_near_a_sm, mse_near_unmoved_i_sm, mse_near_unmoved_a_sm,
     mse_far_i_sm, mse_far_a_sm, mse_far_unmoved_i_sm, mse_far_unmoved_a_sm) = (
        _rolling_mean(x, attribution_window) for x in
        (mse_near_i, mse_near_a, mse_near_unmoved_i, mse_near_unmoved_a,
         mse_far_i, mse_far_a, mse_far_unmoved_i, mse_far_unmoved_a)
    )

    eps = 1e-8
    frac_near_i = mse_near_i_sm / (mse_near_i_sm + mse_near_unmoved_i_sm + eps)
    frac_near_a = mse_near_a_sm / (mse_near_a_sm + mse_near_unmoved_a_sm + eps)
    frac_far_i = mse_far_i_sm / (mse_far_i_sm + mse_far_unmoved_i_sm + eps)
    frac_far_a = mse_far_a_sm / (mse_far_a_sm + mse_far_unmoved_a_sm + eps)

    ax_ratio_near = fig.add_subplot(gs[1:3, 4])
    ax_ratio_near.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray",
                          alpha=0.12, label="pred_err=0 window", zorder=0)
    ax_ratio_near.axhline(0.5, color="gray", ls="-", lw=1, alpha=0.7, label="equal contribution (0.5)")
    ax_ratio_near.plot(t_axis, frac_near_i, color="firebrick", ls="-", alpha=0.6, label="intact")
    ax_ratio_near.plot(t_axis, frac_near_a, color="firebrick", ls=":", label="pred_err=0")
    ax_ratio_near.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_ratio_near.set_ylim(0, 1)
    ax_ratio_near.set_xlabel("Timestep")
    ax_ratio_near.set_ylabel("Fractional error attribution\n(swapped / (swapped + unmoved))")
    ax_ratio_near.set_title("near trial: error attribution fraction")
    ax_ratio_near.legend(fontsize=5.5)
    make_axis_nice(ax_ratio_near)

    ax_ratio_far = fig.add_subplot(gs[3:5, 4])
    ax_ratio_far.axvspan(max(stim_start, t_axis[0]), min(stim_end_plot, t_axis[-1]), color="gray",
                         alpha=0.12, label="pred_err=0 window", zorder=0)
    ax_ratio_far.axhline(0.5, color="gray", ls="-", lw=1, alpha=0.7, label="equal contribution (0.5)")
    ax_ratio_far.plot(t_axis, frac_far_i, color="steelblue", ls="-", alpha=0.6, label="intact")
    ax_ratio_far.plot(t_axis, frac_far_a, color="steelblue", ls=":", label="pred_err=0")
    ax_ratio_far.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax_ratio_far.set_ylim(0, 1)
    ax_ratio_far.set_xlabel("Timestep")
    ax_ratio_far.set_ylabel("Fractional error attribution\n(swapped / (swapped + unmoved))")
    ax_ratio_far.set_title("far trial: error attribution fraction")
    ax_ratio_far.legend(fontsize=5.5)
    make_axis_nice(ax_ratio_far)

    # FOUR full demixed-latent rows: each trial (near-swap, far-swap) contributes its
    # SWAPPED object's row and its OTHER (unmoved that trial) object's row -- both
    # objects' full state predictions were already computed for each trial (the model
    # predicts the whole STATE_DIM vector regardless of which one moved), so this needs no
    # extra forward passes, just demixing the already-computed predictions at a different
    # object's columns. Color is by TRIAL (near-swap=firebrick, far-swap=steelblue),
    # regardless of which object a given row shows, matching the MSE panel's "near"/"far"
    # meaning; each row overlays intact (dashed) vs. pred_err=0 (dotted) against actual.
    # Both of one trial's rows share that trial's own worst-missed timestep (its intact
    # MSE peak, matching the dashed curve already shown in the MSE panel), so it's the
    # same event in both.
    def demix(s_target, s_pred, obj_idx):
        s, e = obj_idx * OBJ_FEATURE_DIM, (obj_idx + 1) * OBJ_FEATURE_DIM
        raw_actual = (s_target[:, s:e] @ MIXING_MATRIX) * RAW_STD
        raw_pred = (s_pred[:, s:e] @ MIXING_MATRIX) * RAW_STD
        return raw_actual, raw_pred

    peak_idx_near = int(np.argmax(mse_near_i))
    peak_idx_far = int(np.argmax(mse_far_i))

    rows = [
        (1, "near trial", near_obj_idx, "swapped", s_target_near, s_pred_near_i, s_pred_near_a,
         "firebrick", peak_idx_near),
        (2, "near trial", far_obj_idx, "unmoved", s_target_near, s_pred_near_i, s_pred_near_a,
         "firebrick", peak_idx_near),
        (3, "far trial", near_obj_idx, "unmoved", s_target_far, s_pred_far_i, s_pred_far_a,
         "steelblue", peak_idx_far),
        (4, "far trial", far_obj_idx, "swapped", s_target_far, s_pred_far_i, s_pred_far_a,
         "steelblue", peak_idx_far),
    ]
    for row_idx, trial_label, obj_idx, moved_label, s_target, s_pred_i, s_pred_a, color, peak_idx in rows:
        raw_actual, raw_pred_i = demix(s_target, s_pred_i, obj_idx)
        _, raw_pred_a = demix(s_target, s_pred_a, obj_idx)
        for feat_idx, feat_name in enumerate(_RAW_NAMES):
            ax = fig.add_subplot(gs[row_idx, feat_idx])
            actual_col = raw_actual[:, feat_idx]
            pred_i_col = raw_pred_i[:, feat_idx]
            pred_a_col = raw_pred_a[:, feat_idx]
            r2_i = _one_minus_r2(actual_col, pred_i_col)
            r2_a = _one_minus_r2(actual_col, pred_a_col)
            ax.plot(t_axis, actual_col, color="black", label="actual")
            ax.plot(t_axis, pred_i_col, color=color, ls="--", alpha=0.7, label="intact")
            ax.plot(t_axis, pred_a_col, color=color, ls=":", label="pred_err=0")
            _mark_worst_point(ax, t_axis, actual_col, pred_a_col, color, peak_idx)
            ax.axvline(t_probe, color="gray", ls=":", lw=1, zorder=1)
            ax.set_title(f"{trial_label} -- obj {obj_idx} ({moved_label}) -- {feat_name} (demixed)\n"
                         f"1-R²(intact)={r2_i:.3f}, 1-R²(pred_err=0)={r2_a:.3f}", fontsize=7)
            ax.set_xlabel("Timestep")
            if feat_idx == 0:
                ax.set_ylabel("Latent value (physical units)")
                ax.legend(fontsize=6)
            make_axis_nice(ax)

    fig.suptitle(f"{title} near-vs-far causal diagnostic (closed-loop, seed={seed}, "
                 f"condition={condition_idx}, pred_err=0 for t in [{stim_start}, {stim_end_plot}))")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "s_pred_near_intact": s_pred_near_i, "s_pred_near_ablated": s_pred_near_a, "s_target_near": s_target_near,
        "s_pred_far_intact": s_pred_far_i, "s_pred_far_ablated": s_pred_far_a, "s_target_far": s_target_far,
        "near_slot": near_slot, "far_slot": far_slot,
        "positions": positions, "headings": headings, "t_probe": t_probe, "seed": seed,
    }


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


def _closed_loop_forward_causal(ec, cb, trans, zero_pred_err=False, stim_start=0, stim_end=None):
    """Autoregressive closed-loop forward pass (see core/trainers/ec_cb.py's
    ECCBTrainer.replay_ec_sequence for the only other precedent of this pattern).

    At each step t: CB makes exactly ONE prediction of s_{t+1}, using the context z_{t-1}
    formed from everything up through step t-1 -- genuinely blind to s_{t+1} itself. This
    is s_pred_t, what's returned and what the loss is computed on. It's then compared
    against the true s_{t+1} to form pred_err_t = s_pred_t - s_{t+1}, which becomes part
    of EC's input for updating z_{t-1} to z_t -- replacing the raw next-state block used
    by the non-closed-loop _shifted_forward, not appended alongside it, so the per-step
    input is [s_t, a_t, pred_err_t] (same total width as [s_t, a_t, s_{t+1}], i.e. same
    TRANS_DIM as the non-closed-loop model). z_t is used at the NEXT step (t+1) to predict
    s_{t+2}, never to re-predict s_{t+1} itself -- pred_err_t only ever influences FUTURE
    predictions, so nothing here lets s_{t+1} leak into its own prediction. (An earlier
    version of this function made a SECOND prediction using z_t -- i.e. AFTER pred_err_t,
    which is a simple invertible function of s_{t+1} itself, had already been folded into
    the context -- and scored that one instead; that let the "prediction" trivially
    recover the answer it had just been handed, rather than genuinely forecasting it.)

    zero_pred_err=True is a CAUSAL ABLATION (not a perturbation, unlike
    _shifted_forward_noisy's noise injection into the non-closed-loop model): for
    timesteps in [stim_start, stim_end) (default: the whole trial), pred_err_t is
    overwritten with exact zeros before it reaches EC's GRU, i.e. EC's update at that step
    proceeds as if CB's prediction had been perfect -- so any resulting change in FUTURE
    predictions is attributable entirely to EC no longer being able to update its context
    off a genuine prediction-error signal, not to EC's own hidden state or CB's weights
    changing at all. zero_pred_err=False (default) reproduces the un-ablated closed-loop
    model exactly.

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
    if stim_end is None:
        stim_end = T
    h = torch.zeros(1, B, ec.gru.hidden_size, device=trans.device, dtype=trans.dtype)

    s_preds, s_targets = [], []
    for t in range(T):
        s_t = trans[:, t, :STATE_DIM]
        a_t = trans[:, t, STATE_DIM:STATE_DIM + ACTION_DIM]
        s_next = trans[:, t, STATE_DIM + ACTION_DIM:]

        z_prev = ec.readout(h.squeeze(0))          # context from before this step
        s_pred = cb(s_t, a_t, z_prev)               # the ONE prediction -- genuinely blind to s_next
        pred_err_t = s_pred - s_next
        if zero_pred_err and stim_start <= t < stim_end:
            pred_err_t = torch.zeros_like(pred_err_t)

        step_input = torch.cat([s_t, a_t, pred_err_t], dim=-1).unsqueeze(1)  # (B, 1, TRANS_DIM)
        _, h = ec.gru(step_input, h)                # update -> z_t, used at the NEXT step only

        s_preds.append(s_pred)
        s_targets.append(s_next)

    return torch.stack(s_preds, dim=1), torch.stack(s_targets, dim=1)


def _closed_loop_forward(ec, cb, trans):
    """Un-ablated closed-loop forward pass -- see _closed_loop_forward_causal (this is
    just that function with zero_pred_err=False, kept as a stable name for train_olm_closed_loop
    and anything else that only ever wants the intact model)."""
    return _closed_loop_forward_causal(ec, cb, trans, zero_pred_err=False)


def _closed_loop_forward_noisy(ec, cb, trans, noise_site, alpha, stim_start, stim_end):
    """Closed-loop counterpart of _shifted_forward_noisy: injects isotropic Gaussian
    noise (expected L2 norm `alpha` per perturbed timestep, active only in
    [stim_start, stim_end)) into the autoregressive closed-loop unroll, rather than
    zeroing pred_err_t outright (_closed_loop_forward_causal's zero_pred_err ablation).
    See _closed_loop_forward_causal for the base one-prediction-per-step mechanic (CB
    predicts s_{t+1} from z_{t-1} alone; that prediction, compared against the true
    s_{t+1}, drives the update to z_t, used only at the NEXT step).

    This is a DIFFERENT intervention from zero_pred_err, not just a graded version of it:
    zero_pred_err only PAUSES new prediction-error-driven updates to an already-formed
    hidden state (so a correctly-encoded memory from before the stim window survives
    untouched); noise injection actively CORRUPTS the currently-held representation
    itself. If a model's prediction survives zero_pred_err intact but degrades under
    noise here, that's not a contradiction -- it means the model doesn't need continuous
    prediction-error feedback to MAINTAIN a memory, but does depend on that memory being
    UNCORRUPTED, which is a real (and different) claim from "CB doesn't matter."

    noise_site:
      - "gru": noise added to EC's GRU hidden state h AFTER each step's update, for every
        t in [stim_start, stim_end) -- compounds forward through the recurrence (as in
        _shifted_forward_noisy), persisting into every SUBSEQUENT step's z_prev (hence
        every later step's prediction and pred_err), since h is the actual persistent
        recurrent state.
      - "cb_input": noise added only to THIS step's z_prev right before CB consumes it to
        make its one prediction -- h itself is left clean, so this site's corruption
        doesn't persist into later steps' z_prev the way "gru" does (it does still show up
        in THIS step's own pred_err_t, since that's computed from the noisy prediction --
        unlike "gru", it has no route to compound forward through the recurrence beyond
        that one step).
      - alpha == 0: identical to _closed_loop_forward_causal(zero_pred_err=False).

    Returns (s_pred, s_target), each (B, T, STATE_DIM) -- no shift (predict_offset=1),
    same convention as _closed_loop_forward_causal.
    """
    B, T, _ = trans.shape
    h = torch.zeros(1, B, ec.gru.hidden_size, device=trans.device, dtype=trans.dtype)
    h_noise_std = alpha / (ec.gru.hidden_size ** 0.5)

    s_preds, s_targets = [], []
    for t in range(T):
        s_t = trans[:, t, :STATE_DIM]
        a_t = trans[:, t, STATE_DIM:STATE_DIM + ACTION_DIM]
        s_next = trans[:, t, STATE_DIM + ACTION_DIM:]

        z_prev = ec.readout(h.squeeze(0))          # context from before this step
        if noise_site == "cb_input" and alpha > 0 and stim_start <= t < stim_end:
            z_noise_std = alpha / (z_prev.shape[-1] ** 0.5)
            z_prev = z_prev + z_noise_std * torch.randn_like(z_prev)
        s_pred = cb(s_t, a_t, z_prev)               # the ONE prediction -- genuinely blind to s_next
        pred_err_t = s_pred - s_next

        step_input = torch.cat([s_t, a_t, pred_err_t], dim=-1).unsqueeze(1)
        _, h = ec.gru(step_input, h)                # update -> z_t, used at the NEXT step only
        if noise_site == "gru" and alpha > 0 and stim_start <= t < stim_end:
            h = h + h_noise_std * torch.randn_like(h)

        s_preds.append(s_pred)
        s_targets.append(s_next)

    return torch.stack(s_preds, dim=1), torch.stack(s_targets, dim=1)


def _closed_loop_forward_periodic_noise(ec, cb, trans, alpha, period, on_duration,
                                         stim_start=0, stim_end=None, anchor=None):
    """Closed-loop forward pass with a PERIODIC prediction-error perturbation -- the
    causal-ablation counterpart of _closed_loop_forward_causal's zero_pred_err, but
    structurally different in two ways:

      1. Instead of a single CONTINUOUS window [stim_start, stim_end) during which the
         manipulation is always active, the manipulation only fires in bursts: active for
         `on_duration` consecutive steps, then inactive for the rest of each
         `period`-length cycle, repeating -- but the phase is pinned to `anchor` (default:
         stim_start), not necessarily stim_start itself: "on" steps are exactly those t
         (within [stim_start, stim_end)) where (t - anchor) % period < on_duration. E.g.
         anchor=stim_start, period=15, on_duration=3 -> on for t in [stim_start,
         stim_start+3), off for [stim_start+3, stim_start+15), on again for
         [stim_start+15, stim_start+18), etc. This lets the model recover with clean
         feedback between bursts, rather than being cut off from real prediction-error
         signal for the whole rest of the trial.

         Pinning the phase to an `anchor` OTHER than stim_start matters when some SPECIFIC
         timestep must always fall inside a burst regardless of period/on_duration/
         stim_start -- e.g. evaluate_causal_ablation anchors to t_probe - 1 (the one row
         whose target first reflects the manipulation) because a location-change effect is
         a single-timestep transient: if that one row happens to land in an "off" gap
         (which, anchored at plain stim_start, depends on essentially arbitrary alignment
         between stim_start and t_probe), the perturbation could miss the only moment that
         actually matters and produce a false null effect purely by scheduling coincidence.

      2. Instead of REPLACING pred_err_t with exact zeros (which only pauses new
         prediction-error-driven updates -- an already-correct hidden state survives
         untouched, per _closed_loop_forward_causal's docstring), this ADDS isotropic
         Gaussian noise directly onto pred_err_t -- the real error signal is still
         present, just corrupted -- during each "on" burst. alpha=0 (or on_duration<=0)
         reproduces the intact model exactly.

    `alpha` is the expected L2 norm of the noise vector injected at EACH perturbed
    timestep -- i.e. the TOTAL noise magnitude summed (in quadrature) across all
    STATE_DIM dimensions together at that step, not the per-dimension magnitude -- same
    convention _shifted_forward_noisy/_closed_loop_forward_noisy already use for their
    "gru"/"cb_input" sites (noise_std = alpha / sqrt(dim), so E[||noise||^2] = alpha**2
    per perturbed step regardless of STATE_DIM). This is a PER-TIMESTEP budget: it does
    NOT renormalize by how many timesteps end up "on" over the whole schedule, so
    changing period/on_duration/stim_start/stim_end (with alpha fixed) changes the TOTAL
    amount of noise injected across the trial, just not the size of any single burst.

    See _periodic_on_intervals for the same on/off schedule expressed as a list of
    (start, end) ranges, used purely for plot shading.

    Returns (s_pred, s_target), each (B, T, STATE_DIM) -- no shift (predict_offset=1),
    same convention as _closed_loop_forward_causal/_closed_loop_forward_noisy. See
    _closed_loop_forward_causal for the base one-prediction-per-step mechanic (CB
    predicts s_{t+1} from z_{t-1} alone; that prediction, compared against the true
    s_{t+1}, drives the update to z_t, used only at the NEXT step) -- the noise here is
    added to pred_err_t AFTER that one prediction, so it still can't leak s_{t+1} into
    its own prediction, only into the update z_t carries forward.
    """
    B, T, _ = trans.shape
    if stim_end is None:
        stim_end = T
    if anchor is None:
        anchor = stim_start
    h = torch.zeros(1, B, ec.gru.hidden_size, device=trans.device, dtype=trans.dtype)
    noise_std = alpha / (STATE_DIM ** 0.5)

    s_preds, s_targets = [], []
    for t in range(T):
        s_t = trans[:, t, :STATE_DIM]
        a_t = trans[:, t, STATE_DIM:STATE_DIM + ACTION_DIM]
        s_next = trans[:, t, STATE_DIM + ACTION_DIM:]

        z_prev = ec.readout(h.squeeze(0))          # context from before this step
        s_pred = cb(s_t, a_t, z_prev)               # the ONE prediction -- genuinely blind to s_next
        pred_err_t = s_pred - s_next
        if (alpha > 0 and on_duration > 0 and stim_start <= t < stim_end
                and (t - anchor) % period < on_duration):
            pred_err_t = pred_err_t + noise_std * torch.randn_like(pred_err_t)

        step_input = torch.cat([s_t, a_t, pred_err_t], dim=-1).unsqueeze(1)  # (B, 1, TRANS_DIM)
        _, h = ec.gru(step_input, h)                # update -> z_t, used at the NEXT step only

        s_preds.append(s_pred)
        s_targets.append(s_next)

    return torch.stack(s_preds, dim=1), torch.stack(s_targets, dim=1)


def _periodic_on_intervals(stim_start, stim_end, period, on_duration, anchor=None):
    """[(start, end), ...] timestep ranges within [stim_start, stim_end) during which
    _closed_loop_forward_periodic_noise's perturbation is "on" (same (t - anchor) % period
    < on_duration condition, anchor defaulting to stim_start) -- used both to compute
    n_on_steps-style diagnostics and to shade the over-time plots with the actual burst
    schedule instead of one solid continuous span."""
    if on_duration <= 0:
        return []
    if anchor is None:
        anchor = stim_start
    intervals = []
    run_start = None
    for t in range(stim_start, stim_end):
        on = (t - anchor) % period < on_duration
        if on and run_start is None:
            run_start = t
        elif not on and run_start is not None:
            intervals.append((run_start, t))
            run_start = None
    if run_start is not None:
        intervals.append((run_start, stim_end))
    return intervals


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

    There's no identity/location sub-group to split here: all OBJ_FEATURE_DIM columns are
    a single mixed readout (see MIXING_MATRIX), and every probe (location/curvature-
    change) perturbs the same columns (just different underlying causes), so a plain
    full-feature error is the right comparison across both probe types.
    """
    start = obj_idx * OBJ_FEATURE_DIM
    end = start + OBJ_FEATURE_DIM
    return ((s_pred[..., start:end] - s_target[..., start:end]) ** 2).mean(dim=-1)  # (B, T-1)


def _fraction_timeseries(s_pred, s_target, changed_idxs):
    """Per-episode, per-timestep fractional error attribution (B, T): err_changed(t) /
    (err_changed(t) + err_other(t)) at every t -- the full time-series version of
    _fractional_preference_index (which reduces to one windowed scalar per episode).
    Averaging this over many episodes (see evaluate_causal_ablation's fraction-vs-time
    plot) gives a much less noisy attribution curve than a single trial's raw
    per-timestep ratio (see plot_probe_trial_causal_diagnostic's fractional-attribution
    panel, whose noise is an inherent property of dividing two small, independently
    noisy per-timestep MSEs at the single-trial level -- not fixable by smoothing alone
    without also smearing away the real signal's timing). `changed_idxs` is the
    manipulated slot (0=left, 1=right), same convention as _fractional_preference_index.
    """
    err0 = _slot_error(s_pred, s_target, 0)  # (B, T)
    err1 = _slot_error(s_pred, s_target, 1)  # (B, T)
    total = torch.clamp(err0 + err1, min=1e-12)
    fractions = torch.stack([err0, err1], dim=1) / total.unsqueeze(1)  # (B, 2, T)
    changed = torch.as_tensor(changed_idxs, dtype=torch.long, device=fractions.device)
    idx = changed.view(-1, 1, 1).expand(-1, 1, fractions.shape[-1])
    return fractions.gather(1, idx).squeeze(1)  # (B, T)


def _fraction_auc(fraction_curve, start, end):
    """Per-episode AUC of a fractional-attribution-over-time curve (see
    _fraction_timeseries) across predicted-timesteps [start, end), normalized by the
    window's duration (trapezoidal integral / (end - start - 1)) so it stays on the same
    bounded [0, 1] scale as the raw fraction / preference index, rather than growing with
    window length -- a convex combination of in-[0,1] values is itself in [0,1].

    Unlike _fractional_preference_index (which averages the raw per-slot errors over a
    short window anchored at t_probe, then takes one ratio), this integrates the ALREADY-
    COMPUTED ratio curve over the full stimulation window -- capturing whether elevated
    attribution to the changed object PERSISTS across the ablation, not just its initial
    transient. Confirmed motivating this: the aggregate fractional-attribution-over-time
    plot shows curvature-change's (ORM) perturbed curve staying elevated well past its
    initial spike, while location-change's (OLM) decays back to baseline almost
    immediately -- a difference a short t_probe-anchored window mostly misses.

    `fraction_curve`: (B, T) numpy array, e.g. _fraction_timeseries(...).cpu().numpy().
    Returns (B,) per-episode scalar.
    """
    window = fraction_curve[:, start:end]
    if window.shape[1] == 0:
        raise ValueError(f"_fraction_auc: empty window [{start}, {end}) for a curve of length {fraction_curve.shape[1]}")
    if window.shape[1] == 1:
        return window[:, 0]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz")
    return trapz(window, dx=1.0, axis=1) / (window.shape[1] - 1)


def _plot_causal_ablation_fraction(t_axis, mean_intact, sem_intact, mean_ablated, sem_ablated,
                                    t_probe, on_intervals, title, fig_path, n_conditions=None,
                                    ylabel="Fractional error attribution\n(changed / (changed + other))",
                                    title_suffix="error attribution over time"):
    """Mean +/- SEM over-time curve across all conditions of one probe -- the aggregate
    (many-episode) counterpart of a single-trial fraction/preference-index panel, shared
    by both the raw per-timestep version (_fraction_timeseries) and the windowed,
    sliding-window version (_fractional_preference_index_sliding) -- `ylabel`/
    `title_suffix` distinguish which one a given call is plotting; the defaults match the
    original raw per-timestep usage exactly. `on_intervals`: [(start, end), ...] -- see
    _periodic_on_intervals -- shaded as the perturbation's actual on/off burst schedule
    rather than one solid continuous span."""
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (start, end) in enumerate(on_intervals):
        ax.axvspan(max(start, t_axis[0]), min(end, t_axis[-1]), color="gray", alpha=0.12,
                   label="noise-perturbed" if i == 0 else None, zorder=0)
    ax.axhline(0.5, color="gray", ls="-", lw=1, alpha=0.7, label="equal contribution (0.5)")
    ax.plot(t_axis, mean_intact, color="firebrick", ls="-", label="intact")
    ax.fill_between(t_axis, mean_intact - sem_intact, mean_intact + sem_intact, color="firebrick", alpha=0.2)
    ax.plot(t_axis, mean_ablated, color="firebrick", ls=":", label="perturbed")
    ax.fill_between(t_axis, mean_ablated - sem_ablated, mean_ablated + sem_ablated, color="firebrick", alpha=0.15)
    ax.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    n_suffix = f" ({n_conditions} conditions)" if n_conditions else ""
    ax.set_title(f"{title} {title_suffix}{n_suffix}")
    ax.legend(fontsize=7)
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


_OVERLAY_COLORS = {"location": "steelblue", "curvature": "firebrick"}
_OVERLAY_LABELS = {"location": "OLM (location)", "curvature": "ORM (curvature)"}


def _plot_fraction_overlay(t_axis, curves, t_probe, on_intervals, title, fig_path,
                            n_conditions=None,
                            ylabel="Fractional error attribution\n(changed / (changed + other))",
                            title_suffix="error attribution over time"):
    """Both probes' mean +/- SEM over-time curves superimposed on one axes -- OLM
    (location-change) in steelblue, ORM (curvature-change) in firebrick, solid=intact /
    dotted=perturbed -- the same curve _plot_causal_ablation_fraction renders per-probe,
    just combined for a direct side-by-side comparison. `curves`: {name: (mean_intact,
    sem_intact, mean_ablated, sem_ablated)} for name in PROBE_NAMES -- both probes share
    one t_axis/t_probe/on_intervals schedule by construction (see evaluate_causal_ablation,
    which calls both probe generators with the same seq_len/t_probe). `n_conditions`, if
    given, is a {name: count} dict (location and curvature have DIFFERENT condition counts
    -- 144 vs 432 -- so a single shared number in the title would misattribute one probe's
    count to the other); baked into each probe's own legend label instead of the title.
    `on_intervals`: [(start, end), ...] -- see _periodic_on_intervals."""
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (start, end) in enumerate(on_intervals):
        ax.axvspan(max(start, t_axis[0]), min(end, t_axis[-1]), color="gray", alpha=0.12,
                   label="noise-perturbed" if i == 0 else None, zorder=0)
    ax.axhline(0.5, color="gray", ls="-", lw=1, alpha=0.7, label="equal contribution (0.5)")
    for name, (mean_intact, sem_intact, mean_ablated, sem_ablated) in curves.items():
        color = _OVERLAY_COLORS[name]
        n_suffix = f" (n={n_conditions[name]})" if n_conditions else ""
        label = f"{_OVERLAY_LABELS[name]}{n_suffix}"
        ax.plot(t_axis, mean_intact, color=color, ls="-", label=f"{label}, intact")
        ax.fill_between(t_axis, mean_intact - sem_intact, mean_intact + sem_intact, color=color, alpha=0.2)
        ax.plot(t_axis, mean_ablated, color=color, ls=":", label=f"{label}, perturbed")
        ax.fill_between(t_axis, mean_ablated - sem_ablated, mean_ablated + sem_ablated, color=color, alpha=0.15)
    ax.axvline(t_probe, color="gray", ls=":", lw=1.5, label=f"manipulation (t={t_probe})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} {title_suffix}")
    ax.legend(fontsize=7)
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def evaluate_displacement(ec, cb, seq_len=SEQ_LEN, t_probe=None, seed=0, device=None, fig_path=FIG_PATH):
    """Deterministic-over-conditions, exhaustive location-change probe
    (generate_probe_location_change_data): compare per-slot prediction error between
    control (never moves) and manipulated (relocated to PROBE_RELOCATION_TARGETS[slot] at
    t_probe) episodes, across every enumerated condition (curvature pair x manipulated slot --
    see _enumerate_probe_pairs). Every condition shares one scripted, controlled agent
    trajectory (see _controlled_trajectory); `seed` only affects observation noise."""
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


def _windowed_fraction(err0, err1, start, end):
    """(B, 2) fractional index [err0, err1] / (err0 + err1), each averaged over
    predicted-timesteps [start, end) -- the shared core both _fractional_preference_index
    and _fractional_preference_index_sliding gather their per-episode changed-slot value
    from."""
    e0 = err0[:, start:end].mean(dim=1)
    e1 = err1[:, start:end].mean(dim=1)
    total = torch.clamp(e0 + e1, min=1e-12)
    return torch.stack([e0, e1], dim=1) / total.unsqueeze(1)  # (B, 2)


def _fractional_preference_index(s_pred, s_target, changed_idxs, t_probe, window=5, predict_offset=2):
    """Per-episode fractional preference index: err_changed / (err_obj0 + err_obj1), where
    each slot's error is averaged over predicted-timesteps [t_probe, t_probe+window)
    (window=None -> average to the end of the sequence), and `changed_idxs` is the
    manipulated slot (0=left, 1=right). ~0.5 = no preference; closer to 1 = more error
    attributed to the changed/novel object ("preference for novel"); closer to 0 = the
    opposite ("preference for familiar"). Only valid for N_OBJECTS == 2 (hardcodes slots
    0/1, same constraint as evaluate_displacement's other_idxs).

    `predict_offset` accounts for which architecture produced (s_pred, s_target): the
    standard model's _shifted_forward predicts s_{t+2} (default predict_offset=2, i.e.
    index j predicts state(j+2)); the closed-loop model's _closed_loop_forward_causal
    predicts s_{t+1} directly with no shift, so callers using it (e.g.
    evaluate_causal_ablation) must pass predict_offset=1 (index j predicts state(j+1)).

    `window`: NOT None (average to end of episode) -- confirmed empirically (on the
    earlier feature representation, before the curvature/bearing noise redesign) that the
    excess prediction error at a manipulation is short-lived, so averaging over a wide
    window (e.g. the old None default, ~30 steps to end of episode) dilutes that signal
    with mostly noise -- measured to weaken the location-change control-vs-changed gap by
    ~7x (0.059 at window=1 down to 0.008 at window=None) for no benefit. Default raised
    from 3 to 5 (still short relative to the episode, just less tightly anchored to
    exactly t_probe) -- see _fractional_preference_index_sliding for a version that slides
    this SAME window size across the rest of the trial instead of anchoring it once, if
    you want to see how the index evolves rather than a single value.

    Plain fraction rather than softmax: automatically scale-invariant (dividing by the
    pair's own total already rescales away the model's absolute error magnitude, so no
    temperature hyperparameter is needed), and directly interpretable as "the fraction of
    total prediction error attributable to the changed object" -- the same style as a
    behavioral discrimination index (time at novel / total time).
    """
    err0 = _slot_error(s_pred, s_target, 0)  # (B, T-1)
    err1 = _slot_error(s_pred, s_target, 1)  # (B, T-1)

    start = max(t_probe - predict_offset, 0)  # index j predicts state(j+predict_offset)
    end = err0.shape[1] if window is None else min(start + window, err0.shape[1])

    fractions = _windowed_fraction(err0, err1, start, end)  # (B, 2)
    changed = torch.as_tensor(changed_idxs, dtype=torch.long, device=fractions.device)
    return fractions.gather(1, changed.unsqueeze(1)).squeeze(1)  # (B,)


def _fractional_preference_index_sliding(s_pred, s_target, changed_idxs, t_probe, window=5, predict_offset=2):
    """(B, n_windows) sliding-window generalization of _fractional_preference_index: the
    SAME fixed window size, but instead of anchoring it once at t_probe, slides it one
    step at a time across the REST of the trial. Window position w covers
    [start + w, start + w + window), for every w where that still fits within the trial,
    where start = max(t_probe - predict_offset, 0) matches _fractional_preference_index's
    own starting point -- so slice w=0 is EXACTLY what _fractional_preference_index
    itself returns.

    Gives a time-resolved view of how the fractional preference index evolves after the
    manipulation (e.g. does it fade back to 0.5 quickly, stay elevated, or oscillate),
    rather than the single scalar _fractional_preference_index reports at one fixed
    window position -- the windowed analog of _fraction_timeseries (which is the
    per-timestep, unwindowed version of the same underlying ratio).

    Returns (B, n_windows), n_windows = T - start - window + 1 (T = err0.shape[1]).
    Raises ValueError if the window doesn't fit at all (n_windows <= 0).
    """
    err0 = _slot_error(s_pred, s_target, 0)  # (B, T)
    err1 = _slot_error(s_pred, s_target, 1)  # (B, T)
    T = err0.shape[1]
    start = max(t_probe - predict_offset, 0)
    n_windows = T - start - window + 1
    if n_windows <= 0:
        raise ValueError(f"window={window} doesn't fit in the remaining trial "
                          f"(T={T}, start={start}, room={T - start})")

    changed = torch.as_tensor(changed_idxs, dtype=torch.long, device=err0.device)
    slices = []
    for w in range(n_windows):
        fractions = _windowed_fraction(err0, err1, start + w, start + w + window)  # (B, 2)
        slices.append(fractions.gather(1, changed.unsqueeze(1)).squeeze(1))  # (B,)
    return torch.stack(slices, dim=1)  # (B, n_windows)


def _plot_preference_group(data, labels, title, fig_path, figsize=(4, 4),
                            ylabel="Preference index\n(0 = familiar, 1 = novel)",
                            title_suffix="preference index"):
    """`ylabel`/`title_suffix` let this same box-plot rendering be reused for other
    bounded-[0,1] per-episode scalars beside the preference index itself (see
    evaluate_causal_ablation's AUC-over-stim-window box plot) -- defaults match the
    original preference-index usage exactly."""
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} {title_suffix}")
    ax.legend()
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


_PROBE_GENERATORS = {
    "location": (generate_probe_location_change_data, "OLM (location-change)", LOCATION_PREF_FIG_PATH, LOCATION_CAUSAL_FIG_PATH),
    "curvature": (generate_probe_curvature_change_data, "ORM (curvature-change)", CURVATURE_PREF_FIG_PATH, CURVATURE_CAUSAL_FIG_PATH),
}


def evaluate_preference_indices(ec, cb, seq_len=SEQ_LEN, t_probe=None, window=5, seed=0,
                                 device=None, fig_paths=None):
    """Compare fractional preference indices across the two deterministic, exhaustive
    probes: location-change (OLM), curvature-change (ORM) -- each against its own
    control baseline (should hover around chance = 0.5). Each probe's conditions share one
    scripted, controlled agent trajectory (see _controlled_trajectory); `seed` now only
    controls observation noise (off by default, OBSERVATION_NOISE_STD=0.0) -- curvature-
    change reuses the same `seed` as location-change, which is fine either way since
    they're independent probes, not paired episodes.

    `window` defaults to 3 -- see _fractional_preference_index for why: the excess
    prediction error at a manipulation is a ~1-timestep spike (CB re-observes the fully
    truthful current state every step, so nothing keeps it wrong once the new state is
    visible again), and the previous window=None default (average to end of episode)
    was measured to dilute that real signal by ~7x. Pass window=None explicitly to
    restore the old end-of-episode averaging if you need it (e.g. to check whether a
    probe's effect is unexpectedly more sustained than location-change's was).

    fig_paths, if given, overrides the default path for one or more of {"location",
    "curvature"} (dict); unspecified probes fall back to their module-level default
    (LOCATION_PREF_FIG_PATH etc).

    Returns per-episode preference-index arrays for both probes' control/change
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
                           window=5, seed=0, device=None, fig_paths=None):
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

    Reuses the same two deterministic probes as evaluate_preference_indices
    (location/curvature-change), but passes them through _shifted_forward_noisy instead
    of _shifted_forward for each (site, alpha) combination -- including the control
    condition, as a sanity check that noise alone (with nothing actually changed) doesn't
    spuriously pull the index away from chance (0.5).

    fig_paths, if given, overrides the default path for one or more of {"location",
    "curvature"} (dict); unspecified probes fall back to their module-level default.

    Returns per-(site, alpha) mean/SEM fractional preference index for both probes'
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


CLOSED_LOOP_CAUSAL_NOISE_FIG_PATHS = {
    "location": "figs/olm_closed_loop_causal_noise_location.png",
    "curvature": "figs/olm_closed_loop_causal_noise_curvature.png",
}


def evaluate_closed_loop_causal_noise(ec, cb, alphas, stim_start, stim_end, seq_len=SEQ_LEN, t_probe=None,
                                       window=5, seed=0, device=None, fig_paths=None):
    """Closed-loop counterpart of evaluate_causal_noise: the SAME dose-response sweep
    (Gaussian noise into "gru" vs. "cb_input", see _closed_loop_forward_noisy) but run on
    the CLOSED-LOOP model instead of the standard one -- for a true apples-to-apples
    comparison against evaluate_causal_ablation's zero_pred_err ablation on the SAME
    architecture. evaluate_causal_noise's standard-model sweep is a different
    architecture AND a different intervention; don't read the two side by side as if
    they were directly comparable.

    ec/cb MUST be the closed-loop model (train_olm_closed_loop/CLOSED_LOOP_CKPT_PATH, or
    train_olm_closed_loop_multi_seed for use with _olm_aggregate_eval_impl.py's
    run_aggregate_causal_ablation_evaluation) -- same requirement as
    evaluate_causal_ablation.

    _fractional_preference_index is called with predict_offset=1 (no 2-step shift),
    matching _closed_loop_forward_noisy/_closed_loop_forward_causal's un-shifted
    convention -- see evaluate_causal_ablation for the same adjustment.

    See evaluate_causal_noise for the full parameter/return-value documentation
    (identical here, only the underlying model/forward pass differs); fig_paths falls
    back to CLOSED_LOOP_CAUSAL_NOISE_FIG_PATHS rather than LOCATION_CAUSAL_FIG_PATH/
    CURVATURE_CAUSAL_FIG_PATH.
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
                    sp_ctrl, st_ctrl = _closed_loop_forward_noisy(ec, cb, ctrl, site, alpha, stim_start, stim_end)
                    sp_manip, st_manip = _closed_loop_forward_noisy(ec, cb, manip, site, alpha, stim_start, stim_end)
                    pref[f"{name}_control"] = _fractional_preference_index(
                        sp_ctrl, st_ctrl, idxs, t_probe, window, predict_offset=1)
                    pref[f"{name}_change"] = _fractional_preference_index(
                        sp_manip, st_manip, idxs, t_probe, window, predict_offset=1)

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
    for name, (_, title, _, _) in _PROBE_GENERATORS.items():
        _plot_causal_noise_group(sweep, f"{name}_change", f"{name}_control", f"{title} (closed-loop)",
                                  fig_paths.get(name, CLOSED_LOOP_CAUSAL_NOISE_FIG_PATHS[name]))
        results[f"{name}_t_probe"] = probe_data[name][3]

    for site in CAUSAL_NOISE_SITES:
        for c in conditions:
            results[f"{site}_{c}_mean"] = sweep[site][c]["mean"]
            results[f"{site}_{c}_sem"] = sweep[site][c]["sem"]

    return results


CAUSAL_ABLATION_FIG_PATHS = {
    "location": "figs/olm_causal_ablation_location.png",
    "curvature": "figs/orm_causal_ablation_curvature.png",
}
CAUSAL_ABLATION_FRACTION_FIG_PATHS = {
    "location": "figs/olm_causal_ablation_fraction_location.png",
    "curvature": "figs/orm_causal_ablation_fraction_curvature.png",
}
CAUSAL_ABLATION_FRACTION_OVERLAY_FIG_PATH = "figs/olm_orm_causal_ablation_fraction_overlay.png"
CAUSAL_ABLATION_SLIDING_PREF_FIG_PATHS = {
    "location": "figs/olm_causal_ablation_sliding_preference_location.png",
    "curvature": "figs/orm_causal_ablation_sliding_preference_curvature.png",
}
CAUSAL_ABLATION_AUC_FIG_PATHS = {
    "location": "figs/olm_causal_ablation_auc_location.png",
    "curvature": "figs/orm_causal_ablation_auc_curvature.png",
}
CAUSAL_ABLATION_PERSIST_FIG_PATHS = {
    "location": "figs/olm_causal_ablation_persist_location.png",
    "curvature": "figs/orm_causal_ablation_persist_curvature.png",
}


def evaluate_causal_ablation(ec, cb, seq_len=SEQ_LEN, t_probe=None, stim_start=None, stim_end=None,
                              alpha=CAUSAL_PERIODIC_ALPHA, period=CAUSAL_PERIODIC_PERIOD,
                              on_duration=CAUSAL_PERIODIC_ON_DURATION,
                              window=5, seed=0, device=None, fig_paths=None, fraction_fig_paths=None,
                              sliding_fig_paths=None, auc_fig_paths=None, persist_fig_paths=None,
                              fraction_overlay_fig_path=None):
    """Closed-loop counterpart of evaluate_causal_noise: instead of injecting Gaussian
    noise into the STANDARD model (_shifted_forward_noisy), this runs the CLOSED-LOOP
    model's PERIODIC prediction-error-noise perturbation (_closed_loop_forward_periodic_
    noise) -- intact vs. perturbed -- across the SAME deterministic, exhaustive probes
    evaluate_preference_indices uses (see _PROBE_GENERATORS), aggregating the fractional
    preference index for both control and manipulated episodes, intact and perturbed.

    ec/cb MUST be the closed-loop model (train_olm_closed_loop/CLOSED_LOOP_CKPT_PATH, or
    train_olm_closed_loop_multi_seed for a seed used with the aggregator in
    _olm_aggregate_eval_impl.py's run_aggregate_causal_ablation_evaluation) --
    architecturally incompatible with the standard model evaluate_preference_indices/
    evaluate_causal_noise use.

    The manipulation itself (see _closed_loop_forward_periodic_noise) is a PERIODIC duty
    cycle within an outer window, not one continuous ablation: on for `on_duration`
    consecutive steps, off for the rest of each `period`-length cycle, repeating from
    stim_start to stim_end -- and each "on" burst ADDS Gaussian noise onto pred_err_t
    rather than zeroing it outright, so the real error signal is corrupted, not silenced.
    `alpha` is the expected L2 norm of the noise injected at EACH perturbed timestep --
    total across all STATE_DIM dimensions together, not per dimension (see
    _closed_loop_forward_periodic_noise) -- so it's a PER-TIMESTEP budget: changing
    period/on_duration/stim_start/stim_end (with alpha fixed) changes how much noise gets
    injected in total across the trial, just not the size of any one burst.
    stim_start (the outer window's start) defaults to CAUSAL_STIM_LEAD timesteps before
    each probe's own t_probe (same lead-in as the single-trial diagnostics, see
    plot_probe_trial_causal_diagnostic) -- but unlike a CONTINUOUS ablation, where
    starting the window long before t_probe dilutes the comparison with a stretch where
    intact/ablated have nothing to differ on, a PERIODIC perturbation doesn't have that
    problem (the model recovers clean feedback between bursts either way), so there's no
    real cost to using a much larger CAUSAL_STIM_LEAD here -- set it >= t_probe (or pass
    stim_start=0 directly) to have the perturbation running well before the object even
    changes, testing whether the periodic noise itself (independent of the manipulation)
    disrupts the model's baseline representation. stim_end still defaults to the rest of
    the trial.

    The on/off PHASE is anchored to t_probe - 1 (see _closed_loop_forward_periodic_noise),
    not to stim_start -- so the one row whose target first reflects the manipulation is
    ALWAYS inside an "on" burst, no matter what stim_start/period/on_duration are set to.
    This matters because a location-change effect is a one-timestep transient: anchoring
    to stim_start instead would make whether that row falls in an "on" burst or an "off"
    gap an accident of arithmetic (does (t_probe - 1 - stim_start) happen to be a multiple
    of period, give or take on_duration?), and a miss would look exactly like "the model
    doesn't care about this perturbation" when it's really just bad luck in the schedule.

    The closed-loop model predicts s_{t+1} directly (no 2-step shift the way the standard
    model's _shifted_forward does), so _fractional_preference_index is called here with
    predict_offset=1, not the standard model's default of 2.

    fig_paths, if given, overrides the default path for one or more of {"location",
    "curvature"} (dict); unspecified probes fall back to CAUSAL_ABLATION_FIG_PATHS.
    fraction_fig_paths does the same for the fractional-attribution-over-time plot below,
    falling back to CAUSAL_ABLATION_FRACTION_FIG_PATHS. sliding_fig_paths does the same
    for the sliding-window preference-index plot, falling back to
    CAUSAL_ABLATION_SLIDING_PREF_FIG_PATHS. auc_fig_paths does the same for the
    AUC-over-stim-window box plot, falling back to CAUSAL_ABLATION_AUC_FIG_PATHS.
    persist_fig_paths does the same for the persistence-only (post-transient) AUC box
    plot, falling back to CAUSAL_ABLATION_PERSIST_FIG_PATHS. fraction_overlay_fig_path
    overrides the single combined-probe path below, falling back to
    CAUSAL_ABLATION_FRACTION_OVERLAY_FIG_PATH.

    ALSO saves, per probe, two mean +/- SEM over-time plots for the changed condition
    (intact vs. perturbed), both averaged over all conditions (144 for location, 432 for
    curvature -- which is what actually makes these curves legible: the single-trial
    version, plot_probe_trial_causal_diagnostic, computes the exact same kind of ratio
    but for just ONE trial, so it's inherently noisy; averaging over the full condition
    ensemble here shrinks that noise by roughly sqrt(n_conditions)):
      - fractional-attribution-over-time (_fraction_timeseries/
        _plot_causal_ablation_fraction): the raw, UNWINDOWED per-timestep ratio.
      - sliding-window preference index (_fractional_preference_index_sliding): the SAME
        `window`-sized average _fractional_preference_index itself uses for the box plot
        above, but slid across the whole rest of the trial instead of anchored once at
        t_probe -- shows whether the windowed index fades back to 0.5, stays elevated, or
        oscillates, at the same smoothing level as the scalar metric.
    Both plots shade the actual periodic on/off burst schedule (_periodic_on_intervals),
    not one solid continuous span.

    ALSO saves ONE combined plot (_plot_fraction_overlay) overlaying both probes'
    fractional-attribution-over-time curves on the same axes -- OLM in steelblue, ORM in
    firebrick -- for a direct side-by-side comparison alongside the two separate
    per-probe versions above (which are still saved as before).

    ALSO computes a 4-group (control/changed x intact/perturbed) box plot of each episode's
    _fraction_auc: the fractional-attribution-over-time curve integrated (and normalized
    by duration) across the WHOLE outer window [stim_start, stim_end), not just a
    short window anchored at t_probe -- motivated by the fractional-attribution-over-time
    plot showing curvature-change's (ORM) elevated attribution PERSISTING across the
    perturbation window while location-change's (OLM) decays back to baseline almost
    immediately; a short window like the preference index's mostly misses that
    persistence difference. See _fraction_auc.

    The full-window AUC above still conflates TWO separable effects -- how big the
    initial transient spike is, and how long any excess attribution lingers after that
    spike -- since it's integrated across both the transient and the tail together. The
    preference-index box plot already isolates the transient (a `window`-sized average
    anchored right at t_probe). This ALSO adds a complementary "persistence" AUC: the
    SAME _fraction_auc computation, but over only the tail of the stim window AFTER the
    transient has already played out ([t_probe + window, stim_end)) -- isolating whether
    excess attribution lingers once the initial spike itself is excluded. Together,
    preference index (transient) + persistence AUC (tail) decompose what the combined
    full-window AUC mixes into one number.

    Returns per-episode preference-index, full-window AUC, AND persistence-AUC arrays for
    both probes' control/change conditions, intact and perturbed, plus each probe's
    t_probe/stim window (and the period/on_duration/alpha schedule) and both over-time
    mean/SEM arrays, and saves one 4-group preference-index box plot, one 4-group
    full-window AUC box plot, one 4-group persistence-AUC box plot, AND both over-time
    plots, per probe.
    """
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()
    fig_paths = fig_paths or {}
    fraction_fig_paths = fraction_fig_paths or {}
    sliding_fig_paths = sliding_fig_paths or {}
    auc_fig_paths = auc_fig_paths or {}
    persist_fig_paths = persist_fig_paths or {}
    overlay_curves = {}  # name -> (mean_intact, sem_intact, mean_ablated, sem_ablated), for _plot_fraction_overlay
    overlay_t_axis = overlay_t_probe = overlay_on_intervals = None
    overlay_n_conditions = {}  # name -> n_conditions, since location/curvature differ (144 vs 432)

    results = {"window": -1 if window is None else window,
               "alpha": alpha, "period": period, "on_duration": on_duration}
    for name, (generator, title, _, _) in _PROBE_GENERATORS.items():
        ctrl, manip, idxs, this_t_probe = generator(seq_len=seq_len, t_probe=t_probe, seed=seed)
        this_stim_start = stim_start if stim_start is not None else max(0, this_t_probe - CAUSAL_STIM_LEAD)
        this_stim_end = seq_len if stim_end is None else stim_end
        # Anchor the on/off phase to t_probe - 1 (the one row whose target first reflects
        # the manipulation -- see _closed_loop_forward_periodic_noise), not stim_start, so
        # that row always falls inside an "on" burst regardless of period/on_duration/
        # stim_start: a location-change effect is a single-timestep transient, so if that
        # row landed in an "off" gap the perturbation could miss the only moment that
        # matters and show a false null effect purely from scheduling coincidence.
        anchor = this_t_probe - 1
        on_intervals = _periodic_on_intervals(this_stim_start, this_stim_end, period, on_duration, anchor=anchor)

        with torch.no_grad():
            sp_ctrl_i, st_ctrl = _closed_loop_forward_causal(ec, cb, ctrl.to(device), zero_pred_err=False)
            sp_ctrl_a, _ = _closed_loop_forward_periodic_noise(ec, cb, ctrl.to(device), alpha, period, on_duration,
                                                                stim_start=this_stim_start, stim_end=this_stim_end,
                                                                anchor=anchor)
            sp_manip_i, st_manip = _closed_loop_forward_causal(ec, cb, manip.to(device), zero_pred_err=False)
            sp_manip_a, _ = _closed_loop_forward_periodic_noise(ec, cb, manip.to(device), alpha, period, on_duration,
                                                                 stim_start=this_stim_start, stim_end=this_stim_end,
                                                                 anchor=anchor)

        pref_control_intact = _fractional_preference_index(
            sp_ctrl_i, st_ctrl, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()
        pref_control_ablated = _fractional_preference_index(
            sp_ctrl_a, st_ctrl, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()
        pref_change_intact = _fractional_preference_index(
            sp_manip_i, st_manip, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()
        pref_change_ablated = _fractional_preference_index(
            sp_manip_a, st_manip, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()

        _plot_preference_group(
            [pref_control_intact, pref_control_ablated, pref_change_intact, pref_change_ablated],
            ["control\n(intact)", "control\n(perturbed)", "changed\n(intact)", "changed\n(perturbed)"],
            f"{title} (closed-loop)",
            fig_paths.get(name, CAUSAL_ABLATION_FIG_PATHS[name]),
            figsize=(6, 4),
        )

        # Fractional-attribution-over-time, averaged over every condition (see docstring).
        frac_change_intact = _fraction_timeseries(sp_manip_i, st_manip, idxs).cpu().numpy()   # (B, T)
        frac_change_ablated = _fraction_timeseries(sp_manip_a, st_manip, idxs).cpu().numpy()  # (B, T)
        n_conditions = frac_change_intact.shape[0]
        frac_t_axis = np.arange(1, frac_change_intact.shape[1] + 1)  # closed-loop, no shift

        frac_mean_intact = frac_change_intact.mean(axis=0)
        frac_sem_intact = frac_change_intact.std(axis=0, ddof=1) / np.sqrt(n_conditions)
        frac_mean_ablated = frac_change_ablated.mean(axis=0)
        frac_sem_ablated = frac_change_ablated.std(axis=0, ddof=1) / np.sqrt(n_conditions)

        _plot_causal_ablation_fraction(
            frac_t_axis, frac_mean_intact, frac_sem_intact, frac_mean_ablated, frac_sem_ablated,
            this_t_probe, on_intervals, f"{title} (closed-loop)",
            fraction_fig_paths.get(name, CAUSAL_ABLATION_FRACTION_FIG_PATHS[name]),
            n_conditions=n_conditions,
        )
        overlay_curves[name] = (frac_mean_intact, frac_sem_intact, frac_mean_ablated, frac_sem_ablated)
        overlay_t_axis, overlay_t_probe = frac_t_axis, this_t_probe
        overlay_on_intervals = on_intervals
        overlay_n_conditions[name] = n_conditions

        # Sliding-window preference index, averaged over every condition -- same window
        # size as the box plot above, slid across the rest of the trial instead of
        # anchored once at t_probe (see _fractional_preference_index_sliding).
        sliding_intact = _fractional_preference_index_sliding(
            sp_manip_i, st_manip, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()   # (B, n_windows)
        sliding_ablated = _fractional_preference_index_sliding(
            sp_manip_a, st_manip, idxs, this_t_probe, window, predict_offset=1).cpu().numpy()  # (B, n_windows)
        sliding_start = max(this_t_probe - 1, 0)  # predict_offset=1, matches the function's own start
        sliding_t_axis = np.arange(sliding_start + 1, sliding_start + 1 + sliding_intact.shape[1])

        sliding_mean_intact = sliding_intact.mean(axis=0)
        sliding_sem_intact = sliding_intact.std(axis=0, ddof=1) / np.sqrt(n_conditions)
        sliding_mean_ablated = sliding_ablated.mean(axis=0)
        sliding_sem_ablated = sliding_ablated.std(axis=0, ddof=1) / np.sqrt(n_conditions)

        _plot_causal_ablation_fraction(
            sliding_t_axis, sliding_mean_intact, sliding_sem_intact, sliding_mean_ablated, sliding_sem_ablated,
            this_t_probe, on_intervals, f"{title} (closed-loop)",
            sliding_fig_paths.get(name, CAUSAL_ABLATION_SLIDING_PREF_FIG_PATHS[name]),
            n_conditions=n_conditions,
            ylabel="Preference index\n(0 = familiar, 1 = novel)",
            title_suffix=f"sliding-window preference index (w={window})",
        )

        # AUC over the FULL stimulation window (not just a short window anchored at
        # t_probe) -- reuses the control conditions' forward passes from the
        # preference-index computation above, so no extra model calls needed.
        frac_control_intact = _fraction_timeseries(sp_ctrl_i, st_ctrl, idxs).cpu().numpy()   # (B, T)
        frac_control_ablated = _fraction_timeseries(sp_ctrl_a, st_ctrl, idxs).cpu().numpy()  # (B, T)
        auc_end = min(this_stim_end, frac_change_intact.shape[1])
        auc_start = min(this_stim_start, auc_end)

        auc_control_intact = _fraction_auc(frac_control_intact, auc_start, auc_end)
        auc_control_ablated = _fraction_auc(frac_control_ablated, auc_start, auc_end)
        auc_change_intact = _fraction_auc(frac_change_intact, auc_start, auc_end)
        auc_change_ablated = _fraction_auc(frac_change_ablated, auc_start, auc_end)

        _plot_preference_group(
            [auc_control_intact, auc_control_ablated, auc_change_intact, auc_change_ablated],
            ["control\n(intact)", "control\n(perturbed)", "changed\n(intact)", "changed\n(perturbed)"],
            f"{title} (closed-loop)",
            auc_fig_paths.get(name, CAUSAL_ABLATION_AUC_FIG_PATHS[name]),
            figsize=(6, 4),
            ylabel="Fractional attribution AUC\n(time-averaged over stim window)",
            title_suffix=f"attribution AUC, stim window [{auc_start}, {auc_end})",
        )

        # Persistence-only AUC: same computation, but over the TAIL of the stim window,
        # after the initial transient (the preference index's own `window`) has already
        # played out -- isolates lingering elevation from the transient's own size.
        persist_start = min(this_t_probe + window, auc_end - 1) if auc_end > auc_start else auc_start
        persist_start = max(persist_start, auc_start)

        persist_control_intact = _fraction_auc(frac_control_intact, persist_start, auc_end)
        persist_control_ablated = _fraction_auc(frac_control_ablated, persist_start, auc_end)
        persist_change_intact = _fraction_auc(frac_change_intact, persist_start, auc_end)
        persist_change_ablated = _fraction_auc(frac_change_ablated, persist_start, auc_end)

        _plot_preference_group(
            [persist_control_intact, persist_control_ablated, persist_change_intact, persist_change_ablated],
            ["control\n(intact)", "control\n(perturbed)", "changed\n(intact)", "changed\n(perturbed)"],
            f"{title} (closed-loop)",
            persist_fig_paths.get(name, CAUSAL_ABLATION_PERSIST_FIG_PATHS[name]),
            figsize=(6, 4),
            ylabel="Fractional attribution AUC\n(persistence, post-transient tail)",
            title_suffix=f"persistence AUC, tail window [{persist_start}, {auc_end})",
        )

        results[f"pref_{name}_control_intact"] = pref_control_intact
        results[f"pref_{name}_control_ablated"] = pref_control_ablated
        results[f"pref_{name}_change_intact"] = pref_change_intact
        results[f"pref_{name}_change_ablated"] = pref_change_ablated
        results[f"{name}_t_probe"] = this_t_probe
        results[f"{name}_stim_start"] = this_stim_start
        results[f"{name}_stim_end"] = this_stim_end
        results[f"{name}_fraction_change_intact_mean"] = frac_mean_intact
        results[f"{name}_fraction_change_intact_sem"] = frac_sem_intact
        results[f"{name}_fraction_change_ablated_mean"] = frac_mean_ablated
        results[f"{name}_fraction_change_ablated_sem"] = frac_sem_ablated
        results[f"{name}_sliding_pref_change_intact_mean"] = sliding_mean_intact
        results[f"{name}_sliding_pref_change_intact_sem"] = sliding_sem_intact
        results[f"{name}_sliding_pref_change_ablated_mean"] = sliding_mean_ablated
        results[f"{name}_sliding_pref_change_ablated_sem"] = sliding_sem_ablated
        results[f"auc_{name}_control_intact"] = auc_control_intact
        results[f"auc_{name}_control_ablated"] = auc_control_ablated
        results[f"auc_{name}_change_intact"] = auc_change_intact
        results[f"auc_{name}_change_ablated"] = auc_change_ablated
        results[f"{name}_auc_stim_start"] = auc_start
        results[f"{name}_auc_stim_end"] = auc_end
        results[f"persist_{name}_control_intact"] = persist_control_intact
        results[f"persist_{name}_control_ablated"] = persist_control_ablated
        results[f"persist_{name}_change_intact"] = persist_change_intact
        results[f"persist_{name}_change_ablated"] = persist_change_ablated
        results[f"{name}_persist_stim_start"] = persist_start
        results[f"{name}_persist_stim_end"] = auc_end

    _plot_fraction_overlay(
        overlay_t_axis, overlay_curves, overlay_t_probe, overlay_on_intervals,
        "OLM vs ORM (closed-loop)",
        fraction_overlay_fig_path or CAUSAL_ABLATION_FRACTION_OVERLAY_FIG_PATH,
        n_conditions=overlay_n_conditions,
    )

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


def train_olm_closed_loop_multi_seed(seeds, out_dir=CLOSED_LOOP_MULTI_SEED_DIR, train_kwargs=None):
    """Train one CLOSED-LOOP OLM model per seed, saving each under
    out_dir/seed_{seed}/model.pt -- the closed-loop counterpart of train_olm_multi_seed,
    for use with _olm_aggregate_eval_impl.py's run_aggregate_causal_ablation_evaluation.

    Unlike train_olm_multi_seed, no probe evaluation is run here -- evaluate_causal_
    ablation is called separately, by the aggregator, against the saved checkpoint (the
    standard model's evaluate_displacement doesn't apply to the closed-loop architecture
    anyway; see _closed_loop_forward_causal vs. _shifted_forward).

    Returns a list of {"seed", "seed_dir", "history"} dicts, one per seed.
    """
    train_kwargs = dict(train_kwargs or {})

    results = []
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'=' * 60}\nOLM closed-loop multi-seed: seed={seed}\n{'=' * 60}")
        ec, cb, history, device = train_olm_closed_loop(
            seed=seed, ckpt_path=os.path.join(seed_dir, "model.pt"), **train_kwargs,
        )
        _save_training_history(seed_dir, history)

        results.append({"seed": seed, "seed_dir": seed_dir, "history": history})

    return results


def main(seeds=None, kind="closed_loop", out_dir=None, train_kwargs=None, eval_kwargs=None):
    """kind="closed_loop" (default): trains the CLOSED-LOOP model -- that's the primary
    model going forward. Pass `seeds` (a list of ints, or an int N for range(N)) for
    train_olm_closed_loop_multi_seed (out_dir defaults to CLOSED_LOOP_MULTI_SEED_DIR);
    omit for a single-seed train_olm_closed_loop() run (no evaluation -- see
    evaluate_causal_ablation/_olm_aggregate_eval_impl.py for the closed-loop equivalent
    of evaluate_displacement, run separately against the saved checkpoint). eval_kwargs
    is ignored for this kind (train_olm_closed_loop_multi_seed doesn't evaluate).

    kind="standard": trains the STANDARD/open-loop model instead (a different
    architecture, not an alternative view of the same one -- kept for backward
    compatibility/comparison). Pass `seeds` for train_olm_multi_seed (out_dir defaults to
    MULTI_SEED_DIR); omit for a single-seed train_olm() + evaluate_displacement() run, as
    this function used to always do before closed-loop became the default.
    """
    set_plt_params(style="fast", file_format="png")

    if kind == "closed_loop":
        out_dir = out_dir or CLOSED_LOOP_MULTI_SEED_DIR
        if seeds is None:
            return train_olm_closed_loop(**dict(train_kwargs or {}))
        if isinstance(seeds, int):
            seeds = np.arange(seeds)
        return train_olm_closed_loop_multi_seed(seeds, out_dir=out_dir, train_kwargs=train_kwargs)

    out_dir = out_dir or MULTI_SEED_DIR
    if seeds is not None:
        if isinstance(seeds, int):
            seeds = np.arange(seeds)

        return train_olm_multi_seed(seeds, out_dir=out_dir, train_kwargs=train_kwargs, eval_kwargs=eval_kwargs)

    ec, cb, history, device = train_olm()
    evaluate_displacement(ec, cb, device=device)


if __name__ == "__main__":
    main()
