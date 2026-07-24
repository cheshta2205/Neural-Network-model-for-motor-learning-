"""
within_context_decode_sro.py — show that action/value below-chance is an XOR
artifact, not missing information, by decoding them THREE ways from h_t (MTL):

  1. pooled         : one linear decoder over all 8 conditions          -> XOR, ~<=0.5
  2. within-context : separate decoder per context, then averaged       -> high (info IS present)
  3. cross-context  : train on context 0, test on context 1 (= CCGP)    -> fails (not abstract)

Context is run through the same three pipelines as a positive control:
it should be high pooled AND high cross-context (abstract), unlike action/value.

Run:  python within_context_decode_sro.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

import geometry_sro as G
from context_inference_sro_rpe import generate_sequences, load

N_SEQS = 300
SEED   = 999
M_COND = 60


def decode(X, y):
    """4-fold CV linear decode accuracy. SHUFFLE so folds don't split by condition
    order (otherwise 'pooled' silently becomes a cross-condition/by-context split)."""
    if len(set(y.tolist())) < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    return cross_val_score(LogisticRegression(max_iter=200), X, y, cv=cv).mean()


def cross_decode(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    return (clf.predict(Xte) == yte).mean()


def main():
    rng = np.random.default_rng(SEED)
    ec, cb, device = load()
    stim, ctx, Ostar, rstar = generate_sequences(N_SEQS, seed=SEED)
    reps = G.rollout_internals(ec, cb, stim, ctx, Ostar, rstar, device)
    X = reps["h_t (MTL)"]
    cond = G.cond_of_trial(stim.numpy(), ctx.numpy())          # ctx*4 + stim, in 0..7
    Xs, cs = G.subsample(X, cond, m=M_COND, rng=rng)

    ctx_id  = cs // 4                                           # 0 or 1
    stim_id = cs % 4
    _, _, op, rew = G.condition_table()                        # per-condition operant/reward
    action = op[cs]                                             # release/hold label per trial
    value  = rew[cs]                                            # rewarded/not label per trial

    print("\n=== Decoding action / value / context from h_t three ways ===\n")
    header = f"{'variable':10s} {'pooled':>10s} {'within-ctx':>12s} {'cross-ctx(CCGP)':>16s}"
    print(header); print("-" * len(header))

    for name, lab in [("action", action), ("value", value), ("context", ctx_id)]:
        # 1. pooled
        pooled = decode(Xs, lab)

        # 2. within-context (skip if the label is constant inside a context, e.g. context-as-label)
        within = []
        for c in (0, 1):
            m = ctx_id == c
            acc = decode(Xs[m], lab[m])
            if not np.isnan(acc):
                within.append(acc)
        within = np.mean(within) if within else np.nan

        # 3. cross-context: train ctx0 -> test ctx1, and ctx1 -> test ctx0, average
        m0, m1 = ctx_id == 0, ctx_id == 1
        cross = np.nan
        if len(set(lab[m0])) > 1 and len(set(lab[m1])) > 1:
            a = cross_decode(Xs[m0], lab[m0], Xs[m1], lab[m1])
            b = cross_decode(Xs[m1], lab[m1], Xs[m0], lab[m0])
            cross = 0.5 * (a + b)

        wtxt = "n/a (const)" if np.isnan(within) else f"{within:.3f}"
        ctxt = "n/a (const)" if np.isnan(cross)  else f"{cross:.3f}"
        print(f"{name:10s} {pooled:>10.3f} {wtxt:>12s} {ctxt:>16s}")

    print("\nchance = 0.50 for all three (binary labels)")
    print("expectation: action/value -> pooled<=.5, within-ctx high, cross-ctx<=.5 (XOR / not abstract)")
    print("             context      -> pooled high, within-ctx n/a,  cross-ctx high (abstract)")


if __name__ == "__main__":
    main()
