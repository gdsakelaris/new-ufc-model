"""Sharpening-recalibration experiment (cross-fitted, on the REAL holdout).

THE QUESTION
------------
The 2026-06-29 calibration audit found the model mildly UNDER-confident
(calibration slope ~1.35). Sharpening would map p -> sigmoid(s * logit(p)) with
s>1, pushing probabilities away from 0.5. Does a single global s actually improve
log-loss/ECE, or is the under-confidence just this holdout's sampling noise?

WHY NOT THE WALK-FORWARD HARNESS
--------------------------------
`_audit_walkforward.py` hardcodes ISOTONIC calibration per window, which would
silently correct the under-confidence and report a misleading "no benefit". This
uses the DEPLOYED ensemble's actual calibrated probabilities instead.

METHOD (leak-free)
------------------
Sharpening is ONE parameter, so the risk isn't overfitting -- it's mistaking
noise for signal. So:
  * s is fit on K-1 folds and scored on the held-out fold (out-of-fold pooled) --
    s is never scored on the fights it was fit on.
  * the per-fight log-loss difference (sharpened - baseline) is bootstrapped for a
    95%% CI. If the CI sits below 0, the gain survives sampling noise.
  * s* stability is reported across repeated folds (is it consistently >1?).

HONEST SCOPE
------------
One ~500-fight holdout = one era. A robust cross-fitted gain here warrants a
multi-window REAL-model confirmation before shipping; a null (CI spans 0) means
leave calibration alone -- under-confidence is the safe direction for picks.

Run:  python Audits/_audit_recalibration.py
"""
import os, io, sys, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import importlib.util

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
MODEL_PATH = os.path.join(HERE, "UFC_Model.py")
EPS = 1e-6
S_GRID = np.linspace(0.50, 2.50, 201)   # candidate sharpening factors
K_FOLDS = 5
N_REPEATS = 5
N_BOOT = 5000
N_BINS_ECE = 10
SEED = 0


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sharpen(p, s):
    return 1.0 / (1.0 + np.exp(-s * _logit(p)))


def _logloss_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logloss(y, p):
    return float(_logloss_vec(y, p).mean())


def _ece(y, p, n_bins=N_BINS_ECE):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    inds = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = inds == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _best_s(y, p):
    """Sharpening factor minimizing log-loss on (y, p), over the grid."""
    lls = [_logloss(y, _sharpen(p, s)) for s in S_GRID]
    return float(S_GRID[int(np.argmin(lls))])


def _oof_sharpened(y, p, seed):
    """One K-fold pass: return out-of-fold sharpened probs (s fit on the OTHER
    folds for each fold) plus the per-fold s* values."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K_FOLDS)
    oof = np.empty(n)
    s_list = []
    for f in range(K_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(K_FOLDS) if g != f])
        s = _best_s(y[tr], p[tr])
        s_list.append(s)
        oof[te] = _sharpen(p[te], s)
    return oof, s_list


def main():
    spec = importlib.util.spec_from_file_location("ufc_model", MODEL_PATH)
    ufc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ufc)

    print("Training winner stage (winner_only; cache-hit if warm)...")
    pipe = ufc.UFCSuperModelPipeline(ufc.DATA_PATH)
    logbuf = io.StringIO()
    try:
        with contextlib.redirect_stdout(logbuf):
            pipe.train(winner_only=True)
    except Exception as e:
        print("TRAIN FAILED:", repr(e)); print(logbuf.getvalue()[-2000:]); raise

    if not hasattr(pipe, "holdout_prob_cal"):
        print("ERROR: holdout arrays missing -- rerun after the UFC_Model.py edits.")
        sys.exit(1)

    y = np.asarray(pipe.holdout_y_true, dtype=int)
    p = np.asarray(pipe.holdout_prob_cal, dtype=float)
    n = len(y)

    print("\n" + "=" * 74)
    print(f"SHARPENING RECALIBRATION  --  cross-fitted on holdout n = {n}")
    print("=" * 74)

    base_ll, base_ece = _logloss(y, p), _ece(y, p)
    s_full = _best_s(y, p)  # in-sample optimum (illustrative; ~ the slope)
    print(f"\n  Baseline (s=1.0)         log-loss {base_ll:.4f}   ECE {base_ece:.4f}")
    print(f"  In-sample optimum        s* = {s_full:.2f}  "
          f"(log-loss {_logloss(y, _sharpen(p, s_full)):.4f}, ECE "
          f"{_ece(y, _sharpen(p, s_full)):.4f})  [illustrative, fit=eval]")

    # --- honest out-of-fold estimate, averaged over repeats ---
    oof_lls, oof_eces, all_s = [], [], []
    canonical_oof = None
    for r in range(N_REPEATS):
        oof, s_list = _oof_sharpened(y, p, seed=SEED + r)
        oof_lls.append(_logloss(y, oof))
        oof_eces.append(_ece(y, oof))
        all_s.extend(s_list)
        if r == 0:
            canonical_oof = oof
    oof_ll_mean, oof_ll_std = np.mean(oof_lls), np.std(oof_lls)
    oof_ece_mean = np.mean(oof_eces)
    s_arr = np.asarray(all_s)

    print(f"\n  Out-of-fold sharpened    log-loss {oof_ll_mean:.4f} (+-{oof_ll_std:.4f})"
          f"   ECE {oof_ece_mean:.4f}")
    print(f"  s* across {len(s_arr)} folds        mean {s_arr.mean():.2f}, "
          f"sd {s_arr.std():.2f}, range [{s_arr.min():.2f}, {s_arr.max():.2f}]")
    print(f"  fraction of folds with s*>1: {np.mean(s_arr > 1.0) * 100:.0f}%")

    # --- significance: bootstrap the per-fight oof log-loss delta ---
    base_pp = _logloss_vec(y, p)
    shrp_pp = _logloss_vec(y, canonical_oof)
    delta_pp = shrp_pp - base_pp          # negative = sharpening helps
    mean_delta = float(delta_pp.mean())
    rng = np.random.default_rng(SEED)
    boots = np.array([delta_pp[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print("\n  --- significance (out-of-fold log-loss delta, sharpened - baseline) ---")
    print(f"  mean delta {mean_delta:+.4f}   95%% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"(negative = improvement)")

    print("\n" + "=" * 74)
    if hi < 0:
        print("  VERDICT: sharpening GENERALIZES within the holdout (CI below 0).")
        print(f"  -> worth a multi-window REAL-model confirmation before shipping; the")
        print(f"     data-driven factor is ~{s_arr.mean():.2f}.")
    elif lo > 0:
        print("  VERDICT: sharpening HURTS (CI above 0). Do not sharpen.")
    else:
        print("  VERDICT: NULL -- the log-loss delta CI spans 0. The under-confidence")
        print("  is within sampling noise on this holdout; no generalizable gain.")
        print("  Leave calibration alone (under-confidence is safe for picks).")
    print("=" * 74)
    print("\n  Reminder: one holdout / one era. Even a positive result is provisional")
    print("  until confirmed across walk-forward windows on the real model.")


if __name__ == "__main__":
    main()
