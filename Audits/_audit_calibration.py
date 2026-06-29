"""Holdout winner-probability CALIBRATION audit.

THE QUESTION
------------
When the model shows "70%", does that fighter actually win ~70% of the time?
That is calibration, and it is the property every EV/Kelly number downstream
silently assumes. This audit measures it on the deployed pipeline's own
out-of-sample holdout.

WHAT IT DOES
------------
Runs the real pipeline once (uses the winner cache, so it is fast when warm),
reads the aligned holdout it just exposed -- the TRUE outcome and the CALIBRATED
P(red), which is the exact probability the GUI prints as "Win %" -- and scores
calibration two ways:

  1. PICK CONFIDENCE (the headline). Fold probs to >=0.5 -- i.e. the confidence
     of the side the model actually picks, which is what you see as "Win %".
     Answers directly: do 70%-confidence picks win 70% of the time?
  2. PER-CORNER P(red). The raw calibration object; also surfaces corner bias
     (does the model systematically over/under-rate the red corner?).

HONEST NOISE CAVEAT (printed with the results)
----------------------------------------------
This is ONE ~500-fight holdout. Per-bin counts are small, so each observed rate
carries a wide Wilson 95% CI. A bar that misses the diagonal but whose CI still
covers it is NOISE, not miscalibration. Trust the whole-set summary -- ECE, and
the calibration slope/intercept -- over any single bin. (For a tighter read,
pool predictions across the walk-forward windows; that is the bigger job.)

Run:  python Audits/_audit_calibration.py
"""
import os, io, sys, math, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import importlib.util

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
MODEL_PATH = os.path.join(HERE, "UFC_Model.py")
PNG_OUT = os.path.join(HERE, "_audit_calibration.png")
N_BINS = 10

# ── thresholds for the heuristic verdict (NOT gospel; calibration is graded on a
#    spectrum and this holdout is noisy). Roughly: the model's own logs treat ECE
#    ~0.05 as good; a calibration slope near 1 and intercept near 0 mean the
#    probabilities are neither over- nor under-confident on average.
ECE_GOOD, ECE_OK = 0.05, 0.08
SLOPE_LO, SLOPE_HI = 0.80, 1.20
INTERCEPT_ABS_OK = 0.20


def _wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial rate k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _reliability(y, p, edges):
    """Per-bin (lo, hi, n, mean_pred, obs_rate, ci_lo, ci_hi, covers_diag)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (p >= lo) & (p <= hi) if last else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), True))
            continue
        pred = float(p[mask].mean())
        k = int(round(y[mask].sum()))
        obs = k / n
        ci_lo, ci_hi = _wilson(k, n)
        covers = (ci_lo <= pred <= ci_hi)
        rows.append((lo, hi, n, pred, obs, ci_lo, ci_hi, covers))
    return rows


def _print_table(title, rows, label):
    print(f"\n{title}")
    print("-" * 78)
    print(f"  {'bin':<13}{'n':>5}   {label:<10}{'observed':<10}"
          f"{'95% CI':<18}{'gap':>7}   flag")
    print("-" * 78)
    miss = 0
    for lo, hi, n, pred, obs, cl, ch, covers in rows:
        if n == 0:
            print(f"  {lo*100:4.0f}-{hi*100:<3.0f}%  {0:>5}   "
                  f"{'--':<10}{'--':<10}{'--':<18}{'--':>7}")
            continue
        flag = "" if covers else "MISS"
        if not covers:
            miss += 1
        gap = obs - pred
        ci = f"[{cl*100:4.1f},{ch*100:5.1f}]"
        print(f"  {lo*100:4.0f}-{hi*100:<3.0f}%  {n:>5}   "
              f"{pred*100:6.1f}%   {obs*100:6.1f}%   {ci:<16}{gap*100:+6.1f}%   {flag}")
    print("-" * 78)
    graded = sum(1 for r in rows if r[2] > 0)
    print(f"  {miss}/{graded} populated bins miss the diagonal beyond their 95% CI "
          f"(0 = consistent with perfect calibration).")
    return miss


def _ece(y, p, n_bins=N_BINS):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    inds = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = mce = 0.0
    for b in range(n_bins):
        m = inds == b
        if not m.any():
            continue
        gap = abs(y[m].mean() - p[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _slope_intercept(y, p):
    """Cox calibration regression: fit y ~ logit(p). slope=1, intercept=0 is
    perfect. slope<1 => over-confident (probs too extreme); intercept!=0 => a
    constant over/under-prediction (here, of the red corner)."""
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y, dtype=int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _save_plot(corner_rows, conf_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib unavailable, skipping PNG: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, rows, title, lo in ((axes[0], corner_rows, "Per-corner P(red)", 0.0),
                                (axes[1], conf_rows, "Pick confidence (Win %)", 0.5)):
        xs = [(r[3]) for r in rows if r[2] > 0]
        ys = [(r[4]) for r in rows if r[2] > 0]
        los = [r[4] - r[5] for r in rows if r[2] > 0]
        his = [r[6] - r[4] for r in rows if r[2] > 0]
        ax.plot([lo, 1], [lo, 1], "k--", lw=1, label="perfect")
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", capsize=3, label="observed")
        ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
        ax.set_xlabel("predicted"); ax.set_ylabel("observed win rate")
        ax.set_title(title); ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Winner-probability calibration (holdout)")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=120)
    print(f"\nReliability diagram saved: {PNG_OUT}")


def main():
    spec = importlib.util.spec_from_file_location("ufc_model", MODEL_PATH)
    ufc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ufc)

    print("Training winner stage (winner_only=True skips the method stage; "
          "fast when the winner cache is warm)...")
    pipe = ufc.UFCSuperModelPipeline(ufc.DATA_PATH)
    logbuf = io.StringIO()
    try:
        with contextlib.redirect_stdout(logbuf):
            pipe.train(winner_only=True)
    except Exception as e:
        print("TRAIN FAILED:", repr(e))
        print(logbuf.getvalue()[-2000:])
        raise

    if not hasattr(pipe, "holdout_y_true"):
        print("ERROR: pipe.holdout_y_true missing -- the UFC_Model.py exposure "
              "edit didn't apply, or an old build is loaded.")
        sys.exit(1)

    y = np.asarray(pipe.holdout_y_true, dtype=int)          # 1 = red corner won
    p_red = np.asarray(pipe.holdout_prob_cal, dtype=float)  # calibrated P(red) = "Win %"
    n = len(y)

    print("\n" + "=" * 78)
    print(f"WINNER-PROBABILITY CALIBRATION  --  holdout n = {n} fights")
    print("=" * 78)

    # 1) Headline: pick-confidence (folded at 0.5, exactly how Win % is displayed).
    pick_red = p_red >= 0.5
    conf = np.where(pick_red, p_red, 1.0 - p_red)
    hit = (y == pick_red.astype(int)).astype(int)   # did the PICKED fighter win?
    conf_rows = _reliability(hit, conf, np.linspace(0.5, 1.0, N_BINS + 1))
    _print_table("[1] PICK CONFIDENCE  -- 'when I show X%, the pick wins X%'",
                 conf_rows, "shown")
    print(f"      Overall pick accuracy: {hit.mean()*100:.1f}%  "
          f"(mean shown confidence: {conf.mean()*100:.1f}%)")

    # 2) Per-corner P(red): cleaner statistical object + corner-bias check.
    corner_rows = _reliability(y, p_red, np.linspace(0.0, 1.0, N_BINS + 1))
    _print_table("[2] PER-CORNER P(red)  -- raw calibration + corner bias",
                 corner_rows, "pred")

    # Whole-set summary metrics (stable; trust these over single bins).
    ece, mce = _ece(y, p_red)
    brier = float(np.mean((p_red - y) ** 2))
    pc = np.clip(p_red, 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    slope, intercept = _slope_intercept(y, p_red)
    mean_pred_red, base_red = float(p_red.mean()), float(y.mean())

    print("\n" + "=" * 78)
    print("WHOLE-SET SUMMARY  (more reliable than any single bin)")
    print("=" * 78)
    print(f"  ECE (expected calib. error) : {ece:.4f}   "
          f"[<{ECE_GOOD:.2f} good, <{ECE_OK:.2f} ok]")
    print(f"  MCE (worst-bin gap)         : {mce:.4f}")
    print(f"  Brier score                 : {brier:.4f}")
    print(f"  Log-loss                    : {logloss:.4f}")
    print(f"  Calibration slope           : {slope:.3f}   "
          f"[1.0 = ideal; <1 over-confident, >1 under-confident]")
    print(f"  Calibration intercept       : {intercept:+.3f}   "
          f"[0 = ideal; sign = systematic red-corner bias]")
    print(f"  Mean P(red) vs red win-rate : {mean_pred_red*100:.1f}% vs "
          f"{base_red*100:.1f}%  (gap {abs(mean_pred_red-base_red)*100:.1f} pts)")

    # Heuristic verdict.
    flags = []
    if ece <= ECE_GOOD:
        verdict = "WELL CALIBRATED"
    elif ece <= ECE_OK:
        verdict = "ACCEPTABLE"
    else:
        verdict = "MISCALIBRATED"
        flags.append(f"ECE {ece:.3f} exceeds {ECE_OK:.2f}")
    if not (SLOPE_LO <= slope <= SLOPE_HI):
        flags.append(f"slope {slope:.2f} outside [{SLOPE_LO},{SLOPE_HI}] "
                     f"({'over' if slope < 1 else 'under'}-confident)")
    if abs(intercept) > INTERCEPT_ABS_OK:
        flags.append(f"intercept {intercept:+.2f} large (corner bias)")
    print("\n  VERDICT: " + verdict)
    for f in flags:
        print(f"    - watch: {f}")
    print("\n  NOTE: one ~500-fight holdout -> per-bin rates are noisy (see Wilson")
    print("  CIs above). The verdict leans on ECE + slope/intercept, which pool")
    print("  the whole set. A bin flagged MISS whose CI still spans the diagonal")
    print("  is sampling noise, not a real calibration defect.")

    _save_plot(corner_rows, conf_rows)


if __name__ == "__main__":
    main()
