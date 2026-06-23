"""Paired multi-window WALK-FORWARD evaluation harness v2 (winner stage).

THE PROBLEM THIS SOLVES
-----------------------
The model is judged on ONE 500-fight holdout. The standard error of accuracy at
n=500, p~=0.66 is ~2.1 points, so a real +0.5..+1.0 pt change is *inside the
noise* and invisible -- which is why nearly every audit lever came back "within
noise (n=500)". This evaluates a change across K rolling windows with PAIRED
per-window deltas, so the "easy/hard batch" variance cancels and small effects
become visible.

v2 -- ALL FIVE rigor fixes from the critique are in:
  #1 LEAK-FREE per-window feature selection. Each window re-runs the real
     selection (corr-prune on TRAIN -> rank by TRAIN-only importance -> take K)
     on its own train slice. The deployed 240 set is NEVER reused, so the
     slim-core selection-leak trap cannot recur. Configs are SELECTION
     STRATEGIES (functions of train data), not fixed column lists.
  #2 FAITHFUL multi-family surrogate. Uses the model's OWN base-model
     constructors (_make_model_specs) for the families the combiner actually
     weights -- LightGBM + XGBoost + CatBoost (boosting) and ExtraTrees +
     ExtraTrees_Deep (bagging) -- blended with weights approximating the
     deployed combiner (AdaBoost removed, renormalized). Closes the "won't
     transfer to the real ensemble" gap that a lone LightGBM+ET blend left open.
  #3 CALIBRATION before ECE. Each window carves a calibration tail from train
     (mirrors the deployed train/val/holdout protocol), fits isotonic on it, and
     applies it before scoring -- so ECE reflects the deployed calibrated
     pipeline, not a raw blend.
  #4 HONEST significance. The paired t over K correlated windows is optimistic,
     so the headline is the WIN-COUNT (k/K) and the empirical NOISE FLOOR. Set
     RUN_NULL_FLOOR=True to measure that floor (A vs A with shifted seeds);
     anything smaller than the floor is nothing.
  #5 PER-DIVISION breakdown. Pools predictions across windows and reports
     men's deltas per weight class, so a change that helps LW but hurts MW
     doesn't hide inside a net "within noise".

REMAINING SCOPE LIMITS (honest):
  * Combiner weights are FIXED (deployed-approx simplex), not OOF-refit per
    window -- refitting the real robust combiner would collapse this into "just
    run the model 5x". Tuned hyperparameters aren't reproduced (no per-window
    Optuna); the base (untuned) family constructors stand in for their _Tuned
    deployed twins. So trust DELTAS, not absolute levels; confirm a winner with
    one full UFC_Model.py run.
  * Cannot test the inference-time odds blend (needs historical odds).
  * Reads the cached bundle (_audit_stage2_bundle_v2.pkl). If feature
    engineering or the era changes in UFC_Model.py, delete it + re-run
    _audit_stage2.py first, else this scores stale features.

HOW TO USE
----------
Two modes, picked by the K_SWEEP knob:
* K-SWEEP (default): K_SWEEP=[200,240,270,300] sweeps leak-free top-K (+ the full
  no-selection set) vs K_SWEEP_BASELINE, with per-division curves -- answers "did
  the optimal feature count move now that calibration + divisions are in view?"
* A/B: set K_SWEEP=[] to compare CONFIG_A (top-240) vs strat_variant(); edit
  strat_variant() to express any other feature change.
RUN_NULL_FLOOR=True adds a baseline-vs-itself(shifted-seeds) pass = the noise
floor; a result is real only if it EXCEEDS that floor.

Run:  python _audit_walkforward.py
"""
import os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

# ============================== KNOBS ==============================
HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
BUNDLE_PATH = os.path.join(HERE, "_audit_stage2_bundle_v2.pkl")
OUT_ROWS_CSV = os.path.join(HERE, "_audit_walkforward_rows.csv")
OUT_SUMMARY_CSV = os.path.join(HERE, "_audit_walkforward_summary.csv")

N_WINDOWS = 5            # rolling holdouts tiling the tail of the timeline
WINDOW_FIGHTS = None     # test-block size; None -> the model's TEST_FIGHTS (500)
CAL_FIGHTS = None        # calibration tail size; None -> the model's VAL_FIGHTS
MIN_TRAIN_FIGHTS = 3000  # refuse a window whose base-train slice is < this
SEEDS = [0]              # base-model seeds (proba averaged); [0] keeps it ~1 pass
N_BINS_ECE = 10
DIV_MIN_N = 60           # min pooled men's fights to report a division
RUN_NULL_FLOOR = True    # True -> also run baseline-vs-itself(shifted seeds) = noise floor
# K-SWEEP MODE: set K_SWEEP to a non-empty list to sweep top-K instead of the
# A/B CONFIG_A/B path. Each K is a leak-free train-selected top-K config; deltas
# are vs K_SWEEP_BASELINE; the full (no-selection) 374 is added as a reference.
# Set K_SWEEP = [] to fall back to the A/B (strat_variant / ablation) comparison.
K_SWEEP = []
K_SWEEP_BASELINE = 240
K_SWEEP_INCLUDE_FULL = True
# FEATURE-GROUP ABLATION (A/B mode only, K_SWEEP=[]): isolate one group's marginal
# value. A = leak-free top-K WITHOUT any column whose name contains a listed
# substring; B = A + exactly those columns. B beating A past the noise floor =
# the group helps. The named cols must be present in the current bundle. [] off.
# Testing the WINNER-eligible altitude features. After FEATURE_ROUTING, the
# winner stage sees the signed acclimatization DIFFERENTIALS + new composites
# (d_accl_shock_kft, d_alt_descent_kft, d_train_alt_kft, r/b_event_camp_gap_kft,
# d_alt_net_edge_kft, d_camp_x_event_kft, train_alt_known) — the raw per-corner
# shocks + absolute venue + mutual_gas/accl_asym are method-routed (this
# winner-only harness can't score those). The substrings below union to ALL
# altitude cols; winner_elig filtering keeps only the winner-eligible ones, so
# the printout shows exactly that winner group — confirm no strays.
# Requires a bundle REBUILT from the camp-altitude data + new features (delete
# _audit_stage2_bundle_v2.pkl, re-run _audit_stage2.py).
ABLATE_PREFIXES = ["alt", "accl", "gas", "camp"]
# Faithful surrogate: family -> deployed-approx combiner weight (AdaBoost removed,
# renormalized from the audit's non-zero weights). Unavailable libs are dropped
# and the rest renormalized.
SURROGATE_WEIGHTS = {
    "LightGBM": 0.24, "XGBoost": 0.15, "CatBoost": 0.10,
    "ExtraTrees": 0.13, "ExtraTrees_Deep": 0.08,
}
# ==================================================================

_T0 = time.time()
def banner(m): print(f"\n>>> [{time.time() - _T0:5.0f}s] {m}", flush=True)

spec = importlib.util.spec_from_file_location("ufc_model", os.path.join(HERE, "UFC_Model.py"))
ufc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ufc)
if WINDOW_FIGHTS is None:
    WINDOW_FIGHTS = int(ufc.TEST_FIGHTS)
if CAL_FIGHTS is None:
    CAL_FIGHTS = int(getattr(ufc, "VAL_FIGHTS", 600))

if not os.path.exists(BUNDLE_PATH):
    raise SystemExit("Missing bundle. Run _audit_stage2.py first to build "
                     f"{os.path.basename(BUNDLE_PATH)}.")
banner(f"Loading cached bundle {os.path.basename(BUNDLE_PATH)}")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
X_full = bundle["X_full"]
y = bundle["y"].astype(int)
n = len(X_full)

# meta (division + gender) straight from the model's own row-meta builder
row_meta = ufc._training_row_meta_from_csv(ufc.DATA_PATH).reset_index(drop=True).iloc[:n]
gender = row_meta["gender"].astype(str).reset_index(drop=True)
division = row_meta["weight_class"].astype(str).reset_index(drop=True)

winner_elig = [c for c in X_full.columns
               if ufc._feature_allowed(c, "winner") and pd.api.types.is_numeric_dtype(X_full[c])]
K_DEFAULT = int(getattr(ufc, "WINNER_MAX_FEATURES", 240))
print(f"rows {n} | winner-eligible {len(winner_elig)} | default K {K_DEFAULT} "
      f"| window {WINDOW_FIGHTS} | cal {CAL_FIGHTS} | seeds {SEEDS}")

ALL_SPECS = dict(ufc._make_model_specs())          # name -> make_fn (untuned)
SURR = {k: w for k, w in SURROGATE_WEIGHTS.items() if k in ALL_SPECS}
_wsum = sum(SURR.values())
SURR = {k: w / _wsum for k, w in SURR.items()}     # renormalize over available
print(f"surrogate families: " + ", ".join(f"{k}={w:.2f}" for k, w in SURR.items()))


def _finalize(cols):
    cols = [c for c in cols if c in winner_elig]
    if hasattr(ufc, "_complete_swap_pairs"):
        cols = ufc._complete_swap_pairs(cols, list(X_full.columns))
    seen, out = set(), []
    for c in cols:
        if c in winner_elig and c not in seen:
            seen.add(c); out.append(c)
    return out


def _seeded(make_fn, seed):
    est = make_fn()
    params = est.get_params()
    for pname in ("random_state", "random_seed"):
        if pname in params:
            try: est.set_params(**{pname: seed})
            except Exception: pass
    return est


def _rank_importance(Xtr_sel, ytr, seed):
    b = _seeded(ALL_SPECS["LightGBM"], seed)
    Xa, ya = ufc._augment_swap(Xtr_sel, ytr)
    b.fit(Xa, ya)
    return np.asarray(b.feature_importances_, float)


# ============ SELECTION STRATEGIES (edit strat_variant for experiments) ======
def strat_topk(Xtr_we, ytr, seed, K=K_DEFAULT):
    """Leak-free: corr-prune on train -> rank by train importance -> top-K."""
    kept = ufc._correlation_prune(Xtr_we[winner_elig], y=ytr,
                                  threshold=ufc.WINNER_CORR_PRUNE_THRESHOLD)
    kept = _finalize(kept)
    imp = _rank_importance(Xtr_we[kept], ytr, seed)
    ranked = [kept[i] for i in np.argsort(imp)[::-1]]
    return _finalize(ranked[:K])

def strat_full(Xtr_we, ytr, seed):
    return list(winner_elig)

def strat_variant(Xtr_we, ytr, seed):
    # DEFAULT = full eligible set (expected WORSE than train-selected top-K ->
    # validates the harness leak-free). To test your change, replace the body,
    # e.g.  return strat_topk(Xtr_we, ytr, seed, K=200)
    #   or  drop = {"d_elo_slope_5"}; return [c for c in strat_topk(Xtr_we,ytr,seed) if c not in drop]
    return strat_full(Xtr_we, ytr, seed)

def make_strat_ablate(prefixes, K=K_DEFAULT):
    """Leak-free top-K selected from the pool with every column whose name
    contains any of `prefixes` REMOVED (the 'without the group' baseline)."""
    drop = set(c for c in winner_elig if any(p in c for p in prefixes))
    def strat(Xtr_we, ytr, seed, _drop=drop, _K=K):
        pool = [c for c in winner_elig if c not in _drop]
        kept = ufc._correlation_prune(Xtr_we[pool], y=ytr,
                                      threshold=ufc.WINNER_CORR_PRUNE_THRESHOLD)
        kept = [c for c in _finalize(kept) if c not in _drop]
        imp = _rank_importance(Xtr_we[kept], ytr, seed)
        ranked = [kept[i] for i in np.argsort(imp)[::-1]]
        return [c for c in _finalize(ranked[:_K]) if c not in _drop]
    return strat

def make_strat_with_group(base_strat, group_cols):
    """B = the ablation baseline's exact selection PLUS the group columns, so the
    A->B delta is purely the marginal effect of adding the group."""
    grp = list(group_cols)
    def strat(Xtr_we, ytr, seed, _base=base_strat, _grp=grp):
        base = _base(Xtr_we, ytr, seed)
        return list(base) + [c for c in _grp if c not in set(base)]
    return strat

CONFIG_A = ("A: train-selected top-%d (leak-free)" % K_DEFAULT, strat_topk)
CONFIG_B = ("B: full winner-eligible (no selection)", strat_variant)
# =============================================================================


def _ece(p, yv, n_bins=N_BINS_ECE):
    p = np.asarray(p, float); yv = np.asarray(yv, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    N = len(p); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum() / N) * abs(p[m].mean() - yv[m].mean())
    return float(e)


def _predict_surrogate(models, Xeval_c):
    Xsw = ufc._swap_features(Xeval_c)
    acc = np.zeros(len(Xeval_c))
    for m, w in models:
        p = (m.predict_proba(Xeval_c)[:, 1] + (1.0 - m.predict_proba(Xsw)[:, 1])) / 2.0
        acc += w * p
    return acc


def _surr_raw_multi(Xtr_c, ytr_c, evals, seed_offset=0):
    """Fit the surrogate ONCE per seed, predict on each eval frame (cal + test).
    Combiner-weighted, swap-averaged, seed-averaged raw P(red)."""
    sums = [np.zeros(len(e)) for e in evals]
    for s in SEEDS:
        Xa, ya = ufc._augment_swap(Xtr_c, ytr_c)
        models = [(_seeded(ALL_SPECS[name], s + seed_offset).fit(Xa, ya), w)
                  for name, w in SURR.items()]
        for i, e in enumerate(evals):
            sums[i] += _predict_surrogate(models, e) / len(SEEDS)
    return [np.clip(x, 1e-6, 1.0 - 1e-6) for x in sums]


def run_config(strat, tr_idx, cal_idx, te_idx, Xwe_imp, seed_offset=0):
    """Leak-free select on train -> fit surrogate ONCE on train -> isotonic-
    calibrate on the cal tail -> return calibrated P(red) on the test block."""
    Xtr = Xwe_imp.iloc[tr_idx].reset_index(drop=True)
    ytr = y.iloc[tr_idx].reset_index(drop=True)
    cols = strat(Xtr, ytr, SEEDS[0] + seed_offset)
    Xtr_c = Xtr[cols]
    Xcal_c = Xwe_imp.iloc[cal_idx][cols].reset_index(drop=True)
    Xte_c = Xwe_imp.iloc[te_idx][cols].reset_index(drop=True)
    ycal = y.iloc[cal_idx].reset_index(drop=True).values
    raw_cal, raw_te = _surr_raw_multi(Xtr_c, ytr, [Xcal_c, Xte_c], seed_offset)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, ycal)
    p_te = np.clip(iso.transform(raw_te), 1e-6, 1.0 - 1e-6)
    return p_te, len(cols)


# ---- rolling windows: [ base-train | cal tail | test ]; window 0 = most recent ----
windows = []
for i in range(N_WINDOWS):
    te_end = n - i * WINDOW_FIGHTS
    te_start = te_end - WINDOW_FIGHTS
    cal_start = te_start - CAL_FIGHTS
    if cal_start < MIN_TRAIN_FIGHTS:
        print(f"  (stopping at {len(windows)} windows: base-train would be < {MIN_TRAIN_FIGHTS})")
        break
    windows.append((np.arange(0, cal_start), np.arange(cal_start, te_start),
                    np.arange(te_start, te_end)))
windows = windows[::-1]
if not windows:
    raise SystemExit("No valid windows -- lower WINDOW_FIGHTS/CAL_FIGHTS/MIN_TRAIN_FIGHTS.")

# ---- build the config list (K-sweep mode or A/B mode) ----
def _make_topk(K):
    return lambda Xtr, ytr, seed, K=K: strat_topk(Xtr, ytr, seed, K=K)

if K_SWEEP:
    SWEEP = True
    cfgs = [(f"K={K}", _make_topk(K), 0) for K in K_SWEEP]
    if f"K={K_SWEEP_BASELINE}" not in [c[0] for c in cfgs]:
        cfgs.insert(0, (f"K={K_SWEEP_BASELINE}", _make_topk(K_SWEEP_BASELINE), 0))
    if K_SWEEP_INCLUDE_FULL:
        cfgs.append(("full(no-sel)", strat_full, 0))
    baseline_label = f"K={K_SWEEP_BASELINE}"
elif ABLATE_PREFIXES:
    SWEEP = False
    ablate_cols = [c for c in winner_elig if any(p in c for p in ABLATE_PREFIXES)]
    if not ablate_cols:
        raise SystemExit(
            f"ABLATE_PREFIXES={ABLATE_PREFIXES} matched 0 columns in the bundle. "
            f"Rebuild it so the new features are captured: delete "
            f"{os.path.basename(BUNDLE_PATH)} and re-run _audit_stage2.py.")
    tag = "/".join(ABLATE_PREFIXES)
    print(f"ABLATION: {len(ablate_cols)} '{tag}' cols -> {ablate_cols}")
    labA = f"A: top-{K_DEFAULT} WITHOUT {tag}"
    labB = f"B: A + {tag} (+{len(ablate_cols)} cols)"
    stratA = make_strat_ablate(ABLATE_PREFIXES)
    stratB = make_strat_with_group(stratA, ablate_cols)
    cfgs = [(labA, stratA, 0), (labB, stratB, 0)]
    baseline_label = labA
else:
    SWEEP = False
    labA, stratA = CONFIG_A
    labB, stratB = CONFIG_B
    cfgs = [(labA, stratA, 0), (labB, stratB, 0)]
    baseline_label = labA

null_label = None
if RUN_NULL_FLOOR:
    base_strat = {c[0]: c[1] for c in cfgs}[baseline_label]
    null_label = baseline_label + "~null"
    cfgs.append((null_label, base_strat, 100))

banner(f"{len(windows)} windows x {len(cfgs)} configs x {len(SEEDS)} seed(s)"
       + ("  [K-SWEEP]" if SWEEP else "") + ("  +null-floor" if RUN_NULL_FLOOR else ""))

records = []
for wi, (tr_idx, cal_idx, te_idx) in enumerate(windows):
    t0 = time.time()
    span = f"[{te_idx[0]}-{te_idx[-1]}]"
    Xwe = X_full[winner_elig].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median").fit(Xwe.iloc[tr_idx])
    Xwe_imp = pd.DataFrame(imp.transform(Xwe), columns=winner_elig)
    yte = y.iloc[te_idx].reset_index(drop=True).values
    g = gender.iloc[te_idx].str.strip().str.lower().reset_index(drop=True).values
    dvn = division.iloc[te_idx].reset_index(drop=True).values
    gid = np.asarray(te_idx)
    kc = {}
    for (label, strat, off) in cfgs:
        p, k = run_config(strat, tr_idx, cal_idx, te_idx, Xwe_imp, seed_offset=off)
        kc[label] = k
        for j in range(len(yte)):
            records.append({"window": wi, "test_span": span, "gid": int(gid[j]),
                            "y": int(yte[j]), "gender": g[j], "division": dvn[j],
                            "config": label, "p": float(p[j])})
    banner(f"window {wi + 1}/{len(windows)} {span} {time.time() - t0:0.0f}s | "
           + ", ".join(f"{lab}={kc[lab]}" for lab, _, _ in cfgs))

R = pd.DataFrame(records)
R["is_men"] = R["gender"].str.strip().str.lower().eq("men")
R.to_csv(OUT_ROWS_CSV, index=False)
WK = sorted(R["window"].unique())


# ----------------------------- metrics & report -----------------------------
def grp(df):
    yv = df["y"].values; p = df["p"].values
    return {"n": len(df), "ll": log_loss(yv, p, labels=[0, 1]),
            "acc": accuracy_score(yv, (p >= 0.5).astype(int)),
            "brier": brier_score_loss(yv, p), "ece": _ece(p, yv)}

def rows_of(label, men_only=True, window=None, division=None):
    m = (R["config"] == label)
    if men_only:
        m = m & R["is_men"]
    if window is not None:
        m = m & (R["window"] == window)
    if division is not None:
        m = m & (R["division"] == division)
    return R[m]

def per_window_full(lb, la, men_only=True):
    rows = []
    for wi in WK:
        a = grp(rows_of(la, men_only, window=wi))
        b = grp(rows_of(lb, men_only, window=wi))
        rows.append({"window": wi, "test_span": R[R["window"] == wi]["test_span"].iloc[0],
                     "n": a["n"],
                     "A_ll": a["ll"], "B_ll": b["ll"], "d_ll": b["ll"] - a["ll"],
                     "A_acc": a["acc"], "B_acc": b["acc"], "d_acc": b["acc"] - a["acc"],
                     "A_ece": a["ece"], "B_ece": b["ece"], "d_ece": b["ece"] - a["ece"]})
    return pd.DataFrame(rows)

def summarize(d, name, la, lb):
    K = len(d)
    print(f"\n================ PAIRED WALK-FORWARD  [{name}]  (B - A) ================")
    print(f"  A = {la}\n  B = {lb}")
    fmt = {c: "{:.4f}".format for c in ["A_ll", "B_ll", "d_ll", "A_ece", "B_ece", "d_ece"]}
    fmt.update({c: "{:.1%}".format for c in ["A_acc", "B_acc", "d_acc"]})
    print(d.to_string(index=False, formatters=fmt))
    print("  ---- summary over windows (lower ll/ece better; higher acc better) ----")
    for col, better, pct in (("d_ll", "lower", False), ("d_ece", "lower", False),
                             ("d_acc", "higher", True)):
        v = d[col].values
        mean = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if K > 1 else 0.0
        wins = int((v < 0).sum()) if better == "lower" else int((v > 0).sum())
        t = mean / (sd / np.sqrt(K)) if sd > 0 else float("nan")
        show = f"{mean:+.2%} +/- {sd:.2%}" if pct else f"{mean:+.4f} +/- {sd:.4f}"
        leans = "B better" if ((mean < 0) == (better == "lower")) else "A better"
        print(f"    {col:<6} mean {show} | B wins {wins}/{K} | t(rough)={t:+.2f} | leans: {leans}")

def noise_floor():
    print("\n================ NOISE FLOOR  [MEN]  (baseline vs itself, shifted seeds) ====")
    print("  Pure fit/selection noise -- a real effect must EXCEED these.")
    d = per_window_full(null_label, baseline_label, men_only=True)
    for col in ("d_ll", "d_ece", "d_acc"):
        v = np.abs(d[col].values)
        print(f"    |{col}| floor: mean {np.mean(v):.4f}  max {np.max(v):.4f}")

sweep_labels = [c[0] for c in cfgs if c[0] != null_label]

if SWEEP:
    print(f"\n================ K-SWEEP (MEN, pooled)  baseline {baseline_label} ================")
    base_ll = grp(rows_of(baseline_label, True))["ll"]
    rows = []
    for label in sweep_labels:
        m = grp(rows_of(label, True))
        pw = per_window_full(label, baseline_label, men_only=True)
        wins = int((pw["d_ll"] < 0).sum())
        rows.append({"config": label, "n": m["n"], "ll": m["ll"], "acc": m["acc"],
                     "ece": m["ece"], "d_ll_vs_base": m["ll"] - base_ll,
                     "win/K": f"{wins}/{len(WK)}"})
    S = pd.DataFrame(rows)
    fmt = {c: "{:.4f}".format for c in ["ll", "ece", "d_ll_vs_base"]}; fmt["acc"] = "{:.1%}".format
    print(S.to_string(index=False, formatters=fmt))
    print(f"  lowest pooled men's log-loss: {S.loc[S['ll'].idxmin(), 'config']}  "
          f"(d_ll_vs_base < 0 = beats {baseline_label}; win/K = windows beating baseline)")

    print("\n  -- overall (men+women) pooled by config --")
    for label in sweep_labels:
        mo = grp(rows_of(label, men_only=False))
        print(f"    {label:<14} ll {mo['ll']:.4f}  acc {mo['acc']:.1%}  ece {mo['ece']:.4f}")

    print("\n================ K-SWEEP PER-DIVISION (MEN, pooled log-loss) ================")
    divs = []
    for d_, dfd in R[R["is_men"]].groupby("division"):
        n_men = int((dfd["config"] == baseline_label).sum())
        if n_men >= DIV_MIN_N:
            divs.append((d_, n_men))
    divs.sort(key=lambda t: -t[1])
    mat = []
    for d_, n_men in divs:
        row = {"division": d_, "n": n_men}
        best_ll, best_lab = np.inf, ""
        for label in sweep_labels:
            ll = grp(rows_of(label, True, division=d_))["ll"]
            row[label] = ll
            if ll < best_ll:
                best_ll, best_lab = ll, label
        row["best"] = best_lab
        mat.append(row)
    M = pd.DataFrame(mat)
    if len(M):
        fmt = {label: "{:.4f}".format for label in sweep_labels}
        print(M.to_string(index=False, formatters=fmt))
        print("  'best' = lowest-log-loss K per division. Thin divisions favoring a LOWER")
        print("  K than deep ones => a lead for division-aware feature counts.")
    else:
        print(f"  (no division reached {DIV_MIN_N} pooled men's fights)")

    if RUN_NULL_FLOOR:
        noise_floor()
        print("  -> compare each K's d_ll_vs_base above to the |d_ll| floor: a K is a real")
        print("     improvement only if its gain EXCEEDS the floor.")
    S.to_csv(OUT_SUMMARY_CSV, index=False)
else:
    dm = per_window_full(labB, labA, men_only=True); summarize(dm, "MEN (primary)", labA, labB)
    do = per_window_full(labB, labA, men_only=False); summarize(do, "OVERALL", labA, labB)
    print("\n  NOTE: paired t is OPTIMISTIC (correlated windows) -- read WIN-COUNT first.")
    if RUN_NULL_FLOOR:
        noise_floor()
    print("\n================ PER-DIVISION (MEN, pooled)  (B - A) ================")
    drows = []
    for d_, dfd in R[R["is_men"]].groupby("division"):
        n_men = int((dfd["config"] == labA).sum())
        if n_men < DIV_MIN_N:
            continue
        a = grp(rows_of(labA, True, division=d_)); b = grp(rows_of(labB, True, division=d_))
        drows.append({"division": d_, "n": n_men, "A_ll": a["ll"], "B_ll": b["ll"],
                      "d_ll": b["ll"] - a["ll"], "A_acc": a["acc"], "B_acc": b["acc"],
                      "d_acc": b["acc"] - a["acc"]})
    D = pd.DataFrame(drows).sort_values("n", ascending=False) if drows else pd.DataFrame()
    if len(D):
        fmt = {c: "{:.4f}".format for c in ["A_ll", "B_ll", "d_ll"]}
        fmt.update({c: "{:.1%}".format for c in ["A_acc", "B_acc", "d_acc"]})
        print(D.to_string(index=False, formatters=fmt))
    else:
        print(f"  (no division reached {DIV_MIN_N})")
    pd.concat([dm.assign(slice="men"), do.assign(slice="overall")]).to_csv(OUT_SUMMARY_CSV, index=False)

print(f"\nPer-row predictions -> {os.path.basename(OUT_ROWS_CSV)} | "
      f"summary -> {os.path.basename(OUT_SUMMARY_CSV)}")
print("Confirm any winner with ONE full UFC_Model.py run (faithful surrogate, not deployed).")
banner("Walk-forward v2 complete")
