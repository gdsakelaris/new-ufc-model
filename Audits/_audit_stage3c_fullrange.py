"""Stage 3c: full-range leak-free feature-count sweep (winner stage, men).

Closes the loop on feature count by sweeping BELOW and ABOVE the current 240.
Mirrors the real pipeline: corr-prune the winner-eligible set on TRAIN (->~317),
rank by train-only importance, sweep K, score men on val + holdout (both clean).

If the curve peaks at ~240, WINNER_MAX_FEATURES is already optimal and feature
count is a closed question. Uses the cached Stage-2 bundle.
Run: python _audit_stage3c_fullrange.py
"""
import os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
BUNDLE_PATH = os.path.join(HERE, "_audit_stage2_bundle_v2.pkl")
BOOST_WEIGHT = 0.8
SEEDS = [0, 1, 2]
VAL_FIGHTS, TEST_FIGHTS = 600, 500

_T0 = time.time()
def banner(m): print(f"\n>>> [{time.time() - _T0:5.0f}s] {m}", flush=True)

spec = importlib.util.spec_from_file_location("ufc_model", os.path.join(HERE, "UFC_Model.py"))
ufc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ufc)

if not os.path.exists(BUNDLE_PATH):
    raise SystemExit("Run _audit_stage2.py first to create the cached bundle.")
banner("Loading cached bundle")
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
X_full, y = bundle["X_full"], bundle["y"].astype(int)
gender = bundle["gender"]

winner_elig = [c for c in X_full.columns
               if ufc._feature_allowed(c, "winner") and pd.api.types.is_numeric_dtype(X_full[c])]
n = len(X_full)
tr = np.arange(0, n - VAL_FIGHTS - TEST_FIGHTS)
va = np.arange(n - VAL_FIGHTS - TEST_FIGHTS, n - TEST_FIGHTS)
ho = np.arange(n - TEST_FIGHTS, n)

Xwe = X_full[winner_elig].apply(pd.to_numeric, errors="coerce")
imp = SimpleImputer(strategy="median").fit(Xwe.iloc[tr])
Xtr = pd.DataFrame(imp.transform(Xwe.iloc[tr]), columns=winner_elig)
Xva = pd.DataFrame(imp.transform(Xwe.iloc[va]), columns=winner_elig)
Xho = pd.DataFrame(imp.transform(Xwe.iloc[ho]), columns=winner_elig)
ytr = y.iloc[tr].reset_index(drop=True)

def men(idx): return gender.iloc[idx].str.strip().str.lower().eq("men").values
mv, mh = men(va), men(ho)
Xva_m = Xva.loc[mv].reset_index(drop=True); yva_m = y.iloc[va].reset_index(drop=True).loc[mv].reset_index(drop=True)
Xho_m = Xho.loc[mh].reset_index(drop=True); yho_m = y.iloc[ho].reset_index(drop=True).loc[mh].reset_index(drop=True)

# ---- mirror the real pipeline: corr-prune on TRAIN, then complete swap pairs ----
banner("Correlation-pruning on train (mirrors winner pipeline)")
kept = ufc._correlation_prune(Xtr[winner_elig], y=ytr, threshold=ufc.WINNER_CORR_PRUNE_THRESHOLD)
kept = ufc._complete_swap_pairs(kept, list(Xtr.columns))
print(f"winner-eligible {len(winner_elig)} -> corr-pruned {len(kept)}")

def make_boost(seed):
    if getattr(ufc, "lgb", None) is not None:
        return ufc.lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.12, reg_lambda=1.2, n_jobs=-1, verbose=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_depth=6, random_state=seed)

def fit_pair(Xc, seed):
    Xa, ya = ufc._augment_swap(Xc, ytr)
    b = make_boost(seed); b.fit(Xa, ya)
    e = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed); e.fit(Xa, ya)
    return b, e

def proba(pair, Xdf):
    b, e = pair; Xsw = ufc._swap_features(Xdf)
    def avg(m): return (m.predict_proba(Xdf)[:, 1] + (1.0 - m.predict_proba(Xsw)[:, 1])) / 2.0
    return np.clip(BOOST_WEIGHT * avg(b) + (1 - BOOST_WEIGHT) * avg(e), 1e-6, 1 - 1e-6)

banner("Ranking corr-pruned set by train-only importance")
imps = []
for s in SEEDS:
    Xa, ya = ufc._augment_swap(Xtr[kept], ytr)
    b = make_boost(s); b.fit(Xa, ya)
    imps.append(np.asarray(b.feature_importances_, dtype=float))
ranked = [kept[i] for i in np.argsort(np.mean(imps, axis=0))[::-1]]

def evaluate(cols):
    vll, vacc, hll, hacc = [], [], [], []
    for s in SEEDS:
        pair = fit_pair(Xtr[cols], s)
        pv, ph = proba(pair, Xva_m[cols]), proba(pair, Xho_m[cols])
        vll.append(log_loss(yva_m, pv)); vacc.append(accuracy_score(yva_m, pv >= 0.5))
        hll.append(log_loss(yho_m, ph)); hacc.append(accuracy_score(yho_m, ph >= 0.5))
    return np.mean(vll), np.mean(vacc), np.mean(hll), np.mean(hacc)

K_GRID = sorted(set([k for k in [160, 200, 220, 240, 260, 280, 300, len(kept)] if k <= len(kept)]))
rows = []
for K in K_GRID:
    banner(f"K={K}")
    vll, vacc, hll, hacc = evaluate(ranked[:K])
    rows.append({"K": K, "val_men_ll": vll, "val_men_acc": vacc, "ho_men_ll": hll, "ho_men_acc": hacc})

R = pd.DataFrame(rows)
ref = R[R["K"] == 240].iloc[0] if (R["K"] == 240).any() else R.iloc[0]
R["d_val_ll"] = R["val_men_ll"] - ref["val_men_ll"]
R["d_ho_ll"] = R["ho_men_ll"] - ref["ho_men_ll"]
R["sum_ll"] = R["val_men_ll"] + R["ho_men_ll"]

banner("RESULTS (both slices clean; deltas vs K=240)")
pd.set_option("display.width", 240)
fmt = {c: "{:.4f}".format for c in ["val_men_ll", "ho_men_ll", "d_val_ll", "d_ho_ll", "sum_ll"]}
fmt.update({c: "{:.1%}".format for c in ["val_men_acc", "ho_men_acc"]})
print(R.to_string(index=False, formatters=fmt))
best = R.loc[R["sum_ll"].idxmin(), "K"]
print(f"\nLowest combined (val+holdout) men's log-loss at K = {int(best)}.")
print("If that's ~240, WINNER_MAX_FEATURES is optimal — feature count is closed.")
R.to_csv(os.path.join(HERE, "_audit_stage3c_results.csv"), index=False)
banner("Stage 3c complete")
