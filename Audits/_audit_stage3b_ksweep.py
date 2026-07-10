"""Stage 3b: LEAK-FREE feature-count sweep (winner stage, men).

Mimics what lowering WINNER_MAX_FEATURES would do: rank the retained 240 by
TRAIN-ONLY importance (LightGBM split importance, same signal the model's
stability selection uses), take the top-K, refit the surrogate, score men.

Because selection uses ONLY train, BOTH val-men and holdout-men are clean eval
sets here (unlike Stage 3, where the core was holdout-selected). If a small K
wins on both, lowering WINNER_MAX_FEATURES in the real model is well-supported.

Uses the cached Stage-2 bundle. Run: python _audit_stage3b_ksweep.py
"""
import os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of Audits/)
BUNDLE_PATH = os.path.join(HERE, "_audit_stage2_bundle_v2.pkl")
BOOST_WEIGHT = 0.8
SEEDS = [0, 1, 2]
VAL_FIGHTS, TEST_FIGHTS = 600, 500
K_GRID = [240, 120, 80, 60, 40, 30, 20]

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
gender, feat_240 = bundle["gender"], bundle["feat_cols_240"]

retained = [c for c in feat_240
            if c in X_full.columns and pd.api.types.is_numeric_dtype(X_full[c])]
n = len(X_full)
tr = np.arange(0, n - VAL_FIGHTS - TEST_FIGHTS)
va = np.arange(n - VAL_FIGHTS - TEST_FIGHTS, n - TEST_FIGHTS)
ho = np.arange(n - TEST_FIGHTS, n)

Xall = X_full[retained].apply(pd.to_numeric, errors="coerce")
imp = SimpleImputer(strategy="median").fit(Xall.iloc[tr])
Xtr = pd.DataFrame(imp.transform(Xall.iloc[tr]), columns=retained)
Xva = pd.DataFrame(imp.transform(Xall.iloc[va]), columns=retained)
Xho = pd.DataFrame(imp.transform(Xall.iloc[ho]), columns=retained)
ytr = y.iloc[tr].reset_index(drop=True)

def men(idx): return gender.iloc[idx].str.strip().str.lower().eq("men").values
mv, mh = men(va), men(ho)
Xva_m = Xva.loc[mv].reset_index(drop=True); yva_m = y.iloc[va].reset_index(drop=True).loc[mv].reset_index(drop=True)
Xho_m = Xho.loc[mh].reset_index(drop=True); yho_m = y.iloc[ho].reset_index(drop=True).loc[mh].reset_index(drop=True)
print(f"train {len(tr)} | val-men {len(yva_m)} | holdout-men {len(yho_m)}")


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

# ---- TRAIN-ONLY importance ranking (leak-free), averaged over seeds ----
banner("Ranking features by train-only importance")
imps = []
for s in SEEDS:
    Xa, ya = ufc._augment_swap(Xtr[retained], ytr)
    b = make_boost(s); b.fit(Xa, ya)
    imps.append(np.asarray(b.feature_importances_, dtype=float))
imp_mean = np.mean(imps, axis=0)
ranked = [retained[i] for i in np.argsort(imp_mean)[::-1]]   # most -> least important

def evaluate(cols):
    vll, vacc, hll, hacc = [], [], [], []
    for s in SEEDS:
        pair = fit_pair(Xtr[cols], s)
        pv, ph = proba(pair, Xva_m[cols]), proba(pair, Xho_m[cols])
        vll.append(log_loss(yva_m, pv)); vacc.append(accuracy_score(yva_m, pv >= 0.5))
        hll.append(log_loss(yho_m, ph)); hacc.append(accuracy_score(yho_m, ph >= 0.5))
    return np.mean(vll), np.mean(vacc), np.mean(hll), np.mean(hacc)

rows = []
for K in K_GRID:
    cols = ranked[:K]
    banner(f"K={K}")
    vll, vacc, hll, hacc = evaluate(cols)
    rows.append({"K": K, "val_men_ll": vll, "val_men_acc": vacc,
                 "ho_men_ll": hll, "ho_men_acc": hacc})

R = pd.DataFrame(rows)
base = R[R["K"] == 240].iloc[0]
R["d_val_ll"] = R["val_men_ll"] - base["val_men_ll"]
R["d_val_acc"] = R["val_men_acc"] - base["val_men_acc"]
R["d_ho_ll"] = R["ho_men_ll"] - base["ho_men_ll"]
R["d_ho_acc"] = R["ho_men_acc"] - base["ho_men_acc"]

banner("RESULTS (both slices clean — selection used train only)")
pd.set_option("display.width", 240)
fmt = {c: "{:.4f}".format for c in ["val_men_ll", "ho_men_ll", "d_val_ll", "d_ho_ll"]}
fmt.update({c: "{:.1%}".format for c in ["val_men_acc", "ho_men_acc", "d_val_acc", "d_ho_acc"]})
print(R.to_string(index=False, formatters=fmt))
print("\nWANT: a K where d_val_ll<0 AND d_ho_ll<0 (log-loss better on BOTH clean slices).")
print("That K is the value to set WINNER_MAX_FEATURES to in UFC_Model.py.\n")
print("Top-30 features by train-only importance (the core that would survive):")
print(ranked[:30])
R.to_csv(os.path.join(HERE, "_audit_stage3b_results.csv"), index=False)
banner("Stage 3b complete")
