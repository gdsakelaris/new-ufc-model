"""Stage 3: targeted prune experiment (WINNER stage, men).

Question: does dropping the low/negative-held-out features from the retained 240
IMPROVE men's generalization, or is it a null (per the 'feature-count is a null
lever' prior)?  We refit the surrogate on reduced feature sets and score men.

Bias control: the Stage-2 permutation importances were computed ON the men's
HOLDOUT, so selecting drop-sets from them and re-scoring the holdout is optimistic.
So we TRAIN on rows[:-1100] and treat the 600-fight VALIDATION slice (rows
[-1100:-500]) as the clean eval (its men were NOT used to pick the drop-sets);
holdout-men is reported as a secondary, optimism-biased check.

Uses the cached Stage-2 bundle (no 10-min rebuild). Run: python _audit_stage3_experiment.py
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
STAGE2_CSV = os.path.join(HERE, "_audit_stage2_results.csv")
BOOST_WEIGHT = 0.8
SEEDS = [0, 1, 2]            # average over model seeds to tame small deltas
VAL_FIGHTS = 600
TEST_FIGHTS = 500

_T0 = time.time()
def banner(m): print(f"\n>>> [{time.time() - _T0:5.0f}s] {m}", flush=True)

spec = importlib.util.spec_from_file_location("ufc_model",
                                              os.path.join(HERE, "UFC_Model.py"))
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

def men_mask(idx):
    return gender.iloc[idx].str.strip().str.lower().eq("men").values
mv, mh = men_mask(va), men_mask(ho)
Xva_m, yva_m = Xva.loc[mv].reset_index(drop=True), y.iloc[va].reset_index(drop=True).loc[mv].reset_index(drop=True)
Xho_m, yho_m = Xho.loc[mh].reset_index(drop=True), y.iloc[ho].reset_index(drop=True).loc[mh].reset_index(drop=True)
print(f"train {len(tr)}  | val-men {len(yva_m)}  | holdout-men {len(yho_m)}")

# ---- drop-sets from Stage-2 perm importances (retained features only) ----
s2 = pd.read_csv(STAGE2_CSV)
perm = s2[s2["status"] == "retained"].set_index("feature")["perm_ll_increase"]
worst_strict = [f for f in retained if perm.get(f, 0) < -0.0005]
worst_neg = [f for f in retained if perm.get(f, 0) < -0.0003]
core = [f for f in retained if perm.get(f, 0) > 0.0005]
momentum = [f for f in ["d_glicko_trend", "d_elo_slope_5", "d_def_str_pm", "d_td_att_p15",
                        "d_oba_ctrl_def", "d_oba_ctrl_off", "elo_win_prob", "d_last5_win_rate"]
            if f in retained]

variants = {
    "baseline_240": retained,
    f"drop momentum-{len(momentum)}": [f for f in retained if f not in set(momentum)],
    f"drop perm<-0.0005 ({len(worst_strict)})": [f for f in retained if f not in set(worst_strict)],
    f"drop perm<-0.0003 ({len(worst_neg)})": [f for f in retained if f not in set(worst_neg)],
    f"core only perm>0.0005 ({len(core)})": core,
}
print("\nmomentum set:", momentum)
print(f"worst_strict ({len(worst_strict)}):", worst_strict)


def make_boost(seed):
    if getattr(ufc, "lgb", None) is not None:
        return ufc.lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.12, reg_lambda=1.2, n_jobs=-1, verbose=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_depth=6,
                                          random_state=seed)

def fit_pair(Xc, yc, seed):
    Xa, ya = ufc._augment_swap(Xc, yc)
    b = make_boost(seed); b.fit(Xa, ya)
    e = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    e.fit(Xa, ya)
    return b, e

def proba(pair, Xdf):
    b, e = pair
    Xsw = ufc._swap_features(Xdf)
    def avg(m): return (m.predict_proba(Xdf)[:, 1] + (1.0 - m.predict_proba(Xsw)[:, 1])) / 2.0
    return np.clip(BOOST_WEIGHT * avg(b) + (1 - BOOST_WEIGHT) * avg(e), 1e-6, 1 - 1e-6)

def evaluate(cols):
    vll, vacc, hll, hacc = [], [], [], []
    for s in SEEDS:
        pair = fit_pair(Xtr[cols], ytr, s)
        pv = proba(pair, Xva_m[cols]); ph = proba(pair, Xho_m[cols])
        vll.append(log_loss(yva_m, pv)); vacc.append(accuracy_score(yva_m, pv >= 0.5))
        hll.append(log_loss(yho_m, ph)); hacc.append(accuracy_score(yho_m, ph >= 0.5))
    return np.mean(vll), np.mean(vacc), np.mean(hll), np.mean(hacc)


rows = []
for name, cols in variants.items():
    banner(f"Evaluating: {name}  ({len(cols)} feats, {len(SEEDS)} seeds)")
    vll, vacc, hll, hacc = evaluate(cols)
    rows.append({"variant": name, "nfeat": len(cols),
                 "val_men_ll": vll, "val_men_acc": vacc,
                 "ho_men_ll": hll, "ho_men_acc": hacc})

R = pd.DataFrame(rows)
b = R.iloc[0]
R["d_val_ll"] = R["val_men_ll"] - b["val_men_ll"]      # negative = better
R["d_val_acc"] = R["val_men_acc"] - b["val_men_acc"]   # positive = better
R["d_ho_acc"] = R["ho_men_acc"] - b["ho_men_acc"]

banner("RESULTS (val-men = clean signal; holdout-men = optimism-biased)")
pd.set_option("display.width", 240)
fmt = {c: "{:.4f}".format for c in ["val_men_ll", "ho_men_ll", "d_val_ll"]}
fmt.update({c: "{:.1%}".format for c in ["val_men_acc", "ho_men_acc", "d_val_acc", "d_ho_acc"]})
print(R.to_string(index=False, formatters=fmt))
print("\nKEY: d_val_ll < 0 AND d_val_acc > 0 on the clean val-men slice => the drop")
print("     genuinely helps men's generalization (not just holdout selection noise).")
R.to_csv(os.path.join(HERE, "_audit_stage3_results.csv"), index=False)
banner("Stage 3 complete")
