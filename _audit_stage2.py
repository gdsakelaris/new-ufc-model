"""Stage 2 audit (WINNER stage). Two questions, one run:

  (A) Among the RETAINED 240 features, which carry no held-out men's signal?
      -> prune candidates.
  (B) Among the DROPPED winner-eligible features (374 -> 240 via corr-prune +
      stability selection), do any carry held-out men's signal the model can't
      see?  -> "the prune threw away something" candidates.

Method: pipeline.model is retrained on ALL data incl. holdout (UFC_Model.py:7358),
so permutation importance on it is leaky. We fit TRAIN-ONLY surrogates and permute
on the MEN'S holdout. Surrogate = LightGBM (boosting, matches your 0.24-weight
LightGBM_Tuned family) + ExtraTrees (bagging, your only weighted bagging family),
blended 0.8/0.2 to match the deployed ~80/20 boosting/bagging split. Corner
symmetry mirrored via _augment_swap (train) and swap-averaged scoring (predict),
exactly like the model's walk-forward.

Throwaway diagnostic harness. Run:  python _audit_stage2.py
"""
import os, io, sys, time, contextlib, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score

_T0 = time.time()
def banner(msg):
    print(f"\n>>> [{time.time() - _T0:5.0f}s] {msg}", flush=True)

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
MODEL_PATH = os.path.join(HERE, "UFC_Model.py")
BUNDLE_PATH = os.path.join(HERE, "_audit_stage2_bundle_v2.pkl")  # full 426-col matrix
STAGE1_CSV = os.path.join(HERE, "_audit_importances.csv")
OUT_CSV = os.path.join(HERE, "_audit_stage2_results.csv")

N_PERM_REPEATS = 3       # shuffles per feature (averaged); 374 feats -> keep modest
BOOST_WEIGHT = 0.8       # boosting/bagging blend, matching deployed ~80/20
SEED = 0

spec = importlib.util.spec_from_file_location("ufc_model", MODEL_PATH)
ufc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ufc)
TEST_FIGHTS = int(ufc.TEST_FIGHTS)


def build_bundle():
    cap = {}
    _orig_build = ufc.build_training_data
    def _build_cap(*a, **k):
        res = _orig_build(*a, **k)
        cap["y"] = pd.Series(np.asarray(res[1])).reset_index(drop=True)
        return res
    ufc.build_training_data = _build_cap

    _orig_aug = ufc._augment_matchup_features
    def _aug_cap(X_df):
        out = _orig_aug(X_df)
        if len(out) > 1000:                       # the full training matrix, not 1-row predict
            cap["X_full"] = out.reset_index(drop=True).copy()
        return out
    ufc._augment_matchup_features = _aug_cap

    real_out = sys.stdout
    _keys = ("Building features", "cache", "Era Selection", "Split Contract",
             "Rows kept", "Walk-forward", "Holdout")
    def _prog(msg):
        s = str(msg)
        if s.startswith("Fold ") or any(k in s for k in _keys):
            print("      " + s, file=real_out, flush=True)
    pipe = ufc.UFCSuperModelPipeline(ufc.DATA_PATH, logger=_prog)
    banner("Building feature matrix via train() — live progress below (~10 min, cache-warm)")
    logbuf = io.StringIO()
    with contextlib.redirect_stdout(logbuf):
        pipe.train()
    banner("Feature matrix built")
    if "X_full" not in cap or "y" not in cap:
        print(logbuf.getvalue()[-2000:]); raise SystemExit("Capture failed.")

    row_meta = ufc._training_row_meta_from_csv(ufc.DATA_PATH).reset_index(drop=True)
    n = min(len(cap["X_full"]), len(cap["y"]), len(row_meta))
    bundle = {
        "X_full": cap["X_full"].iloc[:n],
        "y": cap["y"].iloc[:n].reset_index(drop=True),
        "gender": row_meta["gender"].astype(str).iloc[:n].reset_index(drop=True),
        "feat_cols_240": list(pipe.model.feat_cols),
        "deployed_holdout_acc": float(getattr(pipe.benchmarks, "super_acc", float("nan"))),
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved bundle -> {os.path.basename(BUNDLE_PATH)} ({n} rows x {cap['X_full'].shape[1]} cols)")
    return bundle


if os.path.exists(BUNDLE_PATH):
    banner(f"Loading cached bundle {os.path.basename(BUNDLE_PATH)} (skipping train())")
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
else:
    bundle = build_bundle()

X_full = bundle["X_full"]
y = bundle["y"].astype(int)
gender = bundle["gender"]
feat_240 = bundle["feat_cols_240"]

# ---- winner-eligible (374) via the model's own routing rule ----
winner_elig = [c for c in X_full.columns
               if ufc._feature_allowed(c, "winner") and pd.api.types.is_numeric_dtype(X_full[c])]
retained = [c for c in feat_240 if c in winner_elig]
dropped = [c for c in winner_elig if c not in set(retained)]
print(f"\nWinner-eligible: {len(winner_elig)}  |  retained (model): {len(retained)}  |  "
      f"dropped by prune: {len(dropped)}")

# ---- chronological split: last TEST_FIGHTS = holdout ----
banner("Splitting + imputing feature matrix")
n = len(X_full)
tr_idx = np.arange(0, n - TEST_FIGHTS)
ho_idx = np.arange(n - TEST_FIGHTS, n)

Xwe = X_full[winner_elig].apply(pd.to_numeric, errors="coerce")
imp = SimpleImputer(strategy="median").fit(Xwe.iloc[tr_idx])
Xtr = pd.DataFrame(imp.transform(Xwe.iloc[tr_idx]), columns=winner_elig)
Xho = pd.DataFrame(imp.transform(Xwe.iloc[ho_idx]), columns=winner_elig)
ytr = y.iloc[tr_idx].reset_index(drop=True)
yho = y.iloc[ho_idx].reset_index(drop=True)
g_ho = gender.iloc[ho_idx].reset_index(drop=True)
men = g_ho.str.strip().str.lower().eq("men").values
print(f"Holdout: {len(yho)}  men: {int(men.sum())}  women: {int((~men).sum())}")


def make_boost():
    if getattr(ufc, "lgb", None) is not None:
        return ufc.lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.12, reg_lambda=1.2, n_jobs=-1, verbose=-1, random_state=SEED)
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, max_depth=6, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def fit_pair(Xtr_cols, ytr_):
    Xa, ya = ufc._augment_swap(Xtr_cols, ytr_)
    b = make_boost(); b.fit(Xa, ya)
    e = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
    e.fit(Xa, ya)
    return b, e


def proba(pair, Xdf):
    b, e = pair
    Xsw = ufc._swap_features(Xdf)
    def avg(m):
        return (m.predict_proba(Xdf)[:, 1] + (1.0 - m.predict_proba(Xsw)[:, 1])) / 2.0
    return np.clip(BOOST_WEIGHT * avg(b) + (1.0 - BOOST_WEIGHT) * avg(e), 1e-6, 1 - 1e-6)


def score(pair, Xdf, yv):
    p = proba(pair, Xdf)
    return log_loss(yv, p), accuracy_score(yv, (p >= 0.5).astype(int))


# ---- (A/B) does the prune cost men's accuracy? retained-240 vs full-374 ----
banner("Fitting surrogate on retained-240")
pair240 = fit_pair(Xtr[retained], ytr)
banner("Fitting surrogate on full-374")
pair374 = fit_pair(Xtr[winner_elig], ytr)
banner("Surrogates fitted; running sanity + A/B")

# sanity vs deployed model on full holdout
_, acc_full = score(pair374, Xho[winner_elig], yho)
print(f"Surrogate(374) full-holdout acc: {acc_full:.1%}  "
      f"(deployed reported {bundle.get('deployed_holdout_acc', float('nan')):.1%})")
if not (0.55 <= acc_full <= 0.72):
    print("  WARNING: off expected band — reconstruction may have drifted.")

Xho_men = Xho.loc[men].reset_index(drop=True)
ym = yho.loc[men].reset_index(drop=True)
ll240, acc240 = score(pair240, Xho_men[retained], ym)
ll374, acc374 = score(pair374, Xho_men[winner_elig], ym)
print("\n================ A/B: PRUNE COST ON MEN'S HOLDOUT ================")
print(f"  retained 240 :  log-loss {ll240:.4f}  |  acc {acc240:.1%}")
print(f"  full      374 :  log-loss {ll374:.4f}  |  acc {acc374:.1%}")
print(f"  delta (374-240): log-loss {ll374 - ll240:+.4f}  |  acc {acc374 - acc240:+.1%}")
print("  (374 better => prune is leaving men's signal on the table; "
      "374 worse => prune's variance reduction wins overall)")

# ---- permutation importance on the full-374 surrogate (men's holdout) ----
banner(f"Permutation importance on full-374 surrogate ({N_PERM_REPEATS} reps/feature)")
base_ll, base_acc = ll374, acc374
rng = np.random.default_rng(SEED)
rows = []
_tp = time.time()
for j, feat in enumerate(winner_elig):
    col = Xho_men[feat].values
    d_ll = []
    for _ in range(N_PERM_REPEATS):
        Xp = Xho_men.copy()
        Xp[feat] = rng.permutation(col)
        d_ll.append(log_loss(ym, proba(pair374, Xp[winner_elig])) - base_ll)
    rows.append({"feature": feat, "perm_ll_increase": float(np.mean(d_ll)),
                 "status": "retained" if feat in set(retained) else "dropped"})
    if (j + 1) % 25 == 0 or (j + 1) == len(winner_elig):
        el = time.time() - _tp
        rate = (j + 1) / el if el > 0 else 0.0
        eta = (len(winner_elig) - (j + 1)) / rate if rate > 0 else 0.0
        print(f"      perm {j + 1}/{len(winner_elig)} ({(j + 1) / len(winner_elig):3.0%})"
              f"  elapsed {el:0.0f}s  eta {eta:0.0f}s", flush=True)
res = pd.DataFrame(rows)

banner("Permutation done; building correlation map + writing results")
# corr of each dropped feature with its best retained match (redundant vs unique)
corr_tr = Xtr.corr().abs()
def best_match(f):
    if f in retained:
        return (np.nan, "")
    s = corr_tr.loc[f, retained].drop(labels=[f], errors="ignore")
    return (float(s.max()), str(s.idxmax())) if len(s) else (np.nan, "")
res[["max_corr_retained", "best_retained_match"]] = res["feature"].apply(
    lambda f: pd.Series(best_match(f)))

if os.path.exists(STAGE1_CSV):
    s1 = pd.read_csv(STAGE1_CSV)[["feature", "w_consensus"]]
    res = res.merge(s1, on="feature", how="left")
else:
    res["w_consensus"] = np.nan
res.to_csv(OUT_CSV, index=False)

pd.set_option("display.width", 240)
_f = {c: "{:.4f}".format for c in ["perm_ll_increase", "max_corr_retained", "w_consensus"]}

dropped_df = res[res["status"] == "dropped"].sort_values("perm_ll_increase", ascending=False)
print("\n===== (B) DROPPED FEATURES BY HELD-OUT MEN'S IMPORTANCE (add-back candidates) =====")
print("(high perm_ll_increase + LOW max_corr_retained = prune lost real, non-redundant signal)")
print(dropped_df.head(20).to_string(index=False, formatters=_f))

prune_df = res[(res["status"] == "retained") & (res["perm_ll_increase"] <= 0.0005)] \
    .sort_values("w_consensus")
print(f"\n===== (A) RETAINED FEATURES WITH ~NO HELD-OUT MEN'S SIGNAL ({len(prune_df)}) =====")
print("(prune candidates: shuffling them barely moves men's log-loss)")
print(prune_df.head(30).to_string(index=False, formatters=_f))

print(f"\nFull table -> {os.path.basename(OUT_CSV)}.  Delete {os.path.basename(BUNDLE_PATH)} to rebuild.")
banner("Stage 2 complete")
