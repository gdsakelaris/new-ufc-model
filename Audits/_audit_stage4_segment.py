"""Stage 4: segment-bias check — does the model systematically OVER-pick the
younger / less-experienced / lower-rated fighter (the 'Camilo pattern')?

Method: train a clean surrogate on the first N fights, predict the rest
out-of-sample, orient every fight by the YOUNGER fighter, isolate the segment
where youth is the young fighter's only edge (younger AND fewer fights AND lower
Glicko), and compare the model's confidence in the young fighter to how often
they actually win. A positive calibration gap = real over-pick bias to fix; a
gap near zero = Camilo was just variance, leave the model alone.

Uses the cached Stage-2 bundle. Run: python _audit_stage4_segment.py
"""
import os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
BUNDLE_PATH = os.path.join(HERE, "_audit_stage2_bundle_v2.pkl")
BOOST_WEIGHT = 0.8
SEEDS = [0, 1, 2]
TRAIN_N = 5000            # train on first 5000 fights, test on the rest (~4100)

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
for col in ("d_age", "d_num_fights", "d_glicko_mu"):
    if col not in X_full.columns:
        raise SystemExit(f"Missing column {col} in bundle.")

retained = [c for c in feat_240 if c in X_full.columns and pd.api.types.is_numeric_dtype(X_full[c])]
n = len(X_full)
tr = np.arange(0, TRAIN_N)
te = np.arange(TRAIN_N, n)

Xr = X_full[retained].apply(pd.to_numeric, errors="coerce")
imp = SimpleImputer(strategy="median").fit(Xr.iloc[tr])
Xtr = pd.DataFrame(imp.transform(Xr.iloc[tr]), columns=retained)
Xte = pd.DataFrame(imp.transform(Xr.iloc[te]), columns=retained)
ytr = y.iloc[tr].reset_index(drop=True)
yte = y.iloc[te].reset_index(drop=True).values

def make_boost(seed):
    if getattr(ufc, "lgb", None) is not None:
        return ufc.lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.12, reg_lambda=1.2, n_jobs=-1, verbose=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, max_depth=6, random_state=seed)

def proba_red(Xdf):
    """Swap-averaged P(red win), averaged over seeds."""
    Xsw = ufc._swap_features(Xdf)
    acc = np.zeros(len(Xdf))
    for s in SEEDS:
        Xa, ya = ufc._augment_swap(Xtr, ytr)
        b = make_boost(s); b.fit(Xa, ya)
        e = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=s); e.fit(Xa, ya)
        def avg(m): return (m.predict_proba(Xdf)[:, 1] + (1.0 - m.predict_proba(Xsw)[:, 1])) / 2.0
        acc += (BOOST_WEIGHT * avg(b) + (1 - BOOST_WEIGHT) * avg(e)) / len(SEEDS)
    return np.clip(acc, 1e-6, 1 - 1e-6)

banner(f"Fitting surrogate on first {TRAIN_N}, predicting remaining {len(te)} (out-of-sample)")
p_red = proba_red(Xte)
acc_overall = np.mean((p_red >= 0.5).astype(int) == yte)
cal_overall = np.mean(p_red) - np.mean(yte)   # >0 = over-predicts RED overall
print(f"Test fights: {len(te)} | overall acc {acc_overall:.1%} | "
      f"overall calibration gap (E[P_red]-win_rate) {cal_overall:+.3f}")

# ---- orient every fight by the YOUNGER fighter ----
d_age = X_full["d_age"].iloc[te].to_numpy()            # age_red - age_blue
d_fights = X_full["d_num_fights"].iloc[te].to_numpy()  # fights_red - fights_blue
d_glk = X_full["d_glicko_mu"].iloc[te].to_numpy()      # glicko_red - glicko_blue
g_te = gender.iloc[te].str.strip().str.lower().to_numpy()

red_older = d_age > 0
p_young = np.where(red_older, 1.0 - p_red, p_red)                 # model's P(younger wins)
young_won = np.where(red_older, 1 - yte, yte)                     # did younger win
young_minus_vet_fights = np.where(red_older, -d_fights, d_fights) # younger fights - vet fights
young_minus_vet_glk = np.where(red_older, -d_glk, d_glk)          # younger glicko - vet glicko
age_gap = np.abs(d_age)
is_men = g_te == "men"


def report(mask, label):
    m = mask
    if m.sum() < 20:
        print(f"  {label:<34} n={int(m.sum()):>4}  (too few — skip)")
        return
    pick_young = np.mean(p_young[m] >= 0.5)      # how often model picks the younger fighter
    win_young = np.mean(young_won[m])            # how often younger actually wins
    mean_conf = np.mean(p_young[m])              # model's mean confidence in younger
    cal_gap = mean_conf - win_young              # >0 = model OVER-rates younger (the bias)
    seg_acc = np.mean((p_young[m] >= 0.5).astype(int) == young_won[m])
    print(f"  {label:<34} n={int(m.sum()):>4} | model picks young {pick_young:5.1%} | "
          f"young actually wins {win_young:5.1%} | E[P_young] {mean_conf:5.1%} | "
          f"cal_gap {cal_gap:+.3f} | seg_acc {seg_acc:5.1%}")


# Camilo pattern: notable age gap, AND younger is the lower-experience AND lower-rated one.
camilo = (young_minus_vet_fights < 0) & (young_minus_vet_glk < 0)
banner("SEGMENT: younger fighter is ALSO less-experienced AND lower-rated ('Camilo pattern')")
print("  (cal_gap > 0  =>  model systematically OVER-picks the young/thin fighter)\n")
print("  ALL FIGHTS:")
for thr in (3, 5, 7):
    report(camilo & (age_gap >= thr), f"age_gap>={thr}")
print("  MEN ONLY:")
for thr in (3, 5, 7):
    report(camilo & (age_gap >= thr) & is_men, f"age_gap>={thr}, men")

# Sharpest Camilo analog: young (vet much older) + very thin record gap
sharp = camilo & (age_gap >= 5) & (young_minus_vet_fights <= -3)
print("\n  SHARPEST ANALOG (age_gap>=5 & younger has >=3 fewer fights):")
report(sharp, "all")
report(sharp & is_men, "men")

# ---- calibration table inside the main segment (age_gap>=5) ----
banner("Calibration inside segment (age_gap>=5, Camilo pattern): predicted vs actual for YOUNGER")
seg = camilo & (age_gap >= 5)
pv, wv = p_young[seg], young_won[seg]
bins = [0.0, 0.35, 0.45, 0.55, 0.65, 1.01]
print(f"  segment n={int(seg.sum())}")
print(f"  {'P_young bin':<14}{'n':>5}{'mean_pred':>11}{'actual_win':>12}{'gap':>8}")
for lo, hi in zip(bins[:-1], bins[1:]):
    b = (pv >= lo) & (pv < hi)
    if b.sum() == 0:
        continue
    print(f"  [{lo:.2f},{hi:.2f}){'':<4}{int(b.sum()):>5}{np.mean(pv[b]):>11.1%}"
          f"{np.mean(wv[b]):>12.1%}{np.mean(pv[b]) - np.mean(wv[b]):>+8.3f}")

print("\nINTERPRET: if cal_gap is consistently positive (e.g. > +0.04) in the segment but ~0")
print("overall, the model over-values youth vs proven record => real, fixable bias.")
print("If cal_gap ~ 0, the model is well-calibrated there and Camilo was just variance.")
banner("Stage 4 complete")
