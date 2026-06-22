"""Method-stage audit, pass A (cheap): load the trained method bundle from cache
(no retrain) and report:
  1) BLEND COMPOSITION — how much of the final KO/Sub/Dec prob comes from the
     trained ML models vs. hard-coded history/group/sub priors.
  2) STAGE-1 importances (Finish vs Decision) and STAGE-2 importances (KO vs Sub).
  3) ROUTING read — are the method-only-routed features actually high-importance,
     or is the method stage leaning on features it shares with the winner model?

HGB stages have no native importances, so we read the RF/ET twins in the bundle.
Run: python _audit_method.py
"""
import os, glob, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util

HERE = r"c:\Users\gdsak\OneDrive\Desktop\Glicko-2, Etc"
CACHE = os.path.join(HERE, ".ufc_model_cache")

# import the module so pickled estimators + _feature_allowed are available
spec = importlib.util.spec_from_file_location("ufc_model", os.path.join(HERE, "UFC_Model.py"))
ufc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ufc)

pkls = sorted(glob.glob(os.path.join(CACHE, "method_stage_*.pkl")), key=os.path.getmtime)
if not pkls:
    raise SystemExit("No method_stage cache found — run UFC_Model.py --train-only first.")
print(f"Loading {os.path.basename(pkls[-1])} ...")
with open(pkls[-1], "rb") as f:
    payload = pickle.load(f)
mb = payload.get("method_bundle", payload)
if not isinstance(mb, dict):
    raise SystemExit("Unexpected method bundle structure.")

method_cols = list(mb.get("method_columns", []))
s1_cols = list(mb.get("stage1_cols") or method_cols)
s2_cols = list(mb.get("stage2_cols") or method_cols)
print(f"method feature columns: {len(method_cols)} | stage1: {len(s1_cols)} | stage2: {len(s2_cols)}")

# ---------- 1) BLEND COMPOSITION ----------
w_hist = float(mb.get("w_hist", 0.25)); w_group = float(mb.get("w_group", 0.10))
w_base = float(mb.get("w_base", 0.0)); w_subsig = float(mb.get("w_subsig", 0.10))
w_ml = max(0.0, 1.0 - w_hist - w_group - w_base - w_subsig)
print("\n================ FINAL-PROB BLEND COMPOSITION ================")
print(f"  ML models (stage1/2 + direct + ovr + simple) : w_ml     = {w_ml:.3f}")
print(f"  History prior (winner/loser career methods)  : w_hist   = {w_hist:.3f}")
print(f"  Group prior (weight-class x gender base rate): w_group  = {w_group:.3f}")
print(f"  Base prior                                   : w_base   = {w_base:.3f}")
print(f"  Sub-attempt signal prior                     : w_subsig = {w_subsig:.3f}")
print("  (if w_ml is low, the trained models barely drive the pick — it's mostly priors)")
print("\n  Internal ML mix / shaping:")
for k in ("alpha_stage1", "alpha_stage2", "alpha_direct", "beta_direct", "alpha_ovr",
          "alpha_simple", "temp_decision", "temp_finish", "finish_threshold",
          "sub_threshold", "meta_eta", "sub_boost_k",
          "method_bias_decision", "method_bias_ko_tko", "method_bias_submission"):
    if k in mb:
        print(f"    {k:<20} = {float(mb[k]):.4f}")


def _route(feat):
    # shared with winner (or unrouted default) vs method-only
    return "shared" if ufc._feature_allowed(feat, "winner") else "METHOD-ONLY"


def importance_table(cols, *models, label=""):
    mats = []
    for m in models:
        if m is None:
            continue
        fi = getattr(m, "feature_importances_", None)
        if fi is None:
            continue
        fi = np.asarray(fi, dtype=float)
        if fi.shape[0] != len(cols):
            print(f"  ({label}: importance len {fi.shape[0]} != cols {len(cols)} — skipped a model)")
            continue
        mats.append(fi.argsort().argsort() / max(1, len(cols) - 1))  # rank percentile
    if not mats:
        print(f"  ({label}: no usable RF/ET importances in bundle)")
        return None
    pct = np.vstack(mats).mean(axis=0)
    df = pd.DataFrame({"feature": cols, "imp_pct": pct,
                       "routing": [_route(c) for c in cols]})
    return df.sort_values("imp_pct", ascending=False).reset_index(drop=True)


pd.set_option("display.width", 200)
_fmt = {"imp_pct": "{:.3f}".format}

print("\n================ STAGE 1 importances — FINISH vs DECISION ================")
s1 = importance_table(s1_cols, mb.get("stage1_rf"), mb.get("stage1_et"), label="stage1")
if s1 is not None:
    print(s1.head(25).to_string(index=False, formatters=_fmt))

print("\n================ STAGE 2 importances — KO/TKO vs SUBMISSION (the weak link) ================")
s2 = importance_table(s2_cols, mb.get("stage2_rf"), mb.get("stage2_et"), label="stage2")
if s2 is not None:
    print(s2.head(25).to_string(index=False, formatters=_fmt))

# ---------- 3) ROUTING READ ----------
print("\n================ ROUTING: are method-only features pulling weight? ================")
for name, df in (("stage1 (finish-vs-dec)", s1), ("stage2 (ko-vs-sub)", s2)):
    if df is None:
        continue
    n_mo = int((df["routing"] == "METHOD-ONLY").sum())
    top20_mo = int((df.head(20)["routing"] == "METHOD-ONLY").sum())
    mean_pct_mo = df.loc[df["routing"] == "METHOD-ONLY", "imp_pct"].mean()
    mean_pct_sh = df.loc[df["routing"] == "shared", "imp_pct"].mean()
    print(f"  {name}: {n_mo}/{len(df)} feats are method-only | "
          f"{top20_mo}/20 of the top-20 are method-only | "
          f"mean imp_pct: method-only {mean_pct_mo:.3f} vs shared {mean_pct_sh:.3f}")
print("  (method-only features ranking LOW => the method routing is withholding noise,")
print("   not signal; if they rank HIGH, the routing is earning its keep.)")

if s1 is not None:
    s1.to_csv(os.path.join(HERE, "_audit_method_stage1.csv"), index=False)
if s2 is not None:
    s2.to_csv(os.path.join(HERE, "_audit_method_stage2.csv"), index=False)
print("\nFull ranked tables saved: _audit_method_stage1.csv (finish-vs-dec), "
      "_audit_method_stage2.csv (ko-vs-sub).")

print("\nNEXT: the Submission class is the weak one (F1 ~32%). Stage-2 importances above")
print("show what KO-vs-Sub leans on; if those are weak/low-signal features, that's the")
print("bottleneck. Pass B (reconstruct + permutation/routing test) only if this warrants it.")
