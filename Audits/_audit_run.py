"""Headless driver: run the Sadykhov vs Camilo prediction and dump ensemble
feature importances. Throwaway diagnostic harness (not part of the model)."""
import sys, os, io, contextlib, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import importlib.util

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of Audits/)
P = os.path.join(HERE, "UFC_Model.py")
spec = importlib.util.spec_from_file_location("ufc_model", P)
ufc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ufc)

pipe = ufc.UFCSuperModelPipeline(ufc.DATA_PATH)

# Capture the exact augmented feature dict fed to the winner ensemble.
captured = {}
_orig = ufc.SuperEnsembleModel.predict_proba_single
def _patched(self, feat_dict):
    captured["feat"] = dict(feat_dict)
    return _orig(self, feat_dict)
ufc.SuperEnsembleModel.predict_proba_single = _patched

# Swallow the verbose training log; keep only our output.
logbuf = io.StringIO()
try:
    with contextlib.redirect_stdout(logbuf):
        pipe.train()
except Exception as e:
    tail = logbuf.getvalue()[-2000:]
    print("TRAIN FAILED:", repr(e))
    print(tail)
    raise

res = pipe.predict_matchup("Nazim Sadykhov", "Matheus Camilo",
                           weight_class="Lightweight", gender="Men", rounds=3)

print("================ WINNER / METHOD PREDICTION ================")
print(f"name_a (red): {res['name_a']}   name_b (blue): {res['name_b']}")
print(f"prob_a: {res['prob_a']:.4f}   prob_b: {res['prob_b']:.4f}")
print(f"glicko_mu_a: {res['rating_a']:.1f}   glicko_mu_b: {res['rating_b']:.1f}")
print(f"predicted_method: {res['predicted_method']}")
print("method_probs:", {k: round(v, 4) for k, v in res["method_probs"].items()})

model = pipe.model
feat_cols = list(model.feat_cols)
n = len(feat_cols)
print(f"\nWinner-stage feature count: {n}")
print("Base models in ensemble:", list(model.models.keys()))

# ---- Per-model feature importances, rank-normalised to [0,1] percentile ----
pct_by_model = {}
for name, m in model.models.items():
    fi = getattr(m, "feature_importances_", None)
    if fi is None:
        coef = getattr(m, "coef_", None)
        fi = np.abs(np.ravel(coef)) if coef is not None else None
    if fi is None:
        print(f"  (no importance available: {name})")
        continue
    fi = np.asarray(fi, dtype=float)
    if fi.shape[0] != n:
        print(f"  (importance length {fi.shape[0]} != {n}: {name} -- skipped)")
        continue
    ranks = fi.argsort().argsort().astype(float)  # 0 = least important
    pct = ranks / (n - 1)
    pct_by_model[name] = pct

models_used = list(pct_by_model.keys())
print(f"Models contributing importances ({len(models_used)}):", models_used)
if not models_used:
    raise SystemExit("No base model exposed feature_importances_ — cannot aggregate.")

# ---- Combiner weights: how much each base model actually counts in the final
# ensemble prediction. The audit weights each model's importances by this so a
# heavily-weighted model's view dominates the consensus (matches the model). ----
combiner = model.combiner
print(f"\nCombiner kind: {combiner.get('kind')}")
if combiner.get("kind") == "weighted":
    raw_w = {nm: float(combiner["weights"].get(nm, 0.0)) for nm in model.model_order}
else:
    # stacker/other: no explicit per-model weights -> fall back to equal.
    raw_w = {nm: 1.0 for nm in model.model_order}
    print("  (non-weighted combiner: falling back to EQUAL model weights)")

print("Combiner weights (full ensemble, incl. models w/o importances):")
for nm in sorted(raw_w, key=lambda k: -raw_w[k]):
    flag = "" if nm in pct_by_model else "   <- no native importance (blind spot)"
    print(f"  {nm:<16} {raw_w[nm]:.4f}{flag}")

# Blind-spot: combiner weight sitting on models we can't read natively (e.g. HistGBM).
tot_w = sum(raw_w.values()) or 1.0
blind_w = sum(w for nm, w in raw_w.items() if nm not in pct_by_model)
print(f"\nShare of ensemble weight on models WITHOUT native importances: "
      f"{blind_w / tot_w:.1%}  (permutation importance needed to cover these)")

# Weights restricted+renormalised to models that exposed importances.
w_used = np.array([raw_w.get(nm, 0.0) for nm in models_used], dtype=float)
if w_used.sum() <= 0:
    w_used = np.ones(len(models_used))  # all-zero weight on readable models -> equal
w_used = w_used / w_used.sum()

stack = np.vstack([pct_by_model[k] for k in models_used])  # (n_models, n_feats)
equal_pct = stack.mean(axis=0)                              # naive equal-weight view
weighted_pct = (w_used[:, None] * stack).sum(axis=0)        # combiner-weighted view
min_pct = stack.min(axis=0)
max_pct = stack.max(axis=0)
std_pct = stack.std(axis=0)                                 # cross-model disagreement

imp_df = pd.DataFrame({
    "feature": feat_cols,
    "w_consensus": weighted_pct,   # primary: weighted by combiner importance
    "equal_mean": equal_pct,       # unweighted, for comparison
    "min_pct": min_pct,
    "max_pct": max_pct,
    "cross_model_std": std_pct,    # high std = models disagree on this feature
}).sort_values("w_consensus", ascending=False).reset_index(drop=True)

feat_vec = captured.get("feat", {})
imp_df["matchup_value"] = pd.to_numeric(
    imp_df["feature"].map(lambda f: feat_vec.get(f, np.nan)), errors="coerce")

pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 40)
_fmt = {c: "{:.3f}".format for c in
        ["w_consensus", "equal_mean", "min_pct", "max_pct", "cross_model_std", "matchup_value"]}

print("\n================ TOP 30 (COMBINER-WEIGHTED CONSENSUS) ================")
print(imp_df.head(30).to_string(index=False, formatters=_fmt))

print("\n================ BOTTOM 30 (LOW-IMPORTANCE PRUNE CANDIDATES) ================")
print("(low across models AND low weighted consensus; confirm w/ permutation pass)")
print(imp_df.tail(30).to_string(index=False, formatters=_fmt))

print("\n========== MOST DISAGREED-ON FEATURES (high cross-model std) ==========")
print("(importance is model-dependent here — exactly the case your Q1 asks about)")
print(imp_df.sort_values("cross_model_std", ascending=False).head(20)
      .to_string(index=False, formatters=_fmt))

# Save full table for the audit.
imp_df.to_csv(os.path.join(HERE, "_audit_importances.csv"), index=False)
print(f"\nFull importance table saved to _audit_importances.csv ({len(imp_df)} rows)")
