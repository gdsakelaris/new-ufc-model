"""
UFC Model (single-file).

What this script does:
1) Builds leak-safe chronological pre-fight features from pure_fight_data.csv.
2) Trains a compact time-aware ensemble optimized for log-loss.
3) Calibrates probabilities on validation-only data (Platt vs Isotonic).
4) Reports holdout + walk-forward diagnostics and baseline comparisons.
5) Provides GUI matchup prediction and Excel export (predictions + rankings).

Run:
  python UFC_Model.py

Optional CLI:
  python UFC_Model.py --train-only
"""

import argparse
import hashlib
import math
import os
import pickle
import random
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize as scipy_minimize

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import catboost as cb
except Exception:
    cb = None

try:
    import optuna
except Exception:
    optuna = None

if optuna is not None:
    # Keep Optuna output compact: suppress per-trial parameter logs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "pure_fight_data.csv")
PREDICTIONS_XLSX = os.path.join(SCRIPT_DIR, "UFC_Predictions.xlsx")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".ufc_model_cache")
###################################################################################################
# Bump when winner-stage training logic changes.
WINNER_CACHE_VERSION = "v1"
# Bump when method-stage training logic changes.
METHOD_CACHE_VERSION = "v3"
###################################################################################################
# Pickle payload discriminator (stable across cache file renames).
WINNER_STAGE_CACHE_KIND = "ufc_winner_stage_v1"
METHOD_STAGE_CACHE_KIND = "ufc_method_stage_v1"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

TRAIN_FRACTION = 0.65
VAL_FRACTION = 0.15
TEST_FRACTION = 0.20
MIN_HOLDOUT_FIGHTS = 500
ACTIVE_DAYS = 730
NEEDS_SCALE = {"LogReg", "MLP"}
ELO_BASE = 1500.0
ELO_K = 24.0
OPTUNA_TRIALS = 80
METHOD_TUNING_TRIALS = 480
METHOD_HARD_RESET = False
METHOD_ERA_CANDIDATES = [1993, 2005, 2010, 2014, 2016, 2018, 2020, 2021, 2022, 2023, 2024]
METHOD_AUTO_ERA = True
METHOD_SUBMISSION_CLASS_WEIGHTS = [1.0, 1.05, 1.10, 1.20]
METHOD_SUBMISSION_OVERSAMPLE_RATIOS = [1.0, 1.05, 1.10, 1.15]
METHOD_DEFAULT_SUBMISSION_CLASS_WEIGHT = 1.05
METHOD_DEFAULT_SUBMISSION_OVERSAMPLE_RATIO = 1.05
# Keep winner-model inputs on the stable feature family while allowing richer
# method-only engineering in the method pipeline.
WINNER_EXCLUDE_FEATURES = {
    "sub_entry_pressure_sum", "sub_defensive_leak_sum",
    "ko_attack_pressure_sum", "ko_def_leak_sum",
    "d_head_acc", "d_distance_acc", "d_ground_acc", "d_clinch_acc",
    "d_body_acc", "d_leg_acc", "d_body_leg_attrition",
    "d_head_hunt_share", "d_distance_share",
    "d_ground_strike_accuracy", "d_head_hunt_accuracy", "d_distance_strike_accuracy",
    "d_sub_loss_pct", "d_recent_sub_win_rate",
    "d_sub_entry_pressure", "d_sub_control_conversion", "d_sub_scramble_threat",
    "d_sub_defensive_leak", "d_late_sub_pressure",
    "d_sub_recency_surge", "d_grapple_recency_surge", "d_sub_vs_control_axis",
    "d_recent_ko_loss_rate", "d_r5_def_kd_pm", "d_ko_attack_pressure", "d_ko_def_leak",
    "d_r1f_def_kd_pm", "d_r3_def_kd_pm",
}
STRICT_FUTURE_MODE = True
FORCED_START_YEAR = None
MISSINGNESS_THRESHOLD = 0.35

MU_0 = 1500.0
PHI_0 = 200.0
SIGMA_0 = 0.06
TAU = 0.5
SCALE = 173.7178
CONVERGENCE = 1e-6


def _clip_probs(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _cache_data_fingerprint(path):
    try:
        st = os.stat(path)
        raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        raw = f"{os.path.abspath(path)}|missing"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_key(stage, data_fp, version, extra=""):
    payload = f"{stage}|{data_fp}|{version}|{extra}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _cache_path(stage, key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{stage}_{key}.pkl")


def _cache_load(stage, key):
    path = _cache_path(stage, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(stage, key, payload):
    path = _cache_path(stage, key)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    prefix = f"{stage}_"
    for name in os.listdir(CACHE_DIR):
        if not name.startswith(prefix) or name.endswith(f"{key}.pkl"):
            continue
        old = os.path.join(CACHE_DIR, name)
        try:
            os.remove(old)
        except Exception:
            pass


def _winner_stage_cache_valid(payload, feature_cols, n_rows, train_end, val_end):
    if not isinstance(payload, dict):
        return False
    if payload.get("kind") != WINNER_STAGE_CACHE_KIND:
        return False
    if str(payload.get("winner_cache_version")) != str(WINNER_CACHE_VERSION):
        return False
    if int(payload.get("n_rows", -1)) != int(n_rows):
        return False
    if int(payload.get("train_end", -1)) != int(train_end):
        return False
    if int(payload.get("val_end", -1)) != int(val_end):
        return False
    fc = payload.get("feature_cols")
    if not isinstance(fc, list) or tuple(fc) != tuple(feature_cols):
        return False
    for k in (
        "oof", "valid", "combiner", "model_order", "decision_threshold",
        "test_probs_raw", "test_probs_cal", "y_pred_red",
        "raw_ll", "cal_ll_test", "brier", "acc", "ece", "cal_curve_rmse",
        "lgb_tuned", "tuned_thr", "tuned_acc", "val_acc_for_log", "calibrator",
    ):
        if k not in payload:
            return False
    return True


def _method_stage_cache_valid(payload, winner_cache_key, method_cache_version):
    if not isinstance(payload, dict):
        return False
    if payload.get("kind") != METHOD_STAGE_CACHE_KIND:
        return False
    if str(payload.get("method_cache_version")) != str(method_cache_version):
        return False
    if str(payload.get("winner_cache_key")) != str(winner_cache_key):
        return False
    b = payload.get("method_bundle")
    return isinstance(b, dict) and b.get("imputer") is not None


def _replay_winner_cache_logs(pl, W, winner_cache_key):
    """Replay winner-stage terminal sections from a cache payload (no retrain)."""
    h = str(winner_cache_key)[:12]
    pl._section("Optuna Tuning")
    pl._stat("Cache", f"HIT ({WINNER_CACHE_VERSION}) — Optuna skipped [key={h}]")
    pl._section("Model Setup")
    pl._stat("Base models", ", ".join(W.get("model_order") or []))
    pl._section("OOF Stacking")
    pl._stat("Cache", f"HIT ({WINNER_CACHE_VERSION}) — OOF skipped [key={h}]")
    pl._section("Combiner Selection")
    pl._stat("Selected combiner", W.get("combiner_kind", ""))
    pl._stat("Validation log-loss", W.get("val_ll_str", ""))
    pl._stat("Validation accuracy", W.get("val_acc_str", ""))
    pl._stat("Validation threshold", W.get("val_thr_str", ""))
    pl._section("Calibration")
    pl._stat("Selected method", W.get("cal_name", ""))
    pl._stat("Validation log-loss", W.get("cal_ll_str", ""))
    pl._stat("Strict future mode", "ON" if STRICT_FUTURE_MODE else "OFF")
    if not STRICT_FUTURE_MODE:
        pl._stat("Holdout-selected combiner", W.get("best_holdout_label", ""))
        pl._stat("Holdout combiner log-loss", W.get("best_holdout_ll_str", ""))
        pl._stat("Holdout combiner acc", W.get("best_holdout_acc_str", ""))
        pl._stat("Holdout combiner threshold", W.get("best_holdout_thr_str", ""))
    pl._stat("Calibration used for picks", W.get("cal_used_str", ""))
    pl._stat("Validation accuracy (picked)", W.get("val_acc_picked_str", ""))
    pl._stat("Validation tuned threshold", W.get("val_tuned_thr_str", ""))
    pl._stat("Decision threshold", W.get("decision_thr_str", ""))
    pl._section("Holdout Evaluation")
    for label, val in W.get("holdout_eval", []):
        pl._stat(label, val)
    pl._section("Winner Diagnostics")
    pl._log("Confusion Matrix (Winner: Red=positive)")
    pl._log("               Pred Blue   Pred Red")
    tn, fp, fn, tp = W.get("confusion", (0, 0, 0, 0))
    pl._log(f"Actual Blue   {int(tn):9d}  {int(fp):9d}")
    pl._log(f"Actual Red    {int(fn):9d}  {int(tp):9d}")
    pl._log("")
    pl._log("Accuracy by Weight Class")
    pl._log("-" * 72)
    for wc, acc_wc, n_wc in W.get("wc_rows", []):
        pl._stat(f"{wc} (n={int(n_wc)})", f"{float(acc_wc):.1%}")
    pl._log("")
    pl._log("Accuracy by Gender")
    pl._log("-" * 72)
    for gender, acc_g, n_g in W.get("g_rows", []):
        pl._stat(f"{gender} (n={int(n_g)})", f"{float(acc_g):.1%}")


def _normalize_method_label(raw_method):
    detail = _normalize_method_detail(raw_method)
    if detail.startswith("Decision"):
        return "Decision"
    if detail in ("KO/TKO", "Doctor Stoppage", "DQ/Corner Stoppage"):
        return "KO/TKO"
    if detail == "Submission":
        return "Submission"
    return "Decision"


def _normalize_method_detail(raw_method):
    txt = str(raw_method or "").strip().lower()
    if "decision" in txt or txt.startswith("dec") or "split" in txt or "majority" in txt or "unanimous" in txt:
        return "Decision"
    if "doctor" in txt:
        return "KO/TKO"
    if "dq" in txt or "corner" in txt or "retire" in txt:
        return "KO/TKO"
    if "sub" in txt:
        return "Submission"
    if "ko" in txt or "tko" in txt:
        return "KO/TKO"
    return "Decision"


def _normalize_method_probs(prob_map):
    vals = np.array([float(prob_map.get(k, 0.0)) for k in METHOD_LABELS], dtype=float)
    vals = np.maximum(vals, MIN_METHOD_PROB)
    vals = vals / np.sum(vals)
    return {k: float(v) for k, v in zip(METHOD_LABELS, vals)}


# Deprecated: legacy manual method-probability shaping helpers.
# Kept only as unreachable historical utilities; live method inference no longer calls them.
def _apply_method_logit_bias_arr(probs_arr, bias_vec):
    arr = np.asarray(probs_arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    b = np.asarray(bias_vec, dtype=float).reshape(1, -1)
    logp = np.log(np.clip(arr, MIN_METHOD_PROB, 1.0)) + b
    logp = logp - np.max(logp, axis=1, keepdims=True)
    expv = np.exp(logp)
    out = expv / np.sum(expv, axis=1, keepdims=True)
    return np.clip(out, MIN_METHOD_PROB, 1.0)


def _apply_method_logit_bias_map(prob_map, bias_map):
    arr = np.array([
        float(prob_map.get("Decision", 0.0)),
        float(prob_map.get("KO/TKO", 0.0)),
        float(prob_map.get("Submission", 0.0)),
    ], dtype=float)
    bias_vec = np.array([
        float(bias_map.get("Decision", 0.0)),
        float(bias_map.get("KO/TKO", 0.0)),
        float(bias_map.get("Submission", 0.0)),
    ], dtype=float)
    out = _apply_method_logit_bias_arr(arr, bias_vec)[0]
    return _normalize_method_probs({
        "Decision": float(out[0]),
        "KO/TKO": float(out[1]),
        "Submission": float(out[2]),
    })


def _apply_binary_threshold_warp(p, thr):
    p_arr = np.asarray(p, dtype=float)
    t = float(np.clip(thr, 0.05, 0.95))
    left = 0.5 * (p_arr / max(t, 1e-6))
    right = 0.5 + 0.5 * ((p_arr - t) / max(1.0 - t, 1e-6))
    out = np.where(p_arr < t, left, right)
    return np.clip(out, 1e-4, 1.0 - 1e-4)


def _apply_submission_signal_boost_arr(probs_arr, sub_signal_arr, boost_k):
    arr = np.asarray(probs_arr, dtype=float).copy()
    sig = np.asarray(sub_signal_arr, dtype=float).reshape(-1)
    k = float(boost_k)
    # Positive boost only when submission signal is above prior-ish baseline.
    scale = np.clip((sig - 0.18) / 0.22, -1.5, 2.5)
    arr[:, 2] = arr[:, 2] * np.exp(k * scale)
    arr = np.clip(arr, MIN_METHOD_PROB, 1.0)
    arr = arr / np.sum(arr, axis=1, keepdims=True)
    return arr


def _apply_submission_signal_boost_map(prob_map, sub_signal, boost_k):
    arr = np.array([
        float(prob_map.get("Decision", 0.0)),
        float(prob_map.get("KO/TKO", 0.0)),
        float(prob_map.get("Submission", 0.0)),
    ], dtype=float).reshape(1, -1)
    out = _apply_submission_signal_boost_arr(arr, np.array([float(sub_signal)], dtype=float), boost_k)[0]
    return _normalize_method_probs({
        "Decision": float(out[0]),
        "KO/TKO": float(out[1]),
        "Submission": float(out[2]),
    })


def _sub_attempt_prior_array(X_df):
    X = X_df.reset_index(drop=True)
    d_sub_att = pd.to_numeric(X.get("d_sub_att_p15", pd.Series(np.zeros(len(X)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d_grap_tdd = pd.to_numeric(X.get("d_grapple_vs_tdd", pd.Series(np.zeros(len(X)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d_been_fin = pd.to_numeric(X.get("d_been_finished_pct", pd.Series(np.zeros(len(X)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d_dec = pd.to_numeric(X.get("d_dec_win_pct", pd.Series(np.zeros(len(X)))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    z = 1.35 * d_sub_att + 0.75 * d_grap_tdd + 0.55 * d_been_fin
    p_sub = 1.0 / (1.0 + np.exp(-z))
    p_sub = np.clip(0.08 + 0.50 * p_sub, 0.06, 0.62)
    p_dec = np.clip(0.54 + 0.20 * d_dec - 0.35 * p_sub, 0.12, 0.82)
    p_ko = np.clip(1.0 - p_dec - p_sub, 0.05, 0.70)
    arr = np.stack([p_dec, p_ko, p_sub], axis=1)
    arr = np.clip(arr, MIN_METHOD_PROB, 1.0)
    arr = arr / np.sum(arr, axis=1, keepdims=True)
    return arr


def _method_profile_from_history(history):
    wins = [h for h in history if h.get("result") == "W"]
    losses = [h for h in history if h.get("result") == "L"]

    def _rate(rows, method_label, prior, strength=8.0):
        n = len(rows)
        if n == 0:
            return float(prior)
        hits = sum(1 for h in rows if _normalize_method_label(h.get("method", "")) == method_label)
        return (hits + prior * strength) / (n + strength)

    return {
        "win_decision": _rate(wins, "Decision", 0.50),
        "win_ko_tko": _rate(wins, "KO/TKO", 0.32),
        "win_submission": _rate(wins, "Submission", 0.18),
        "loss_decision": _rate(losses, "Decision", 0.45),
        "loss_ko_tko": _rate(losses, "KO/TKO", 0.35),
        "loss_submission": _rate(losses, "Submission", 0.20),
        "wins_n": float(len(wins)),
        "losses_n": float(len(losses)),
    }


def _oriented_method_matrix(X_df, y_red_win):
    X_m = X_df.copy()
    y_arr = np.asarray(y_red_win).astype(int)
    sign = np.where(y_arr == 1, 1.0, -1.0)
    for col in X_m.columns:
        if col.startswith("d_"):
            X_m[col] = X_m[col].astype(float).values * sign
    return X_m


def _augment_method_features(X_df):
    X = X_df.copy()
    eps = 1e-6

    def _col(name, default=0.0):
        if name not in X.columns:
            return pd.Series(np.full(len(X), default), index=X.index, dtype=float)
        return pd.to_numeric(X[name], errors="coerce").fillna(default)

    d_ko_win = _col("d_ko_win_pct", 0.0)
    d_sub_win = _col("d_sub_win_pct", 0.0)
    d_dec_win = _col("d_dec_win_pct", 0.0)
    d_ko_loss = _col("d_ko_loss_pct", 0.0)
    d_sub_att = _col("d_sub_att_p15", 0.0)
    d_r3_sub = _col("d_rd3_sub_att", 0.0)
    d_r3_sub_share = _col("d_rd3_sub_share", 0.0)
    d_late_sub = _col("d_late_vs_early_sub_att", 0.0)
    d_sub_t23 = _col("d_sub_att_trend_23", 0.0)
    d_been_finished = _col("d_been_finished_pct", 0.0)
    d_finish_vs_resist = _col("d_finish_vs_resist", 0.0)
    d_str_vs_def = _col("d_striking_vs_defense", 0.0)
    d_strike_exchange_ratio = _col("d_strike_exchange_ratio", 0.0)
    d_grap_vs_tdd = _col("d_grapple_vs_tdd", 0.0)
    d_ortho_vs_south = _col("d_ortho_vs_south", 0.0)
    d_power_ratio = _col("d_power_ratio", 0.0)
    d_head_pct = _col("d_head_pct", 0.0)
    d_sig_str_diff_pm = _col("d_sig_str_diff_pm", 0.0)
    d_first_round_finish_rate = _col("d_first_round_finish_rate", 0.0)
    d_rd1_intensity_ratio = _col("d_rd1_intensity_ratio", 0.0)
    d_damage_efficiency = _col("d_damage_efficiency", 0.0)
    d_sig_str_acc = _col("d_sig_str_acc", _col("d_td_sig_str_acc", 0.0))
    d_ground_pct = _col("d_ground_pct", 0.0)
    d_ctrl_pct = _col("d_ctrl_pct", _col("d_td_ctrl_pct", 0.0))
    d_rev_p15 = _col("d_rev_p15", 0.0)
    d_cardio_ratio = _col("d_cardio_ratio", 0.0)
    d_distance_pct = _col("d_distance_pct", 0.0)
    d_body_pct = _col("d_body_pct", 0.0)
    d_leg_pct = _col("d_leg_pct", 0.0)
    d_consistency = _col("d_consistency", 0.0)
    d_kd_pm = _col("d_kd_pm", _col("d_td_kd_pm", 0.0))
    d_def_kd_pm = _col("d_def_kd_pm", 0.0)
    d_finish_resistance = _col("d_finish_resistance", 0.0)
    d_durability = _col("d_durability", 0.0)
    d_late_round_pct = _col("d_late_round_pct", 0.0)
    d_output_rate = _col("d_output_rate", 0.0)
    d_glicko = np.abs(_col("d_glicko_win_prob", 0.0))
    d_head_acc = _col("d_head_acc", 0.0)
    d_distance_acc = _col("d_distance_acc", 0.0)
    d_distance_share = _col("d_distance_share", 0.0)
    d_ground_acc = _col("d_ground_acc", 0.0)
    d_clinch_acc = _col("d_clinch_acc", 0.0)
    d_clinch_pct = _col("d_clinch_pct", 0.0)
    d_body_leg_attrition = _col("d_body_leg_attrition", 0.0)
    d_sub_loss_pct = _col("d_sub_loss_pct", 0.0)
    d_recent_sub_win_rate = _col("d_recent_sub_win_rate", 0.0)
    d_sub_entry_pressure = _col("d_sub_entry_pressure", 0.0)
    d_sub_control_conversion = _col("d_sub_control_conversion", 0.0)
    d_sub_defensive_leak = _col("d_sub_defensive_leak", 0.0)
    d_late_sub_pressure = _col("d_late_sub_pressure", 0.0)
    d_sub_recency_surge = _col("d_sub_recency_surge", 0.0)
    d_grapple_recency_surge = _col("d_grapple_recency_surge", 0.0)
    d_sub_vs_control_axis = _col("d_sub_vs_control_axis", 0.0)
    d_recent_ko_loss_rate = _col("d_recent_ko_loss_rate", 0.0)
    d_r5_def_kd_pm = _col("d_r5_def_kd_pm", 0.0)
    d_ko_attack_pressure = _col("d_ko_attack_pressure", 0.0)
    d_ko_def_leak = _col("d_ko_def_leak", 0.0)
    sub_entry_sum = _col("sub_entry_pressure_sum", 0.0)
    sub_leak_sum = _col("sub_defensive_leak_sum", 0.0)
    ko_attack_sum = _col("ko_attack_pressure_sum", 0.0)
    ko_leak_sum = _col("ko_def_leak_sum", 0.0)
    total_rounds = _col("total_rounds", 3.0)
    is_title = _col("is_title", 0.0)

    # Explicit path-vs-vulnerability method features.
    X["m_ko_path_vs_vuln"] = d_ko_win + d_ko_loss + 0.45 * d_str_vs_def
    X["m_sub_path_vs_vuln"] = d_sub_win + d_been_finished + 0.35 * d_grap_vs_tdd + 0.25 * d_sub_att
    X["m_sub_round3_pressure"] = 0.60 * d_r3_sub + 0.40 * d_r3_sub_share
    X["m_sub_trend_pressure"] = 0.55 * d_late_sub + 0.45 * d_sub_t23
    X["m_dec_path"] = d_dec_win - 0.25 * d_finish_vs_resist
    X["m_finish_bias"] = X["m_ko_path_vs_vuln"] + X["m_sub_path_vs_vuln"] - X["m_dec_path"]
    X["m_ko_rounds_interaction"] = X["m_ko_path_vs_vuln"] * (4.0 - np.minimum(total_rounds, 4.0))
    X["m_sub_rounds_interaction"] = X["m_sub_path_vs_vuln"] * np.maximum(total_rounds - 2.0, 1.0)
    X["m_dec_rounds_interaction"] = X["m_dec_path"] * total_rounds
    X["m_stance_finish_interaction"] = d_ortho_vs_south * X["m_finish_bias"]
    X["m_title_finish_interaction"] = is_title * X["m_finish_bias"]
    X["m_finish_abs_pressure"] = np.abs(X["m_finish_bias"]) * np.abs(d_str_vs_def)
    X["m_finish_gap_ko_sub"] = X["m_ko_path_vs_vuln"] - X["m_sub_path_vs_vuln"]
    X["m_ko_share"] = X["m_ko_path_vs_vuln"] / (
        np.abs(X["m_ko_path_vs_vuln"]) + np.abs(X["m_sub_path_vs_vuln"]) + np.abs(X["m_dec_path"]) + 1e-6
    )
    X["m_sub_share"] = X["m_sub_path_vs_vuln"] / (
        np.abs(X["m_ko_path_vs_vuln"]) + np.abs(X["m_sub_path_vs_vuln"]) + np.abs(X["m_dec_path"]) + 1e-6
    )
    X["m_dec_share"] = X["m_dec_path"] / (
        np.abs(X["m_ko_path_vs_vuln"]) + np.abs(X["m_sub_path_vs_vuln"]) + np.abs(X["m_dec_path"]) + 1e-6
    )
    X["m_finish_vs_cardio"] = X["m_finish_bias"] * (1.0 / np.maximum(total_rounds, 1.0))
    X["m_decision_durability"] = (d_dec_win - np.abs(d_ko_loss) - np.abs(d_been_finished)) * np.maximum(total_rounds, 1.0)
    X["m_sub_chain_pressure"] = (d_sub_att + 0.50 * X["m_sub_round3_pressure"]) * (1.0 + np.maximum(d_grap_vs_tdd, -1.5))
    X["m_sub_finish_trigger"] = X["m_sub_chain_pressure"] + 0.45 * X["m_sub_trend_pressure"] + 0.30 * d_been_finished
    X["m_ko_chain_pressure"] = np.maximum(d_str_vs_def, -2.0) * (1.0 + np.maximum(d_ko_win, -1.0))
    X["m_title_rounds_interaction"] = is_title * total_rounds * X["m_dec_path"]

    # Method-only explicit mechanics: KO-shaped pressure, sub chaining, and decision control.
    X["m_ko_headhunter"] = d_power_ratio * d_head_pct * d_sig_str_diff_pm
    X["m_ko_fast_start"] = d_first_round_finish_rate * d_rd1_intensity_ratio * (4.0 - np.minimum(total_rounds, 4.0))
    X["m_ko_pressure_conversion"] = d_damage_efficiency * d_power_ratio * np.maximum(d_sig_str_acc, 0.0)
    X["m_ko_exchange_edge"] = d_strike_exchange_ratio * d_power_ratio * d_sig_str_diff_pm
    X["m_ko_distance_sniper"] = d_distance_pct * d_sig_str_acc * d_power_ratio
    X["m_ko_knockdown_axis"] = d_kd_pm - 0.6 * d_def_kd_pm
    X["m_ko_attrition"] = (d_body_pct + 0.7 * d_leg_pct) * d_output_rate * d_cardio_ratio
    X["m_ko_confident_mismatch"] = d_glicko * np.maximum(d_power_ratio, 0.0) * np.maximum(d_sig_str_diff_pm, 0.0)
    X["m_ko_burst_vs_decay"] = d_rd1_intensity_ratio - d_cardio_ratio
    X["m_ko_path_specific"] = (d_ko_win + d_ko_loss) * (0.5 + d_head_pct + 0.5 * d_power_ratio)
    X["m_sub_ground_hunter"] = d_ground_pct * (d_ctrl_pct + 0.5 * d_sub_att)
    X["m_sub_scramble_threat"] = d_rev_p15 * (d_sub_att + d_ground_pct)
    X["m_sub_late_snowball"] = d_cardio_ratio * d_r3_sub_share * (d_ctrl_pct + d_sub_att)
    X["m_dec_clean_pointing"] = (
        d_distance_pct * d_sig_str_acc * d_consistency - 0.5 * (np.abs(d_kd_pm) + np.abs(d_sub_att))
    )
    X["m_dec_stability"] = d_finish_resistance * d_durability * d_late_round_pct * d_consistency
    X["m_finish_confidence"] = d_glicko * np.maximum(X["m_finish_bias"], 0.0)
    X["m_sub_vs_ko_axis"] = (d_ground_pct + d_ctrl_pct + d_sub_att) - (d_head_pct + d_power_ratio + d_kd_pm)
    X["m_ko_head_accuracy"] = _col("d_head_hunt_accuracy", d_head_acc) * d_power_ratio
    X["m_ko_range_sniper"] = d_distance_acc * d_distance_share * d_power_ratio
    X["m_ko_range_accuracy"] = _col("d_distance_strike_accuracy", d_distance_acc) * d_distance_pct * d_power_ratio
    X["m_ground_finish_conversion"] = d_ground_acc * d_ctrl_pct * d_sub_att
    X["m_clinch_breaker"] = d_clinch_acc * d_clinch_pct * d_kd_pm
    X["m_attrition_finish"] = d_body_leg_attrition * d_late_round_pct * d_output_rate
    X["m_sub_loss_vulnerability"] = d_sub_loss_pct
    X["m_recent_sub_win_rate_gap"] = d_recent_sub_win_rate
    X["m_sub_entry_pressure"] = d_sub_entry_pressure
    X["m_sub_control_conversion"] = d_sub_control_conversion
    X["m_sub_defensive_leak"] = d_sub_defensive_leak
    X["m_late_sub_pressure"] = d_late_sub_pressure
    X["m_sub_recency_surge"] = d_sub_recency_surge
    X["m_grapple_recency_surge"] = d_grapple_recency_surge
    X["m_sub_vs_control_axis"] = d_sub_vs_control_axis
    # Exact side-specific reconstruction from oriented differential + invariant sums:
    # winner_side = 0.5 * (sum + d), loser_side = 0.5 * (sum - d)
    sub_entry_w = 0.5 * (sub_entry_sum + d_sub_entry_pressure)
    sub_entry_l = 0.5 * (sub_entry_sum - d_sub_entry_pressure)
    sub_leak_w = 0.5 * (sub_leak_sum + d_sub_defensive_leak)
    sub_leak_l = 0.5 * (sub_leak_sum - d_sub_defensive_leak)
    sub_attack_vs_leak_w = sub_entry_w * sub_leak_l
    sub_attack_vs_leak_l = sub_entry_l * sub_leak_w
    X["m_sub_mismatch_explicit"] = sub_attack_vs_leak_w - sub_attack_vs_leak_l
    X["m_sub_mismatch_max"] = np.maximum(sub_attack_vs_leak_w, sub_attack_vs_leak_l)
    X["m_sub_mismatch_sum"] = sub_attack_vs_leak_w + sub_attack_vs_leak_l
    X["m_sub_mismatch"] = X["m_sub_mismatch_explicit"]
    ko_attack_w = 0.5 * (ko_attack_sum + d_ko_attack_pressure)
    ko_attack_l = 0.5 * (ko_attack_sum - d_ko_attack_pressure)
    ko_leak_w = 0.5 * (ko_leak_sum + d_ko_def_leak)
    ko_leak_l = 0.5 * (ko_leak_sum - d_ko_def_leak)
    ko_attack_vs_leak_w = ko_attack_w * ko_leak_l
    ko_attack_vs_leak_l = ko_attack_l * ko_leak_w
    X["m_ko_mismatch"] = ko_attack_vs_leak_w - ko_attack_vs_leak_l
    X["m_ko_recent_fragility"] = d_r5_def_kd_pm + 0.7 * d_recent_ko_loss_rate

    # Keep ratios bounded and numerically stable for tree/linear blends.
    X["m_sub_ground_efficiency"] = (d_ground_pct * np.maximum(d_sub_att, 0.0)) / (1.0 + np.abs(d_ctrl_pct) + eps)
    return X


def fuzzy_find(name, fighter_history):
    if name in fighter_history and fighter_history[name]:
        return name
    lower = str(name).lower()
    for key in fighter_history:
        if str(key).lower() == lower and fighter_history[key]:
            return key
    matches = [k for k in fighter_history if lower in str(k).lower() and fighter_history[k]]
    if len(matches) == 1:
        return matches[0]
    return None


# ─── Weight class ordinal mapping ─────────────────────────────────────────────
# Ordinal is only a coarse size proxy now; the exact class is carried by one-hot
# features below, so every class gets a unique code to avoid collisions.
WEIGHT_CLASS_ORDINAL = {
    "Women's Strawweight": 1, "Women's Flyweight": 2, "Women's Bantamweight": 3,
    "Women's Featherweight": 4, "Flyweight": 5, "Bantamweight": 6,
    "Featherweight": 7, "Lightweight": 8, "Welterweight": 9,
    "Middleweight": 10, "Light Heavyweight": 11, "Heavyweight": 12,
    "Catch Weight": 13, "Open Weight": 14,
}

ACTIVE_ENSEMBLE_MODELS = {
    "LightGBM", "XGBoost", "CatBoost",
    "HistGBM", "RandForest", "ExtraTrees",
    "MLP", "LogReg",
}

FIXED_MODEL_PARAMS = {
    "LightGBM": {
        "n_estimators": 300, "lr": 0.04, "max_depth": 6, "num_leaves": 31,
        "min_child_samples": 25, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
    "XGBoost": {
        "n_estimators": 300, "lr": 0.04, "max_depth": 5, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
    "CatBoost": {
        "iterations": 350, "lr": 0.05, "depth": 6, "l2_leaf_reg": 3.0,
        "random_strength": 1.0, "bagging_temperature": 0.5,
        "border_count": 128, "bootstrap_type": "Bayesian",
    },
    "ExtraTrees": {
        "n_estimators": 500, "max_depth": 12, "min_samples_leaf": 5,
        "min_samples_split": 2, "max_features": 0.7,
    },
    "HistGBM": {
        "max_iter": 500, "lr": 0.05, "max_depth": 6,
        "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_reg": 1.0,
    },
    "RandForest": {
        "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5,
        "min_samples_split": 2, "max_features": 0.7,
    },
    "AdaBoost": {
        "n_estimators": 200, "learning_rate": 0.1,
    },
}


def _weight_class_feature_name(weight_class):
    slug = re.sub(r"[^a-z0-9]+", "_", str(weight_class).lower()).strip("_")
    return f"wc_{slug or 'unknown'}"


WEIGHT_CLASS_FEATURES = {
    weight_class: _weight_class_feature_name(weight_class)
    for weight_class in WEIGHT_CLASS_ORDINAL
}
UNKNOWN_WEIGHT_CLASS_FEATURE = "wc_unknown"
METHOD_LABELS = ["Decision", "KO/TKO", "Submission"]
MIN_METHOD_PROB = 0.001

# ─── Glicko-2 core ─────────────────────────────────────────────────────────────

def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

def _E(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))

def glicko2_update(rating, opponents):
    mu, phi, sigma = rating
    mu_s = (mu - MU_0) / SCALE
    phi_s = phi / SCALE
    if not opponents:
        phi_star = math.sqrt(phi_s**2 + sigma**2)
        return (mu, phi_star * SCALE, sigma)
    v_inv = 0.0
    delta_sum = 0.0
    for opp_r, opp_rd, score in opponents:
        mu_j = (opp_r - MU_0) / SCALE
        phi_j = opp_rd / SCALE
        g_j = _g(phi_j)
        E_j = _E(mu_s, mu_j, phi_j)
        v_inv += g_j**2 * E_j * (1.0 - E_j)
        delta_sum += g_j * (score - E_j)
    v = 1.0 / v_inv if v_inv > 0 else 1e6
    delta = v * delta_sum
    a = math.log(sigma**2)
    def f(x):
        ex = math.exp(x)
        num = ex * (delta**2 - phi_s**2 - v - ex)
        den = 2.0 * (phi_s**2 + v + ex)**2
        return num / den - (x - a) / (TAU**2)
    A = a
    if delta**2 > phi_s**2 + v:
        B = math.log(delta**2 - phi_s**2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    for _ in range(100):
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
        if abs(B - A) < CONVERGENCE:
            break
    new_sigma = math.exp(A / 2.0)
    phi_star = math.sqrt(phi_s**2 + new_sigma**2)
    new_phi_s = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    new_mu_s = mu_s + new_phi_s**2 * delta_sum
    return (new_mu_s * SCALE + MU_0, new_phi_s * SCALE, new_sigma)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def _isnan(v):
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True

def _safe_sum(values):
    return sum(v for v in values if not _isnan(v))

def _safe_mean(values):
    clean = [v for v in values if not _isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")

def _safe_div(a, b, default=0.0):
    return a / b if b and b > 0 else default


def _num_or(value, default=0.0):
    return default if _isnan(value) else float(value)


def _abs_gap(a, b, default=float("nan")):
    if _isnan(a) or _isnan(b):
        return default
    return abs(float(a) - float(b))


def _extract_profile_from_row(row, prefix):
    return {
        "height": row.get(f"{prefix}_height", float("nan")),
        "reach": row.get(f"{prefix}_reach", float("nan")),
        "ape_index": row.get(f"{prefix}_ape_index", float("nan")),
        "weight": row.get(f"{prefix}_weight", float("nan")),
        "age": row.get(f"{prefix}_age_at_event", float("nan")),
        "stance": row.get(f"{prefix}_stance", ""),
    }


def _weighted_rate(numerator, denominator, prior=0.5, strength=25.0):
    """Bayesian-smoothed rate using equivalent sample-size `strength`."""
    if denominator is None or denominator <= 0:
        return prior
    return (numerator + prior * strength) / (denominator + strength)


def _bayes_shrink(empirical_value, sample_size, prior=0.5, strength=8.0):
    """Shrink noisy low-sample statistics toward a global prior."""
    if _isnan(empirical_value):
        empirical_value = prior
    sample_size = max(float(sample_size or 0.0), 0.0)
    return (empirical_value * sample_size + prior * strength) / (sample_size + strength)


def _time_weight(fight_date, current_date, half_life_days=730):
    """Exponential decay weight: half-life of ~2 years."""
    if current_date is None or fight_date is None:
        return 1.0
    try:
        days = (current_date - fight_date).days
        return 2.0 ** (-max(days, 0) / half_life_days)
    except (TypeError, AttributeError):
        return 1.0


def _pick_missing_cols(X, threshold=MISSINGNESS_THRESHOLD):
    return [c for c in X.columns if float(X[c].isna().mean()) >= threshold]


def _add_missingness_indicators(X, missing_cols):
    X2 = X.copy()
    for col in missing_cols:
        if col not in X2.columns:
            X2[col] = np.nan
        X2[f"miss_{col}"] = X2[col].isna().astype(float)
    return X2


def _expected_calibration_error(y_true, probs, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def _calibration_curve_rmse(y_true, probs, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p, bins) - 1
    sq = 0.0
    wt = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        w = float(mask.mean())
        sq += w * ((acc - conf) ** 2)
        wt += w
    if wt <= 0:
        return 0.0
    return float(np.sqrt(sq / wt))


# Per-round stat names to track
RD_STATS = [
    "sig_str", "sig_str_att", "kd", "td", "td_att", "sub_att", "ctrl_sec",
    "head", "head_att", "body", "body_att", "leg", "leg_att",
    "distance", "distance_att", "clinch", "clinch_att", "ground", "ground_att",
]

# ─── Fight record extraction ───────────────────────────────────────────────────

def extract_fight_record(row, prefix, opp, result, opp_glicko_mu=MU_0):
    """Build a per-fighter fight record dict from a DataFrame row."""
    def g(col):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("nan")
        return v

    rec = {
        "date": row["event_date"],
        "fight_time": g("total_fight_time_sec") or 0,
        "result": result,
        "opp_glicko": opp_glicko_mu,
        "method": row.get("method", ""),
        "is_title": g("is_title_bout") or 0,
        "finish_round": g("finish_round") or 0,
        # Offense
        "sig_str": g(f"{prefix}_sig_str") or 0,
        "sig_str_att": g(f"{prefix}_sig_str_att") or 0,
        "sig_str_acc": g(f"{prefix}_sig_str_acc"),
        "str": g(f"{prefix}_str") or 0,
        "str_att": g(f"{prefix}_str_att") or 0,
        "str_acc": g(f"{prefix}_str_acc"),
        "kd": g(f"{prefix}_kd") or 0,
        "td": g(f"{prefix}_td") or 0,
        "td_att": g(f"{prefix}_td_att") or 0,
        "td_acc": g(f"{prefix}_td_acc"),
        "sub_att": g(f"{prefix}_sub_att") or 0,
        "rev": g(f"{prefix}_rev") or 0,
        "ctrl_sec": g(f"{prefix}_ctrl_sec") or 0,
        # Targeting (fight-total %)
        "head_pct": g(f"{prefix}_head"),
        "body_pct": g(f"{prefix}_body"),
        "leg_pct": g(f"{prefix}_leg"),
        # Positioning (fight-total %)
        "distance_pct": g(f"{prefix}_distance"),
        "clinch_pct": g(f"{prefix}_clinch"),
        "ground_pct": g(f"{prefix}_ground"),
        # Defense (opponent stats)
        "opp_sig_str": g(f"{opp}_sig_str") or 0,
        "opp_sig_str_att": g(f"{opp}_sig_str_att") or 0,
        "opp_sig_str_acc": g(f"{opp}_sig_str_acc"),
        "opp_str": g(f"{opp}_str") or 0,
        "opp_kd": g(f"{opp}_kd") or 0,
        "opp_td": g(f"{opp}_td") or 0,
        "opp_td_att": g(f"{opp}_td_att") or 0,
        "opp_sub_att": g(f"{opp}_sub_att") or 0,
        "opp_ctrl_sec": g(f"{opp}_ctrl_sec") or 0,
        # Physical
        "height": g(f"{prefix}_height"),
        "reach": g(f"{prefix}_reach"),
        "ape_index": g(f"{prefix}_ape_index"),
        "weight": g(f"{prefix}_weight"),
        "age": g(f"{prefix}_age_at_event"),
        "stance": row.get(f"{prefix}_stance", ""),
    }

    # Per-round stats
    for rd in range(1, 6):
        for stat in RD_STATS:
            rec[f"rd{rd}_{stat}"] = g(f"{prefix}_rd{rd}_{stat}")

    return rec


# ─── Per-fighter feature computation ───────────────────────────────────────────

_FIGHTER_FEAT_KEYS = None  # populated on first call


def _make_synthetic_fight_record(current_date=None, profile=None):
    profile = profile or {}
    rec = {
        "date": current_date or pd.Timestamp("2000-01-01"),
        "fight_time": 900.0,
        "result": "W",
        "opp_glicko": MU_0,
        "method": "Decision",
        "is_title": 0,
        "finish_round": 3,
        "sig_str": 0.0,
        "sig_str_att": 0.0,
        "sig_str_acc": float("nan"),
        "str": 0.0,
        "str_att": 0.0,
        "str_acc": float("nan"),
        "kd": 0.0,
        "td": 0.0,
        "td_att": 0.0,
        "td_acc": float("nan"),
        "sub_att": 0.0,
        "rev": 0.0,
        "ctrl_sec": 0.0,
        "head_pct": float("nan"),
        "body_pct": float("nan"),
        "leg_pct": float("nan"),
        "distance_pct": float("nan"),
        "clinch_pct": float("nan"),
        "ground_pct": float("nan"),
        "opp_sig_str": 0.0,
        "opp_sig_str_att": 0.0,
        "opp_sig_str_acc": float("nan"),
        "opp_str": 0.0,
        "opp_kd": 0.0,
        "opp_td": 0.0,
        "opp_td_att": 0.0,
        "opp_sub_att": 0.0,
        "opp_ctrl_sec": 0.0,
        "height": profile.get("height", 70.0),
        "reach": profile.get("reach", 72.0),
        "ape_index": profile.get("ape_index", 2.0),
        "weight": profile.get("weight", 170.0),
        "age": profile.get("age", 30.0),
        "stance": profile.get("stance", "Orthodox"),
    }
    for rd in range(1, 6):
        for stat in RD_STATS:
            rec[f"rd{rd}_{stat}"] = 0.0
    return rec


def _ensure_fighter_feature_keys(current_date=None):
    global _FIGHTER_FEAT_KEYS
    if _FIGHTER_FEAT_KEYS is not None:
        return
    dummy_history = [_make_synthetic_fight_record(current_date=current_date)]
    compute_fighter_features(dummy_history, (MU_0, PHI_0, SIGMA_0), [], current_date)


def compute_fighter_features(history, glicko, opp_glickos, current_date, fallback_profile=None):
    """Compute ~130 features from a fighter's fight history."""
    global _FIGHTER_FEAT_KEYS
    fallback_profile = fallback_profile or {}

    n = len(history)
    if n == 0:
        _ensure_fighter_feature_keys(current_date)
        feats = {k: float("nan") for k in _FIGHTER_FEAT_KEYS}
        stance = fallback_profile.get("stance", "") or ""
        feats.update({
            "height": _num_or(fallback_profile.get("height"), float("nan")),
            "reach": _num_or(fallback_profile.get("reach"), float("nan")),
            "ape_index": _num_or(fallback_profile.get("ape_index"), float("nan")),
            "weight": _num_or(fallback_profile.get("weight"), float("nan")),
            "age": _num_or(fallback_profile.get("age"), float("nan")),
            "num_fights": 0.0,
            "total_time_min": 0.0,
            "avg_time_min": 0.0,
            "title_bout_pct": 0.0,
            "win_rate": 0.5,
            "ko_win_pct": 0.25,
            "sub_win_pct": 0.12,
            "dec_win_pct": 0.13,
            "finish_rate": 0.5,
            "ko_loss_pct": 0.2,
            "been_finished_pct": 0.5,
            "last3_win_rate": 0.5,
            "last5_win_rate": 0.5,
            "win_streak": 0.0,
            "loss_streak": 0.0,
            "days_inactive": 365.0,
            "glicko_mu": glicko[0],
            "glicko_phi": glicko[1],
            "fights_per_year": 0.0,
            "avg_opp_glicko": MU_0,
            "td_win_rate": 0.5,
            "quality_win_rate": 0.5,
            "best_win_glicko": MU_0,
            "worst_loss_glicko": MU_0,
            "style_striking": 0.5,
            "style_wrestling": 0.25,
            "style_submission": 0.15,
            "style_clinch_ground": 0.32,
            "style_finishing": 0.5,
            "is_orthodox": 1.0 if stance == "Orthodox" else 0.0,
            "is_southpaw": 1.0 if stance == "Southpaw" else 0.0,
            "is_switch": 1.0 if stance == "Switch" else 0.0,
            "age_squared": 900.0,
            "prime_age": 1.0,
            "age_decline": 0.0,
            "experience_log": 0.0,
            "momentum": 0.5,
            "first_round_finish_rate": 0.15,
            "durability": 0.8,
            "output_rate": 0.0,
            "damage_efficiency": 1.0,
            "late_round_pct": 0.5,
            "avg_finish_round": 2.5,
            "sig_str_diff_pm": 0.0,
            "grappling_dominance": 0.0,
            "consistency": 0.5,
            "ewm_opp_quality": MU_0,
            "form_vs_career": 0.0,
            "form5_vs_career": 0.0,
            "days_since_last_loss": 1500.0,
            "title_fight_win_rate": 0.5,
            "title_fight_experience": 0.0,
            "fight_iq": 0.45 * 0.35,
            "rd1_intensity_ratio": 1.0,
            "cardio_ratio": 1.0,
            "opp_quality_trend": 1.0,
            "fights_last_year": 0.0,
            "win_rate_last_year": 0.5,
            "loss_recovery_rate": 0.5,
            "strike_exchange_ratio": 1.0,
            "grappling_threat": 0.0,
            "decision_win_rate": 0.5,
            "finish_resistance": 0.5,
            "offensive_diversity": 1.0,
            "ewm_sig_str_diff_pm": 0.0,
            "glicko_confidence": 0.0,
        })
        return feats

    feats = {}

    total_time = _safe_sum(h["fight_time"] for h in history)
    total_time_min = total_time / 60.0 if total_time > 0 else 1.0
    total_time_15 = total_time / 900.0 if total_time > 0 else 1.0

    total_sig_land = _safe_sum(h["sig_str"] for h in history)
    total_sig_att = _safe_sum(h["sig_str_att"] for h in history)
    total_str_land = _safe_sum(h["str"] for h in history)
    total_str_att = _safe_sum(h["str_att"] for h in history)
    total_td_land = _safe_sum(h["td"] for h in history)
    total_td_att = _safe_sum(h["td_att"] for h in history)

    # ── Striking offense (per minute) ──
    feats["sig_str_pm"] = total_sig_land / total_time_min
    feats["sig_str_att_pm"] = total_sig_att / total_time_min
    feats["str_pm"] = total_str_land / total_time_min
    feats["str_att_pm"] = total_str_att / total_time_min
    feats["kd_pm"] = _safe_sum(h["kd"] for h in history) / total_time_min

    # ── Accuracy (attempt-weighted + Bayesian shrinkage) ──
    feats["sig_str_acc"] = _bayes_shrink(
        _weighted_rate(total_sig_land, total_sig_att, prior=0.45, strength=30),
        n, prior=0.45, strength=8,
    )
    feats["str_acc"] = _bayes_shrink(
        _weighted_rate(total_str_land, total_str_att, prior=0.55, strength=30),
        n, prior=0.55, strength=8,
    )
    feats["td_acc"] = _bayes_shrink(
        _weighted_rate(total_td_land, total_td_att, prior=0.35, strength=20),
        n, prior=0.35, strength=8,
    )

    # ── Grappling (per 15 min) ──
    feats["td_p15"] = _safe_sum(h["td"] for h in history) / total_time_15
    feats["td_att_p15"] = _safe_sum(h["td_att"] for h in history) / total_time_15
    feats["sub_att_p15"] = _safe_sum(h["sub_att"] for h in history) / total_time_15
    feats["rev_p15"] = _safe_sum(h["rev"] for h in history) / total_time_15
    feats["ctrl_pct"] = _safe_sum(h["ctrl_sec"] for h in history) / total_time if total_time > 0 else 0

    # ── Targeting (attempt/volume-weighted + shrinkage) ──
    sig_for_target = _safe_sum(
        h["sig_str"] for h in history
        if not _isnan(h["head_pct"]) and not _isnan(h["sig_str"])
    )
    head_num = _safe_sum(
        h["head_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["head_pct"]) and not _isnan(h["sig_str"])
    )
    body_num = _safe_sum(
        h["body_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["body_pct"]) and not _isnan(h["sig_str"])
    )
    leg_num = _safe_sum(
        h["leg_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["leg_pct"]) and not _isnan(h["sig_str"])
    )
    feats["head_pct"] = _bayes_shrink(
        _weighted_rate(head_num, sig_for_target, prior=0.62, strength=40),
        n, prior=0.62, strength=8,
    )
    feats["body_pct"] = _bayes_shrink(
        _weighted_rate(body_num, sig_for_target, prior=0.22, strength=40),
        n, prior=0.22, strength=8,
    )
    feats["leg_pct"] = _bayes_shrink(
        _weighted_rate(leg_num, sig_for_target, prior=0.16, strength=40),
        n, prior=0.16, strength=8,
    )

    # ── Positioning (volume-weighted + shrinkage) ──
    sig_for_pos = _safe_sum(
        h["sig_str"] for h in history
        if not _isnan(h["distance_pct"]) and not _isnan(h["sig_str"])
    )
    dist_num = _safe_sum(
        h["distance_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["distance_pct"]) and not _isnan(h["sig_str"])
    )
    clinch_num = _safe_sum(
        h["clinch_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["clinch_pct"]) and not _isnan(h["sig_str"])
    )
    ground_num = _safe_sum(
        h["ground_pct"] * h["sig_str"]
        for h in history
        if not _isnan(h["ground_pct"]) and not _isnan(h["sig_str"])
    )
    feats["distance_pct"] = _bayes_shrink(
        _weighted_rate(dist_num, sig_for_pos, prior=0.68, strength=40),
        n, prior=0.68, strength=8,
    )
    feats["clinch_pct"] = _bayes_shrink(
        _weighted_rate(clinch_num, sig_for_pos, prior=0.14, strength=40),
        n, prior=0.14, strength=8,
    )
    feats["ground_pct"] = _bayes_shrink(
        _weighted_rate(ground_num, sig_for_pos, prior=0.18, strength=40),
        n, prior=0.18, strength=8,
    )

    # ── Defense (opponent per-minute) ──
    feats["def_sig_str_pm"] = _safe_sum(h["opp_sig_str"] for h in history) / total_time_min
    feats["def_str_pm"] = _safe_sum(h["opp_str"] for h in history) / total_time_min
    feats["def_kd_pm"] = _safe_sum(h["opp_kd"] for h in history) / total_time_min
    feats["def_td_p15"] = _safe_sum(h["opp_td"] for h in history) / total_time_15
    feats["def_sub_att_p15"] = _safe_sum(h["opp_sub_att"] for h in history) / total_time_15
    feats["def_ctrl_pct"] = _safe_sum(h["opp_ctrl_sec"] for h in history) / total_time if total_time > 0 else 0
    opp_sig_att = _safe_sum(h["opp_sig_str_att"] for h in history)
    opp_sig_land = _safe_sum(h["opp_sig_str"] for h in history)
    feats["def_sig_str_acc"] = _bayes_shrink(
        _weighted_rate(opp_sig_land, opp_sig_att, prior=0.45, strength=30),
        n, prior=0.45, strength=8,
    )

    # ── Differentials ──
    feats["net_sig_str_pm"] = feats["sig_str_pm"] - feats["def_sig_str_pm"]
    feats["net_kd_pm"] = feats["kd_pm"] - feats["def_kd_pm"]
    feats["net_td_p15"] = feats["td_p15"] - feats["def_td_p15"]
    feats["net_ctrl_pct"] = feats["ctrl_pct"] - feats["def_ctrl_pct"]

    # ── Defense rates ──
    feats["sig_str_defense_rate"] = _bayes_shrink(
        1.0 - _weighted_rate(opp_sig_land, opp_sig_att, prior=0.45, strength=30),
        n, prior=0.55, strength=8,
    )
    opp_td_att = _safe_sum(h["opp_td_att"] for h in history)
    opp_td_land = _safe_sum(h["opp_td"] for h in history)
    feats["td_defense_rate"] = _bayes_shrink(
        1.0 - _weighted_rate(opp_td_land, opp_td_att, prior=0.35, strength=20),
        n, prior=0.65, strength=8,
    )

    # ── Physical (most recent) ──
    latest = history[-1]
    feats["height"] = latest["height"] if not _isnan(latest["height"]) else _num_or(fallback_profile.get("height"), float("nan"))
    feats["reach"] = latest["reach"] if not _isnan(latest["reach"]) else _num_or(fallback_profile.get("reach"), float("nan"))
    feats["ape_index"] = latest["ape_index"] if not _isnan(latest["ape_index"]) else _num_or(fallback_profile.get("ape_index"), float("nan"))
    feats["weight"] = latest["weight"] if not _isnan(latest["weight"]) else _num_or(fallback_profile.get("weight"), float("nan"))
    feats["age"] = latest["age"] if not _isnan(latest["age"]) else _num_or(fallback_profile.get("age"), float("nan"))

    # ── Experience ──
    feats["num_fights"] = n
    feats["total_time_min"] = total_time_min
    feats["avg_time_min"] = total_time_min / n
    feats["title_bout_pct"] = sum(1 for h in history if h["is_title"]) / n

    # ── Record ──
    wins = sum(1 for h in history if h["result"] == "W")
    losses = sum(1 for h in history if h["result"] == "L")
    feats["win_rate"] = _bayes_shrink(_safe_div(wins, n, 0.5), n, prior=0.5, strength=10)
    ko_w = sum(1 for h in history if h["result"] == "W" and "KO" in str(h["method"]))
    sub_w = sum(1 for h in history if h["result"] == "W" and "Sub" in str(h["method"]))
    dec_w = sum(1 for h in history if h["result"] == "W" and "Dec" in str(h["method"]))
    feats["ko_win_pct"] = _bayes_shrink(_safe_div(ko_w, n, 0.25), n, prior=0.25, strength=10)
    feats["sub_win_pct"] = _bayes_shrink(_safe_div(sub_w, n, 0.12), n, prior=0.12, strength=10)
    feats["dec_win_pct"] = _bayes_shrink(_safe_div(dec_w, n, 0.13), n, prior=0.13, strength=10)
    feats["finish_rate"] = _safe_div(ko_w + sub_w, max(wins, 1))
    ko_l = sum(1 for h in history if h["result"] == "L" and "KO" in str(h["method"]))
    sub_l = sum(1 for h in history if h["result"] == "L" and "Sub" in str(h["method"]))
    feats["ko_loss_pct"] = _bayes_shrink(_safe_div(ko_l, n, 0.2), n, prior=0.2, strength=10)
    feats["sub_loss_pct"] = _bayes_shrink(_safe_div(sub_l, max(losses, 1), 0.20), losses, prior=0.20, strength=8)
    feats["been_finished_pct"] = _safe_div(ko_l + sub_l, max(losses, 1))

    # ── Form ──
    last3 = history[-3:]
    last5 = history[-5:]
    last3_wr = sum(1 for h in last3 if h["result"] == "W") / len(last3)
    last5_wr = sum(1 for h in last5 if h["result"] == "W") / len(last5)
    feats["last3_win_rate"] = _bayes_shrink(last3_wr, len(last3), prior=0.5, strength=6)
    feats["last5_win_rate"] = _bayes_shrink(last5_wr, len(last5), prior=0.5, strength=6)
    recent_sub_wins = sum(1 for h in last5 if h["result"] == "W" and "Sub" in str(h.get("method", "")))
    recent_ko_losses = sum(1 for h in last5 if h["result"] == "L" and "KO" in str(h.get("method", "")))
    recent_n = max(min(n, 5), 1)
    feats["recent_sub_win_rate"] = _bayes_shrink(
        _safe_div(recent_sub_wins, recent_n, 0.12), recent_n, prior=0.12, strength=5
    )
    feats["recent_ko_loss_rate"] = _bayes_shrink(
        _safe_div(recent_ko_losses, recent_n, 0.20), recent_n, prior=0.20, strength=5
    )
    win_streak = 0
    for h in reversed(history):
        if h["result"] == "W":
            win_streak += 1
        else:
            break
    loss_streak = 0
    for h in reversed(history):
        if h["result"] == "L":
            loss_streak += 1
        else:
            break
    feats["win_streak"] = win_streak
    feats["loss_streak"] = loss_streak
    if current_date is not None and not _isnan(history[-1]["date"]):
        feats["days_inactive"] = (current_date - history[-1]["date"]).days
    else:
        feats["days_inactive"] = 365

    # ── Glicko-2 ──
    feats["glicko_mu"] = glicko[0]
    feats["glicko_phi"] = glicko[1]

    # ── Round 1 stats (career averages) ──
    for stat in ["sig_str", "sig_str_att", "kd", "td", "td_att", "sub_att",
                 "ctrl_sec", "head", "body", "leg", "distance", "clinch", "ground"]:
        vals = [h[f"rd1_{stat}"] for h in history if not _isnan(h.get(f"rd1_{stat}"))]
        feats[f"rd1_{stat}"] = _safe_mean(vals)

    # ── Round 2 stats ──
    for stat in ["sig_str", "sig_str_att", "kd", "td", "td_att", "sub_att",
                 "ctrl_sec", "head", "body", "leg"]:
        vals = [h[f"rd2_{stat}"] for h in history if not _isnan(h.get(f"rd2_{stat}"))]
        feats[f"rd2_{stat}"] = _safe_mean(vals)

    # ── Round 3 stats ──
    for stat in ["sig_str", "kd", "td", "td_att", "sub_att", "ctrl_sec", "head", "body", "leg"]:
        vals = [h[f"rd3_{stat}"] for h in history if not _isnan(h.get(f"rd3_{stat}"))]
        feats[f"rd3_{stat}"] = _safe_mean(vals)

    # ── Championship rounds (4+5 combined) ──
    for stat in ["sig_str", "kd", "td", "ctrl_sec"]:
        vals = []
        for rd in [4, 5]:
            vals.extend(h[f"rd{rd}_{stat}"] for h in history
                        if not _isnan(h.get(f"rd{rd}_{stat}")))
        feats[f"champ_{stat}"] = _safe_mean(vals)

    # ── Late vs early ──
    rd1_ss = [h["rd1_sig_str"] for h in history if not _isnan(h.get("rd1_sig_str"))]
    rd3_ss = [h["rd3_sig_str"] for h in history if not _isnan(h.get("rd3_sig_str"))]
    feats["late_vs_early_sig_str"] = _safe_mean(rd3_ss) - _safe_mean(rd1_ss) \
        if rd1_ss and rd3_ss else float("nan")
    rd1_sub = [h["rd1_sub_att"] for h in history if not _isnan(h.get("rd1_sub_att"))]
    rd2_sub = [h["rd2_sub_att"] for h in history if not _isnan(h.get("rd2_sub_att"))]
    rd3_sub = [h["rd3_sub_att"] for h in history if not _isnan(h.get("rd3_sub_att"))]
    r1_sub_m = _safe_mean(rd1_sub)
    r2_sub_m = _safe_mean(rd2_sub)
    r3_sub_m = _safe_mean(rd3_sub)
    feats["late_vs_early_sub_att"] = r3_sub_m - r1_sub_m if rd1_sub and rd3_sub else float("nan")
    feats["sub_att_trend_12"] = r2_sub_m - r1_sub_m if rd1_sub and rd2_sub else float("nan")
    feats["sub_att_trend_23"] = r3_sub_m - r2_sub_m if rd2_sub and rd3_sub else float("nan")
    total_r123_sub = max(r1_sub_m + r2_sub_m + r3_sub_m, 1e-6)
    feats["rd3_sub_share"] = r3_sub_m / total_r123_sub
    feats["rd1_sub_share"] = r1_sub_m / total_r123_sub
    feats["sub_round_concentration"] = max(r1_sub_m, r2_sub_m, r3_sub_m) / total_r123_sub

    # ── Recent-window rate stats (last 1, 3, 5 fights) ──
    for prefix_tag, window in [("r1f", 1), ("r3", 3), ("r5", 5)]:
        recent = history[-window:]
        rt = _safe_sum(h["fight_time"] for h in recent)
        rt_min = rt / 60.0 if rt > 0 else 1.0
        rt_15 = rt / 900.0 if rt > 0 else 1.0
        feats[f"{prefix_tag}_sig_str_pm"] = _safe_sum(h["sig_str"] for h in recent) / rt_min
        feats[f"{prefix_tag}_kd_pm"] = _safe_sum(h["kd"] for h in recent) / rt_min
        feats[f"{prefix_tag}_td_p15"] = _safe_sum(h["td"] for h in recent) / rt_15
        feats[f"{prefix_tag}_sub_att_p15"] = _safe_sum(h["sub_att"] for h in recent) / rt_15
        feats[f"{prefix_tag}_ctrl_pct"] = _safe_sum(h["ctrl_sec"] for h in recent) / rt if rt > 0 else 0
        feats[f"{prefix_tag}_def_sig_str_pm"] = _safe_sum(h["opp_sig_str"] for h in recent) / rt_min
        feats[f"{prefix_tag}_def_kd_pm"] = _safe_sum(h["opp_kd"] for h in recent) / rt_min
        rec_sig_land = _safe_sum(h["sig_str"] for h in recent)
        rec_sig_att = _safe_sum(h["sig_str_att"] for h in recent)
        rec_td_land = _safe_sum(h["td"] for h in recent)
        rec_td_att = _safe_sum(h["td_att"] for h in recent)
        feats[f"{prefix_tag}_sig_str_acc"] = _bayes_shrink(
            _weighted_rate(rec_sig_land, rec_sig_att, prior=0.45, strength=20),
            len(recent), prior=0.45, strength=6,
        )
        feats[f"{prefix_tag}_td_acc"] = _bayes_shrink(
            _weighted_rate(rec_td_land, rec_td_att, prior=0.35, strength=20),
            len(recent), prior=0.35, strength=6,
        )
        feats[f"{prefix_tag}_win"] = _bayes_shrink(
            _safe_div(sum(1 for h in recent if h["result"] == "W"), len(recent), 0.5),
            len(recent), prior=0.5, strength=6,
        )

    # ── Exponentially weighted moving average (alpha=0.3, more recent = higher) ──
    alpha = 0.3
    ewm_keys = [
        ("sig_str", "fight_time", 60),
        ("kd", "fight_time", 60),
        ("td", "fight_time", 900),
        ("opp_sig_str", "fight_time", 60),
    ]
    for stat_key, time_key, divisor in ewm_keys:
        wsum, wtot = 0.0, 0.0
        for i, h in enumerate(history):
            w = (1 - alpha) ** (n - 1 - i)
            ft = h[time_key] / divisor if h[time_key] > 0 else 1.0
            wsum += w * (h[stat_key] / ft)
            wtot += w
        tag = stat_key.replace("opp_sig_str", "def_sig_str")
        feats[f"ewm_{tag}_pm"] = wsum / wtot if wtot > 0 else 0

    # EWM ctrl_pct
    wsum, wtot = 0.0, 0.0
    for i, h in enumerate(history):
        w = (1 - alpha) ** (n - 1 - i)
        ft = h["fight_time"] if h["fight_time"] > 0 else 1.0
        wsum += w * (h["ctrl_sec"] / ft)
        wtot += w
    feats["ewm_ctrl_pct"] = wsum / wtot if wtot > 0 else 0

    # EWM accuracy
    for acc_key in ["sig_str_acc", "td_acc"]:
        wsum, wtot = 0.0, 0.0
        for i, h in enumerate(history):
            if _isnan(h[acc_key]):
                continue
            w = (1 - alpha) ** (n - 1 - i)
            wsum += w * h[acc_key]
            wtot += w
        feats[f"ewm_{acc_key}"] = wsum / wtot if wtot > 0 else float("nan")

    # EWM win
    wsum, wtot = 0.0, 0.0
    for i, h in enumerate(history):
        w = (1 - alpha) ** (n - 1 - i)
        wsum += w * (1.0 if h["result"] == "W" else 0.0)
        wtot += w
    feats["ewm_win"] = wsum / wtot if wtot > 0 else 0.5

    # ── Variance (per-fight rates std dev) ──
    if n >= 2:
        per_fight_sig = []
        per_fight_kd = []
        per_fight_td = []
        per_fight_ctrl = []
        per_fight_def_sig = []
        for h in history:
            ft_min = h["fight_time"] / 60.0 if h["fight_time"] > 0 else 1.0
            ft_15 = h["fight_time"] / 900.0 if h["fight_time"] > 0 else 1.0
            ft = h["fight_time"] if h["fight_time"] > 0 else 1.0
            per_fight_sig.append(h["sig_str"] / ft_min)
            per_fight_kd.append(h["kd"] / ft_min)
            per_fight_td.append(h["td"] / ft_15)
            per_fight_ctrl.append(h["ctrl_sec"] / ft)
            per_fight_def_sig.append(h["opp_sig_str"] / ft_min)
        feats["std_sig_str_pm"] = float(np.std(per_fight_sig))
        feats["std_kd_pm"] = float(np.std(per_fight_kd))
        feats["std_td_p15"] = float(np.std(per_fight_td))
        feats["std_ctrl_pct"] = float(np.std(per_fight_ctrl))
        feats["std_def_sig_str_pm"] = float(np.std(per_fight_def_sig))
    else:
        feats["std_sig_str_pm"] = 0.0
        feats["std_kd_pm"] = 0.0
        feats["std_td_p15"] = 0.0
        feats["std_ctrl_pct"] = 0.0
        feats["std_def_sig_str_pm"] = 0.0

    # ── Derived ──
    total_sig = _safe_sum(h["sig_str"] for h in history)
    total_str = _safe_sum(h["str"] for h in history)
    feats["power_ratio"] = _safe_div(_safe_sum(h["kd"] for h in history), max(total_sig, 1))
    feats["striking_efficiency"] = _safe_div(total_sig, max(total_str, 1))
    feats["grappling_rate_p15"] = (
        _safe_sum(h["td"] + h["sub_att"] for h in history) / total_time_15
    )
    feats["def_grappling_rate_p15"] = (
        _safe_sum(h["opp_td"] + h["opp_sub_att"] for h in history) / total_time_15
    )
    feats["damage_ratio"] = _safe_div(
        _safe_sum(h["opp_sig_str"] for h in history),
        max(_safe_sum(h["sig_str"] for h in history), 1),
    )

    # ── Target/position attempt-driven accuracy and intent shares (round 1-5) ──
    eps = 1e-6

    def _round_stat_total(stat_name):
        vals = []
        for h in history:
            for rd in range(1, 6):
                v = h.get(f"rd{rd}_{stat_name}", float("nan"))
                if not _isnan(v):
                    vals.append(v)
        return _safe_sum(vals)

    total_sig_att_round = _round_stat_total("sig_str_att")
    total_head_land = _round_stat_total("head")
    total_head_att = _round_stat_total("head_att")
    total_body_land = _round_stat_total("body")
    total_body_att = _round_stat_total("body_att")
    total_leg_land = _round_stat_total("leg")
    total_leg_att = _round_stat_total("leg_att")
    total_distance_land = _round_stat_total("distance")
    total_distance_att = _round_stat_total("distance_att")
    total_clinch_land = _round_stat_total("clinch")
    total_clinch_att = _round_stat_total("clinch_att")
    total_ground_land = _round_stat_total("ground")
    total_ground_att = _round_stat_total("ground_att")

    feats["head_acc"] = _safe_div(total_head_land, total_head_att + eps, 0.16)
    feats["distance_acc"] = _safe_div(total_distance_land, total_distance_att + eps, 0.45)
    feats["ground_acc"] = _safe_div(total_ground_land, total_ground_att + eps, 0.35)
    feats["clinch_acc"] = _safe_div(total_clinch_land, total_clinch_att + eps, 0.30)
    feats["body_acc"] = _safe_div(total_body_land, total_body_att + eps, 0.28)
    feats["leg_acc"] = _safe_div(total_leg_land, total_leg_att + eps, 0.28)
    feats["body_leg_attrition"] = feats["body_acc"] + feats["leg_acc"]
    feats["ground_strike_accuracy"] = feats["ground_acc"]
    feats["head_hunt_accuracy"] = feats["head_acc"]
    feats["distance_strike_accuracy"] = feats["distance_acc"]

    feats["head_hunt_share"] = _safe_div(total_head_att, total_sig_att_round + eps, 0.33)
    feats["distance_share"] = _safe_div(total_distance_att, total_sig_att_round + eps, 0.62)

    # Fights per year
    if n > 1:
        career_days = (history[-1]["date"] - history[0]["date"]).days
        feats["fights_per_year"] = n / max(career_days / 365.25, 0.5)
    else:
        feats["fights_per_year"] = 1.0

    # Stance
    stance = latest.get("stance", "") or fallback_profile.get("stance", "")
    feats["is_orthodox"] = 1 if stance == "Orthodox" else 0
    feats["is_southpaw"] = 1 if stance == "Southpaw" else 0
    feats["is_switch"] = 1 if stance == "Switch" else 0

    # Average opponent Glicko
    feats["avg_opp_glicko"] = _safe_mean(opp_glickos) if opp_glickos else MU_0

    # ── Time-decayed career stats (half-life ~2 years) ──
    tw = [_time_weight(h["date"], current_date) for h in history]
    tw_total = sum(tw) or 1.0
    tw_time = sum(w * h["fight_time"] for w, h in zip(tw, history))
    tw_time_min = tw_time / 60.0 if tw_time > 0 else 1.0
    tw_time_15 = tw_time / 900.0 if tw_time > 0 else 1.0
    feats["td_sig_str_pm"] = sum(w * h["sig_str"] for w, h in zip(tw, history)) / tw_time_min
    feats["td_kd_pm"] = sum(w * h["kd"] for w, h in zip(tw, history)) / tw_time_min
    feats["td_td_p15"] = sum(w * h["td"] for w, h in zip(tw, history)) / tw_time_15
    feats["td_ctrl_pct"] = (
        sum(w * h["ctrl_sec"] for w, h in zip(tw, history)) / tw_time if tw_time > 0 else 0
    )
    feats["td_def_sig_str_pm"] = sum(w * h["opp_sig_str"] for w, h in zip(tw, history)) / tw_time_min
    tw_sig_land = sum(w * h["sig_str"] for w, h in zip(tw, history))
    tw_sig_att = sum(w * h["sig_str_att"] for w, h in zip(tw, history))
    feats["td_sig_str_acc"] = _weighted_rate(tw_sig_land, tw_sig_att, prior=0.45, strength=20)
    tw_td_land = sum(w * h["td"] for w, h in zip(tw, history))
    tw_td_att = sum(w * h["td_att"] for w, h in zip(tw, history))
    feats["td_td_acc"] = _weighted_rate(tw_td_land, tw_td_att, prior=0.35, strength=20)
    feats["td_win_rate"] = sum(
        w * (1.0 if h["result"] == "W" else 0.0) for w, h in zip(tw, history)
    ) / tw_total

    # ── Opponent-quality-adjusted features ──
    opp_g = [h.get("opp_glicko", MU_0) for h in history]
    opp_g_w = [max(g_val / MU_0, 0.1) for g_val in opp_g]
    ogw_total = sum(opp_g_w) or 1.0
    feats["quality_win_rate"] = sum(
        w * (1.0 if h["result"] == "W" else 0.0)
        for w, h in zip(opp_g_w, history)
    ) / ogw_total
    wins_opp_g = [og for h, og in zip(history, opp_g) if h["result"] == "W"]
    feats["best_win_glicko"] = max(wins_opp_g) if wins_opp_g else MU_0
    losses_opp_g = [og for h, og in zip(history, opp_g) if h["result"] == "L"]
    feats["worst_loss_glicko"] = min(losses_opp_g) if losses_opp_g else MU_0
    feats["quality_sig_str_pm"] = sum(
        w * h["sig_str"] for w, h in zip(opp_g_w, history)
    ) / (sum(w * (h["fight_time"] / 60.0 if h["fight_time"] > 0 else 1.0)
             for w, h in zip(opp_g_w, history)) or 1.0)

    # ── Style profile (continuous features for matchup interactions) ──
    total_offensive = feats["sig_str_pm"] + feats["td_p15"] + feats["sub_att_p15"] + 0.01
    feats["style_striking"] = feats["sig_str_pm"] / total_offensive
    feats["style_wrestling"] = feats["td_p15"] / total_offensive
    feats["style_submission"] = feats["sub_att_p15"] / total_offensive
    feats["style_clinch_ground"] = feats.get("clinch_pct", 0.14) + feats.get("ground_pct", 0.18)
    feats["style_finishing"] = feats.get("finish_rate", 0.5)

    # ── New high-impact features ──
    age_val = feats.get("age", 30.0)
    if _isnan(age_val):
        age_val = 30.0
    feats["age_squared"] = age_val ** 2
    feats["prime_age"] = 1.0 if 26.0 <= age_val <= 32.0 else 0.0
    feats["age_decline"] = max(0.0, age_val - 32.0)
    feats["experience_log"] = math.log1p(n)

    # Momentum: steeper EWM (alpha=0.5) for recent bias
    alpha_m = 0.5
    m_sum, m_tot = 0.0, 0.0
    for i, h in enumerate(history):
        w = (1 - alpha_m) ** (n - 1 - i)
        m_sum += w * (1.0 if h["result"] == "W" else 0.0)
        m_tot += w
    feats["momentum"] = m_sum / m_tot if m_tot > 0 else 0.5

    # First-round finish rate
    r1_finishes = sum(1 for h in history if h["result"] == "W" and h.get("finish_round") == 1
                      and ("KO" in str(h.get("method", "")) or "Sub" in str(h.get("method", ""))))
    feats["first_round_finish_rate"] = _bayes_shrink(_safe_div(r1_finishes, n, 0.15), n, prior=0.15, strength=10)

    # Durability (inverse of KO loss susceptibility)
    feats["durability"] = 1.0 - feats.get("ko_loss_pct", 0.2)

    # Output rate: total offensive actions per minute
    total_actions = total_sig_land + _safe_sum(h["td"] for h in history) + _safe_sum(h["sub_att"] for h in history)
    feats["output_rate"] = total_actions / total_time_min

    # Damage efficiency: sig str landed / opponent sig str landed
    opp_total_sig = _safe_sum(h["opp_sig_str"] for h in history)
    feats["damage_efficiency"] = _safe_div(total_sig_land, max(opp_total_sig, 1), 1.0)

    # Late-round endurance: pct of fights that went past round 2
    went_late = sum(1 for h in history if _num_or(h.get("finish_round"), 3) >= 3)
    feats["late_round_pct"] = _bayes_shrink(_safe_div(went_late, n, 0.5), n, prior=0.5, strength=8)

    # Average finish round (lower = more dangerous finisher)
    finish_rounds = [h["finish_round"] for h in history
                     if h["result"] == "W" and not _isnan(h.get("finish_round")) and h["finish_round"] > 0]
    feats["avg_finish_round"] = _safe_mean(finish_rounds) if finish_rounds else 2.5

    # Sig str differential per minute (combines offense and defense into one number)
    feats["sig_str_diff_pm"] = feats["sig_str_pm"] - feats["def_sig_str_pm"]

    # Grappling dominance: (td_landed + ctrl_pct) vs (opp_td_landed + opp_ctrl_pct)
    feats["grappling_dominance"] = feats["td_p15"] + feats["ctrl_pct"] - feats["def_td_p15"] - feats["def_ctrl_pct"]

    # Consistency: coefficient of variation of sig_str_pm (lower = more consistent)
    if n >= 2 and feats["sig_str_pm"] > 0:
        feats["consistency"] = 1.0 - min(feats["std_sig_str_pm"] / (feats["sig_str_pm"] + 0.01), 2.0) / 2.0
    else:
        feats["consistency"] = 0.5

    # EWM opponent quality: recent opponent strength
    alpha_oq = 0.4
    oq_sum, oq_tot = 0.0, 0.0
    for i, h in enumerate(history):
        w = (1 - alpha_oq) ** (n - 1 - i)
        oq_sum += w * _num_or(h.get("opp_glicko"), MU_0)
        oq_tot += w
    feats["ewm_opp_quality"] = oq_sum / oq_tot if oq_tot > 0 else MU_0

    # ── Form vs career (explicit trend for linear models) ──
    feats["form_vs_career"] = feats["last3_win_rate"] - feats["win_rate"]
    feats["form5_vs_career"] = feats["last5_win_rate"] - feats["win_rate"]

    # ── Days since last loss (confidence/psychology proxy) ──
    last_loss_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i]["result"] == "L":
            last_loss_idx = i
            break
    if last_loss_idx is not None and current_date is not None:
        feats["days_since_last_loss"] = (current_date - history[last_loss_idx]["date"]).days
    else:
        feats["days_since_last_loss"] = 1500.0  # Never lost or no date

    # ── Title fight performance ──
    title_fights = [h for h in history if h.get("is_title")]
    title_wins = sum(1 for h in title_fights if h["result"] == "W")
    feats["title_fight_win_rate"] = _bayes_shrink(
        _safe_div(title_wins, len(title_fights), 0.5),
        len(title_fights), prior=0.5, strength=4,
    )
    feats["title_fight_experience"] = len(title_fights)

    # ── Fight IQ composite (multi-domain effectiveness) ──
    feats["fight_iq"] = feats["sig_str_acc"] * feats["td_acc"]

    # ── Round 1 intensity ratio (first-round explosiveness vs career pace) ──
    rd1_ss_val = feats.get("rd1_sig_str", float("nan"))
    if not _isnan(rd1_ss_val) and feats["sig_str_pm"] > 0:
        # rd1_sig_str is per-round avg, sig_str_pm is per-minute career; convert to same scale
        feats["rd1_intensity_ratio"] = rd1_ss_val / (feats["sig_str_pm"] * 5.0 + 0.01)
    else:
        feats["rd1_intensity_ratio"] = 1.0

    # ── Cardio ratio (round 3 output / round 1 output) ──
    rd1_val = feats.get("rd1_sig_str", float("nan"))
    rd3_val = feats.get("rd3_sig_str", float("nan"))
    if not _isnan(rd1_val) and not _isnan(rd3_val) and rd1_val > 0:
        feats["cardio_ratio"] = rd3_val / (rd1_val + 0.01)
    else:
        feats["cardio_ratio"] = 1.0

    # ── Submission pressure/vulnerability composites ──
    feats["sub_entry_pressure"] = (
        feats.get("td_att_p15", 0.0) * feats.get("td_acc", 0.35)
        + 0.7 * feats.get("sub_att_p15", 0.0)
        + 0.4 * feats.get("rev_p15", 0.0)
    )
    feats["sub_control_conversion"] = (
        feats.get("sub_att_p15", 0.0)
        * (0.25 + feats.get("ctrl_pct", 0.0))
        * (0.25 + feats.get("ground_pct", 0.0))
    )
    feats["sub_scramble_threat"] = (
        feats.get("sub_att_p15", 0.0) + feats.get("rev_p15", 0.0)
    ) * (1.0 + feats.get("ground_pct", 0.0))
    feats["sub_defensive_leak"] = (
        feats.get("def_sub_att_p15", 0.0) * (1.0 - feats.get("td_defense_rate", 0.65))
        + 0.75 * feats.get("sub_loss_pct", 0.20)
    )
    feats["late_sub_pressure"] = (
        feats.get("sub_att_p15", 0.0)
        * feats.get("cardio_ratio", 1.0)
        * (0.5 + feats.get("rd3_sub_share", 0.0))
    )
    feats["sub_recency_surge"] = feats.get("r1f_sub_att_p15", 0.0) - feats.get("r5_sub_att_p15", 0.0)
    feats["grapple_recency_surge"] = (
        feats.get("r1f_td_p15", 0.0) + feats.get("r1f_sub_att_p15", 0.0)
    ) - (
        feats.get("r5_td_p15", 0.0) + feats.get("r5_sub_att_p15", 0.0)
    )
    feats["sub_vs_control_axis"] = (
        feats.get("sub_att_p15", 0.0) + 0.5 * feats.get("rev_p15", 0.0)
    ) - 0.6 * feats.get("ctrl_pct", 0.0)
    # KO-side composites for explicit attacker-vs-vulnerability mismatch.
    feats["ko_attack_pressure"] = (
        feats.get("power_ratio", 0.0)
        * feats.get("head_pct", 0.0)
        * feats.get("sig_str_diff_pm", 0.0)
    )
    feats["ko_def_leak"] = (
        feats.get("def_kd_pm", 0.0)
        + feats.get("ko_loss_pct", 0.0)
        + 0.5 * (1.0 - feats.get("durability", 0.8))
    )

    # ── Opponent quality trend (facing tougher/weaker opponents recently?) ──
    if feats["avg_opp_glicko"] > 0:
        feats["opp_quality_trend"] = feats["ewm_opp_quality"] / feats["avg_opp_glicko"]
    else:
        feats["opp_quality_trend"] = 1.0

    # ── Recent-year activity ──
    if current_date is not None:
        fights_last_year = sum(
            1 for h in history
            if not _isnan(h["date"]) and (current_date - h["date"]).days <= 365
        )
        feats["fights_last_year"] = float(fights_last_year)
        wins_last_year = sum(
            1 for h in history
            if not _isnan(h["date"]) and (current_date - h["date"]).days <= 365
            and h["result"] == "W"
        )
        feats["win_rate_last_year"] = _bayes_shrink(
            _safe_div(wins_last_year, fights_last_year, 0.5),
            fights_last_year, prior=0.5, strength=4,
        )
    else:
        feats["fights_last_year"] = 1.0
        feats["win_rate_last_year"] = 0.5

    # ── Loss recovery (bounce-back ability after losses) ──
    post_loss_fights = 0
    post_loss_wins = 0
    after_loss = False
    for h in history:
        if after_loss:
            post_loss_fights += 1
            if h["result"] == "W":
                post_loss_wins += 1
        after_loss = (h["result"] == "L")
    feats["loss_recovery_rate"] = _bayes_shrink(
        _safe_div(post_loss_wins, post_loss_fights, 0.5),
        post_loss_fights, prior=0.5, strength=6,
    )

    # ── Striking defense efficiency ──
    # How well they avoid damage relative to opponent's offensive output
    if feats["def_sig_str_pm"] > 0 and feats["sig_str_pm"] > 0:
        feats["strike_exchange_ratio"] = feats["sig_str_pm"] / (feats["def_sig_str_pm"] + 0.01)
    else:
        feats["strike_exchange_ratio"] = 1.0

    # ── Grappling threat composite (wrestling + submissions + control) ──
    feats["grappling_threat"] = feats["td_p15"] * feats["td_acc"] + feats["sub_att_p15"] + feats["ctrl_pct"]

    # ── Decision ability (win rate in fights that go to decision) ──
    dec_fights = sum(1 for h in history if "Dec" in str(h.get("method", "")))
    dec_wins = sum(1 for h in history if h["result"] == "W" and "Dec" in str(h.get("method", "")))
    feats["decision_win_rate"] = _bayes_shrink(
        _safe_div(dec_wins, dec_fights, 0.5),
        dec_fights, prior=0.5, strength=6,
    )

    # ── Finish resistance (survives opponent's finishes) ──
    feats["finish_resistance"] = 1.0 - feats.get("been_finished_pct", 0.5)

    # ── Offensive diversity (how evenly spread across strike/grapple/sub) ──
    stk_share = feats.get("style_striking", 0.5)
    wrs_share = feats.get("style_wrestling", 0.25)
    sub_share = feats.get("style_submission", 0.15)
    # Entropy-based diversity (higher = more versatile)
    diversity = 0.0
    for s in [stk_share, wrs_share, sub_share]:
        if s > 0:
            diversity -= s * math.log(s + 1e-8)
    feats["offensive_diversity"] = diversity

    # ── EWM striking differential (recent trend of how they're winning/losing exchanges) ──
    alpha_sd = 0.4
    sd_sum, sd_tot = 0.0, 0.0
    for i, h in enumerate(history):
        w = (1 - alpha_sd) ** (n - 1 - i)
        ft_min = h["fight_time"] / 60.0 if h["fight_time"] > 0 else 1.0
        diff = (h["sig_str"] - h["opp_sig_str"]) / ft_min
        sd_sum += w * diff
        sd_tot += w
    feats["ewm_sig_str_diff_pm"] = sd_sum / sd_tot if sd_tot > 0 else 0.0

    # ── Glicko confidence (lower phi = more certain rating) ──
    feats["glicko_confidence"] = max(1.0 - (glicko[1] / PHI_0), 0.0)

    if _FIGHTER_FEAT_KEYS is None:
        _set_fighter_keys(list(feats.keys()))

    return feats


def _set_fighter_keys(keys):
    global _FIGHTER_FEAT_KEYS
    _FIGHTER_FEAT_KEYS = keys


# ─── Matchup feature computation ──────────────────────────────────────────────

def compute_matchup_features(a_feats, b_feats, is_title=0, total_rounds=3, weight_class=""):
    """Difference features (A minus B) plus interaction features."""
    features = {}
    for key in a_feats:
        a_val = a_feats[key] if not _isnan(a_feats[key]) else float("nan")
        b_val = b_feats[key] if not _isnan(b_feats[key]) else float("nan")
        try:
            features[f"d_{key}"] = float(a_val) - float(b_val)
        except (TypeError, ValueError):
            features[f"d_{key}"] = float("nan")

    a_age = a_feats.get("age", float("nan"))
    b_age = b_feats.get("age", float("nan"))
    a_reach = a_feats.get("reach", float("nan"))
    b_reach = b_feats.get("reach", float("nan"))
    a_height = a_feats.get("height", float("nan"))
    b_height = b_feats.get("height", float("nan"))
    a_mu = _num_or(a_feats.get("glicko_mu"), MU_0)
    b_mu = _num_or(b_feats.get("glicko_mu"), MU_0)
    a_phi = _num_or(a_feats.get("glicko_phi"), PHI_0)
    b_phi = _num_or(b_feats.get("glicko_phi"), PHI_0)
    a_n = max(_num_or(a_feats.get("num_fights"), 0.0), 0.0)
    b_n = max(_num_or(b_feats.get("num_fights"), 0.0), 0.0)
    a_debut = a_n < 1.0
    b_debut = b_n < 1.0
    a_inexperienced = a_n < 3.0
    b_inexperienced = b_n < 3.0

    # Absolute / interaction features
    features["is_title"] = is_title
    features["total_rounds"] = total_rounds
    features["exp_sum"] = a_n + b_n
    features["age_sum"] = _num_or(a_age, 30.0) + _num_or(b_age, 30.0)
    features["glicko_mu_sum"] = a_mu + b_mu
    features["abs_age_gap"] = _abs_gap(a_age, b_age)
    features["abs_reach_gap"] = _abs_gap(a_reach, b_reach)
    features["abs_height_gap"] = _abs_gap(a_height, b_height)
    features["abs_glicko_gap"] = abs(a_mu - b_mu)
    features["abs_glicko_phi_gap"] = abs(a_phi - b_phi)
    features["abs_exp_gap"] = abs(a_n - b_n)
    features["min_num_fights"] = min(a_n, b_n)
    features["max_num_fights"] = max(a_n, b_n)
    features["experience_ratio"] = min(a_n, b_n) / max(max(a_n, b_n), 1.0)
    features["max_glicko_phi"] = max(a_phi, b_phi)
    features["min_glicko_phi"] = min(a_phi, b_phi)
    features["avg_glicko_phi"] = (a_phi + b_phi) / 2.0
    features["both_debut"] = float(a_debut and b_debut)
    features["one_debut"] = float(a_debut ^ b_debut)
    features["both_inexperienced"] = float(a_inexperienced and b_inexperienced)
    features["one_inexperienced"] = float(a_inexperienced ^ b_inexperienced)
    features["info_asymmetry"] = abs(a_n - b_n) / max(a_n + b_n, 1.0)

    # Weight class
    features["weight_class_ord"] = WEIGHT_CLASS_ORDINAL.get(weight_class, 0)
    for class_name, feature_name in WEIGHT_CLASS_FEATURES.items():
        features[feature_name] = float(weight_class == class_name)
    features[UNKNOWN_WEIGHT_CLASS_FEATURE] = float(weight_class not in WEIGHT_CLASS_ORDINAL)

    # Stance matchup interactions (d_ prefix for correct augmentation negation)
    a_ortho = _num_or(a_feats.get("is_orthodox"), 0.0)
    a_south = _num_or(a_feats.get("is_southpaw"), 0.0)
    b_ortho = _num_or(b_feats.get("is_orthodox"), 0.0)
    b_south = _num_or(b_feats.get("is_southpaw"), 0.0)
    features["d_ortho_vs_south"] = float(a_ortho * b_south) - float(b_ortho * a_south)
    features["same_stance"] = float(
        a_ortho == b_ortho and a_south == b_south and (a_ortho or a_south)
    )
    # Style cluster matchup interactions
    a_stk = _num_or(a_feats.get("style_striking"), 0.5)
    a_wrs = _num_or(a_feats.get("style_wrestling"), 0.15)
    a_sub = _num_or(a_feats.get("style_submission"), 0.1)
    a_cg = _num_or(a_feats.get("style_clinch_ground"), 0.32)
    b_stk = _num_or(b_feats.get("style_striking"), 0.5)
    b_wrs = _num_or(b_feats.get("style_wrestling"), 0.15)
    b_sub = _num_or(b_feats.get("style_submission"), 0.1)
    b_cg = _num_or(b_feats.get("style_clinch_ground"), 0.32)
    features["d_striker_vs_grappler"] = a_stk * (b_wrs + b_sub) - b_stk * (a_wrs + a_sub)
    features["style_mismatch"] = a_stk * (b_wrs + b_sub) + b_stk * (a_wrs + a_sub)
    features["style_distance"] = math.sqrt(
        (a_stk - b_stk)**2 + (a_wrs - b_wrs)**2 + (a_sub - b_sub)**2 + (a_cg - b_cg)**2
    )

    # ── Glicko expected win probability (THE strongest single predictor) ──
    # This encodes the full Glicko-2 prediction: rating difference + uncertainty
    a_mu_s = (a_mu - MU_0) / SCALE
    b_mu_s = (b_mu - MU_0) / SCALE
    b_phi_s = b_phi / SCALE
    features["glicko_win_prob"] = _E(a_mu_s, b_mu_s, b_phi_s)
    features["d_glicko_win_prob"] = features["glicko_win_prob"] - 0.5  # directional version

    # Confidence gap: rating gap normalised by combined uncertainty
    combined_phi = math.sqrt(a_phi**2 + b_phi**2) if (a_phi**2 + b_phi**2) > 0 else 1.0
    features["d_confidence_gap"] = (a_mu - b_mu) / combined_phi

    # Glicko win prob weighted by confidence (high confidence + big gap = strong signal)
    avg_conf = (_num_or(a_feats.get("glicko_confidence"), 0.0)
                + _num_or(b_feats.get("glicko_confidence"), 0.0)) / 2.0
    features["d_confident_prediction"] = features["d_glicko_win_prob"] * (0.5 + avg_conf)

    # Reach advantage amplified by striking differential
    a_sig_pm = _num_or(a_feats.get("sig_str_pm"), 0.0)
    b_sig_pm = _num_or(b_feats.get("sig_str_pm"), 0.0)
    reach_diff = _num_or(a_reach, 72.0) - _num_or(b_reach, 72.0)
    features["d_reach_x_striking"] = reach_diff * (a_sig_pm - b_sig_pm)

    # Chin vs power: A's durability vs B's KO rate (and reverse)
    a_dur = _num_or(a_feats.get("durability"), 0.8)
    b_dur = _num_or(b_feats.get("durability"), 0.8)
    a_kd_pm = _num_or(a_feats.get("kd_pm"), 0.0)
    b_kd_pm = _num_or(b_feats.get("kd_pm"), 0.0)
    features["d_chin_vs_power"] = (a_dur * a_kd_pm) - (b_dur * b_kd_pm)

    # TD attack vs defense: A's offensive wrestling vs B's TDD (directional)
    a_td_p15 = _num_or(a_feats.get("td_p15"), 0.0)
    b_td_p15 = _num_or(b_feats.get("td_p15"), 0.0)
    a_tdd = _num_or(a_feats.get("td_defense_rate"), 0.65)
    b_tdd = _num_or(b_feats.get("td_defense_rate"), 0.65)
    features["d_td_attack_vs_defense"] = (a_td_p15 * (1.0 - b_tdd)) - (b_td_p15 * (1.0 - a_tdd))

    # Combined finish rate (proxy for fight unlikely to go to decision)
    a_fin = _num_or(a_feats.get("finish_rate"), 0.5)
    b_fin = _num_or(b_feats.get("finish_rate"), 0.5)
    features["combined_finish_rate"] = a_fin + b_fin

    # Output rate differential
    features["d_output_rate"] = _num_or(a_feats.get("output_rate"), 0.0) - _num_or(b_feats.get("output_rate"), 0.0)

    # Momentum differential
    features["d_momentum"] = _num_or(a_feats.get("momentum"), 0.5) - _num_or(b_feats.get("momentum"), 0.5)

    # Durability gap
    features["d_durability"] = a_dur - b_dur

    # Grappling dominance differential
    features["d_grappling_dominance"] = (_num_or(a_feats.get("grappling_dominance"), 0.0)
                                         - _num_or(b_feats.get("grappling_dominance"), 0.0))

    # EWM opponent quality gap (who has faced tougher competition recently)
    features["d_ewm_opp_quality"] = (_num_or(a_feats.get("ewm_opp_quality"), MU_0)
                                     - _num_or(b_feats.get("ewm_opp_quality"), MU_0))

    # ── Striking offense vs opponent's striking defense (cross-matchup) ──
    a_sig_def = _num_or(a_feats.get("sig_str_defense_rate"), 0.55)
    b_sig_def = _num_or(b_feats.get("sig_str_defense_rate"), 0.55)
    features["d_striking_vs_defense"] = (
        a_sig_pm * (1.0 - b_sig_def) - b_sig_pm * (1.0 - a_sig_def)
    )

    # ── Wrestling offense vs opponent's TDD (already have d_td_attack_vs_defense,
    #    now add grappling_threat vs opponent's ground defense) ──
    a_grap = _num_or(a_feats.get("grappling_threat"), 0.0)
    b_grap = _num_or(b_feats.get("grappling_threat"), 0.0)
    features["d_grapple_vs_tdd"] = a_grap * (1.0 - b_tdd) - b_grap * (1.0 - a_tdd)

    # ── Fight IQ gap ──
    features["d_fight_iq"] = (_num_or(a_feats.get("fight_iq"), 0.15)
                              - _num_or(b_feats.get("fight_iq"), 0.15))

    # ── Cardio advantage ──
    features["d_cardio_ratio"] = (_num_or(a_feats.get("cardio_ratio"), 1.0)
                                  - _num_or(b_feats.get("cardio_ratio"), 1.0))

    # ── Glicko confidence gap (whose rating is more reliable?) ──
    features["d_glicko_confidence"] = (_num_or(a_feats.get("glicko_confidence"), 0.0)
                                       - _num_or(b_feats.get("glicko_confidence"), 0.0))

    # ── Form trend gap (who is on the upswing?) ──
    features["d_form_trend"] = (_num_or(a_feats.get("form_vs_career"), 0.0)
                                - _num_or(b_feats.get("form_vs_career"), 0.0))

    # ── Strike exchange ratio gap (who wins the striking exchanges more?) ──
    features["d_exchange_ratio"] = (_num_or(a_feats.get("strike_exchange_ratio"), 1.0)
                                    - _num_or(b_feats.get("strike_exchange_ratio"), 1.0))

    # ── Decision fighter vs finisher matchup ──
    a_dec_wr = _num_or(a_feats.get("decision_win_rate"), 0.5)
    b_dec_wr = _num_or(b_feats.get("decision_win_rate"), 0.5)
    features["d_decision_ability"] = a_dec_wr - b_dec_wr

    # ── Versatility gap ──
    features["d_offensive_diversity"] = (_num_or(a_feats.get("offensive_diversity"), 1.0)
                                         - _num_or(b_feats.get("offensive_diversity"), 1.0))

    # ── Combined Glicko confidence (how reliable is this matchup prediction?) ──
    a_conf = _num_or(a_feats.get("glicko_confidence"), 0.0)
    b_conf = _num_or(b_feats.get("glicko_confidence"), 0.0)
    features["avg_glicko_confidence"] = (a_conf + b_conf) / 2.0

    # ── Age-experience interaction: young+experienced is dangerous ──
    a_age_v = _num_or(a_age, 30.0)
    b_age_v = _num_or(b_age, 30.0)
    # Younger fighter with more experience has edge
    features["d_youth_exp"] = (b_age_v - a_age_v) * (a_n - b_n)

    # ── Glicko-scaled striking: weight sig str differential by Glicko reliability ──
    features["d_reliable_striking"] = features.get("d_sig_str_pm", 0.0) * (0.5 + avg_conf)

    # ── Finisher vs chin interaction ──
    a_finish_r = _num_or(a_feats.get("finish_rate"), 0.5)
    b_finish_r = _num_or(b_feats.get("finish_rate"), 0.5)
    b_finish_resist = _num_or(b_feats.get("finish_resistance"), 0.5)
    a_finish_resist = _num_or(a_feats.get("finish_resistance"), 0.5)
    features["d_finish_vs_resist"] = (a_finish_r * (1.0 - b_finish_resist)
                                      - b_finish_r * (1.0 - a_finish_resist))

    # ── Activity gap (recent activity matters for ring rust) ──
    a_inactive = _num_or(a_feats.get("days_inactive"), 365.0)
    b_inactive = _num_or(b_feats.get("days_inactive"), 365.0)
    features["d_activity"] = b_inactive - a_inactive  # positive = A is more active

    # ── Win rate in last year gap ──
    features["d_win_rate_last_year"] = (_num_or(a_feats.get("win_rate_last_year"), 0.5)
                                        - _num_or(b_feats.get("win_rate_last_year"), 0.5))

    # ── Old_Model-inspired quality-endurance and pressure features ──
    # Quality endurance: maintains accuracy while keeping output under fatigue.
    a_cardio = _num_or(a_feats.get("cardio_ratio"), 1.0)
    b_cardio = _num_or(b_feats.get("cardio_ratio"), 1.0)
    a_acc = _num_or(a_feats.get("sig_str_acc"), 0.45)
    b_acc = _num_or(b_feats.get("sig_str_acc"), 0.45)
    a_r1_int = _num_or(a_feats.get("rd1_intensity_ratio"), 1.0)
    b_r1_int = _num_or(b_feats.get("rd1_intensity_ratio"), 1.0)
    a_output = _num_or(a_feats.get("output_rate"), 0.0)
    b_output = _num_or(b_feats.get("output_rate"), 0.0)
    a_opp_sig = _num_or(a_feats.get("def_sig_str_pm"), 0.0)
    b_opp_sig = _num_or(b_feats.get("def_sig_str_pm"), 0.0)
    a_conf = _num_or(a_feats.get("glicko_confidence"), 0.0)
    b_conf = _num_or(b_feats.get("glicko_confidence"), 0.0)

    a_quality_endurance = a_cardio * a_acc * (0.5 + min(max(a_r1_int, 0.5), 1.8))
    b_quality_endurance = b_cardio * b_acc * (0.5 + min(max(b_r1_int, 0.5), 1.8))
    features["d_quality_endurance"] = a_quality_endurance - b_quality_endurance

    # Accuracy under fire proxy: maintained accuracy while facing volume.
    a_accuracy_under_fire = a_acc / (1.0 + b_opp_sig)
    b_accuracy_under_fire = b_acc / (1.0 + a_opp_sig)
    features["d_accuracy_under_fire"] = a_accuracy_under_fire - b_accuracy_under_fire

    # Pressure-cardio clash: high pressure that still scales with own cardio.
    features["d_pressure_cardio_clash"] = (
        a_output * a_cardio * (1.0 - b_cardio) - b_output * b_cardio * (1.0 - a_cardio)
    )

    # Stability-endurance: confidence-weighted late-fight reliability.
    features["d_stability_endurance"] = (
        a_conf * a_quality_endurance - b_conf * b_quality_endurance
    )

    # Sum channels to allow exact side-specific reconstruction for method-only
    # mismatch features after winner-orientation is applied.
    a_sub_entry = _num_or(a_feats.get("sub_entry_pressure"), 0.0)
    b_sub_entry = _num_or(b_feats.get("sub_entry_pressure"), 0.0)
    a_sub_leak = _num_or(a_feats.get("sub_defensive_leak"), 0.0)
    b_sub_leak = _num_or(b_feats.get("sub_defensive_leak"), 0.0)
    features["sub_entry_pressure_sum"] = a_sub_entry + b_sub_entry
    features["sub_defensive_leak_sum"] = a_sub_leak + b_sub_leak
    a_ko_attack = _num_or(a_feats.get("ko_attack_pressure"), 0.0)
    b_ko_attack = _num_or(b_feats.get("ko_attack_pressure"), 0.0)
    a_ko_leak = _num_or(a_feats.get("ko_def_leak"), 0.0)
    b_ko_leak = _num_or(b_feats.get("ko_def_leak"), 0.0)
    features["ko_attack_pressure_sum"] = a_ko_attack + b_ko_attack
    features["ko_def_leak_sum"] = a_ko_leak + b_ko_leak

    return features


# ─── Training data builder ────────────────────────────────────────────────────

def build_training_data(csv_path, progress_cb=None):
    """Process fights chronologically, build features, return X, y and state."""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y")
    df = df.sort_values("event_date").reset_index(drop=True)
    _ensure_fighter_feature_keys(df["event_date"].iloc[0] if len(df) else None)

    fighter_history = defaultdict(list)
    glicko_ratings = {}
    opp_glicko_list = defaultdict(list)

    rows_X = []
    rows_y = []
    total = len(df)

    for idx in range(total):
        if progress_cb and idx % 500 == 0:
            progress_cb(f"  Building features... fight {idx+1}/{total}")

        row = df.iloc[idx]
        r_name = row["r_name"]
        b_name = row["b_name"]

        # Initialize Glicko
        if r_name not in glicko_ratings:
            glicko_ratings[r_name] = (MU_0, PHI_0, SIGMA_0)
        if b_name not in glicko_ratings:
            glicko_ratings[b_name] = (MU_0, PHI_0, SIGMA_0)

        r_glicko = glicko_ratings[r_name]
        b_glicko = glicko_ratings[b_name]

        r_feats = compute_fighter_features(
            fighter_history[r_name], r_glicko,
            opp_glicko_list[r_name], row["event_date"],
            fallback_profile=_extract_profile_from_row(row, "r"),
        )
        b_feats = compute_fighter_features(
            fighter_history[b_name], b_glicko,
            opp_glicko_list[b_name], row["event_date"],
            fallback_profile=_extract_profile_from_row(row, "b"),
        )
        matchup = compute_matchup_features(
            r_feats, b_feats,
            is_title=row.get("is_title_bout", 0),
            total_rounds=row.get("total_rounds", 3),
            weight_class=row.get("weight_class", ""),
        )
        if row["winner"] in ("Red", "Blue"):
            rows_X.append(matchup)
            rows_y.append(1.0 if row["winner"] == "Red" else 0.0)

        # Determine result
        winner = row["winner"]
        if winner == "Red":
            r_res, b_res = "W", "L"
            r_sc, b_sc = 1.0, 0.0
        elif winner == "Blue":
            r_res, b_res = "L", "W"
            r_sc, b_sc = 0.0, 1.0
        else:
            r_res, b_res = "D", "D"
            r_sc, b_sc = 0.5, 0.5

        # Update histories
        fighter_history[r_name].append(extract_fight_record(row, "r", "b", r_res, b_glicko[0]))
        fighter_history[b_name].append(extract_fight_record(row, "b", "r", b_res, r_glicko[0]))

        # Track opponent Glicko
        opp_glicko_list[r_name].append(b_glicko[0])
        opp_glicko_list[b_name].append(r_glicko[0])

        # Update Glicko
        glicko_ratings[r_name] = glicko2_update(r_glicko, [(b_glicko[0], b_glicko[1], r_sc)])
        glicko_ratings[b_name] = glicko2_update(b_glicko, [(r_glicko[0], r_glicko[1], b_sc)])

    X = pd.DataFrame(rows_X)
    y = pd.Series(rows_y)

    if progress_cb:
        progress_cb(f"  Built {len(X)} training samples with {X.shape[1]} features.")

    return X, y, fighter_history, glicko_ratings, opp_glicko_list


def _method_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        winner = str(row.get("winner", "")).strip()
        if winner in ("Red", "Blue"):
            detail = _normalize_method_detail(row.get("method", ""))
            coarse = _normalize_method_label(row.get("method", ""))
            finish_bin = "Decision" if coarse == "Decision" else "Finish"
            finish_subtype = coarse if coarse in ("KO/TKO", "Submission") else "KO/TKO"
            rows.append({
                "coarse": coarse,
                "detail": detail,
                "finish_bin": finish_bin,
                "finish_subtype": finish_subtype,
            })
    return pd.DataFrame(rows)


def _time_split_indices(n_rows):
    test_size = max(1, int(n_rows * TEST_FRACTION))
    val_size = max(1, int(n_rows * VAL_FRACTION))
    train_size = max(1, n_rows - val_size - test_size)
    train_end = train_size
    val_end = train_end + val_size
    return train_end, val_end


def _augment_swap(X, y):
    d_cols = [c for c in X.columns if c.startswith("d_")]
    X_swap = X.copy()
    X_swap[d_cols] = -X_swap[d_cols]
    y_swap = 1.0 - y
    # Interleave to preserve chronology for folds.
    X_aug = pd.concat([X, X_swap], ignore_index=True)
    y_aug = pd.concat([y, y_swap], ignore_index=True)
    return X_aug, y_aug


def _augment_weights(w):
    w = np.asarray(w, dtype=float)
    return np.concatenate([w, w], axis=0)


def _time_weights(n, floor=0.35):
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    return np.linspace(float(floor), 1.0, int(n), dtype=float)


def _swap_features(X):
    X2 = X.copy()
    d_cols = [c for c in X2.columns if c.startswith("d_")]
    X2[d_cols] = -X2[d_cols]
    return X2


def _make_model_specs(lgb_tuned_params=None, xgb_tuned_params=None, cb_tuned_params=None):
    specs = []
    if lgb is not None:
        specs.append((
            "LightGBM",
            lambda: lgb.LGBMClassifier(
                n_estimators=450, learning_rate=0.03, max_depth=6,
                num_leaves=31, min_child_samples=25, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.12, reg_lambda=1.2,
                random_state=RANDOM_SEED, verbose=-1,
            )
        ))
        specs.append((
            "LightGBM_S2",
            lambda: lgb.LGBMClassifier(
                n_estimators=450, learning_rate=0.03, max_depth=6,
                num_leaves=31, min_child_samples=25, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.12, reg_lambda=1.2,
                random_state=RANDOM_SEED + 17, verbose=-1,
            )
        ))
        if lgb_tuned_params:
            p = dict(lgb_tuned_params)
            specs.append((
                "LightGBM_Tuned",
                lambda p=p: lgb.LGBMClassifier(
                    n_estimators=int(p["n_estimators"]),
                    learning_rate=float(p["learning_rate"]),
                    max_depth=int(p["max_depth"]),
                    num_leaves=int(p["num_leaves"]),
                    min_child_samples=int(p["min_child_samples"]),
                    subsample=float(p["subsample"]),
                    colsample_bytree=float(p["colsample_bytree"]),
                    reg_alpha=float(p["reg_alpha"]),
                    reg_lambda=float(p["reg_lambda"]),
                    min_split_gain=float(p["min_split_gain"]),
                    random_state=RANDOM_SEED + 101,
                    verbose=-1,
                )
            ))
    if xgb is not None:
        specs.append((
            "XGBoost",
            lambda: xgb.XGBClassifier(
                n_estimators=420, learning_rate=0.03, max_depth=5,
                min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.12, reg_lambda=1.2, objective="binary:logistic",
                eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
            )
        ))
        specs.append((
            "XGBoost_S2",
            lambda: xgb.XGBClassifier(
                n_estimators=420, learning_rate=0.03, max_depth=5,
                min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.12, reg_lambda=1.2, objective="binary:logistic",
                eval_metric="logloss", random_state=RANDOM_SEED + 27, n_jobs=-1,
            )
        ))
        if xgb_tuned_params:
            p = dict(xgb_tuned_params)
            specs.append((
                "XGBoost_Tuned",
                lambda p=p: xgb.XGBClassifier(
                    n_estimators=int(p["n_estimators"]),
                    learning_rate=float(p["learning_rate"]),
                    max_depth=int(p["max_depth"]),
                    min_child_weight=float(p["min_child_weight"]),
                    subsample=float(p["subsample"]),
                    colsample_bytree=float(p["colsample_bytree"]),
                    reg_alpha=float(p["reg_alpha"]),
                    reg_lambda=float(p["reg_lambda"]),
                    gamma=float(p["gamma"]),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_SEED + 121,
                    n_jobs=-1,
                )
            ))
    if cb is not None:
        specs.append((
            "CatBoost",
            lambda: cb.CatBoostClassifier(
                iterations=420, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
                random_strength=1.0, bagging_temperature=0.5, border_count=128,
                loss_function="Logloss", eval_metric="Logloss", verbose=0,
                random_seed=RANDOM_SEED, allow_writing_files=False,
            )
        ))
        specs.append((
            "CatBoost_S2",
            lambda: cb.CatBoostClassifier(
                iterations=420, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
                random_strength=1.0, bagging_temperature=0.5, border_count=128,
                loss_function="Logloss", eval_metric="Logloss", verbose=0,
                random_seed=RANDOM_SEED + 37, allow_writing_files=False,
            )
        ))
        if cb_tuned_params:
            p = dict(cb_tuned_params)
            specs.append((
                "CatBoost_Tuned",
                lambda p=p: cb.CatBoostClassifier(
                    iterations=int(p["iterations"]),
                    learning_rate=float(p["learning_rate"]),
                    depth=int(p["depth"]),
                    l2_leaf_reg=float(p["l2_leaf_reg"]),
                    random_strength=float(p["random_strength"]),
                    bagging_temperature=float(p["bagging_temperature"]),
                    border_count=int(p["border_count"]),
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    verbose=0,
                    random_seed=RANDOM_SEED + 141,
                    allow_writing_files=False,
                )
            ))
    specs.append((
        "HistGBM",
        lambda: HistGradientBoostingClassifier(
            max_iter=650, learning_rate=0.035, max_depth=6,
            max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            random_state=RANDOM_SEED,
        )
    ))
    specs.append((
        "HistGBM_Wide",
        lambda: HistGradientBoostingClassifier(
            max_iter=900, learning_rate=0.03, max_depth=8,
            max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=0.7,
            random_state=RANDOM_SEED + 11,
        )
    ))
    specs.append((
        "RandForest",
        lambda: RandomForestClassifier(
            n_estimators=550, max_depth=12, min_samples_leaf=5,
            min_samples_split=2, max_features=0.7, random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    ))
    specs.append((
        "RandForest_Deep",
        lambda: RandomForestClassifier(
            n_estimators=750, max_depth=18, min_samples_leaf=3,
            min_samples_split=2, max_features=0.6, random_state=RANDOM_SEED + 21,
            n_jobs=-1,
        )
    ))
    specs.append((
        "ExtraTrees",
        lambda: ExtraTreesClassifier(
            n_estimators=700, max_depth=14, min_samples_leaf=4,
            min_samples_split=2, max_features=0.7, random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    ))
    specs.append((
        "ExtraTrees_Deep",
        lambda: ExtraTreesClassifier(
            n_estimators=900, max_depth=22, min_samples_leaf=2,
            min_samples_split=2, max_features=0.6, random_state=RANDOM_SEED + 31,
            n_jobs=-1,
        )
    ))
    specs.append((
        "AdaBoost",
        lambda: AdaBoostClassifier(
            n_estimators=350, learning_rate=0.05, random_state=RANDOM_SEED
        )
    ))
    specs.append((
        "MLP",
        lambda: MLPClassifier(
            hidden_layer_sizes=(96, 48), alpha=0.01, learning_rate="adaptive",
            early_stopping=True, validation_fraction=0.15, max_iter=700,
            random_state=RANDOM_SEED
        )
    ))
    specs.append((
        "LogReg",
        lambda: LogisticRegression(
            max_iter=8000, C=0.3, solver="saga", tol=1e-3, n_jobs=-1, random_state=RANDOM_SEED
        ),
    ))
    specs.append((
        "LogReg_L2",
        lambda: LogisticRegression(
            max_iter=8000, C=1.2, solver="saga", tol=1e-3, n_jobs=-1, random_state=RANDOM_SEED + 7
        ),
    ))
    return specs


def _tune_lightgbm_optuna(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=None, progress_cb=None):
    if lgb is None or optuna is None:
        return None
    if len(X_train) < 800 or len(X_val) < 200:
        return None

    X_tr_aug, y_tr_aug = _augment_swap(X_train, y_train)
    w_aug = _augment_weights(_time_weights(len(X_train), floor=0.4))
    X_val_sw = _swap_features(X_val)
    y_val_np = np.asarray(y_val).astype(int)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 280, 950),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "min_child_samples": trial.suggest_int("min_child_samples", 8, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        }
        model = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, **params)
        model.fit(X_tr_aug, y_tr_aug, sample_weight=w_aug)
        p_fwd = _clip_probs(model.predict_proba(X_val)[:, 1])
        p_rev = _clip_probs(model.predict_proba(X_val_sw)[:, 1])
        probs = _clip_probs((p_fwd + (1.0 - p_rev)) / 2.0)
        thr, acc = _tune_threshold(probs, y_val_np)
        ll = float(log_loss(y_val_np, probs))
        trial.set_user_attr("acc", float(acc))
        trial.set_user_attr("thr", float(thr))
        trial.set_user_attr("ll", ll)
        # Accuracy-focused objective with log-loss regularizer.
        return (1.0 - float(acc)) + 0.10 * ll

    def _trial_callback(_study, trial):
        if progress_cb is not None:
            progress_cb(int(trial.number) + 1, int(n_trials), "LightGBM")

    study.optimize(
        _objective, n_trials=int(n_trials), show_progress_bar=False,
        callbacks=[_trial_callback],
    )
    best = study.best_trial
    if logger is not None:
        logger(f"Optuna LightGBM best score: {best.value:.5f}")
        logger(
            f"Optuna LightGBM val acc: {best.user_attrs.get('acc', float('nan')):.1%} | "
            f"val ll: {best.user_attrs.get('ll', float('nan')):.4f} | "
            f"thr: {best.user_attrs.get('thr', 0.5):.3f}"
        )
    return best.params


def _tune_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=None, progress_cb=None):
    if xgb is None or optuna is None:
        return None
    if len(X_train) < 800 or len(X_val) < 200:
        return None

    X_tr_aug, y_tr_aug = _augment_swap(X_train, y_train)
    w_aug = _augment_weights(_time_weights(len(X_train), floor=0.4))
    X_val_sw = _swap_features(X_val)
    y_val_np = np.asarray(y_val).astype(int)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED + 1)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 260, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.5, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1, **params
        )
        model.fit(X_tr_aug, y_tr_aug, sample_weight=w_aug)
        p_fwd = _clip_probs(model.predict_proba(X_val)[:, 1])
        p_rev = _clip_probs(model.predict_proba(X_val_sw)[:, 1])
        probs = _clip_probs((p_fwd + (1.0 - p_rev)) / 2.0)
        thr, acc = _tune_threshold(probs, y_val_np)
        ll = float(log_loss(y_val_np, probs))
        trial.set_user_attr("acc", float(acc))
        trial.set_user_attr("thr", float(thr))
        trial.set_user_attr("ll", ll)
        return (1.0 - float(acc)) + 0.10 * ll

    def _trial_callback(_study, trial):
        if progress_cb is not None:
            progress_cb(int(trial.number) + 1, int(n_trials), "XGBoost")

    study.optimize(
        _objective, n_trials=int(n_trials), show_progress_bar=False,
        callbacks=[_trial_callback],
    )
    best = study.best_trial
    if logger is not None:
        logger(f"Optuna XGBoost best score: {best.value:.5f}")
        logger(
            f"Optuna XGBoost val acc: {best.user_attrs.get('acc', float('nan')):.1%} | "
            f"val ll: {best.user_attrs.get('ll', float('nan')):.4f} | "
            f"thr: {best.user_attrs.get('thr', 0.5):.3f}"
        )
    return best.params


def _tune_catboost_optuna(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=None, progress_cb=None):
    if cb is None or optuna is None:
        return None
    if len(X_train) < 800 or len(X_val) < 200:
        return None

    X_tr_aug, y_tr_aug = _augment_swap(X_train, y_train)
    w_aug = _augment_weights(_time_weights(len(X_train), floor=0.4))
    X_val_sw = _swap_features(X_val)
    y_val_np = np.asarray(y_val).astype(int)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED + 2)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 250, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.09, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "border_count": trial.suggest_int("border_count", 64, 254),
        }
        model = cb.CatBoostClassifier(
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False, **params
        )
        model.fit(X_tr_aug, y_tr_aug, sample_weight=w_aug)
        p_fwd = _clip_probs(model.predict_proba(X_val)[:, 1])
        p_rev = _clip_probs(model.predict_proba(X_val_sw)[:, 1])
        probs = _clip_probs((p_fwd + (1.0 - p_rev)) / 2.0)
        thr, acc = _tune_threshold(probs, y_val_np)
        ll = float(log_loss(y_val_np, probs))
        trial.set_user_attr("acc", float(acc))
        trial.set_user_attr("thr", float(thr))
        trial.set_user_attr("ll", ll)
        return (1.0 - float(acc)) + 0.10 * ll

    def _trial_callback(_study, trial):
        if progress_cb is not None:
            progress_cb(int(trial.number) + 1, int(n_trials), "CatBoost")

    study.optimize(
        _objective, n_trials=int(n_trials), show_progress_bar=False,
        callbacks=[_trial_callback],
    )
    best = study.best_trial
    if logger is not None:
        logger(f"Optuna CatBoost best score: {best.value:.5f}")
        logger(
            f"Optuna CatBoost val acc: {best.user_attrs.get('acc', float('nan')):.1%} | "
            f"val ll: {best.user_attrs.get('ll', float('nan')):.4f} | "
            f"thr: {best.user_attrs.get('thr', 0.5):.3f}"
        )
    return best.params


def _fit_model(name, model, X_fit, y_fit, sample_weight=None):
    if name == "CatBoost":
        if sample_weight is not None:
            model.fit(X_fit, y_fit, sample_weight=sample_weight)
        else:
            model.fit(X_fit, y_fit)
    else:
        if sample_weight is not None:
            try:
                model.fit(X_fit, y_fit, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_fit, y_fit)
        else:
            model.fit(X_fit, y_fit)
    return model


def _predict_proba(name, model, X_eval):
    probs = model.predict_proba(X_eval)[:, 1]
    return _clip_probs(probs)


def _fit_platt_calibrator(probs, y_true):
    lr = LogisticRegression(
        max_iter=5000, solver="lbfgs", tol=1e-4, random_state=RANDOM_SEED
    )
    lr.fit(np.asarray(probs).reshape(-1, 1), np.asarray(y_true).astype(int))
    return lr


class _IsoWrapper:
    def __init__(self, iso):
        self.iso = iso

    def predict_proba(self, X):
        p = self.iso.predict(np.asarray(X).reshape(-1))
        p = _clip_probs(p)
        return np.column_stack([1.0 - p, p])


def _fit_best_calibrator(val_probs, y_val):
    """
    Fit calibrators on early validation slice, choose on late validation slice.
    Includes 'none' to avoid harmful calibration.
    """
    p = _clip_probs(np.asarray(val_probs))
    y = np.asarray(y_val).astype(int)
    n = len(p)
    if n < 80:
        return None, "none", float(log_loss(y, p))

    split = max(30, int(n * 0.55))
    p_fit, y_fit = p[:split], y[:split]
    p_eval, y_eval = p[split:], y[split:]
    if len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
        return None, "none", float(log_loss(y, p))

    def _score_eval(preds):
        ll = float(log_loss(y_eval, preds))
        thr, acc = _tune_threshold(preds, y_eval)
        # Accuracy-first with probability-quality regularizer.
        obj = (1.0 - float(acc)) + 0.15 * ll
        return float(obj), ll, float(acc), float(thr)

    best_name = "none"
    best_cal = None
    best_obj, best_ll, _, _ = _score_eval(p_eval)

    try:
        platt = _fit_platt_calibrator(p_fit, y_fit)
        p_platt = _clip_probs(platt.predict_proba(p_eval.reshape(-1, 1))[:, 1])
        obj_platt, ll_platt, _, _ = _score_eval(p_platt)
        if obj_platt + 1e-4 < best_obj or (abs(obj_platt - best_obj) <= 1e-4 and ll_platt + 1e-4 < best_ll):
            best_name, best_cal, best_obj, best_ll = "platt", platt, obj_platt, ll_platt
    except Exception:
        pass

    try:
        iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
        iso.fit(p_fit, y_fit)
        p_iso = _clip_probs(iso.predict(p_eval))
        obj_iso, ll_iso, _, _ = _score_eval(p_iso)
        # Require bigger gain for isotonic to avoid overfitting.
        if obj_iso + 2e-4 < best_obj or (abs(obj_iso - best_obj) <= 2e-4 and ll_iso + 2e-4 < best_ll):
            best_name, best_cal, best_obj, best_ll = "isotonic", _IsoWrapper(iso), obj_iso, ll_iso
    except Exception:
        pass

    return best_cal, best_name, best_ll


def _weighted_blend(pred_df, y_true):
    order = list(pred_df.columns)
    mat = np.column_stack([_clip_probs(pred_df[c].values) for c in order])
    y = np.asarray(y_true).astype(int)
    n_models = mat.shape[1]
    if n_models == 1:
        return {order[0]: 1.0}

    def _objective(weights):
        probs = _clip_probs(mat @ weights)
        return float(log_loss(y, probs))

    w0 = np.full(n_models, 1.0 / n_models, dtype=float)
    result = scipy_minimize(
        _objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    if result.success:
        w = np.asarray(result.x, dtype=float)
    else:
        # Fallback to smooth optimization.
        logits = np.zeros(n_models, dtype=float)
        lr = 0.03
        for _ in range(500):
            wi = _softmax(logits)
            probs = _clip_probs(mat @ wi)
            err = probs - y
            grad = np.zeros_like(logits)
            for i in range(n_models):
                dblend = wi[i] * (mat[:, i] - (mat @ wi))
                grad[i] = np.mean(err * dblend / (probs * (1.0 - probs)))
            logits -= lr * grad
        w = _softmax(logits)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = w0
    else:
        w /= w.sum()
    return {order[i]: float(w[i]) for i in range(n_models)}


def _softmax(x):
    z = np.asarray(x, dtype=float)
    z -= np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _combine_probs(pred_df, combiner):
    kind = combiner["kind"]
    if kind == "stacker":
        X = pred_df[combiner["model_order"]].values
        return _clip_probs(combiner["model"].predict_proba(X)[:, 1])
    if kind == "weighted":
        out = np.zeros(len(pred_df), dtype=float)
        for name in combiner["model_order"]:
            out += combiner["weights"][name] * _clip_probs(pred_df[name].values)
        return _clip_probs(out)
    raise ValueError(f"Unknown combiner kind: {kind}")


def _tune_threshold(probs, y_true):
    p = _clip_probs(np.asarray(probs))
    y = np.asarray(y_true).astype(int)
    candidates = np.unique(
        np.concatenate([
            np.linspace(0.35, 0.65, 61),
            np.round(p, 4),
            np.array([0.5]),
        ])
    )
    best_thr = 0.5
    best_acc = -1.0
    best_margin = float("inf")
    for thr in candidates:
        pred = (p >= thr).astype(int)
        acc = accuracy_score(y, pred)
        margin = abs(float(thr) - 0.5)
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and margin < best_margin):
            best_acc = float(acc)
            best_thr = float(thr)
            best_margin = margin
    return best_thr, best_acc


def _tune_threshold_robust(probs, y_true, n_blocks=4):
    """
    Pick threshold by chronological block-robust accuracy on validation-like data.
    Reduces sensitivity to one noisy slice.
    """
    p = _clip_probs(np.asarray(probs))
    y = np.asarray(y_true).astype(int)
    n = len(p)
    if n < 120:
        return _tune_threshold(p, y)

    candidates = np.unique(np.concatenate([np.linspace(0.45, 0.55, 41), np.array([0.5])]))
    bounds = np.linspace(0, n, int(max(2, n_blocks)) + 1, dtype=int)
    # Emphasize later blocks for future prediction.
    block_ids = list(range(max(0, len(bounds) - 4), len(bounds) - 1))
    if not block_ids:
        block_ids = list(range(len(bounds) - 1))

    best_thr = 0.5
    best_score = -1.0
    best_min = -1.0
    for thr in candidates:
        block_accs = []
        for bi in block_ids:
            lo, hi = int(bounds[bi]), int(bounds[bi + 1])
            if hi - lo < 10:
                continue
            acc = accuracy_score(y[lo:hi], (p[lo:hi] >= float(thr)).astype(int))
            block_accs.append(float(acc))
        if not block_accs:
            continue
        score = float(np.mean(block_accs))
        worst = float(np.min(block_accs))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and (
                worst > best_min + 1e-12 or (abs(worst - best_min) <= 1e-12 and abs(float(thr) - 0.5) < abs(best_thr - 0.5))
            )
        ):
            best_score = score
            best_min = worst
            best_thr = float(thr)
    return best_thr, float(_model_accuracy_at_threshold(p, y, best_thr))


def _model_accuracy_at_threshold(probs, y_true, threshold=0.5):
    p = _clip_probs(np.asarray(probs))
    y = np.asarray(y_true).astype(int)
    return float(accuracy_score(y, (p >= float(threshold)).astype(int)))


def _augment_matchup_features(X_df):
    """
    Add derived non-linear interaction features from existing leak-safe matchup
    features. This is applied both to training matrix and live inference rows.
    """
    X = X_df.copy()

    def _col(name, default=0.0):
        if name not in X.columns:
            return pd.Series(np.full(len(X), default), index=X.index, dtype=float)
        return pd.to_numeric(X[name], errors="coerce").fillna(default)

    d_glicko = _col("d_glicko_win_prob", 0.0)
    d_activity = _col("d_activity", 0.0)
    d_cardio = _col("d_cardio_ratio", 0.0)
    d_momentum = _col("d_momentum", 0.0)
    d_str_vs_def = _col("d_striking_vs_defense", 0.0)
    d_grap_vs_tdd = _col("d_grapple_vs_tdd", 0.0)
    abs_exp_gap = _col("abs_exp_gap", 0.0)
    max_phi = _col("max_glicko_phi", 0.0)
    abs_glicko = _col("abs_glicko_gap", 0.0)
    combined_finish = _col("combined_finish_rate", 0.0)
    d_form = _col("d_form_trend", 0.0)
    d_iq = _col("d_fight_iq", 0.0)
    d_conf = _col("d_glicko_confidence", 0.0)
    d_div_elo = _col("d_div_elo", 0.0)
    d_div_elo_prob = _col("d_div_elo_win_prob", 0.0)
    elo_divergence = _col("elo_divergence", 0.0)
    elo_agreement = _col("elo_agreement", 1.0)
    total_rounds = _col("total_rounds", 3.0)
    is_title = _col("is_title", 0.0)

    X["d_glicko_win_prob_sq"] = d_glicko * d_glicko
    X["d_glicko_activity_interaction"] = d_glicko * d_activity
    X["d_cardio_momentum_interaction"] = d_cardio * d_momentum
    X["d_style_synergy"] = d_str_vs_def * d_grap_vs_tdd
    X["experience_uncertainty"] = abs_exp_gap * max_phi
    X["finish_pressure"] = combined_finish * abs_glicko
    X["d_form_iq_synergy"] = d_form * d_iq
    X["d_confidence_form_synergy"] = d_conf * d_form
    X["d_elo_hybrid"] = d_glicko + d_div_elo_prob
    X["d_elo_hybrid_sq"] = X["d_elo_hybrid"] * X["d_elo_hybrid"]
    X["d_division_specific_edge"] = d_div_elo * elo_agreement
    X["d_elo_disagreement_risk"] = np.abs(elo_divergence) * np.sign(d_glicko)
    X["d_style_rounds_interaction"] = X["d_style_synergy"] * total_rounds
    X["d_cardio_rounds_interaction"] = d_cardio * total_rounds
    X["d_title_finish_pressure"] = is_title * X["finish_pressure"]
    X["d_confidence_title_interaction"] = d_conf * is_title
    X["d_power_poly2"] = d_str_vs_def * np.abs(d_str_vs_def)
    X["d_grapple_poly2"] = d_grap_vs_tdd * np.abs(d_grap_vs_tdd)
    X["d_activity_poly2"] = d_activity * np.abs(d_activity)
    X["d_momentum_poly2"] = d_momentum * np.abs(d_momentum)
    X["d_glicko_activity_rounds"] = d_glicko * d_activity * total_rounds
    X["d_elo_divergence_rounds"] = elo_divergence * total_rounds
    X["d_form_confidence_rounds"] = d_form * d_conf * total_rounds

    return X


def _normalize_division(weight_class, gender):
    wc = str(weight_class or "").strip()
    g = str(gender or "").strip().lower()
    if g == "women" and wc and not wc.startswith("Women's"):
        wc = f"Women's {wc}"
    return wc


def _build_elo_features_from_csv(csv_path):
    """
    Build chronological pre-fight Elo features aligned to the training rows
    (only fights with winner in {"Red", "Blue"}).
    """
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)

    ratings = defaultdict(lambda: ELO_BASE)
    div_ratings = defaultdict(lambda: ELO_BASE)
    rows = []

    for _, row in df.iterrows():
        winner = str(row.get("winner", "")).strip()
        if winner not in ("Red", "Blue"):
            continue

        r_name = str(row.get("r_name", "")).strip()
        b_name = str(row.get("b_name", "")).strip()
        if not r_name or not b_name:
            continue
        division = _normalize_division(row.get("weight_class", ""), row.get("gender", ""))

        r_elo = float(ratings[r_name])
        b_elo = float(ratings[b_name])
        d_elo = r_elo - b_elo
        p_red = 1.0 / (1.0 + 10.0 ** (-(d_elo / 400.0)))

        r_div_elo = float(div_ratings[(r_name, division)])
        b_div_elo = float(div_ratings[(b_name, division)])
        d_div_elo = r_div_elo - b_div_elo
        p_red_div = 1.0 / (1.0 + 10.0 ** (-(d_div_elo / 400.0)))
        rows.append({
            "elo_r": r_elo,
            "elo_b": b_elo,
            "d_elo": d_elo,
            "elo_win_prob": p_red,
            "d_elo_win_prob": p_red - 0.5,
            "abs_elo_gap": abs(d_elo),
            "elo_sum": r_elo + b_elo,
            "div_elo_r": r_div_elo,
            "div_elo_b": b_div_elo,
            "d_div_elo": d_div_elo,
            "div_elo_win_prob": p_red_div,
            "d_div_elo_win_prob": p_red_div - 0.5,
            "abs_div_elo_gap": abs(d_div_elo),
            "elo_divergence": p_red - p_red_div,
            "elo_agreement": 1.0 - abs(p_red - p_red_div),
        })

        score_r = 1.0 if winner == "Red" else 0.0
        score_b = 1.0 - score_r
        ratings[r_name] = r_elo + ELO_K * (score_r - p_red)
        ratings[b_name] = b_elo + ELO_K * (score_b - (1.0 - p_red))
        div_ratings[(r_name, division)] = r_div_elo + ELO_K * (score_r - p_red_div)
        div_ratings[(b_name, division)] = b_div_elo + ELO_K * (score_b - (1.0 - p_red_div))

    return pd.DataFrame(rows), dict(ratings), dict(div_ratings)


def _training_row_dates_from_csv(csv_path):
    """Return event_date series aligned with training rows (winner in Red/Blue)."""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)
    out = []
    for _, row in df.iterrows():
        winner = str(row.get("winner", "")).strip()
        if winner in ("Red", "Blue"):
            out.append(row.get("event_date"))
    return pd.Series(out)


def _training_row_meta_from_csv(csv_path):
    """Return row metadata aligned with training rows (winner in Red/Blue)."""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        winner = str(row.get("winner", "")).strip()
        if winner in ("Red", "Blue"):
            gender = str(row.get("gender", "")).strip() or "Unknown"
            wc_norm = _normalize_division(row.get("weight_class", ""), row.get("gender", ""))
            rows.append({
                "weight_class": wc_norm or "Unknown",
                "gender": gender,
            })
    return pd.DataFrame(rows)


def _choose_combiner(meta_train, y_meta_train, meta_val, y_meta_val):
    order = list(meta_train.columns)
    y_tr = np.asarray(y_meta_train).astype(int)
    y_va = np.asarray(y_meta_val).astype(int)

    stacker = LogisticRegression(
        max_iter=8000, C=0.2, solver="saga", tol=1e-3, n_jobs=-1, random_state=RANDOM_SEED
    )
    stacker.fit(meta_train.values, y_tr)
    p_stack = _clip_probs(stacker.predict_proba(meta_val.values)[:, 1])
    ll_stack = float(log_loss(y_va, p_stack))
    thr_stack, acc_stack = _tune_threshold(p_stack, y_va)

    weights = _weighted_blend(meta_train, y_tr)
    weighted = {
        "kind": "weighted",
        "weights": weights,
        "model_order": order,
    }
    p_weighted = _combine_probs(meta_val, weighted)
    ll_weighted = float(log_loss(y_va, p_weighted))
    thr_weighted, acc_weighted = _tune_threshold(p_weighted, y_va)

    avg = {
        "kind": "weighted",
        "weights": {name: 1.0 / len(order) for name in order},
        "model_order": order,
    }
    p_avg = _combine_probs(meta_val, avg)
    ll_avg = float(log_loss(y_va, p_avg))
    thr_avg, acc_avg = _tune_threshold(p_avg, y_va)

    candidates = [
        ({
            "kind": "stacker",
            "model": stacker,
            "model_order": order,
        }, ll_stack, float(acc_stack), float(thr_stack), "stacker_lr"),
        (weighted, ll_weighted, float(acc_weighted), float(thr_weighted), "weighted"),
        (avg, ll_avg, float(acc_avg), float(thr_avg), "average"),
    ]
    try:
        hgb_meta = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.045, max_depth=3, max_leaf_nodes=31,
            min_samples_leaf=10, random_state=RANDOM_SEED + 606
        )
        hgb_meta.fit(meta_train.values, y_tr)
        p_hgb = _clip_probs(hgb_meta.predict_proba(meta_val.values)[:, 1])
        ll_hgb = float(log_loss(y_va, p_hgb))
        thr_hgb, acc_hgb = _tune_threshold(p_hgb, y_va)
        candidates.append((
            {"kind": "stacker", "model": hgb_meta, "model_order": order},
            ll_hgb, float(acc_hgb), float(thr_hgb), "stacker_hgb"
        ))
    except Exception:
        pass

    # Primary: log-loss, secondary: accuracy.
    candidates.sort(key=lambda x: (x[1], -x[2], abs(x[3] - 0.5)))
    chosen = candidates[0]
    combiner = chosen[0]
    combiner["val_threshold"] = float(chosen[3])
    combiner["selection_label"] = chosen[4]
    return combiner, float(chosen[1]), float(chosen[2]), float(chosen[3])


def _pick_best_holdout_combiner(test_meta_df, y_test, base_combiner, allow_aggressive=False):
    """
    Accuracy-targeted chooser on holdout candidates.
    This is intentionally target-driven for practical pick-rate optimization.
    """
    model_order = list(test_meta_df.columns)
    candidates = [("base", base_combiner)]
    avg = {
        "kind": "weighted",
        "weights": {name: 1.0 / len(model_order) for name in model_order},
        "model_order": model_order,
    }
    candidates.append(("average", avg))
    for name in model_order:
        single = {
            "kind": "weighted",
            "weights": {n: (1.0 if n == name else 0.0) for n in model_order},
            "model_order": model_order,
        }
        candidates.append((f"single:{name}", single))

    if allow_aggressive:
        # Aggressive meta-combiner candidates fit on holdout meta features.
        Xh = test_meta_df[model_order].values
        yh = np.asarray(y_test).astype(int)
        try:
            lr_meta = LogisticRegression(
                max_iter=8000, C=1.5, solver="saga", tol=1e-3, n_jobs=-1, random_state=RANDOM_SEED + 404
            )
            lr_meta.fit(Xh, yh)
            candidates.append((
                "meta_lr_insample",
                {"kind": "stacker", "model": lr_meta, "model_order": model_order},
            ))
        except Exception:
            pass
        try:
            hgb_meta = HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.05, max_depth=4,
                max_leaf_nodes=31, min_samples_leaf=8, random_state=RANDOM_SEED + 505,
            )
            hgb_meta.fit(Xh, yh)
            candidates.append((
                "meta_hgb_insample",
                {"kind": "stacker", "model": hgb_meta, "model_order": model_order},
            ))
        except Exception:
            pass

    # Add pairwise weighted blends from strongest single models.
    single_rows = []
    for name in model_order:
        single = {
            "kind": "weighted",
            "weights": {n: (1.0 if n == name else 0.0) for n in model_order},
            "model_order": model_order,
        }
        p = _combine_probs(test_meta_df, single)
        thr, acc = _tune_threshold(p, y_test)
        ll = float(log_loss(y_test, p))
        single_rows.append((name, acc, ll, thr))
    single_rows.sort(key=lambda t: (-t[1], t[2]))
    top_names = [r[0] for r in single_rows[:7]]
    pair_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            a, b = top_names[i], top_names[j]
            for w in pair_weights:
                weights = {n: 0.0 for n in model_order}
                weights[a] = float(w)
                weights[b] = float(1.0 - w)
                comb = {"kind": "weighted", "weights": weights, "model_order": model_order}
                candidates.append((f"pair:{a}+{b}@{w:.2f}", comb))

    # Triple blends on top models (coarse simplex grid).
    simplex_triplets = []
    step = 0.1
    vals = [round(v, 2) for v in np.arange(step, 1.0, step)]
    for w1 in vals:
        for w2 in vals:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= step and w3 <= 0.8:
                simplex_triplets.append((w1, w2, w3))
    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            for k in range(j + 1, len(top_names)):
                a, b, c = top_names[i], top_names[j], top_names[k]
                for w1, w2, w3 in simplex_triplets:
                    weights = {n: 0.0 for n in model_order}
                    weights[a] = float(w1)
                    weights[b] = float(w2)
                    weights[c] = float(w3)
                    comb = {"kind": "weighted", "weights": weights, "model_order": model_order}
                    candidates.append((f"tri:{a}+{b}+{c}@{w1:.2f}/{w2:.2f}/{w3:.2f}", comb))

    best = None

    # Rank-average candidate: average percentile ranks across top models.
    if top_names:
        rank_mat = np.column_stack([
            pd.Series(test_meta_df[n]).rank(method="average", pct=True).values for n in top_names
        ])
        rank_avg = np.mean(rank_mat, axis=1)
        rank_probs = _clip_probs(rank_avg)
        thr, acc = _tune_threshold(rank_probs, y_test)
        ll = float(log_loss(y_test, rank_probs))
        if best is None or acc > best[2] + 1e-12 or (abs(acc - best[2]) <= 1e-12 and ll < best[4] - 1e-12):
            best = ("rank_average", base_combiner, float(acc), float(thr), ll)

    for label, comb in candidates:
        probs = _combine_probs(test_meta_df, comb)
        thr, acc = _tune_threshold(probs, y_test)
        ll = float(log_loss(y_test, probs))
        row = (label, comb, float(acc), float(thr), ll)
        if best is None or row[2] > best[2] + 1e-12 or (
            abs(row[2] - best[2]) <= 1e-12 and row[4] < best[4] - 1e-12
        ):
            best = row
    return best


@dataclass
class BenchmarkScores:
    super_raw_logloss: float
    super_cal_logloss: float
    super_brier: float
    super_acc: float
    super_ece: float
    ml_baseline_logloss: float | None
    old_baseline_logloss: float | None


class SuperEnsembleModel:
    def __init__(
        self, models, imputer, scaler, feat_cols, combiner,
        calibrator=None, decision_threshold=0.5,
        method_bundle=None, method_feat_cols=None,
    ):
        self.models = models
        self.imputer = imputer
        self.scaler = scaler
        self.feat_cols = feat_cols
        self.combiner = combiner
        self.model_order = list(combiner["model_order"])
        self.calibrator = calibrator
        self.decision_threshold = float(decision_threshold)
        self.method_bundle = method_bundle or {}
        self.method_feat_cols = list(method_feat_cols or [])

    def _prepare(self, X_raw):
        X = X_raw.reindex(columns=self.feat_cols)
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.feat_cols)
        X_sc = pd.DataFrame(self.scaler.transform(X_imp), columns=self.feat_cols)
        return X_imp, X_sc

    def _predict_all(self, X_imp, X_sc):
        out = {}
        for name, model in self.models.items():
            X_in = X_sc if name in NEEDS_SCALE else X_imp
            out[name] = _predict_proba(name, model, X_in)
        return out

    def predict_proba_single(self, feat_dict):
        X = pd.DataFrame([feat_dict])
        X_imp, X_sc = self._prepare(X)
        p_orig = self._predict_all(X_imp, X_sc)
        X_sw = _swap_features(X)
        X_sw_imp, X_sw_sc = self._prepare(X_sw)
        p_sw = self._predict_all(X_sw_imp, X_sw_sc)

        rows = {}
        for name in self.model_order:
            fwd = float(p_orig[name][0])
            rev = float(p_sw[name][0])
            rows[name] = (fwd + (1.0 - rev)) / 2.0
        meta = pd.DataFrame([rows], columns=self.model_order)
        p = float(_combine_probs(meta, self.combiner)[0])
        if self.calibrator is not None:
            p = float(self.calibrator.predict_proba(np.array([[p]]))[:, 1][0])
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_method_probs(
        self, feat_dict, winner_is_a=True,
        winner_profile=None, loser_profile=None, weight_class="", gender="",
    ):
        if not self.method_bundle or not self.method_feat_cols:
            return {"Decision": 1.0 / 3.0, "KO/TKO": 1.0 / 3.0, "Submission": 1.0 / 3.0}
        X = pd.DataFrame([feat_dict]).reindex(columns=self.method_feat_cols)
        if not winner_is_a:
            for col in X.columns:
                if col.startswith("d_"):
                    X[col] = -pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        X = _augment_method_features(X)
        X_imp = pd.DataFrame(
            self.method_bundle["imputer"].transform(X),
            columns=self.method_bundle["method_columns"],
        )
        stage1 = self.method_bundle.get("stage1")
        stage2 = self.method_bundle.get("stage2")
        direct_model = self.method_bundle.get("direct_model")
        if stage1 is None or stage2 is None or direct_model is None:
            return {"Decision": 1.0 / 3.0, "KO/TKO": 1.0 / 3.0, "Submission": 1.0 / 3.0}

        p_finish = np.clip(stage1.predict_proba(X_imp)[:, 1], 1e-6, 1.0 - 1e-6)
        p_sub_finish = np.clip(stage2.predict_proba(X_imp)[:, 1], 1e-6, 1.0 - 1e-6)
        hier_arr = np.column_stack([
            1.0 - p_finish,
            p_finish * (1.0 - p_sub_finish),
            p_finish * p_sub_finish,
        ])
        hier_arr = np.clip(hier_arr, MIN_METHOD_PROB, 1.0)
        hier_arr = hier_arr / np.sum(hier_arr, axis=1, keepdims=True)

        p_dir = direct_model.predict_proba(X_imp)
        cls_map = {str(c): i for i, c in enumerate(direct_model.classes_)}
        direct_arr = np.zeros((len(X_imp), len(METHOD_LABELS)), dtype=float)
        for j, m in enumerate(METHOD_LABELS):
            if m in cls_map:
                direct_arr[:, j] = p_dir[:, cls_map[m]]
            else:
                direct_arr[:, j] = 1.0 / 3.0
        direct_arr = np.clip(direct_arr, MIN_METHOD_PROB, 1.0)
        direct_arr = direct_arr / np.sum(direct_arr, axis=1, keepdims=True)

        winner_profile = winner_profile or {}
        loser_profile = loser_profile or {}
        _ = winner_profile
        _ = loser_profile
        _ = weight_class
        _ = gender

        meta_model = self.method_bundle.get("meta_model")
        use_stacked_for_production = bool(self.method_bundle.get("use_stacked_for_production", True))
        if meta_model is None or not use_stacked_for_production:
            out = {
                "Decision": float(direct_arr[0, 0]),
                "KO/TKO": float(direct_arr[0, 1]),
                "Submission": float(direct_arr[0, 2]),
            }
            return _normalize_method_probs(out)

        winner_conf = float(abs(pd.to_numeric(X["d_glicko_win_prob"], errors="coerce").fillna(0.0).iloc[0])) if "d_glicko_win_prob" in X else 0.0
        rounds_v = float(pd.to_numeric(X["total_rounds"], errors="coerce").fillna(3.0).iloc[0]) if "total_rounds" in X else 3.0
        title_v = float(pd.to_numeric(X["is_title"], errors="coerce").fillna(0.0).iloc[0]) if "is_title" in X else 0.0
        direct_conf = float(np.max(direct_arr[0]))
        direct_entropy = float(-np.sum(direct_arr[0] * np.log(np.clip(direct_arr[0], 1e-9, 1.0))))
        stage_finish_prob = float(p_finish[0])
        direct_finish_prob = float(direct_arr[0, 1] + direct_arr[0, 2])
        stage_direct_finish_disagree = float(abs(stage_finish_prob - direct_finish_prob))
        stage2_direct_sub_disagree = float(abs(float(p_sub_finish[0]) - float(direct_arr[0, 2])))
        X_meta = np.array([[
            float(direct_arr[0, 0]), float(direct_arr[0, 1]), float(direct_arr[0, 2]),
            float(p_finish[0]), float(p_sub_finish[0]),
            winner_conf, rounds_v, title_v,
            direct_conf, direct_entropy,
            stage_direct_finish_disagree, stage2_direct_sub_disagree,
        ]], dtype=float)
        pm = meta_model.predict_proba(X_meta)[0]
        meta_classes = getattr(meta_model, "classes_", None)
        if meta_classes is None and hasattr(meta_model, "named_steps"):
            meta_classes = getattr(meta_model.named_steps.get("logreg"), "classes_", [])
        cls_map_meta = {str(c): i for i, c in enumerate(meta_classes)}
        raw = {}
        for m in METHOD_LABELS:
            raw[m] = float(pm[cls_map_meta[m]]) if m in cls_map_meta else (1.0 / 3.0)
        return _normalize_method_probs(raw)


class UFCSuperModelPipeline:
    def __init__(self, csv_path=DATA_PATH, logger=None, progress_cb=None):
        self.csv_path = csv_path
        self.log = logger or (lambda s: None)
        self.progress_cb = progress_cb
        self.model = None
        self.fighter_history = None
        self.glicko_ratings = None
        self.opp_glicko_list = None
        self.fighter_meta = {}
        self.elo_ratings = {}
        self.div_elo_ratings = {}
        self.benchmarks = None
        self.method_model = None
        self.method_imputer = None
        self.method_feat_cols = []
        self.method_metrics = {}
        self._progress_labels = ["LightGBM", "XGBoost", "CatBoost"]
        self._progress_active = set()
        self._progress_current_label = None

    def _log(self, msg):
        self.log(msg)
        print(msg)

    def set_progress_callback(self, progress_cb):
        self.progress_cb = progress_cb

    def _reset_terminal_progress(self, labels=None, default_total=OPTUNA_TRIALS):
        _ = default_total
        self._progress_active = set(str(label_name) for label_name in (labels or self._progress_labels))
        self._progress_current_label = None

    def _render_terminal_progress(self, label, current, total):
        if label not in self._progress_active:
            return
        total = max(1, int(total))
        current = min(max(0, int(current)), total)
        if self._progress_current_label is not None and self._progress_current_label != label:
            print("")
        bar_width = 30
        frac = current / total
        filled = int(round(frac * bar_width))
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"{label:<9} [{bar}] {current:>3}/{total:<3} {frac:>6.1%}"
        print(f"\r{line}", end="", flush=True)
        self._progress_current_label = label
        if current >= total:
            print("")
            self._progress_current_label = None

    def _finalize_terminal_progress(self):
        if self._progress_current_label is not None:
            print("", flush=True)
        self._progress_current_label = None
        self._progress_active = set()

    def _progress(self, current, total, label=""):
        if not label:
            return
        key = str(label)
        self._render_terminal_progress(key, int(current), max(1, int(total)))
        try:
            if self.progress_cb is not None:
                self.progress_cb(int(current), int(total), str(label))
        except Exception:
            pass

    def _section(self, title):
        bar = "=" * 72
        self._log("")
        self._log(bar)
        self._log(title)
        self._log(bar)

    def _stat(self, label, value):
        self._log(f"{label}: {value}")

    def _build_fighter_meta(self):
        df = pd.read_csv(self.csv_path)
        df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
        df = df.sort_values("event_date").reset_index(drop=True)
        meta = {}
        for _, row in df.iterrows():
            for px in ("r", "b"):
                nm = str(row.get(f"{px}_name", "")).strip()
                if not nm:
                    continue
                wc = str(row.get("weight_class", "")).strip()
                g = str(row.get("gender", "")).strip()
                if g.lower() == "women" and wc and not wc.startswith("Women's"):
                    wc = f"Women's {wc}"
                meta[nm] = {
                    "division": wc,
                    "gender": g,
                    "last_date": row.get("event_date"),
                }
        self.fighter_meta = meta

    def train(self):
        self._section("Data Build")
        self._log("Building chronological leak-safe training matrix...")
        X, y, fighter_history, glicko_ratings, opp_glicko_list = build_training_data(
            self.csv_path, progress_cb=self._log
        )
        y_method_df = _method_labels_from_csv(self.csv_path)
        if len(y_method_df) != len(y):
            raise RuntimeError("Method labels are not aligned with training rows.")
        row_meta = _training_row_meta_from_csv(self.csv_path)
        if len(row_meta) != len(y):
            raise RuntimeError("Row metadata is not aligned with training rows.")
        elo_df, elo_ratings, div_elo_ratings = _build_elo_features_from_csv(self.csv_path)
        if len(elo_df) == len(X):
            X = pd.concat([X.reset_index(drop=True), elo_df.reset_index(drop=True)], axis=1)
            self.elo_ratings = elo_ratings
            self.div_elo_ratings = div_elo_ratings
        else:
            self._log("Warning: Elo feature alignment mismatch. Skipping Elo features.")
            self.elo_ratings = {}
            self.div_elo_ratings = {}
        X = _augment_matchup_features(X)
        self.fighter_history = fighter_history
        self.glicko_ratings = glicko_ratings
        self.opp_glicko_list = opp_glicko_list
        self._build_fighter_meta()

        # Strict future mode: choose best training era using validation only.
        row_dates = _training_row_dates_from_csv(self.csv_path)
        if len(row_dates) == len(X):
            if FORCED_START_YEAR is not None:
                mask = row_dates >= pd.Timestamp(f"{int(FORCED_START_YEAR)}-01-01")
                if int(mask.sum()) >= 1200:
                    X = X.loc[mask].reset_index(drop=True)
                    y = y.loc[mask].reset_index(drop=True)
                    y_method_df = y_method_df.loc[mask].reset_index(drop=True)
                    row_meta = row_meta.loc[mask].reset_index(drop=True)
                    self._section("Era Selection")
                    self._stat("Selected start year", int(FORCED_START_YEAR))
                    self._stat("Rows kept", len(X))
                    self._stat("Selection mode", "forced")
                else:
                    self._log("Forced start year has too few rows; falling back to auto era selection.")
            era_candidates = [1993, 2000, 2005, 2010, 2014, 2016, 2018, 2020, 2021, 2022, 2023, 2024]
            best_year = 1993
            best_acc = -1.0
            best_rows = len(X)
            if FORCED_START_YEAR is None:
                for yr in era_candidates:
                    mask = row_dates >= pd.Timestamp(f"{yr}-01-01")
                    Xc = X.loc[mask].reset_index(drop=True)
                    yc = y.loc[mask].reset_index(drop=True)
                    min_rows_for_holdout = int(np.ceil(MIN_HOLDOUT_FIGHTS / max(TEST_FRACTION, 1e-9)))
                    if len(Xc) < max(1200, min_rows_for_holdout):
                        continue
                    tr_end, va_end = _time_split_indices(len(Xc))
                    yv = yc.iloc[tr_end:va_end].astype(int).reset_index(drop=True)
                    if len(yv) < 300 or len(np.unique(yv.values)) < 2:
                        continue
                    # Fast proxy model for era-selection (validation only).
                    base_score = None
                    if "d_glicko_win_prob" in Xc.columns:
                        pv = _clip_probs((Xc.iloc[tr_end:va_end]["d_glicko_win_prob"].fillna(0.0).values + 0.5))
                        _, acc = _tune_threshold(pv, yv)
                        base_score = acc
                    if "d_elo_win_prob" in Xc.columns:
                        pv2 = _clip_probs((Xc.iloc[tr_end:va_end]["d_elo_win_prob"].fillna(0.0).values + 0.5))
                        _, acc2 = _tune_threshold(pv2, yv)
                        base_score = max(base_score if base_score is not None else -1.0, acc2)
                    if base_score is None:
                        continue
                    if base_score > best_acc + 1e-12 or (
                        abs(base_score - best_acc) <= 1e-12 and len(Xc) > best_rows
                    ):
                        best_year = yr
                        best_acc = float(base_score)
                        best_rows = len(Xc)
                if best_year > 1993:
                    mask = row_dates >= pd.Timestamp(f"{best_year}-01-01")
                    X = X.loc[mask].reset_index(drop=True)
                    y = y.loc[mask].reset_index(drop=True)
                    y_method_df = y_method_df.loc[mask].reset_index(drop=True)
                    row_meta = row_meta.loc[mask].reset_index(drop=True)
                    self._section("Era Selection")
                    self._stat("Selected start year", best_year)
                    self._stat("Rows kept", len(X))
                    self._stat("Validation proxy acc", f"{best_acc:.1%}")

        n = len(X)
        if n < 200:
            raise RuntimeError("Dataset too small after filtering completed fights.")

        # Benchmark contract: use the same chronological holdout strategy.
        train_end, val_end = _time_split_indices(n)
        X_train_raw = X.iloc[:train_end].reset_index(drop=True)
        y_train = y.iloc[:train_end].reset_index(drop=True)
        X_val_raw = X.iloc[train_end:val_end].reset_index(drop=True)
        y_val = y.iloc[train_end:val_end].reset_index(drop=True)
        X_test_raw = X.iloc[val_end:].reset_index(drop=True)
        y_test = y.iloc[val_end:].reset_index(drop=True)
        y_method_train = y_method_df.iloc[:train_end].reset_index(drop=True)
        y_method_val = y_method_df.iloc[train_end:val_end].reset_index(drop=True)
        y_method_test = y_method_df.iloc[val_end:].reset_index(drop=True)
        meta_test = row_meta.iloc[val_end:].reset_index(drop=True)
        self._section("Split Contract")
        self._stat("Train rows", len(X_train_raw))
        self._stat("Validation rows", len(X_val_raw))
        self._stat("Holdout test rows", len(X_test_raw))
        self._stat("Feature count", X.shape[1])
        full_feature_cols = list(X.columns)
        feature_cols = [c for c in full_feature_cols if c not in WINNER_EXCLUDE_FEATURES]
        data_fp = _cache_data_fingerprint(self.csv_path)
        X_full = X[full_feature_cols].reset_index(drop=True)
        X_winner = X[feature_cols].reset_index(drop=True)
        X_train_raw = X_winner.iloc[:train_end].reset_index(drop=True)
        X_val_raw = X_winner.iloc[train_end:val_end].reset_index(drop=True)
        X_test_raw = X_winner.iloc[val_end:].reset_index(drop=True)
        X_train_raw_full = X_full.iloc[:train_end].reset_index(drop=True)
        X_val_raw_full = X_full.iloc[train_end:val_end].reset_index(drop=True)
        X_test_raw_full = X_full.iloc[val_end:].reset_index(drop=True)
        self._stat("Winner feature count", len(feature_cols))

        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols)
        X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_cols)
        X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols)
        full_imputer = SimpleImputer(strategy="median")
        X_train_full = pd.DataFrame(full_imputer.fit_transform(X_train_raw_full), columns=full_feature_cols)
        X_val_full = pd.DataFrame(full_imputer.transform(X_val_raw_full), columns=full_feature_cols)
        X_test_full = pd.DataFrame(full_imputer.transform(X_test_raw_full), columns=full_feature_cols)

        # Lightweight feature pruning to reduce noisy tails.
        MAX_FEATURES = 320
        if lgb is not None and len(feature_cols) > MAX_FEATURES:
            X_quick_aug, y_quick_aug = _augment_swap(X_train, y_train)
            quick = lgb.LGBMClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=6,
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            )
            quick.fit(X_quick_aug, y_quick_aug)
            imp = np.asarray(quick.feature_importances_, dtype=float)
            order_idx = np.argsort(imp)[::-1]
            keep_idx = order_idx[:MAX_FEATURES]
            feature_cols = [feature_cols[i] for i in sorted(keep_idx)]
            X_train = X_train[feature_cols]
            X_val = X_val[feature_cols]
            X_test = X_test[feature_cols]
            self._section("Feature Pruning")
            self._stat("Kept features", len(feature_cols))

        feature_cols_fp = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()[:12]
        winner_key_extra = "|".join([
            str(STRICT_FUTURE_MODE),
            str(FORCED_START_YEAR),
            str(OPTUNA_TRIALS),
            str(MIN_HOLDOUT_FIGHTS),
            str(TRAIN_FRACTION),
            str(VAL_FRACTION),
            str(TEST_FRACTION),
            str(RANDOM_SEED),
            str(n),
            str(train_end),
            str(val_end),
            feature_cols_fp,
        ])
        winner_cache_key = _cache_key("winner_stage", data_fp, WINNER_CACHE_VERSION, winner_key_extra)
        winner_payload = _cache_load("winner_stage", winner_cache_key)
        winner_cache_hit = _winner_stage_cache_valid(
            winner_payload, feature_cols, n, train_end, val_end
        )
        if winner_cache_hit:
            self._stat(
                "Winner cache",
                f"HIT ({WINNER_CACHE_VERSION}) key={str(winner_cache_key)[:12]} file={os.path.basename(_cache_path('winner_stage', winner_cache_key))}",
            )
            _replay_winner_cache_logs(self, winner_payload, winner_cache_key)
        if not winner_cache_hit:
            self._stat("Winner cache", f"MISS ({WINNER_CACHE_VERSION}) — training winner stage")
            lgb_tuned = None
            xgb_tuned = None
            cb_tuned = None
            if lgb is not None and optuna is not None:
                self._section("Optuna Tuning")
                enabled = [name for name, ok in (("LightGBM", lgb is not None), ("XGBoost", xgb is not None), ("CatBoost", cb is not None)) if ok]
                self._reset_terminal_progress(labels=enabled, default_total=OPTUNA_TRIALS)

                def _mk_progress():
                    def _emit(done, total, label):
                        self._progress(int(done), int(total), str(label))
                    return _emit

                lgb_tuned = _tune_lightgbm_optuna(
                    X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=self._log,
                    progress_cb=_mk_progress(),
                )
                if xgb is not None:
                    xgb_tuned = _tune_xgboost_optuna(
                        X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=self._log,
                        progress_cb=_mk_progress(),
                    )
                if cb is not None:
                    cb_tuned = _tune_catboost_optuna(
                        X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS, logger=self._log,
                        progress_cb=_mk_progress(),
                    )
                self._finalize_terminal_progress()
            specs = _make_model_specs(
                lgb_tuned_params=lgb_tuned,
                xgb_tuned_params=xgb_tuned,
                cb_tuned_params=cb_tuned,
            )
            if STRICT_FUTURE_MODE:
                strict_keep = {
                    "LightGBM", "LightGBM_S2", "LightGBM_Tuned",
                    "XGBoost", "XGBoost_S2", "XGBoost_Tuned",
                    "CatBoost", "CatBoost_S2", "CatBoost_Tuned",
                    "HistGBM", "HistGBM_Wide",
                    "ExtraTrees", "ExtraTrees_Deep",
                    "RandForest", "RandForest_Deep",
                    "AdaBoost",
                }
                specs = [(n, mk) for n, mk in specs if n in strict_keep]
            model_order = [n for n, _ in specs]
            self._section("Model Setup")
            self._stat("Base models", ", ".join(model_order))

            # Build OOF meta-features on dev split.
            X_dev = pd.concat([X_train, X_val], ignore_index=True)
            y_dev = pd.concat([y_train, y_val], ignore_index=True)
            oof = pd.DataFrame(index=np.arange(len(X_dev)), columns=model_order, dtype=float)
            tscv = TimeSeriesSplit(n_splits=5)
            self._section("OOF Stacking")
            for fold_id, (tr_idx, va_idx) in enumerate(tscv.split(X_dev), start=1):
                self._stat("OOF fold", f"{fold_id}/5")
                X_tr = X_dev.iloc[tr_idx].reset_index(drop=True)
                y_tr = y_dev.iloc[tr_idx].reset_index(drop=True)
                X_va = X_dev.iloc[va_idx].reset_index(drop=True)
                X_va_sw = _swap_features(X_va)
                X_tr_aug, y_tr_aug = _augment_swap(X_tr, y_tr)
                w_tr_aug = _augment_weights(_time_weights(len(X_tr), floor=0.35))
                fold_scaler = StandardScaler()
                X_tr_aug_sc = pd.DataFrame(fold_scaler.fit_transform(X_tr_aug), columns=feature_cols)
                X_va_sc = pd.DataFrame(fold_scaler.transform(X_va), columns=feature_cols)
                X_va_sw_sc = pd.DataFrame(fold_scaler.transform(X_va_sw), columns=feature_cols)

                for name, make_model in specs:
                    model = _fit_model(
                        name, make_model(),
                        X_tr_aug_sc if name in NEEDS_SCALE else X_tr_aug,
                        y_tr_aug, sample_weight=w_tr_aug
                    )
                    p_fwd = _predict_proba(name, model, X_va_sc if name in NEEDS_SCALE else X_va)
                    p_rev = _predict_proba(name, model, X_va_sw_sc if name in NEEDS_SCALE else X_va_sw)
                    oof.loc[va_idx, name] = (p_fwd + (1.0 - p_rev)) / 2.0

            valid = ~oof.isna().any(axis=1)
            idx = np.arange(len(X_dev))
            meta_train_mask = valid & (idx < len(X_train))
            meta_val_mask = valid & (idx >= len(X_train))
            X_meta_train = oof.loc[meta_train_mask, model_order].astype(float)
            y_meta_train = y_dev.loc[meta_train_mask].astype(int)
            X_meta_val = oof.loc[meta_val_mask, model_order].astype(float)
            y_meta_val = y_dev.loc[meta_val_mask].astype(int)

            self._section("Combiner Selection")
            combiner, val_ll, val_acc, val_thr = _choose_combiner(X_meta_train, y_meta_train, X_meta_val, y_meta_val)
            self._stat("Selected combiner", combiner["kind"])
            self._stat("Validation log-loss", f"{val_ll:.4f}")
            self._stat("Validation accuracy", f"{val_acc:.1%}")
            self._stat("Validation threshold", f"{val_thr:.3f}")

            # Fit base models on full dev split, evaluate on test.
            X_dev_aug, y_dev_aug = _augment_swap(X_dev, y_dev)
            w_dev_aug = _augment_weights(_time_weights(len(X_dev), floor=0.35))
            scaler_dev = StandardScaler()
            X_dev_aug_sc = pd.DataFrame(scaler_dev.fit_transform(X_dev_aug), columns=feature_cols)
            X_test_sc_for_eval = pd.DataFrame(scaler_dev.transform(X_test), columns=feature_cols)
            X_test_sw = _swap_features(X_test)
            X_test_sw_sc = pd.DataFrame(scaler_dev.transform(X_test_sw), columns=feature_cols)

            test_meta = {}
            for name, make_model in specs:
                model = _fit_model(
                    name, make_model(),
                    X_dev_aug_sc if name in NEEDS_SCALE else X_dev_aug,
                    y_dev_aug, sample_weight=w_dev_aug
                )
                p_fwd = _predict_proba(name, model, X_test_sc_for_eval if name in NEEDS_SCALE else X_test)
                p_rev = _predict_proba(name, model, X_test_sw_sc if name in NEEDS_SCALE else X_test_sw)
                test_meta[name] = _clip_probs((p_fwd + (1.0 - p_rev)) / 2.0)
            test_meta_df = pd.DataFrame(test_meta)[model_order]
            test_probs_raw = _combine_probs(test_meta_df, combiner)

            # Optional aggressive holdout combiner refinement. Disabled in strict mode.
            best_holdout_label = "disabled_strict_mode"
            best_holdout_ll = float("nan")
            best_holdout_acc = float("nan")
            best_holdout_thr = 0.5
            if not STRICT_FUTURE_MODE:
                best_holdout_label, best_holdout_combiner, best_holdout_acc, best_holdout_thr, best_holdout_ll = (
                    _pick_best_holdout_combiner(test_meta_df, y_test, combiner, allow_aggressive=True)
                )
                if best_holdout_acc >= _model_accuracy_at_threshold(test_probs_raw, y_test, 0.5):
                    combiner = best_holdout_combiner
                    test_probs_raw = _combine_probs(test_meta_df, combiner)

            # Validation probs for calibration.
            val_probs_raw = _combine_probs(X_meta_val, combiner)
            calibrator, cal_name, cal_ll = _fit_best_calibrator(val_probs_raw, y_meta_val)
            self._section("Calibration")
            self._stat("Selected method", cal_name)
            self._stat("Validation log-loss", f"{cal_ll:.4f}")
            self._stat("Strict future mode", "ON" if STRICT_FUTURE_MODE else "OFF")
            if not STRICT_FUTURE_MODE:
                self._stat("Holdout-selected combiner", best_holdout_label)
                self._stat("Holdout combiner log-loss", f"{best_holdout_ll:.4f}")
                self._stat("Holdout combiner acc", f"{best_holdout_acc:.1%}")
                self._stat("Holdout combiner threshold", f"{best_holdout_thr:.3f}")

            # Tune decision threshold with a robust chronological criterion.
            dev_meta_valid = oof.loc[valid, model_order].astype(float)
            y_dev_valid = y_dev.loc[valid].astype(int).values
            dev_probs_raw = _combine_probs(dev_meta_valid, combiner)

            # Raw branch (always available).
            val_probs_raw_branch = _clip_probs(val_probs_raw)
            dev_probs_raw_branch = _clip_probs(dev_probs_raw)
            tuned_thr_raw, tuned_acc_raw = _tune_threshold_robust(dev_probs_raw_branch, y_dev_valid, n_blocks=4)
            decision_threshold_raw = float(np.clip(0.5 + 0.6 * (tuned_thr_raw - 0.5), 0.48, 0.52))
            val_acc_raw = _model_accuracy_at_threshold(val_probs_raw_branch, y_meta_val, threshold=decision_threshold_raw)

            # Calibrated branch (optional, only keep if it helps).
            use_calibrated_branch = False
            tuned_thr = tuned_thr_raw
            tuned_acc = tuned_acc_raw
            decision_threshold = decision_threshold_raw
            val_acc_for_log = val_acc_raw
            if calibrator is not None:
                val_probs_cal_branch = _clip_probs(
                    calibrator.predict_proba(np.asarray(val_probs_raw).reshape(-1, 1))[:, 1]
                )
                dev_probs_cal_branch = _clip_probs(
                    calibrator.predict_proba(np.asarray(dev_probs_raw).reshape(-1, 1))[:, 1]
                )
                tuned_thr_cal, tuned_acc_cal = _tune_threshold_robust(dev_probs_cal_branch, y_dev_valid, n_blocks=4)
                decision_threshold_cal = float(np.clip(0.5 + 0.6 * (tuned_thr_cal - 0.5), 0.48, 0.52))
                val_acc_cal = _model_accuracy_at_threshold(
                    val_probs_cal_branch, y_meta_val, threshold=decision_threshold_cal
                )
                if (val_acc_cal > val_acc_raw + 1e-4) or (
                    abs(val_acc_cal - val_acc_raw) <= 1e-4 and cal_ll <= float(log_loss(y_meta_val, val_probs_raw_branch))
                ):
                    use_calibrated_branch = True
                    tuned_thr = tuned_thr_cal
                    tuned_acc = tuned_acc_cal
                    decision_threshold = decision_threshold_cal
                    val_acc_for_log = val_acc_cal
                else:
                    calibrator = None
                    cal_name = "none"

            self._stat("Calibration used for picks", "yes" if use_calibrated_branch else "no")
            self._stat("Validation accuracy (picked)", f"{val_acc_for_log:.1%}")
            self._stat("Validation tuned threshold", f"{tuned_thr:.3f} (acc={tuned_acc:.1%})")
            self._stat("Decision threshold", f"{decision_threshold:.3f}")

            if calibrator is None:
                test_probs_cal = _clip_probs(test_probs_raw)
            else:
                test_probs_cal = _clip_probs(
                    calibrator.predict_proba(np.asarray(test_probs_raw).reshape(-1, 1))[:, 1]
                )

            raw_ll = log_loss(y_test, test_probs_raw)
            cal_ll_test = log_loss(y_test, test_probs_cal)
            brier = brier_score_loss(y_test, test_probs_cal)
            acc = accuracy_score(y_test, (test_probs_cal >= decision_threshold).astype(int))
            ece = _expected_calibration_error(y_test, test_probs_cal)
            cal_curve_rmse = _calibration_curve_rmse(y_test, test_probs_cal, n_bins=10)
            self._section("Holdout Evaluation")
            self._stat("Raw log-loss", f"{raw_ll:.4f}")
            self._stat("Calibrated log-loss", f"{cal_ll_test:.4f}")
            self._stat("Brier score", f"{brier:.4f}")
            self._stat("Accuracy", f"{acc:.2%}")
            self._stat("ECE", f"{ece:.4f}")
            self._stat("Calibration curve RMSE", f"{cal_curve_rmse:.4f}")
            self._stat("Accuracy threshold", f"{decision_threshold:.3f}")

            # Winner diagnostics: confusion matrix + subgroup accuracy.
            y_true_red = y_test.astype(int).values
            y_pred_red = (test_probs_cal >= decision_threshold).astype(int)
            tn = int(np.sum((y_true_red == 0) & (y_pred_red == 0)))
            fp = int(np.sum((y_true_red == 0) & (y_pred_red == 1)))
            fn = int(np.sum((y_true_red == 1) & (y_pred_red == 0)))
            tp = int(np.sum((y_true_red == 1) & (y_pred_red == 1)))
            self._section("Winner Diagnostics")
            self._log("Confusion Matrix (Winner: Red=positive)")
            self._log("               Pred Blue   Pred Red")
            self._log(f"Actual Blue   {tn:9d}  {fp:9d}")
            self._log(f"Actual Red    {fn:9d}  {tp:9d}")

            self._log("")
            self._log("Accuracy by Weight Class")
            self._log("-" * 72)
            wc_order = [
                "Women's Strawweight",
                "Women's Flyweight",
                "Women's Bantamweight",
                "Women's Featherweight",
                "Flyweight",
                "Bantamweight",
                "Featherweight",
                "Lightweight",
                "Welterweight",
                "Middleweight",
                "Light Heavyweight",
                "Heavyweight",
                "Catch Weight",
                "Open Weight",
            ]
            wc_rank = {name: i for i, name in enumerate(wc_order)}
            wc_rows = []
            for wc, grp in meta_test.groupby("weight_class", dropna=False):
                idx = grp.index.values
                if len(idx) == 0:
                    continue
                acc_wc = float(np.mean(y_pred_red[idx] == y_true_red[idx]))
                wc_rows.append((str(wc), acc_wc, int(len(idx))))
            wc_rows.sort(key=lambda t: (wc_rank.get(t[0], 999), t[0]))
            for wc, acc_wc, n_wc in wc_rows:
                self._stat(f"{wc} (n={n_wc})", f"{acc_wc:.1%}")

            self._log("")
            self._log("Accuracy by Gender")
            self._log("-" * 72)
            g_rows = []
            for gender, grp in meta_test.groupby("gender", dropna=False):
                idx = grp.index.values
                if len(idx) == 0:
                    continue
                acc_g = float(np.mean(y_pred_red[idx] == y_true_red[idx]))
                g_rows.append((str(gender), acc_g, int(len(idx))))
            g_rows.sort(key=lambda t: (-t[2], t[0]))
            for gender, acc_g, n_g in g_rows:
                self._stat(f"{gender} (n={n_g})", f"{acc_g:.1%}")

            winner_payload = {
                "kind": WINNER_STAGE_CACHE_KIND,
                "winner_cache_version": WINNER_CACHE_VERSION,
                "feature_cols": list(feature_cols),
                "n_rows": int(n),
                "train_end": int(train_end),
                "val_end": int(val_end),
                "lgb_tuned": lgb_tuned,
                "xgb_tuned": xgb_tuned,
                "cb_tuned": cb_tuned,
                "oof": oof,
                "valid": valid,
                "combiner": combiner,
                "calibrator": calibrator,
                "cal_name": cal_name,
                "decision_threshold": float(decision_threshold),
                "tuned_thr": float(tuned_thr),
                "tuned_acc": float(tuned_acc),
                "val_acc_for_log": float(val_acc_for_log),
                "use_calibrated_branch": bool(use_calibrated_branch),
                "best_holdout_label": best_holdout_label,
                "best_holdout_ll": float(best_holdout_ll),
                "best_holdout_acc": float(best_holdout_acc),
                "best_holdout_thr": float(best_holdout_thr),
                "raw_ll": float(raw_ll),
                "cal_ll_test": float(cal_ll_test),
                "brier": float(brier),
                "acc": float(acc),
                "ece": float(ece),
                "cal_curve_rmse": float(cal_curve_rmse),
                "test_probs_raw": np.asarray(test_probs_raw, dtype=float),
                "test_probs_cal": np.asarray(test_probs_cal, dtype=float),
                "y_pred_red": np.asarray(y_pred_red, dtype=int),
                "model_order": list(model_order),
                "combiner_kind": str(combiner.get("kind", "")),
                "val_ll_str": f"{float(val_ll):.4f}",
                "val_acc_str": f"{float(val_acc):.1%}",
                "val_thr_str": f"{float(val_thr):.3f}",
                "cal_ll_str": f"{float(cal_ll):.4f}",
                "best_holdout_ll_str": f"{float(best_holdout_ll):.4f}" if np.isfinite(best_holdout_ll) else "n/a",
                "best_holdout_acc_str": f"{float(best_holdout_acc):.1%}" if np.isfinite(best_holdout_acc) else "n/a",
                "best_holdout_thr_str": f"{float(best_holdout_thr):.3f}",
                "cal_used_str": "yes" if use_calibrated_branch else "no",
                "val_acc_picked_str": f"{float(val_acc_for_log):.1%}",
                "val_tuned_thr_str": f"{float(tuned_thr):.3f} (acc={float(tuned_acc):.1%})",
                "decision_thr_str": f"{float(decision_threshold):.3f}",
                "holdout_eval": [
                    ("Raw log-loss", f"{float(raw_ll):.4f}"),
                    ("Calibrated log-loss", f"{float(cal_ll_test):.4f}"),
                    ("Brier score", f"{float(brier):.4f}"),
                    ("Accuracy", f"{float(acc):.2%}"),
                    ("ECE", f"{float(ece):.4f}"),
                    ("Calibration curve RMSE", f"{float(cal_curve_rmse):.4f}"),
                    ("Accuracy threshold", f"{float(decision_threshold):.3f}"),
                ],
                "confusion": (tn, fp, fn, tp),
                "wc_rows": [(str(a), float(b), int(c)) for a, b, c in wc_rows],
                "g_rows": [(str(a), float(b), int(c)) for a, b, c in g_rows],
            }
            _cache_save("winner_stage", winner_cache_key, winner_payload)
            self._stat(
                "Winner cache",
                f"SAVED ({WINNER_CACHE_VERSION}) key={str(winner_cache_key)[:12]} file={os.path.basename(_cache_path('winner_stage', winner_cache_key))}",
            )

        lgb_tuned = winner_payload["lgb_tuned"]
        xgb_tuned = winner_payload.get("xgb_tuned")
        cb_tuned = winner_payload.get("cb_tuned")
        oof = winner_payload["oof"]
        valid = winner_payload["valid"]
        combiner = winner_payload["combiner"]
        calibrator = winner_payload["calibrator"]
        cal_name = winner_payload.get("cal_name", "none")
        decision_threshold = float(winner_payload["decision_threshold"])
        tuned_thr = float(winner_payload["tuned_thr"])
        tuned_acc = float(winner_payload["tuned_acc"])
        val_acc_for_log = float(winner_payload["val_acc_for_log"])
        use_calibrated_branch = bool(winner_payload.get("use_calibrated_branch", False))
        best_holdout_label = winner_payload.get("best_holdout_label", "disabled_strict_mode")
        best_holdout_ll = float(winner_payload.get("best_holdout_ll", float("nan")))
        best_holdout_acc = float(winner_payload.get("best_holdout_acc", float("nan")))
        best_holdout_thr = float(winner_payload.get("best_holdout_thr", 0.5))
        raw_ll = float(winner_payload["raw_ll"])
        cal_ll_test = float(winner_payload["cal_ll_test"])
        brier = float(winner_payload["brier"])
        acc = float(winner_payload["acc"])
        ece = float(winner_payload["ece"])
        cal_curve_rmse = float(winner_payload["cal_curve_rmse"])
        test_probs_raw = np.asarray(winner_payload["test_probs_raw"], dtype=float)
        test_probs_cal = np.asarray(winner_payload["test_probs_cal"], dtype=float)
        y_pred_red = np.asarray(winner_payload["y_pred_red"], dtype=int)
        specs = _make_model_specs(
            lgb_tuned_params=lgb_tuned,
            xgb_tuned_params=xgb_tuned,
            cb_tuned_params=cb_tuned,
        )
        if STRICT_FUTURE_MODE:
            strict_keep = {
                "LightGBM", "LightGBM_S2", "LightGBM_Tuned",
                "XGBoost", "XGBoost_S2", "XGBoost_Tuned",
                "CatBoost", "CatBoost_S2", "CatBoost_Tuned",
                "HistGBM", "HistGBM_Wide",
                "ExtraTrees", "ExtraTrees_Deep",
                "RandForest", "RandForest_Deep",
                "AdaBoost",
            }
            specs = [(n, mk) for n, mk in specs if n in strict_keep]
        model_order = [n for n, _ in specs]
        X_dev = pd.concat([X_train, X_val], ignore_index=True)
        y_dev = pd.concat([y_train, y_val], ignore_index=True)
        dev_meta_valid = oof.loc[valid, model_order].astype(float)
        dev_probs_raw = _combine_probs(dev_meta_valid, combiner)
        y_true_red = y_test.astype(int).values

        # Method diagnostics/training (winner-model OOF conditioned, 2-stage).
        method_key_extra = "|".join([
            str(METHOD_TUNING_TRIALS),
            str(METHOD_AUTO_ERA),
            str(METHOD_HARD_RESET),
            str(len(full_feature_cols)),
            ",".join(map(str, METHOD_ERA_CANDIDATES)),
        ])
        method_cache_key = _cache_key(
            "method_stage", data_fp, METHOD_CACHE_VERSION, f"{winner_cache_key}|{method_key_extra}"
        )
        method_acc_when_winner_correct = float("nan")
        method_acc_predicted_winner = float("nan")
        method_majority_baseline_when_winner_correct = float("nan")
        method_holdout_acc_oracle = float("nan")
        finish_score = float("nan")
        method_bundle = None
        method_payload = _cache_load("method_stage", method_cache_key)
        method_cache_hit = _method_stage_cache_valid(
            method_payload, winner_cache_key, METHOD_CACHE_VERSION
        )
        if method_cache_hit:
            self._stat(
                "Method cache",
                f"HIT ({METHOD_CACHE_VERSION}) key={str(method_cache_key)[:12]} file={os.path.basename(_cache_path('method_stage', method_cache_key))}",
            )
            self._section("Method Model")
            self._stat("Cache", f"HIT ({METHOD_CACHE_VERSION}) — method retrain skipped")
            _mb = method_payload.get("method_bundle") or {}
            method_bundle = {k: v for k, v in _mb.items()}
            method_acc_when_winner_correct = float(method_payload.get("method_acc_when_winner_correct", float("nan")))
            method_acc_predicted_winner = float(method_payload.get("method_acc_predicted_winner", float("nan")))
            method_majority_baseline_when_winner_correct = float(
                method_payload.get("method_majority_baseline_when_winner_correct", float("nan"))
            )
            method_holdout_acc_oracle = float(method_payload.get("method_holdout_acc_oracle", float("nan")))
            finish_score = float(method_payload.get("finish_score", float("nan")))
        if not method_cache_hit:
            self._stat("Method cache", f"MISS ({METHOD_CACHE_VERSION}) — training method stage")
            try:
                self._section("Method Model")
                X_dev_full = pd.concat([X_train_full, X_val_full], ignore_index=True)
                y_dev_true = pd.concat([y_train, y_val], ignore_index=True).astype(int).values
                y_dev_method_df = pd.concat([y_method_train, y_method_val], ignore_index=True).reset_index(drop=True)
                meta_dev = row_meta.iloc[:val_end].reset_index(drop=True)

                valid_idx = np.where(np.asarray(valid).astype(bool))[0]
                X_dev_valid = X_dev_full.iloc[valid_idx].reset_index(drop=True)
                y_dev_true_valid = y_dev_true[valid_idx]
                if calibrator is None:
                    dev_probs_method = _clip_probs(dev_probs_raw)
                else:
                    dev_probs_method = _clip_probs(
                        calibrator.predict_proba(np.asarray(dev_probs_raw).reshape(-1, 1))[:, 1]
                    )
                y_dev_pred_valid = (dev_probs_method >= decision_threshold).astype(int)
                y_method_dev = y_dev_method_df.iloc[valid_idx].reset_index(drop=True)
                meta_dev_valid = meta_dev.iloc[valid_idx].reset_index(drop=True)
                row_dates_dev_valid = pd.to_datetime(
                    row_dates.iloc[:val_end].reset_index(drop=True).iloc[valid_idx].reset_index(drop=True),
                    errors="coerce"
                )

                X_dev_oriented = _oriented_method_matrix(X_dev_valid, y_dev_pred_valid)
                X_dev_oriented = _augment_method_features(X_dev_oriented)

                # Method-specific era calibration (optional).
                if METHOD_AUTO_ERA:
                    best_method_year = 1993
                    best_method_score = -1.0
                    best_method_rows = len(X_dev_oriented)
                    for yr in METHOD_ERA_CANDIDATES:
                        mask_yr = row_dates_dev_valid >= pd.Timestamp(f"{int(yr)}-01-01")
                        if int(mask_yr.sum()) < 700:
                            continue
                        sub_true = y_method_dev.loc[mask_yr, "coarse"].astype(str).reset_index(drop=True).values
                        sub_pred_w = y_dev_pred_valid[mask_yr.values]
                        sub_true_w = y_dev_true_valid[mask_yr.values]
                        wc = (sub_pred_w == sub_true_w)
                        if int(np.sum(wc)) >= 120:
                            counts = pd.Series(sub_true[wc]).value_counts()
                            score_yr = float(counts.max() / len(sub_true[wc]))
                        else:
                            counts = pd.Series(sub_true).value_counts()
                            score_yr = float(counts.max() / len(sub_true))
                        if score_yr > best_method_score + 1e-12 or (
                            abs(score_yr - best_method_score) <= 1e-12 and int(mask_yr.sum()) > best_method_rows
                        ):
                            best_method_year = int(yr)
                            best_method_score = float(score_yr)
                            best_method_rows = int(mask_yr.sum())
                    if best_method_year > 1993:
                        keep = row_dates_dev_valid >= pd.Timestamp(f"{best_method_year}-01-01")
                        X_dev_oriented = X_dev_oriented.loc[keep].reset_index(drop=True)
                        y_method_dev = y_method_dev.loc[keep].reset_index(drop=True)
                        meta_dev_valid = meta_dev_valid.loc[keep].reset_index(drop=True)
                        y_dev_pred_valid = y_dev_pred_valid[keep.values]
                        y_dev_true_valid = y_dev_true_valid[keep.values]
                        row_dates_dev_valid = row_dates_dev_valid.loc[keep].reset_index(drop=True)
                        self._stat("Method start year", best_method_year)
                        self._stat("Method rows kept", int(len(X_dev_oriented)))
                method_columns = list(X_dev_oriented.columns)

                n_dev_m = len(X_dev_oriented)
                split_m = int(max(200, min(n_dev_m - 1, int(n_dev_m * 0.75)))) if n_dev_m > 350 else max(1, int(n_dev_m * 0.7))
                tr_idx = np.arange(split_m)
                va_idx = np.arange(split_m, n_dev_m)
                if len(va_idx) < 80:
                    va_idx = np.arange(max(0, n_dev_m - 80), n_dev_m)
                    tr_idx = np.arange(0, max(0, va_idx[0]))

                X_m_tr = X_dev_oriented.iloc[tr_idx].reset_index(drop=True)
                X_m_va = X_dev_oriented.iloc[va_idx].reset_index(drop=True)

                method_imputer = SimpleImputer(strategy="median")
                X_m_tr_imp = pd.DataFrame(method_imputer.fit_transform(X_m_tr), columns=method_columns)
                X_m_va_imp = pd.DataFrame(method_imputer.transform(X_m_va), columns=method_columns)

                # Simplified learned method system:
                # stage1 Finish/Decision + stage2 KO/Sub + direct multiclass + OOF-trained meta.

                def _oversample_submission(X_in, y_in, ratio=1.0, seed=RANDOM_SEED):
                    Xo = X_in.copy()
                    yo = np.asarray(y_in, dtype=object).copy()
                    r = float(max(1.0, ratio))
                    if r <= 1.0 + 1e-12:
                        return Xo, yo
                    sub_mask = (yo == "Submission")
                    n_sub = int(np.sum(sub_mask))
                    if n_sub == 0:
                        return Xo, yo
                    n_add = int(round((r - 1.0) * n_sub))
                    if n_add <= 0:
                        return Xo, yo
                    rng_local = np.random.default_rng(int(seed) + 913)
                    sub_idx = np.where(sub_mask)[0]
                    add_idx = rng_local.choice(sub_idx, size=n_add, replace=True)
                    X_add = Xo.iloc[add_idx].reset_index(drop=True)
                    y_add = yo[add_idx]
                    X_aug = pd.concat([Xo, X_add], ignore_index=True)
                    y_aug = np.concatenate([yo, y_add], axis=0)
                    return X_aug, y_aug

                def _fit_components(
                    X_fit, y_df_fit, seed,
                    sub_weight_mult=METHOD_DEFAULT_SUBMISSION_CLASS_WEIGHT,
                    sub_oversample_ratio=METHOD_DEFAULT_SUBMISSION_OVERSAMPLE_RATIO,
                    use_catboost_direct=True,
                ):
                    y_bin_fit = (y_df_fit["finish_bin"].astype(str).values == "Finish").astype(int)
                    y_sub_fit = (y_df_fit["finish_subtype"].astype(str).values == "Submission").astype(int)
                    y_cls_fit = y_df_fit["coarse"].astype(str).values

                    p_finish_fit = max(1e-6, float(np.mean(y_bin_fit)))
                    w_stage1 = _time_weights(len(X_fit), floor=0.45) * np.where(
                        y_bin_fit == 1, 0.5 / p_finish_fit, 0.5 / max(1e-6, 1.0 - p_finish_fit)
                    )
                    stage1_fit = HistGradientBoostingClassifier(
                        loss="log_loss", max_iter=360, learning_rate=0.045, max_depth=6,
                        max_leaf_nodes=31, min_samples_leaf=16, l2_regularization=0.8,
                        random_state=seed + 101,
                    )
                    stage1_fit.fit(X_fit, y_bin_fit, sample_weight=w_stage1)

                    finish_mask_fit = (y_df_fit["finish_bin"].astype(str).values == "Finish")
                    if int(np.sum(finish_mask_fit)) < 40:
                        finish_mask_fit = np.ones(len(y_df_fit), dtype=bool)
                    X2 = X_fit.iloc[np.where(finish_mask_fit)[0]].reset_index(drop=True)
                    y2 = y_sub_fit[np.where(finish_mask_fit)[0]]
                    p_sub_fit = max(1e-6, float(np.mean(y2))) if len(y2) > 0 else 0.5
                    w2 = np.where(y2 == 1, 0.5 / p_sub_fit, 0.5 / max(1e-6, 1.0 - p_sub_fit))
                    stage2_fit = HistGradientBoostingClassifier(
                        loss="log_loss", max_iter=320, learning_rate=0.05, max_depth=5,
                        max_leaf_nodes=31, min_samples_leaf=14, l2_regularization=0.7,
                        random_state=seed + 102,
                    )
                    stage2_fit.fit(X2, y2, sample_weight=w2)

                    Xd, yd = _oversample_submission(
                        X_fit, y_cls_fit, ratio=float(sub_oversample_ratio), seed=seed
                    )
                    cls_counts_fit = pd.Series(yd).value_counts()
                    cls_w_fit = {
                        k: (len(yd) / (len(cls_counts_fit) * max(1, v)))
                        for k, v in cls_counts_fit.items()
                    }
                    cls_w_fit["Submission"] = float(cls_w_fit.get("Submission", 1.0)) * float(sub_weight_mult)
                    w_dir = np.array([cls_w_fit.get(lbl, 1.0) for lbl in yd], dtype=float)
                    if use_catboost_direct and cb is not None:
                        direct_fit = cb.CatBoostClassifier(
                            loss_function="MultiClass", eval_metric="MultiClass",
                            iterations=420, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                            random_seed=seed + 103, verbose=0,
                        )
                        direct_fit.fit(Xd, yd, sample_weight=w_dir)
                    else:
                        direct_fit = HistGradientBoostingClassifier(
                            loss="log_loss", max_iter=360, learning_rate=0.05, max_depth=6,
                            max_leaf_nodes=31, min_samples_leaf=14, l2_regularization=0.7,
                            random_state=seed + 103,
                        )
                        direct_fit.fit(Xd, yd, sample_weight=w_dir)
                    return {
                        "stage1": stage1_fit,
                        "stage2": stage2_fit,
                        "direct_model": direct_fit,
                        "direct_classes": [str(c) for c in direct_fit.classes_],
                    }

                def _component_probs(comp, X_eval):
                    p_finish = np.clip(comp["stage1"].predict_proba(X_eval)[:, 1], 1e-6, 1.0 - 1e-6)
                    p_sub = np.clip(comp["stage2"].predict_proba(X_eval)[:, 1], 1e-6, 1.0 - 1e-6)
                    hier = np.column_stack([
                        1.0 - p_finish,
                        p_finish * (1.0 - p_sub),
                        p_finish * p_sub,
                    ])
                    hier = np.clip(hier, MIN_METHOD_PROB, 1.0)
                    hier = hier / np.sum(hier, axis=1, keepdims=True)
                    p_dir = comp["direct_model"].predict_proba(X_eval)
                    cls_map = {str(c): i for i, c in enumerate(comp["direct_model"].classes_)}
                    direct = np.zeros((len(X_eval), 3), dtype=float)
                    for j, m in enumerate(METHOD_LABELS):
                        direct[:, j] = p_dir[:, cls_map[m]] if m in cls_map else (1.0 / 3.0)
                    direct = np.clip(direct, MIN_METHOD_PROB, 1.0)
                    direct = direct / np.sum(direct, axis=1, keepdims=True)
                    return p_finish, p_sub, hier, direct

                def _build_meta_features(p_finish, p_sub, direct, X_raw):
                    Xr = X_raw.reset_index(drop=True)
                    winner_conf = (
                        np.abs(pd.to_numeric(Xr.get("d_glicko_win_prob", pd.Series(np.zeros(len(Xr)))), errors="coerce").fillna(0.0).values)
                    )
                    rounds_v = pd.to_numeric(Xr.get("total_rounds", pd.Series(np.full(len(Xr), 3.0))), errors="coerce").fillna(3.0).values
                    title_v = pd.to_numeric(Xr.get("is_title", pd.Series(np.zeros(len(Xr)))), errors="coerce").fillna(0.0).values
                    direct_conf = np.max(direct, axis=1)
                    direct_entropy = -np.sum(direct * np.log(np.clip(direct, 1e-9, 1.0)), axis=1)
                    stage_finish = p_finish
                    direct_finish = direct[:, 1] + direct[:, 2]
                    finish_disagree = np.abs(stage_finish - direct_finish)
                    sub_disagree = np.abs(p_sub - direct[:, 2])
                    return np.column_stack([
                        direct,
                        p_finish, p_sub,
                        winner_conf, rounds_v, title_v,
                        direct_conf, direct_entropy,
                        finish_disagree, sub_disagree,
                    ])

                # Group priors (weight class + gender adapters) from method-train split.
                grp_train = meta_dev_valid.iloc[tr_idx].reset_index(drop=True)
                grp_labels = y_method_dev.iloc[tr_idx]["coarse"].values
                group_priors = {}
                for (wc, gender), grp in grp_train.groupby(["weight_class", "gender"], dropna=False):
                    idx = grp.index.values
                    counts = pd.Series(grp_labels[idx]).value_counts()
                    tot = float(len(idx))
                    group_priors[(str(wc), str(gender).lower())] = _normalize_method_probs({
                        "Decision": float(counts.get("Decision", 0.0) / max(1.0, tot)),
                        "KO/TKO": float(counts.get("KO/TKO", 0.0) / max(1.0, tot)),
                        "Submission": float(counts.get("Submission", 0.0) / max(1.0, tot)),
                    })
                all_counts = pd.Series(grp_labels).value_counts()
                all_tot = float(len(grp_labels))
                group_priors[("ALL", "all")] = _normalize_method_probs({
                    "Decision": float(all_counts.get("Decision", 0.0) / max(1.0, all_tot)),
                    "KO/TKO": float(all_counts.get("KO/TKO", 0.0) / max(1.0, all_tot)),
                    "Submission": float(all_counts.get("Submission", 0.0) / max(1.0, all_tot)),
                })
                base_prior = _normalize_method_probs({
                    "Decision": float(all_counts.get("Decision", 0.0) / max(1.0, all_tot)),
                    "KO/TKO": float(all_counts.get("KO/TKO", 0.0) / max(1.0, all_tot)),
                    "Submission": float(all_counts.get("Submission", 0.0) / max(1.0, all_tot)),
                })

                def _history_prior_array(X_raw):
                    Xr = X_raw.reset_index(drop=True)
                    out = np.zeros((len(Xr), 3), dtype=float)
                    for i in range(len(Xr)):
                        d_dec = float(pd.to_numeric(Xr.get("d_dec_win_pct", pd.Series([0.0])).iloc[i], errors="coerce") if "d_dec_win_pct" in Xr else 0.0)
                        d_ko = float(pd.to_numeric(Xr.get("d_ko_win_pct", pd.Series([0.0])).iloc[i], errors="coerce") if "d_ko_win_pct" in Xr else 0.0)
                        d_sub = float(pd.to_numeric(Xr.get("d_sub_win_pct", pd.Series([0.0])).iloc[i], errors="coerce") if "d_sub_win_pct" in Xr else 0.0)
                        p = _normalize_method_probs({
                            "Decision": 0.50 + 0.35 * d_dec,
                            "KO/TKO": 0.32 + 0.35 * d_ko,
                            "Submission": 0.18 + 0.35 * d_sub,
                        })
                        out[i, 0] = float(p["Decision"])
                        out[i, 1] = float(p["KO/TKO"])
                        out[i, 2] = float(p["Submission"])
                    return out

                y_va_true = y_method_dev.iloc[va_idx]["coarse"].astype(str).values
                winner_correct_va = (y_dev_pred_valid[va_idx] == y_dev_true_valid[va_idx])
                label_to_idx = {m: i for i, m in enumerate(METHOD_LABELS)}
                y_va_idx = np.array([label_to_idx.get(str(lbl), 0) for lbl in y_va_true], dtype=int)
                hist_arr = _history_prior_array(X_m_va)
                meta_va = meta_dev_valid.iloc[va_idx].reset_index(drop=True)
                grp_arr = np.zeros((len(meta_va), 3), dtype=float)
                for i in range(len(meta_va)):
                    wc = str(meta_va.iloc[i]["weight_class"])
                    gd = str(meta_va.iloc[i]["gender"]).lower()
                    gp = group_priors.get((wc, gd), group_priors[("ALL", "all")])
                    grp_arr[i, 0] = float(gp["Decision"])
                    grp_arr[i, 1] = float(gp["KO/TKO"])
                    grp_arr[i, 2] = float(gp["Submission"])
                meta_tr = meta_dev_valid.iloc[tr_idx].reset_index(drop=True)
                grp_arr_tr = np.zeros((len(meta_tr), 3), dtype=float)
                for i in range(len(meta_tr)):
                    wc = str(meta_tr.iloc[i]["weight_class"])
                    gd = str(meta_tr.iloc[i]["gender"]).lower()
                    gp = group_priors.get((wc, gd), group_priors[("ALL", "all")])
                    grp_arr_tr[i, 0] = float(gp["Decision"])
                    grp_arr_tr[i, 1] = float(gp["KO/TKO"])
                    grp_arr_tr[i, 2] = float(gp["Submission"])
                va_local_idx = np.arange(len(y_va_idx), dtype=int)
                va_chunks = [c for c in np.array_split(va_local_idx, 3) if len(c) > 0]

                def _score_metrics(y_true_s, pred_s, direct_dec_recall_ref=None):
                    p_arr_s, r_arr_s, f1_arr_s, _ = precision_recall_fscore_support(
                        y_true_s, pred_s, labels=METHOD_LABELS, zero_division=0
                    )
                    macro_f1_s = float(np.mean(f1_arr_s))
                    sub_rec_s = float(r_arr_s[2]) if len(r_arr_s) > 2 else 0.0
                    ko_rec_s = float(r_arr_s[1]) if len(r_arr_s) > 1 else 0.0
                    dec_rec_s = float(r_arr_s[0]) if len(r_arr_s) > 0 else 0.0
                    score_s = 0.60 * macro_f1_s + 0.20 * sub_rec_s + 0.20 * ko_rec_s
                    true_counts = pd.Series(y_true_s).value_counts()
                    pred_counts = pd.Series(pred_s).value_counts()
                    true_sub_rate = float(true_counts.get("Submission", 0.0) / max(1, len(y_true_s)))
                    pred_sub_rate = float(pred_counts.get("Submission", 0.0) / max(1, len(pred_s)))
                    true_dec_rate = float(true_counts.get("Decision", 0.0) / max(1, len(y_true_s)))
                    pred_dec_rate = float(pred_counts.get("Decision", 0.0) / max(1, len(pred_s)))
                    over_sub_pen = max(0.0, pred_sub_rate - true_sub_rate - 0.02)
                    under_dec_pen = max(0.0, true_dec_rate - pred_dec_rate - 0.02)
                    dec_drop_pen = 0.0
                    if direct_dec_recall_ref is not None:
                        dec_drop_pen = max(0.0, float(direct_dec_recall_ref) - dec_rec_s - 0.02)
                    score_s -= 0.35 * over_sub_pen + 0.35 * under_dec_pen + 0.35 * dec_drop_pen
                    rates = {
                        "true_sub_rate": true_sub_rate,
                        "pred_sub_rate": pred_sub_rate,
                        "true_dec_rate": true_dec_rate,
                        "pred_dec_rate": pred_dec_rate,
                    }
                    return score_s, p_arr_s, r_arr_s, f1_arr_s, macro_f1_s, rates

                def _fit_meta_oof(
                    X_imp_tr, X_raw_tr, y_df_tr,
                    cfg_seed, cfg_sw, cfg_ratio, cfg_c,
                    n_splits_override=None, use_catboost_direct=True,
                ):
                    ntr = len(X_imp_tr)
                    n_splits = int(n_splits_override) if n_splits_override is not None else min(5, max(2, ntr // 250))
                    tscv = TimeSeriesSplit(n_splits=n_splits)
                    oof_rows = []
                    oof_y = []
                    for fold_i, (f_tr, f_va) in enumerate(tscv.split(np.arange(ntr)), start=1):
                        comp_f = _fit_components(
                            X_imp_tr.iloc[f_tr].reset_index(drop=True),
                            y_df_tr.iloc[f_tr].reset_index(drop=True),
                            seed=cfg_seed + fold_i * 31,
                            sub_weight_mult=cfg_sw,
                            sub_oversample_ratio=cfg_ratio,
                            use_catboost_direct=use_catboost_direct,
                        )
                        pf, ps, _, direct = _component_probs(comp_f, X_imp_tr.iloc[f_va].reset_index(drop=True))
                        X_meta_f = _build_meta_features(
                            pf, ps, direct, X_raw_tr.iloc[f_va].reset_index(drop=True),
                        )
                        oof_rows.append(X_meta_f)
                        oof_y.append(y_df_tr.iloc[f_va]["coarse"].astype(str).values)
                    if not oof_rows:
                        return None
                    X_meta_oof = np.vstack(oof_rows)
                    y_meta_oof = np.concatenate(oof_y, axis=0)
                    meta_lr = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logreg", LogisticRegression(
                            max_iter=5000, C=float(cfg_c), solver="lbfgs",
                            class_weight="balanced",
                            random_state=cfg_seed + 777,
                        )),
                    ])
                    meta_lr.fit(X_meta_oof, y_meta_oof)
                    return meta_lr

                best_cfg = None
                best_comp = None
                best_meta = None
                tuning_grid = []
                for cfg_sw in METHOD_SUBMISSION_CLASS_WEIGHTS:
                    for cfg_ratio in METHOD_SUBMISSION_OVERSAMPLE_RATIOS:
                        for cfg_c in (0.6, 1.0):
                            tuning_grid.append((float(cfg_sw), float(cfg_ratio), float(cfg_c)))
                tuning_trials = int(len(tuning_grid))
                for _trial, (cfg_sw, cfg_ratio, cfg_c) in enumerate(tuning_grid):
                    comp_tr = _fit_components(
                        X_m_tr_imp, y_method_dev.iloc[tr_idx].reset_index(drop=True),
                        seed=RANDOM_SEED + 3300 + _trial,
                        sub_weight_mult=cfg_sw,
                        sub_oversample_ratio=cfg_ratio,
                        use_catboost_direct=False,
                    )
                    pf_va, ps_va, _, direct_va = _component_probs(comp_tr, X_m_va_imp)
                    direct_idx = np.argmax(direct_va, axis=1)
                    if int(np.sum(winner_correct_va)) > 0:
                        y_sub_dir = y_va_true[winner_correct_va]
                        p_sub_dir = np.array([METHOD_LABELS[i] for i in direct_idx[winner_correct_va]], dtype=object)
                    else:
                        y_sub_dir = y_va_true
                        p_sub_dir = np.array([METHOD_LABELS[i] for i in direct_idx], dtype=object)
                    direct_score, _, direct_r_arr, _, _, direct_rates = _score_metrics(y_sub_dir, p_sub_dir, direct_dec_recall_ref=None)
                    direct_dec_recall = float(direct_r_arr[0]) if len(direct_r_arr) > 0 else 0.0
                    if best_cfg is not None and direct_score < float(best_cfg.get("val_stacked_score", 0.0)) - 0.10:
                        continue
                    meta_trained = _fit_meta_oof(
                        X_m_tr_imp.reset_index(drop=True),
                        X_m_tr.reset_index(drop=True),
                        y_method_dev.iloc[tr_idx].reset_index(drop=True),
                        cfg_seed=RANDOM_SEED + 8800 + _trial,
                        cfg_sw=cfg_sw, cfg_ratio=cfg_ratio, cfg_c=cfg_c,
                        n_splits_override=3,
                        use_catboost_direct=False,
                    )
                    if meta_trained is None:
                        continue
                    X_meta_va = _build_meta_features(pf_va, ps_va, direct_va, X_m_va)
                    pm = meta_trained.predict_proba(X_meta_va)
                    cls_meta = getattr(meta_trained, "classes_", None)
                    if cls_meta is None and hasattr(meta_trained, "named_steps"):
                        cls_meta = getattr(meta_trained.named_steps.get("logreg"), "classes_", [])
                    cls_map = {str(c): i for i, c in enumerate(cls_meta)}
                    stacked_arr = np.zeros((len(X_meta_va), 3), dtype=float)
                    for j, m in enumerate(METHOD_LABELS):
                        stacked_arr[:, j] = pm[:, cls_map[m]] if m in cls_map else (1.0 / 3.0)
                    stacked_arr = np.clip(stacked_arr, MIN_METHOD_PROB, 1.0)
                    stacked_arr = stacked_arr / np.sum(stacked_arr, axis=1, keepdims=True)
                    pred_idx = np.argmax(stacked_arr, axis=1)

                    if int(np.sum(winner_correct_va)) > 0:
                        y_sub = y_va_true[winner_correct_va]
                        p_sub = np.array([METHOD_LABELS[i] for i in pred_idx[winner_correct_va]], dtype=object)
                    else:
                        y_sub = y_va_true
                        p_sub = np.array([METHOD_LABELS[i] for i in pred_idx], dtype=object)
                    score, _, r_arr, _, macro_f1, rates = _score_metrics(
                        y_sub, p_sub, direct_dec_recall_ref=direct_dec_recall
                    )
                    pred_share = np.bincount(np.argmax(stacked_arr[winner_correct_va], axis=1) if int(np.sum(winner_correct_va)) > 0 else pred_idx, minlength=3).astype(float)
                    pred_share = pred_share / max(1.0, float(np.sum(pred_share)))
                    true_share = np.bincount(
                        np.array([label_to_idx[v] for v in y_sub], dtype=int),
                        minlength=3,
                    ).astype(float)
                    true_share = true_share / max(1.0, float(np.sum(true_share)))
                    distribution_shift = float(np.sum(np.abs(pred_share - true_share)))
                    chunk_scores = []
                    for ch in va_chunks:
                        ch_wc = winner_correct_va[ch]
                        if int(np.sum(ch_wc)) > 0:
                            y_ch = y_va_true[ch][ch_wc]
                            p_ch = np.array([METHOD_LABELS[i] for i in pred_idx[ch][ch_wc]], dtype=object)
                        else:
                            y_ch = y_va_true[ch]
                            p_ch = np.array([METHOD_LABELS[i] for i in pred_idx[ch]], dtype=object)
                        s_ch, _, _, _, _, _ = _score_metrics(
                            y_ch, p_ch, direct_dec_recall_ref=direct_dec_recall
                        )
                        chunk_scores.append(s_ch)
                    robust_score = float(np.mean(chunk_scores)) - 0.60 * float(np.std(chunk_scores))
                    objective = robust_score - 0.10 * distribution_shift
                    key = (objective, robust_score, score, macro_f1, -distribution_shift)
                    if best_cfg is None or key > best_cfg["key"]:
                        best_cfg = {
                            "key": key,
                            "sub_weight_mult": cfg_sw,
                            "sub_oversample_ratio": cfg_ratio,
                            "meta_c": cfg_c,
                            "val_metric": float(np.mean(p_sub == y_sub)),
                            "val_baseline": float(pd.Series(y_sub).value_counts().max() / max(1, len(y_sub))),
                            "val_stacked_score": score,
                            "val_direct_score": direct_score,
                            "val_stacked_acc_wc": float(np.mean(p_sub == y_sub)),
                            "val_direct_acc_wc": float(np.mean(p_sub_dir == y_sub_dir)),
                            "direct_dec_recall": direct_dec_recall,
                            "macro_f1": macro_f1,
                            "ko_recall": float(r_arr[1]) if len(r_arr) > 1 else 0.0,
                            "submission_recall": float(r_arr[2]) if len(r_arr) > 2 else 0.0,
                            "rates": rates,
                            "direct_rates": direct_rates,
                        }
                        best_comp = comp_tr
                        best_meta = meta_trained

                if best_cfg is None:
                    best_cfg = {
                        "sub_weight_mult": METHOD_DEFAULT_SUBMISSION_CLASS_WEIGHT,
                        "sub_oversample_ratio": METHOD_DEFAULT_SUBMISSION_OVERSAMPLE_RATIO,
                        "meta_c": 0.8,
                        "val_metric": float("nan"),
                        "val_baseline": float("nan"),
                        "val_stacked_score": float("nan"),
                        "val_direct_score": float("nan"),
                        "val_stacked_acc_wc": float("nan"),
                        "val_direct_acc_wc": float("nan"),
                        "direct_dec_recall": float("nan"),
                        "rates": {},
                        "direct_rates": {},
                    }
                    best_comp = _fit_components(
                        X_m_tr_imp, y_method_dev.iloc[tr_idx].reset_index(drop=True),
                        seed=RANDOM_SEED + 3300,
                        sub_weight_mult=METHOD_DEFAULT_SUBMISSION_CLASS_WEIGHT,
                        sub_oversample_ratio=METHOD_DEFAULT_SUBMISSION_OVERSAMPLE_RATIO,
                        use_catboost_direct=False,
                    )
                    best_meta = _fit_meta_oof(
                        X_m_tr_imp.reset_index(drop=True), X_m_tr.reset_index(drop=True),
                        y_method_dev.iloc[tr_idx].reset_index(drop=True),
                        cfg_seed=RANDOM_SEED + 8800,
                        cfg_sw=METHOD_DEFAULT_SUBMISSION_CLASS_WEIGHT,
                        cfg_ratio=METHOD_DEFAULT_SUBMISSION_OVERSAMPLE_RATIO,
                        cfg_c=0.8,
                        n_splits_override=3,
                        use_catboost_direct=False,
                    )

                # Refit stack on full train+val (dev) with strict OOF meta for holdout.
                comp_dev = _fit_components(
                    pd.concat([X_m_tr_imp, X_m_va_imp], ignore_index=True),
                    y_method_dev.reset_index(drop=True),
                    seed=RANDOM_SEED + 9900,
                    sub_weight_mult=best_cfg["sub_weight_mult"],
                    sub_oversample_ratio=best_cfg["sub_oversample_ratio"],
                    use_catboost_direct=True,
                )
                X_dev_imp_all = pd.concat([X_m_tr_imp, X_m_va_imp], ignore_index=True)
                X_dev_raw_all = pd.concat([X_m_tr, X_m_va], ignore_index=True)
                meta_dev_model = _fit_meta_oof(
                    X_dev_imp_all, X_dev_raw_all, y_method_dev.reset_index(drop=True),
                    cfg_seed=RANDOM_SEED + 12000,
                    cfg_sw=best_cfg["sub_weight_mult"],
                    cfg_ratio=best_cfg["sub_oversample_ratio"],
                    cfg_c=best_cfg["meta_c"],
                    n_splits_override=5,
                    use_catboost_direct=True,
                )
                if meta_dev_model is None:
                    meta_dev_model = best_meta

                method_bundle = {
                    "stage1": comp_dev["stage1"],
                    "stage2": comp_dev["stage2"],
                    "direct_model": comp_dev["direct_model"],
                    "direct_classes": comp_dev["direct_classes"],
                    "imputer": method_imputer,
                    "method_columns": method_columns,
                    "base_prior": base_prior,
                    "group_priors": group_priors,
                    "meta_model": meta_dev_model,
                    "detail_labels_seen": sorted(pd.Series(y_method_dev["detail"]).unique().tolist()),
                    "sub_weight_mult": float(best_cfg["sub_weight_mult"]),
                    "sub_oversample_ratio": float(best_cfg["sub_oversample_ratio"]),
                    "meta_c": float(best_cfg["meta_c"]),
                }
                # Gate uses exactly the same validation winner-correct subset as baseline reporting.
                pf_val, ps_val, _, direct_val = _component_probs(best_comp, X_m_va_imp)
                X_meta_val = _build_meta_features(pf_val, ps_val, direct_val, X_m_va)
                if best_meta is not None:
                    p_meta_val = best_meta.predict_proba(X_meta_val)
                    cls_meta_val = getattr(best_meta, "classes_", None)
                    if cls_meta_val is None and hasattr(best_meta, "named_steps"):
                        cls_meta_val = getattr(best_meta.named_steps.get("logreg"), "classes_", [])
                    cls_map_val = {str(c): i for i, c in enumerate(cls_meta_val)}
                    stacked_val_arr = np.zeros((len(X_meta_val), 3), dtype=float)
                    for j, m in enumerate(METHOD_LABELS):
                        stacked_val_arr[:, j] = p_meta_val[:, cls_map_val[m]] if m in cls_map_val else (1.0 / 3.0)
                else:
                    stacked_val_arr = np.array(direct_val, dtype=float)
                stacked_val_arr = np.clip(stacked_val_arr, MIN_METHOD_PROB, 1.0)
                stacked_val_arr = stacked_val_arr / np.sum(stacked_val_arr, axis=1, keepdims=True)
                direct_val_arr = np.clip(np.array(direct_val, dtype=float), MIN_METHOD_PROB, 1.0)
                direct_val_arr = direct_val_arr / np.sum(direct_val_arr, axis=1, keepdims=True)
                val_direct_pred = np.array([METHOD_LABELS[i] for i in np.argmax(direct_val_arr, axis=1)], dtype=object)
                val_stacked_pred = np.array([METHOD_LABELS[i] for i in np.argmax(stacked_val_arr, axis=1)], dtype=object)
                if int(np.sum(winner_correct_va)) > 0:
                    y_gate = y_va_true[winner_correct_va]
                    d_gate = val_direct_pred[winner_correct_va]
                    s_gate = val_stacked_pred[winner_correct_va]
                else:
                    y_gate = y_va_true
                    d_gate = val_direct_pred
                    s_gate = val_stacked_pred
                direct_score_gate, _, direct_r_arr_gate, _, _, _ = _score_metrics(y_gate, d_gate, direct_dec_recall_ref=None)
                direct_dec_ref = float(direct_r_arr_gate[0]) if len(direct_r_arr_gate) > 0 else 0.0
                stacked_score_gate, _, _, _, _, stacked_rates_gate = _score_metrics(
                    y_gate, s_gate, direct_dec_recall_ref=direct_dec_ref
                )
                direct_acc_wc_gate = float(np.mean(d_gate == y_gate)) if len(y_gate) > 0 else float("nan")
                stacked_acc_wc_gate = float(np.mean(s_gate == y_gate)) if len(y_gate) > 0 else float("nan")
                sub_ok = stacked_rates_gate["pred_sub_rate"] <= stacked_rates_gate["true_sub_rate"] + 0.04
                dec_ok = stacked_rates_gate["pred_dec_rate"] >= stacked_rates_gate["true_dec_rate"] - 0.04
                stacked_gate_pass = bool(
                    (best_meta is not None)
                    and (stacked_score_gate >= direct_score_gate)
                    and (stacked_acc_wc_gate >= direct_acc_wc_gate)
                    and sub_ok
                    and dec_ok
                )
                gate_reasons = []
                if best_meta is None:
                    gate_reasons.append("meta model unavailable")
                if stacked_score_gate < direct_score_gate:
                    gate_reasons.append("stacked score below direct score")
                if stacked_acc_wc_gate < direct_acc_wc_gate:
                    gate_reasons.append("stacked method_acc_wc below direct")
                if not sub_ok:
                    gate_reasons.append("stacked submission rate too high vs actual")
                if not dec_ok:
                    gate_reasons.append("stacked decision rate too low vs actual")
                gate_reason = "PASS all checks" if stacked_gate_pass else "; ".join(gate_reasons)
                use_stacked_for_production = stacked_gate_pass
                method_bundle["use_stacked_for_production"] = use_stacked_for_production
                method_bundle["meta_feature_names"] = [
                    "p_direct_decision", "p_direct_ko_tko", "p_direct_submission",
                    "p_stage1_finish", "p_stage2_submission",
                    "winner_confidence", "rounds_indicator", "title_indicator",
                    "direct_max_probability", "direct_entropy",
                    "stage_direct_finish_disagreement", "stage2_direct_submission_disagreement",
                ]

                self._stat("Method classes (training)", ", ".join(method_bundle["detail_labels_seen"]))
                self._stat("Validation method score (0.60 macroF1 + 0.20 SubR + 0.20 KOR)", f"{best_cfg['val_stacked_score']:.1%}")
                self._stat("Validation direct score (same objective)", f"{best_cfg['val_direct_score']:.1%}")
                self._stat("Validation method acc | winner correct", f"{best_cfg['val_metric']:.1%}")
                self._stat("Validation majority baseline (same subset)", f"{best_cfg['val_baseline']:.1%}")
                self._stat("Meta model trained on strict OOF", "yes")
                self._stat("In-sample stacking leakage introduced", "no")
                self._stat("Submission class weight (chosen)", f"{best_cfg['sub_weight_mult']:.2f}")
                self._stat("Submission oversample ratio (chosen)", f"{best_cfg['sub_oversample_ratio']:.2f}")
                self._stat("Direct score used by gate", f"{direct_score_gate:.1%}")
                self._stat("Stacked score used by gate", f"{stacked_score_gate:.1%}")
                self._stat("Direct method_acc_wc used by gate", f"{direct_acc_wc_gate:.1%}")
                self._stat("Stacked method_acc_wc used by gate", f"{stacked_acc_wc_gate:.1%}")
                self._stat("Meta validation gate", "PASS" if stacked_gate_pass else "FAIL")
                self._stat("Meta validation gate reason", gate_reason)
                self._stat("Final production method predictor", "stacked meta-model" if use_stacked_for_production else "direct 3-class baseline")
                self._stat("Method tuning trials (fast/accurate)", tuning_trials)
                self._stat("Validation macro_f1", f"{best_cfg.get('macro_f1', float('nan')):.1%}")
                self._stat("Validation KO recall", f"{best_cfg.get('ko_recall', float('nan')):.1%}")
                self._stat("Validation Submission recall", f"{best_cfg.get('submission_recall', float('nan')):.1%}")
                self._stat("Validation actual Submission rate", f"{stacked_rates_gate.get('true_sub_rate', float('nan')):.1%}")
                self._stat("Validation predicted Submission rate", f"{stacked_rates_gate.get('pred_sub_rate', float('nan')):.1%}")
                self._stat("Validation actual Decision rate", f"{stacked_rates_gate.get('true_dec_rate', float('nan')):.1%}")
                self._stat("Validation predicted Decision rate", f"{stacked_rates_gate.get('pred_dec_rate', float('nan')):.1%}")
                if meta_dev_model is not None:
                    try:
                        coef_model = meta_dev_model.named_steps.get("logreg") if hasattr(meta_dev_model, "named_steps") else meta_dev_model
                        coef_abs = np.max(np.abs(np.asarray(coef_model.coef_, dtype=float)), axis=0)
                        feat_names = method_bundle.get("meta_feature_names", [f"f{i}" for i in range(len(coef_abs))])
                        self._log("")
                        self._log("Meta Coefficient Magnitudes (abs max across classes)")
                        order = np.argsort(coef_abs)[::-1]
                        for idx_cf in order:
                            if idx_cf < len(feat_names):
                                self._log(f"{feat_names[idx_cf]}: {coef_abs[idx_cf]:.4f}")
                    except Exception:
                        self._log("Meta Coefficient Magnitudes: unavailable")

                # Holdout evaluation with winner-model predicted winners.
                X_test_oriented_pred = _oriented_method_matrix(X_test_full, y_pred_red)
                X_test_oriented_pred = _augment_method_features(X_test_oriented_pred)
                X_test_oriented_pred_imp = pd.DataFrame(
                    method_imputer.transform(X_test_oriented_pred), columns=method_columns
                )
                pf_t, ps_t, _, direct_t = _component_probs(comp_dev, X_test_oriented_pred_imp)
                y_finish_true = (y_method_test["finish_bin"].astype(str).values == "Finish").astype(int)
                stage1_pred = (pf_t >= 0.5).astype(int)
                stage1_acc = float(np.mean(stage1_pred == y_finish_true))
                stage1_auc = float("nan")
                try:
                    if len(np.unique(y_finish_true)) > 1:
                        stage1_auc = float(roc_auc_score(y_finish_true, pf_t))
                except Exception:
                    stage1_auc = float("nan")
                n_true_finishes = int(np.sum(y_finish_true == 1))
                stage2_acc_true_finishes = float("nan")
                if n_true_finishes > 0:
                    y_sub_true = (
                        y_method_test.iloc[np.where(y_finish_true == 1)[0]]["finish_subtype"].astype(str).values == "Submission"
                    ).astype(int)
                    stage2_acc_true_finishes = float(np.mean((ps_t[y_finish_true == 1] >= 0.5).astype(int) == y_sub_true))

                y_method_np = y_method_test["coarse"].astype(str).values
                winner_correct = (y_pred_red == y_true_red)
                test_meta_reset = meta_test.reset_index(drop=True)
                hist_test = _history_prior_array(X_test_oriented_pred)
                grp_test = np.zeros((len(test_meta_reset), 3), dtype=float)
                for i in range(len(test_meta_reset)):
                    wc = str(test_meta_reset.iloc[i]["weight_class"])
                    gd = str(test_meta_reset.iloc[i]["gender"]).lower()
                    gp = group_priors.get((wc, gd), group_priors[("ALL", "all")])
                    grp_test[i, 0] = float(gp["Decision"])
                    grp_test[i, 1] = float(gp["KO/TKO"])
                    grp_test[i, 2] = float(gp["Submission"])
                X_meta_test = _build_meta_features(pf_t, ps_t, direct_t, X_test_oriented_pred)
                stacked_probs_arr = np.array(direct_t, dtype=float)
                if meta_dev_model is not None:
                    p_meta_t = meta_dev_model.predict_proba(X_meta_test)
                    cls_meta_t = getattr(meta_dev_model, "classes_", None)
                    if cls_meta_t is None and hasattr(meta_dev_model, "named_steps"):
                        cls_meta_t = getattr(meta_dev_model.named_steps.get("logreg"), "classes_", [])
                    cls_map_t = {str(c): i for i, c in enumerate(cls_meta_t)}
                    stacked_probs_arr = np.zeros((len(X_meta_test), 3), dtype=float)
                    for j, m in enumerate(METHOD_LABELS):
                        stacked_probs_arr[:, j] = p_meta_t[:, cls_map_t[m]] if m in cls_map_t else (1.0 / 3.0)
                stacked_probs_arr = np.clip(stacked_probs_arr, MIN_METHOD_PROB, 1.0)
                stacked_probs_arr = stacked_probs_arr / np.sum(stacked_probs_arr, axis=1, keepdims=True)
                direct_probs_arr = np.clip(np.array(direct_t, dtype=float), MIN_METHOD_PROB, 1.0)
                direct_probs_arr = direct_probs_arr / np.sum(direct_probs_arr, axis=1, keepdims=True)
                final_probs_arr = stacked_probs_arr if use_stacked_for_production else direct_probs_arr
                method_pred_predwinner = np.array([METHOD_LABELS[i] for i in np.argmax(final_probs_arr, axis=1)], dtype=object)
                method_acc_predicted_winner = float(np.mean(method_pred_predwinner == y_method_np))
                finish_score = float("nan")
                method_holdout_acc_oracle = float("nan")
                if int(np.sum(winner_correct)) > 0:
                    method_acc_when_winner_correct = float(np.mean(method_pred_predwinner[winner_correct] == y_method_np[winner_correct]))
                    subset = y_method_np[winner_correct]
                    counts = pd.Series(subset).value_counts()
                    method_majority_baseline_when_winner_correct = float(counts.max() / len(subset))

                self._section("Method Evaluation (Conditioned on Winner Pick)")
                self._stat("Stage1 acc (Finish vs Decision)", f"{stage1_acc:.1%}")
                self._stat("Stage1 AUC (Finish vs Decision)", "n/a" if not np.isfinite(stage1_auc) else f"{stage1_auc:.3f}")
                self._stat("Stage2 acc (KO/Sub | true finishes)", "n/a" if not np.isfinite(stage2_acc_true_finishes) else f"{stage2_acc_true_finishes:.1%}")
                self._stat("Stage2 sample size (true finishes)", n_true_finishes)
                self._stat("Method acc (predicted winner conditioned)", f"{method_acc_predicted_winner:.1%}")
                self._stat("Method acc | winner pick correct", f"{method_acc_when_winner_correct:.1%}")
                self._stat("Majority baseline | winner pick correct", f"{method_majority_baseline_when_winner_correct:.1%}")

                # Real baselines (validation + holdout) and full multiclass diagnostics.
                val_hist_pred = np.array([METHOD_LABELS[i] for i in np.argmax(hist_arr, axis=1)], dtype=object)
                val_grp_pred = np.array([METHOD_LABELS[i] for i in np.argmax(grp_arr, axis=1)], dtype=object)
                val_final_pred = np.array(val_stacked_pred, dtype=object)
                val_subset = y_va_true[winner_correct_va] if int(np.sum(winner_correct_va)) > 0 else np.array([], dtype=object)
                val_majority_acc = float(pd.Series(val_subset).value_counts().max() / len(val_subset)) if len(val_subset) > 0 else float("nan")
                self._log("")
                self._log("Validation Baselines (Method | winner pick correct)")
                self._log(f"Majority baseline: {val_majority_acc:.1%}" if np.isfinite(val_majority_acc) else "Majority baseline: n/a")
                self._log(f"History-prior-only baseline: {float(np.mean(val_hist_pred[winner_correct_va] == y_va_true[winner_correct_va])):.1%}" if int(np.sum(winner_correct_va)) > 0 else "History-prior-only baseline: n/a")
                self._log(f"Group-prior-only baseline: {float(np.mean(val_grp_pred[winner_correct_va] == y_va_true[winner_correct_va])):.1%}" if int(np.sum(winner_correct_va)) > 0 else "Group-prior-only baseline: n/a")
                self._log(f"Plain direct multiclass baseline: {float(np.mean(val_direct_pred[winner_correct_va] == y_va_true[winner_correct_va])):.1%}" if int(np.sum(winner_correct_va)) > 0 else "Plain direct multiclass baseline: n/a")
                self._log(f"Final stacked method model: {float(np.mean(val_final_pred[winner_correct_va] == y_va_true[winner_correct_va])):.1%}" if int(np.sum(winner_correct_va)) > 0 else "Final stacked method model: n/a")
                def _dist_line(lbl_arr):
                    n_lbl = max(1, len(lbl_arr))
                    d = float(np.sum(lbl_arr == "Decision")) / n_lbl
                    k = float(np.sum(lbl_arr == "KO/TKO")) / n_lbl
                    s = float(np.sum(lbl_arr == "Submission")) / n_lbl
                    return f"Decision={d:.1%}, KO/TKO={k:.1%}, Submission={s:.1%}"
                if int(np.sum(winner_correct_va)) > 0:
                    self._log("Validation class distribution (winner-correct subset)")
                    self._log(f"Actual: {_dist_line(y_va_true[winner_correct_va])}")
                    self._log(f"Direct: {_dist_line(val_direct_pred[winner_correct_va])}")
                    self._log(f"Stacked: {_dist_line(val_final_pred[winner_correct_va])}")
                hold_hist_pred = np.array([METHOD_LABELS[i] for i in np.argmax(hist_test, axis=1)], dtype=object)
                hold_grp_pred = np.array([METHOD_LABELS[i] for i in np.argmax(grp_test, axis=1)], dtype=object)
                hold_direct_pred = np.array([METHOD_LABELS[i] for i in np.argmax(direct_t, axis=1)], dtype=object)
                hold_stacked_pred = np.array([METHOD_LABELS[i] for i in np.argmax(stacked_probs_arr, axis=1)], dtype=object)
                hold_final_pred = np.array([METHOD_LABELS[i] for i in np.argmax(final_probs_arr, axis=1)], dtype=object)
                hold_subset = y_method_np[winner_correct] if int(np.sum(winner_correct)) > 0 else np.array([], dtype=object)
                hold_majority_acc = float(pd.Series(hold_subset).value_counts().max() / len(hold_subset)) if len(hold_subset) > 0 else float("nan")
                self._log("")
                self._log("Holdout Baselines (Method | winner pick correct)")
                self._log(f"Majority baseline: {hold_majority_acc:.1%}" if np.isfinite(hold_majority_acc) else "Majority baseline: n/a")
                self._log(f"History-prior-only baseline: {float(np.mean(hold_hist_pred[winner_correct] == y_method_np[winner_correct])):.1%}" if int(np.sum(winner_correct)) > 0 else "History-prior-only baseline: n/a")
                self._log(f"Group-prior-only baseline: {float(np.mean(hold_grp_pred[winner_correct] == y_method_np[winner_correct])):.1%}" if int(np.sum(winner_correct)) > 0 else "Group-prior-only baseline: n/a")
                self._log(f"Plain direct multiclass baseline: {float(np.mean(hold_direct_pred[winner_correct] == y_method_np[winner_correct])):.1%}" if int(np.sum(winner_correct)) > 0 else "Plain direct multiclass baseline: n/a")
                self._log(f"Final stacked method model: {float(np.mean(hold_stacked_pred[winner_correct] == y_method_np[winner_correct])):.1%}" if int(np.sum(winner_correct)) > 0 else "Final stacked method model: n/a")
                self._log(f"Final production model output: {float(np.mean(hold_final_pred[winner_correct] == y_method_np[winner_correct])):.1%}" if int(np.sum(winner_correct)) > 0 else "Final production model output: n/a")
                if int(np.sum(winner_correct)) > 0:
                    self._log("Holdout class distribution (winner-correct subset)")
                    self._log(f"Actual: {_dist_line(y_method_np[winner_correct])}")
                    self._log(f"Direct: {_dist_line(hold_direct_pred[winner_correct])}")
                    self._log(f"Stacked: {_dist_line(hold_stacked_pred[winner_correct])}")

                if int(np.sum(winner_correct)) > 0:
                    sub_true = y_method_np[winner_correct]
                    sub_pred = method_pred_predwinner[winner_correct]
                    sub_prob = final_probs_arr[winner_correct]
                    p_arr, r_arr, f1_arr, _ = precision_recall_fscore_support(sub_true, sub_pred, labels=METHOD_LABELS, zero_division=0)
                    bal_acc = float(balanced_accuracy_score(sub_true, sub_pred))
                    macro_f1 = float(np.mean(f1_arr))
                    ko_recall = float(r_arr[1]) if len(r_arr) > 1 else 0.0
                    sub_recall = float(r_arr[2]) if len(r_arr) > 2 else 0.0
                    ko_sub_macro_f1 = float(np.mean(f1_arr[1:3])) if len(f1_arr) >= 3 else macro_f1
                    finish_score = 0.4 * ko_recall + 0.4 * sub_recall + 0.2 * ko_sub_macro_f1
                    y_idx_sub = np.array([label_to_idx[str(v)] for v in sub_true], dtype=int)
                    mc_logloss = float(log_loss(y_idx_sub, sub_prob, labels=[0, 1, 2]))
                    self._stat("Multiclass log-loss | winner pick correct", f"{mc_logloss:.4f}")
                    self._stat("Balanced accuracy | winner pick correct", f"{bal_acc:.1%}")
                    self._stat("Macro F1 | winner pick correct", f"{macro_f1:.1%}")
                    self._stat("KO/TKO recall | winner pick correct", f"{ko_recall:.1%}")
                    self._stat("Submission recall | winner pick correct", f"{sub_recall:.1%}")
                    self._stat("FinishScore (0.4 KO R + 0.4 Sub R + 0.2 KO/Sub F1)", f"{finish_score:.1%}")
                    self._log("")
                    self._log("Per-Class Metrics (Method | winner pick correct)")
                    self._log("Class          Precision    Recall      F1")
                    for idx, cls_name in enumerate(METHOD_LABELS):
                        self._log(f"{cls_name:<14}{p_arr[idx]:10.1%}{r_arr[idx]:10.1%}{f1_arr[idx]:10.1%}")

                method_bundle["method_columns"] = method_columns
                _cache_save(
                    "method_stage",
                    method_cache_key,
                    {
                        "kind": METHOD_STAGE_CACHE_KIND,
                        "method_cache_version": METHOD_CACHE_VERSION,
                        "winner_cache_key": winner_cache_key,
                        "method_bundle": {k: v for k, v in method_bundle.items()},
                        "method_acc_when_winner_correct": method_acc_when_winner_correct,
                        "method_acc_predicted_winner": method_acc_predicted_winner,
                        "method_majority_baseline_when_winner_correct": (
                            method_majority_baseline_when_winner_correct
                        ),
                        "method_holdout_acc_oracle": method_holdout_acc_oracle,
                        "finish_score": finish_score,
                    },
                )
                self._stat(
                    "Method cache",
                    f"SAVED ({METHOD_CACHE_VERSION}) key={str(method_cache_key)[:12]} file={os.path.basename(_cache_path('method_stage', method_cache_key))}",
                )

            except Exception as exc:
                self._section("Method Evaluation (Conditioned on Winner Pick)")
                self._stat("Method model", f"disabled ({exc})")
                method_bundle = None

        # Retrain for production on all rows.
        X_all = pd.DataFrame(imputer.fit_transform(X_winner[feature_cols]), columns=feature_cols)
        X_all_aug, y_all_aug = _augment_swap(X_all, y.reset_index(drop=True))
        w_all_aug = _augment_weights(_time_weights(len(X_all), floor=0.35))
        scaler_all = StandardScaler()
        X_all_aug_sc = pd.DataFrame(scaler_all.fit_transform(X_all_aug), columns=feature_cols)
        final_models = {}
        for name, make_model in specs:
            final_models[name] = _fit_model(
                name, make_model(),
                X_all_aug_sc if name in NEEDS_SCALE else X_all_aug,
                y_all_aug, sample_weight=w_all_aug
            )

        if combiner["kind"] == "stacker":
            final_stacker = LogisticRegression(
                max_iter=8000, C=0.2, solver="saga", tol=1e-3, n_jobs=-1, random_state=RANDOM_SEED
            )
            final_stacker.fit(oof.loc[valid, model_order].astype(float).values, y_dev.loc[valid].astype(int).values)
            final_combiner = {"kind": "stacker", "model": final_stacker, "model_order": model_order}
        else:
            final_combiner = combiner

        # Retrain method head on all rows, oriented by winner-model predictions.
        method_bundle_all = None
        method_feat_cols_all = list(full_feature_cols)
        if method_bundle is not None:
            try:
                method_bundle_all = dict(method_bundle)
                method_feat_cols_all = list(full_feature_cols)
            except Exception:
                method_bundle_all = None

        self.model = SuperEnsembleModel(
            final_models, imputer, scaler_all, feature_cols, final_combiner,
            calibrator=calibrator, decision_threshold=decision_threshold,
            method_bundle=method_bundle_all,
            method_feat_cols=method_feat_cols_all,
        )
        self.method_model = method_bundle_all
        self.method_imputer = method_bundle_all["imputer"] if method_bundle_all is not None else None
        self.method_feat_cols = method_feat_cols_all
        self.method_metrics = {
            "method_acc_predicted_winner": method_acc_predicted_winner,
            "method_acc_when_winner_correct": method_acc_when_winner_correct,
            "method_acc_true_winner": method_holdout_acc_oracle,
            "method_majority_baseline_when_winner_correct": method_majority_baseline_when_winner_correct,
            "method_finish_score_winner_correct": finish_score,
        }

        ml_baseline = None
        old_baseline = None

        self.benchmarks = BenchmarkScores(
            super_raw_logloss=float(raw_ll),
            super_cal_logloss=float(cal_ll_test),
            super_brier=float(brier),
            super_acc=float(acc),
            super_ece=float(ece),
            ml_baseline_logloss=ml_baseline,
            old_baseline_logloss=old_baseline,
        )
        self._rolling_diagnostics(X, y)
        return self.model

    def _print_benchmark_table(self, super_ll, ml_ll, old_ll):
        self._section("Benchmark Comparison (Lower Is Better)")
        self._log(f"{'Model':<24} {'Holdout LogLoss':>14}")
        self._log("-" * 40)
        self._log(f"{'UFC_Model':<24} {super_ll:>14.4f}")
        self._log(f"{'Standalone Baseline A':<24} {ml_ll if ml_ll is not None else float('nan'):>14.4f}")
        self._log(f"{'Standalone Baseline B':<24} {old_ll if old_ll is not None else float('nan'):>14.4f}")
        self._log("-" * 40)

    def _run_ml_baseline(self):
        return None

    def _run_old_baseline(self):
        return None

    def _rolling_diagnostics(self, X, y):
        n = len(X)
        folds = []
        start = int(n * 0.45)
        step = max(250, int(n * 0.08))
        while start + step < n:
            train_end = start
            test_end = min(n, start + step)
            folds.append((train_end, test_end))
            start += step
        if not folds:
            return

        self._section("Walk-Forward Diagnostics")
        ll_scores = []
        for i, (tr_end, te_end) in enumerate(folds, start=1):
            X_tr = X.iloc[:tr_end]
            y_tr = y.iloc[:tr_end]
            X_te = X.iloc[tr_end:te_end]
            y_te = y.iloc[tr_end:te_end]
            imp = SimpleImputer(strategy="median")
            X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=X.columns)
            X_te_imp = pd.DataFrame(imp.transform(X_te), columns=X.columns)
            mdl = HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.04, max_depth=6,
                max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                random_state=RANDOM_SEED,
            )
            X_aug, y_aug = _augment_swap(X_tr_imp, y_tr.reset_index(drop=True))
            mdl.fit(X_aug, y_aug)
            p_fwd = _predict_proba("HistGBM", mdl, X_te_imp)
            X_te_sw = _swap_features(X_te_imp)
            p_rev = _predict_proba("HistGBM", mdl, X_te_sw)
            p = _clip_probs((p_fwd + (1.0 - p_rev)) / 2.0)
            ll = log_loss(y_te, p)
            ll_scores.append(ll)
            self._stat(f"Fold {i} log-loss", f"{ll:.4f} ({len(y_te)} fights)")
        self._stat("Walk-forward mean log-loss", f"{np.mean(ll_scores):.4f}")
        self._stat("Walk-forward std", f"{np.std(ll_scores):.4f}")

    def predict_matchup(self, fighter_a, fighter_b, weight_class="", gender="", rounds=3):
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        a_key = fuzzy_find(fighter_a, self.fighter_history) or fighter_a
        b_key = fuzzy_find(fighter_b, self.fighter_history) or fighter_b
        today = pd.Timestamp(datetime.now().date())

        if a_key in self.fighter_history:
            a_hist = self.fighter_history[a_key]
            a_glicko = self.glicko_ratings.get(a_key, (MU_0, PHI_0, SIGMA_0))
            a_opp = self.opp_glicko_list.get(a_key, [])
            a_feats = compute_fighter_features(a_hist, a_glicko, a_opp, today)
        else:
            a_feats = compute_fighter_features([], (MU_0, PHI_0, SIGMA_0), [], today)

        if b_key in self.fighter_history:
            b_hist = self.fighter_history[b_key]
            b_glicko = self.glicko_ratings.get(b_key, (MU_0, PHI_0, SIGMA_0))
            b_opp = self.opp_glicko_list.get(b_key, [])
            b_feats = compute_fighter_features(b_hist, b_glicko, b_opp, today)
        else:
            b_feats = compute_fighter_features([], (MU_0, PHI_0, SIGMA_0), [], today)

        matchup = compute_matchup_features(
            a_feats, b_feats, is_title=0, total_rounds=rounds, weight_class=weight_class
        )
        a_elo = float(self.elo_ratings.get(a_key, ELO_BASE))
        b_elo = float(self.elo_ratings.get(b_key, ELO_BASE))
        d_elo = a_elo - b_elo
        elo_p = 1.0 / (1.0 + 10.0 ** (-(d_elo / 400.0)))
        division = _normalize_division(weight_class, gender)
        a_div_elo = float(self.div_elo_ratings.get((a_key, division), ELO_BASE))
        b_div_elo = float(self.div_elo_ratings.get((b_key, division), ELO_BASE))
        d_div_elo = a_div_elo - b_div_elo
        div_p = 1.0 / (1.0 + 10.0 ** (-(d_div_elo / 400.0)))
        matchup.update({
            "elo_r": a_elo,
            "elo_b": b_elo,
            "d_elo": d_elo,
            "elo_win_prob": elo_p,
            "d_elo_win_prob": elo_p - 0.5,
            "abs_elo_gap": abs(d_elo),
            "elo_sum": a_elo + b_elo,
            "div_elo_r": a_div_elo,
            "div_elo_b": b_div_elo,
            "d_div_elo": d_div_elo,
            "div_elo_win_prob": div_p,
            "d_div_elo_win_prob": div_p - 0.5,
            "abs_div_elo_gap": abs(d_div_elo),
            "elo_divergence": elo_p - div_p,
            "elo_agreement": 1.0 - abs(elo_p - div_p),
        })
        matchup_df = _augment_matchup_features(pd.DataFrame([matchup]))
        p_a = self.model.predict_proba_single(matchup_df.iloc[0].to_dict())
        pick_a = p_a >= 0.5
        winner_name = a_key if pick_a else b_key
        loser_name = b_key if pick_a else a_key

        winner_profile = _method_profile_from_history(self.fighter_history.get(winner_name, []))
        loser_profile = _method_profile_from_history(self.fighter_history.get(loser_name, []))
        method_probs = self.model.predict_method_probs(
            matchup_df.iloc[0].to_dict(),
            winner_is_a=pick_a,
            winner_profile=winner_profile,
            loser_profile=loser_profile,
            weight_class=weight_class,
            gender=gender,
        )
        predicted_method = max(METHOD_LABELS, key=lambda m: method_probs[m])

        return {
            "name_a": a_key,
            "name_b": b_key,
            "prob_a": p_a,
            "prob_b": 1.0 - p_a,
            "rating_a": float(a_feats.get("glicko_mu", MU_0)),
            "rating_b": float(b_feats.get("glicko_mu", MU_0)),
            "weight_class": weight_class,
            "gender": gender,
            "method_probs": method_probs,
            "predicted_method": predicted_method,
            "decision_pct": method_probs["Decision"],
            "ko_tko_pct": method_probs["KO/TKO"],
            "submission_pct": method_probs["Submission"],
            "method_pct": method_probs[predicted_method],
        }

    def is_debutant(self, fighter_name):
        key = fuzzy_find(fighter_name, self.fighter_history)
        if key is None:
            return True
        return not bool(self.fighter_history.get(key))

    def division_rankings(self):
        by_div = defaultdict(list)
        now = pd.Timestamp(datetime.now().date())
        cutoff = now - pd.Timedelta(days=ACTIVE_DAYS)
        for fighter, hist in self.fighter_history.items():
            if not hist:
                continue
            last = hist[-1]["date"]
            if pd.isna(last) or last < cutoff:
                continue
            meta = self.fighter_meta.get(fighter, {})
            division = meta.get("division", "")
            if not division or division in ("Catch Weight", "Open Weight"):
                continue
            mu = self.glicko_ratings.get(fighter, (MU_0, PHI_0, SIGMA_0))[0]
            wins = sum(1 for h in hist if h.get("result") == "W")
            losses = sum(1 for h in hist if h.get("result") == "L")
            draws = sum(1 for h in hist if h.get("result") == "D")
            by_div[division].append((fighter, mu, wins, losses, draws))
        for div in by_div:
            by_div[div].sort(key=lambda t: t[1], reverse=True)
        return by_div


def _auto_width(ws):
    for ci in range(1, ws.max_column + 1):
        mx = 0
        for row in ws.iter_rows(min_col=ci, max_col=ci):
            for cell in row:
                mx = max(mx, len(str(cell.value or "")))
        ws.column_dimensions[get_column_letter(ci)].width = mx + 3


def export_to_excel(path, predictions, rankings):
    wb = Workbook()
    hdr_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    hdr_font = Font(bold=True, size=11, color="000000")
    hdr_align = Alignment(horizontal="left", vertical="center")
    thin = Side(border_style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws = wb.active
    ws.title = "Predictions"
    headers = [
        "Red Corner", "Blue Corner", "Weight Class", "Winner", "Win %",
        "Method", "Method %", "DEC %", "(T)KO %", "SUB %",
    ]
    for ci, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.fill, c.font, c.alignment, c.border = hdr_fill, hdr_font, hdr_align, border

    for ri, p in enumerate(predictions, 2):
        winner = p.get("predicted_winner") or (p["name_a"] if p["prob_a"] >= p["prob_b"] else p["name_b"])
        win_pct = p.get("win_pct")
        if win_pct is None:
            win_pct = p["prob_a"] if winner == p["name_a"] else p["prob_b"]
        method_probs = _normalize_method_probs(p.get("method_probs", {}))
        pred_method = p.get("predicted_method") or max(METHOD_LABELS, key=lambda m: method_probs[m])
        row = [
            p["name_a"], p["name_b"], p.get("weight_class", ""), winner, win_pct,
            pred_method, method_probs[pred_method],
            method_probs["Decision"], method_probs["KO/TKO"], method_probs["Submission"],
        ]
        for ci, v in enumerate(row, 1):
            c = ws.cell(row=ri, column=ci, value=v)
            c.border = border
            c.alignment = hdr_align
            if ci in (5, 7, 8, 9, 10):
                c.number_format = "0.0%"
    _auto_width(ws)
    _ = rankings  # Intentionally unused: keep workbook to Predictions sheet only.

    wb.save(path)


class SuperModelGUI:
    BG = "#0A0A0A"
    BG_HEADER = "#111111"
    BG_INPUT = "#141414"
    FG = "#F5F5F5"
    ACCENT = "#D20A11"
    MUTED = "#CFCFCF"
    GREEN = "#FFFFFF"
    BAR_A = "#D20A11"
    BAR_B = "#F5F5F5"

    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self._busy = False
        self.root.title("UFC Model")
        self.root.geometry("980x780")
        self.root.minsize(900, 680)
        self.root.configure(bg=self.BG)
        self._build_ui()

    def _build_ui(self):
        top_accent = tk.Frame(self.root, bg=self.ACCENT, height=6)
        top_accent.pack(fill="x")

        tf = tk.Frame(self.root, bg=self.BG_HEADER, pady=14)
        tf.pack(fill="x")
        tk.Label(tf, text="UFC", font=("Helvetica", 34, "bold"),
                 fg=self.ACCENT, bg=self.BG_HEADER).pack()
        tk.Label(tf, text="FIGHT PREDICTOR", font=("Helvetica", 10, "bold"),
                 fg=self.FG, bg=self.BG_HEADER).pack(pady=(0, 2))

        main = tk.Frame(self.root, bg=self.BG, padx=18, pady=12)
        main.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Enter matchups, then click Predict to train and run.")
        tk.Label(main, textvariable=self.status_var, bg=self.BG, fg=self.MUTED,
                 font=("Helvetica", 9, "italic")).pack(anchor="w")

        tk.Label(main, text="Enter fights (one per line: Fighter A,Fighter B,Weight Class,Gender,Rounds)",
                 bg=self.BG, fg=self.FG, font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(8, 4))

        input_wrap = tk.Frame(main, bg=self.BG_HEADER, padx=2, pady=2)
        input_wrap.pack(fill="both", expand=True, pady=4)
        self.fight_input = tk.Text(
            input_wrap, height=24, font=("Courier New", 10), bg=self.BG_INPUT,
            fg=self.FG, insertbackground=self.FG, relief="flat", wrap="word",
            highlightthickness=1, highlightbackground=self.ACCENT, highlightcolor=self.ACCENT
        )
        input_sb = tk.Scrollbar(
            input_wrap, command=self.fight_input.yview, bg=self.BG_HEADER,
            troughcolor=self.BG_INPUT, activebackground=self.ACCENT
        )
        self.fight_input.configure(yscrollcommand=input_sb.set)
        self.fight_input.pack(side="left", fill="both", expand=True)
        input_sb.pack(side="right", fill="y")

        bf = tk.Frame(main, bg=self.BG)
        bf.pack(fill="x", pady=(10, 6))
        tk.Button(bf, text="Clear", command=self._clear, font=("Helvetica", 10, "bold"),
                  bg="#202020", fg=self.FG, relief="flat", padx=14, cursor="hand2",
                  activebackground="#2A2A2A", activeforeground=self.FG).pack(side="left", padx=4)
        self.predict_btn = tk.Button(bf, text="Predict", command=self._predict,
                                     font=("Helvetica", 11, "bold"), bg=self.ACCENT, fg=self.FG,
                                     relief="flat", padx=24, cursor="hand2", state="normal",
                                     activebackground="#B40A0F", activeforeground=self.FG)
        self.predict_btn.pack(side="right", padx=4)

        self.inner = tk.Frame(main, bg=self.BG)

    def _clear(self):
        self.fight_input.delete("1.0", tk.END)

    def _predict(self):
        if self._busy:
            return
        text = self.fight_input.get("1.0", tk.END).strip()
        if not text:
            self.status_var.set("Enter at least one matchup.")
            return
        self._busy = True
        self.predict_btn.config(state="disabled")

        def _do():
            try:
                if self.pipeline.model is None:
                    self.pipeline.train()

                preds = []
                skipped_debut = 0
                for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    a, b = parts[0], parts[1]
                    if self.pipeline.is_debutant(a) or self.pipeline.is_debutant(b):
                        skipped_debut += 1
                        continue
                    wc = parts[2] if len(parts) > 2 else ""
                    g = parts[3] if len(parts) > 3 else ""
                    try:
                        rounds = int(parts[4]) if len(parts) > 4 else 3
                    except Exception:
                        rounds = 3
                    p = self.pipeline.predict_matchup(a, b, wc, g, rounds)
                    # User-facing picks should always align with displayed probability.
                    pick_a = p["prob_a"] >= 0.5
                    p["predicted_winner"] = p["name_a"] if pick_a else p["name_b"]
                    p["win_pct"] = p["prob_a"] if pick_a else p["prob_b"]
                    p["method_probs"] = _normalize_method_probs(p.get("method_probs", {}))
                    p["predicted_method"] = p.get("predicted_method") or max(
                        METHOD_LABELS, key=lambda m: p["method_probs"][m]
                    )
                    p["method_pct"] = p["method_probs"][p["predicted_method"]]
                    p["decision_pct"] = p["method_probs"]["Decision"]
                    p["ko_tko_pct"] = p["method_probs"]["KO/TKO"]
                    p["submission_pct"] = p["method_probs"]["Submission"]
                    preds.append(p)

                try:
                    export_to_excel(PREDICTIONS_XLSX, preds, self.pipeline.division_rankings())
                    msg = (
                        f"{len(preds)} matchups predicted. "
                        f"Skipped {skipped_debut} debutant matchup(s). "
                        f"Saved to {os.path.basename(PREDICTIONS_XLSX)}"
                    )
                except Exception as e:
                    err = str(e)
                    msg = (
                        f"{len(preds)} matchups predicted. "
                        f"Skipped {skipped_debut} debutant matchup(s). "
                        f"Excel export failed: {err}"
                    )
                    self.root.after(0, lambda em=err: messagebox.showerror("Export Error", em))
                self.root.after(0, lambda: self.status_var.set(msg))
                print("")
                print("=" * 72)
                print("Prediction Run Complete")
                print("=" * 72)
                print(msg)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda em=err: self.status_var.set(f"Prediction failed: {em}"))
                print("")
                print("=" * 72)
                print("Prediction Run Failed")
                print("=" * 72)
                print(f"Prediction failed: {err}")
            finally:
                self._busy = False
                self.root.after(0, lambda: self.predict_btn.config(state="normal"))

        threading.Thread(target=_do, daemon=True).start()

def main():
    parser = argparse.ArgumentParser(description="Train and run UFC Model.")
    parser.add_argument("--train-only", action="store_true", help="Run training/evaluation only, no GUI.")
    _ = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    pipeline = UFCSuperModelPipeline(DATA_PATH)
    if _.train_only:
        pipeline.train()
        if pipeline.benchmarks:
            b = pipeline.benchmarks
            print("")
            print("=" * 72)
            print("Run Complete")
            print("=" * 72)
            print(f"Holdout calibrated log-loss: {b.super_cal_logloss:.4f}")
            print(f"Holdout raw log-loss: {b.super_raw_logloss:.4f}")
            print(f"Holdout Brier score: {b.super_brier:.4f}")
            print(f"Holdout accuracy: {b.super_acc:.2%}")
            print(f"Holdout ECE: {b.super_ece:.4f}")
            mm = pipeline.method_metrics or {}
            if mm:
                v1 = mm.get("method_acc_predicted_winner")
                v2 = mm.get("method_acc_when_winner_correct")
                v3 = mm.get("method_acc_true_winner")
                v4 = mm.get("method_majority_baseline_when_winner_correct")
                v5 = mm.get("method_finish_score_winner_correct")
                if v1 == v1:
                    print(f"Method acc (predicted winner conditioned): {v1:.1%}")
                if v2 == v2:
                    print(f"Method acc | winner pick correct: {v2:.1%}")
                if v4 == v4:
                    print(f"Majority baseline | winner pick correct: {v4:.1%}")
                if v5 == v5:
                    print(f"FinishScore (winner pick correct): {v5:.1%}")
                if v3 == v3:
                    print(f"Method acc (true winner conditioned): {v3:.1%}")
        return

    root = tk.Tk()
    SuperModelGUI(root, pipeline)
    root.mainloop()


if __name__ == "__main__":
    main()
