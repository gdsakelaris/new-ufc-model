"""
UFC ML Fight Predictor.

Loads pure_fight_data.csv, engineers 150+ features per matchup (career rates,
defense, Glicko-2, round stats, form, EWM, variance, etc.), trains an
Optuna-tuned LightGBM + XGBoost + Logistic Regression ensemble, then
predicts upcoming fights via a tkinter GUI and exports to Excel.
"""

import csv, math, os, threading, warnings, logging
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from scipy.optimize import minimize as scipy_minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from pytorch_tabnet.tab_model import TabNetClassifier
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Glicko-2 constants ────────────────────────────────────────────────────────
MU_0 = 1500.0
PHI_0 = 200.0
SIGMA_0 = 0.06
TAU = 0.5
SCALE = 173.7178
CONVERGENCE = 1e-6

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

# Per-round stat names to track
RD_STATS = [
    "sig_str", "sig_str_att", "kd", "td", "td_att", "sub_att", "ctrl_sec",
    "head", "body", "leg", "distance", "clinch", "ground",
]

# ─── Fight record extraction ───────────────────────────────────────────────────

def extract_fight_record(row, prefix, opp, result):
    """Build a per-fighter fight record dict from a DataFrame row."""
    def g(col):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("nan")
        return v

    rec = {
        "date": row["event_date"],
        "fight_time": g(f"total_fight_time_sec") or 0,
        "result": result,
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


def compute_fighter_features(history, glicko, opp_glickos, current_date):
    """Compute ~130 features from a fighter's fight history."""
    global _FIGHTER_FEAT_KEYS

    n = len(history)
    if n == 0:
        if _FIGHTER_FEAT_KEYS:
            return {k: float("nan") for k in _FIGHTER_FEAT_KEYS}
        return {}

    feats = {}

    total_time = _safe_sum(h["fight_time"] for h in history)
    total_time_min = total_time / 60.0 if total_time > 0 else 1.0
    total_time_15 = total_time / 900.0 if total_time > 0 else 1.0

    # ── Striking offense (per minute) ──
    feats["sig_str_pm"] = _safe_sum(h["sig_str"] for h in history) / total_time_min
    feats["sig_str_att_pm"] = _safe_sum(h["sig_str_att"] for h in history) / total_time_min
    feats["str_pm"] = _safe_sum(h["str"] for h in history) / total_time_min
    feats["str_att_pm"] = _safe_sum(h["str_att"] for h in history) / total_time_min
    feats["kd_pm"] = _safe_sum(h["kd"] for h in history) / total_time_min

    # ── Accuracy (career average) ──
    feats["sig_str_acc"] = _safe_mean([h["sig_str_acc"] for h in history])
    feats["str_acc"] = _safe_mean([h["str_acc"] for h in history])
    feats["td_acc"] = _safe_mean([h["td_acc"] for h in history])

    # ── Grappling (per 15 min) ──
    feats["td_p15"] = _safe_sum(h["td"] for h in history) / total_time_15
    feats["td_att_p15"] = _safe_sum(h["td_att"] for h in history) / total_time_15
    feats["sub_att_p15"] = _safe_sum(h["sub_att"] for h in history) / total_time_15
    feats["rev_p15"] = _safe_sum(h["rev"] for h in history) / total_time_15
    feats["ctrl_pct"] = _safe_sum(h["ctrl_sec"] for h in history) / total_time if total_time > 0 else 0

    # ── Targeting (average %) ──
    feats["head_pct"] = _safe_mean([h["head_pct"] for h in history])
    feats["body_pct"] = _safe_mean([h["body_pct"] for h in history])
    feats["leg_pct"] = _safe_mean([h["leg_pct"] for h in history])

    # ── Positioning (average %) ──
    feats["distance_pct"] = _safe_mean([h["distance_pct"] for h in history])
    feats["clinch_pct"] = _safe_mean([h["clinch_pct"] for h in history])
    feats["ground_pct"] = _safe_mean([h["ground_pct"] for h in history])

    # ── Defense (opponent per-minute) ──
    feats["def_sig_str_pm"] = _safe_sum(h["opp_sig_str"] for h in history) / total_time_min
    feats["def_str_pm"] = _safe_sum(h["opp_str"] for h in history) / total_time_min
    feats["def_kd_pm"] = _safe_sum(h["opp_kd"] for h in history) / total_time_min
    feats["def_td_p15"] = _safe_sum(h["opp_td"] for h in history) / total_time_15
    feats["def_sub_att_p15"] = _safe_sum(h["opp_sub_att"] for h in history) / total_time_15
    feats["def_ctrl_pct"] = _safe_sum(h["opp_ctrl_sec"] for h in history) / total_time if total_time > 0 else 0
    feats["def_sig_str_acc"] = _safe_mean([h["opp_sig_str_acc"] for h in history])

    # ── Differentials ──
    feats["net_sig_str_pm"] = feats["sig_str_pm"] - feats["def_sig_str_pm"]
    feats["net_kd_pm"] = feats["kd_pm"] - feats["def_kd_pm"]
    feats["net_td_p15"] = feats["td_p15"] - feats["def_td_p15"]
    feats["net_ctrl_pct"] = feats["ctrl_pct"] - feats["def_ctrl_pct"]

    # ── Defense rates ──
    opp_sig_att = _safe_sum(h["opp_sig_str_att"] for h in history)
    opp_sig_land = _safe_sum(h["opp_sig_str"] for h in history)
    feats["sig_str_defense_rate"] = 1.0 - _safe_div(opp_sig_land, opp_sig_att)
    opp_td_att = _safe_sum(h["opp_td_att"] for h in history)
    opp_td_land = _safe_sum(h["opp_td"] for h in history)
    feats["td_defense_rate"] = 1.0 - _safe_div(opp_td_land, opp_td_att)

    # ── Physical (most recent) ──
    latest = history[-1]
    feats["height"] = latest["height"]
    feats["reach"] = latest["reach"]
    feats["ape_index"] = latest["ape_index"]
    feats["weight"] = latest["weight"]
    feats["age"] = latest["age"]

    # ── Experience ──
    feats["num_fights"] = n
    feats["total_time_min"] = total_time_min
    feats["avg_time_min"] = total_time_min / n
    feats["title_bout_pct"] = sum(1 for h in history if h["is_title"]) / n

    # ── Record ──
    wins = sum(1 for h in history if h["result"] == "W")
    losses = sum(1 for h in history if h["result"] == "L")
    feats["win_rate"] = wins / n
    ko_w = sum(1 for h in history if h["result"] == "W" and "KO" in str(h["method"]))
    sub_w = sum(1 for h in history if h["result"] == "W" and "Sub" in str(h["method"]))
    dec_w = sum(1 for h in history if h["result"] == "W" and "Dec" in str(h["method"]))
    feats["ko_win_pct"] = ko_w / n
    feats["sub_win_pct"] = sub_w / n
    feats["dec_win_pct"] = dec_w / n
    feats["finish_rate"] = _safe_div(ko_w + sub_w, max(wins, 1))
    ko_l = sum(1 for h in history if h["result"] == "L" and "KO" in str(h["method"]))
    sub_l = sum(1 for h in history if h["result"] == "L" and "Sub" in str(h["method"]))
    feats["ko_loss_pct"] = ko_l / n
    feats["been_finished_pct"] = _safe_div(ko_l + sub_l, max(losses, 1))

    # ── Form ──
    last3 = history[-3:]
    last5 = history[-5:]
    feats["last3_win_rate"] = sum(1 for h in last3 if h["result"] == "W") / len(last3)
    feats["last5_win_rate"] = sum(1 for h in last5 if h["result"] == "W") / len(last5)
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
    for stat in ["sig_str", "kd", "td", "ctrl_sec", "head", "body", "leg"]:
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

    # ── Recent-window rate stats (last 1, 3, 5 fights) ──
    for prefix_tag, window in [("r1f", 1), ("r3", 3), ("r5", 5)]:
        recent = history[-window:]
        rt = _safe_sum(h["fight_time"] for h in recent)
        rt_min = rt / 60.0 if rt > 0 else 1.0
        rt_15 = rt / 900.0 if rt > 0 else 1.0
        feats[f"{prefix_tag}_sig_str_pm"] = _safe_sum(h["sig_str"] for h in recent) / rt_min
        feats[f"{prefix_tag}_kd_pm"] = _safe_sum(h["kd"] for h in recent) / rt_min
        feats[f"{prefix_tag}_td_p15"] = _safe_sum(h["td"] for h in recent) / rt_15
        feats[f"{prefix_tag}_ctrl_pct"] = _safe_sum(h["ctrl_sec"] for h in recent) / rt if rt > 0 else 0
        feats[f"{prefix_tag}_def_sig_str_pm"] = _safe_sum(h["opp_sig_str"] for h in recent) / rt_min
        feats[f"{prefix_tag}_sig_str_acc"] = _safe_mean([h["sig_str_acc"] for h in recent])
        feats[f"{prefix_tag}_td_acc"] = _safe_mean([h["td_acc"] for h in recent])
        feats[f"{prefix_tag}_win"] = sum(1 for h in recent if h["result"] == "W") / len(recent)

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

    # Fights per year
    if n > 1:
        career_days = (history[-1]["date"] - history[0]["date"]).days
        feats["fights_per_year"] = n / max(career_days / 365.25, 0.5)
    else:
        feats["fights_per_year"] = 1.0

    # Stance
    stance = latest.get("stance", "")
    feats["is_orthodox"] = 1 if stance == "Orthodox" else 0
    feats["is_southpaw"] = 1 if stance == "Southpaw" else 0
    feats["is_switch"] = 1 if stance == "Switch" else 0

    # Average opponent Glicko
    feats["avg_opp_glicko"] = _safe_mean(opp_glickos) if opp_glickos else MU_0

    if _FIGHTER_FEAT_KEYS is None:
        _set_fighter_keys(list(feats.keys()))

    return feats


def _set_fighter_keys(keys):
    global _FIGHTER_FEAT_KEYS
    _FIGHTER_FEAT_KEYS = keys


# ─── Matchup feature computation ──────────────────────────────────────────────

def compute_matchup_features(a_feats, b_feats, is_title=0, total_rounds=3):
    """Difference features (A minus B) plus interaction features."""
    features = {}
    for key in a_feats:
        a_val = a_feats[key] if not _isnan(a_feats[key]) else float("nan")
        b_val = b_feats[key] if not _isnan(b_feats[key]) else float("nan")
        try:
            features[f"d_{key}"] = float(a_val) - float(b_val)
        except (TypeError, ValueError):
            features[f"d_{key}"] = float("nan")
    # Absolute / interaction features
    features["is_title"] = is_title
    features["total_rounds"] = total_rounds
    features["exp_sum"] = (a_feats.get("num_fights", 0) or 0) + (b_feats.get("num_fights", 0) or 0)
    features["age_sum"] = (a_feats.get("age", 30) or 30) + (b_feats.get("age", 30) or 30)
    features["glicko_mu_sum"] = (a_feats.get("glicko_mu", MU_0) or MU_0) + (b_feats.get("glicko_mu", MU_0) or MU_0)
    return features


# ─── Training data builder ────────────────────────────────────────────────────

def build_training_data(csv_path, progress_cb=None):
    """Process fights chronologically, build features, return X, y and state."""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y")
    df = df.sort_values("event_date").reset_index(drop=True)

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

        # Compute features if both have prior fights
        if fighter_history[r_name] and fighter_history[b_name]:
            r_feats = compute_fighter_features(
                fighter_history[r_name], r_glicko,
                opp_glicko_list[r_name], row["event_date"],
            )
            b_feats = compute_fighter_features(
                fighter_history[b_name], b_glicko,
                opp_glicko_list[b_name], row["event_date"],
            )
            matchup = compute_matchup_features(
                r_feats, b_feats,
                is_title=row.get("is_title_bout", 0),
                total_rounds=row.get("total_rounds", 3),
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
        fighter_history[r_name].append(extract_fight_record(row, "r", "b", r_res))
        fighter_history[b_name].append(extract_fight_record(row, "b", "r", b_res))

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


# ─── Optuna tuning + ensemble ─────────────────────────────────────────────────

NEEDS_SCALE = {"MLP", "SVM", "LogReg", "TabNet"}


def _augment_swap(X, y):
    """Double the dataset by interleaving swapped-corner mirrors.

    For every row the d_* (difference) columns are negated and the label
    is flipped.  Interleaving (orig_0, swap_0, orig_1, swap_1 ...) keeps
    the chronological ordering intact for TimeSeriesSplit.
    """
    d_cols = [c for c in X.columns if c.startswith("d_")]
    X_swap = X.copy()
    X_swap[d_cols] = -X_swap[d_cols]
    y_swap = 1.0 - y

    n = len(X)
    X_aug = pd.DataFrame(
        np.empty((n * 2, X.shape[1]), dtype=np.float64), columns=X.columns,
    )
    y_aug = pd.Series(np.empty(n * 2, dtype=np.float64))
    X_aug.iloc[0::2] = X.values
    X_aug.iloc[1::2] = X_swap.values
    y_aug.iloc[0::2] = y.values
    y_aug.iloc[1::2] = y_swap.values
    return X_aug.reset_index(drop=True), y_aug.reset_index(drop=True)


def _swap_features(X):
    """Return a copy with all d_* columns negated (corner swap)."""
    X2 = X.copy()
    d_cols = [c for c in X2.columns if c.startswith("d_")]
    X2[d_cols] = -X2[d_cols]
    return X2


class EnsembleModel:
    def __init__(self, models, imputer, scaler, feat_cols, weights):
        self.models = models          # dict  name -> fitted model
        self.imputer = imputer
        self.scaler = scaler
        self.feat_cols = feat_cols
        self.weights = weights        # dict  name -> float weight

    def _predict_all(self, X_imp, X_sc):
        probas = {}
        for name, model in self.models.items():
            X_in = X_sc if name in NEEDS_SCALE else X_imp
            probas[name] = (model.predict_proba(X_in.values)[:, 1]
                            if name == "TabNet"
                            else model.predict_proba(X_in)[:, 1])
        return probas

    def predict_proba_single(self, features_dict):
        """Symmetric prediction: average P(A wins|A=red) and 1-P(A wins|A=blue)."""
        # Original orientation
        X = pd.DataFrame([features_dict])[self.feat_cols]
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.feat_cols)
        X_sc = pd.DataFrame(self.scaler.transform(X_imp), columns=self.feat_cols)
        p_orig = self._predict_all(X_imp, X_sc)

        # Swapped orientation
        X_sw = _swap_features(X)
        X_sw_imp = pd.DataFrame(self.imputer.transform(X_sw), columns=self.feat_cols)
        X_sw_sc = pd.DataFrame(self.scaler.transform(X_sw_imp), columns=self.feat_cols)
        p_swap = self._predict_all(X_sw_imp, X_sw_sc)

        total = 0.0
        for n in self.models:
            p_fwd = p_orig[n][0]
            p_rev = p_swap[n][0]
            total += self.weights[n] * (p_fwd + (1.0 - p_rev)) / 2.0
        return total


def _optimize_weights(probas_dict, y_true):
    """SLSQP-optimize ensemble weights that minimize log-loss on held-out data."""
    names = list(probas_dict.keys())
    n_models = len(names)
    preds = np.column_stack([probas_dict[k] for k in names])

    def objective(w):
        blend = np.clip(preds @ w, 1e-15, 1 - 1e-15)
        return log_loss(y_true, blend)

    result = scipy_minimize(
        objective,
        x0=np.ones(n_models) / n_models,
        method="SLSQP",
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    weights = {names[i]: round(float(result.x[i]), 3) for i in range(n_models)}
    return weights, result.fun


def _fit_model(name, model, X, y):
    """Fit a model, handling TabNet's numpy requirement."""
    if name == "TabNet":
        model.fit(X.values, y.astype(int).values,
                  max_epochs=100, patience=15, batch_size=1024)
    else:
        model.fit(X, y)
    return model


def _predict_model(name, model, X):
    if name == "TabNet":
        return model.predict_proba(X.values)[:, 1]
    return model.predict_proba(X)[:, 1]


def _trial_bar(model_name, n_trials, status_var=None):
    """Return an Optuna callback that prints a single updating progress bar."""
    def callback(study, trial):
        done = trial.number + 1
        pct = done / n_trials
        filled = int(30 * pct)
        bar = "#" * filled + "-" * (30 - filled)
        best = study.best_value
        msg = f"  {model_name}  [{bar}] {done}/{n_trials}  best={best:.4f}"
        print(f"\r{msg}", end="", flush=True)
        if status_var:
            status_var.set(msg)
        if done == n_trials:
            print()
    return callback


def tune_and_train(X, y, n_trials=50, progress_cb=None, status_var=None):
    """Optuna-tune LightGBM / XGBoost / CatBoost, train 11-model ensemble.

    Holds out the most recent 20 % of fights as a true test set that is
    never used during tuning or training.  Reports test-set metrics, then
    retrains final models on ALL data for production predictions.
    """
    n = len(X)
    test_size = int(n * 0.2)
    train_size = n - test_size

    X_train_raw, X_test_raw = X.iloc[:train_size], X.iloc[train_size:]
    y_train_raw = y.iloc[:train_size].reset_index(drop=True)
    y_test_raw = y.iloc[train_size:].reset_index(drop=True)

    if progress_cb:
        progress_cb("")
        progress_cb(f"  Train set: {train_size} fights  |  Test set: {test_size} fights (most recent 20%)")
        progress_cb("")

    # Impute on train, transform both
    imputer = SimpleImputer(strategy="median")
    X_train_orig = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns)

    # Augment training data — interleave (orig, swap) to eliminate red-corner bias
    X_train, y_train_raw = _augment_swap(X_train_orig, y_train_raw)

    if progress_cb:
        progress_cb(f"  Augmented train set: {len(X_train)} rows (corner-swap debiasing)")

    # Scaled versions for models that need it
    scaler_eval = StandardScaler()
    X_train_sc = pd.DataFrame(scaler_eval.fit_transform(X_train), columns=X.columns)
    X_test_sc = pd.DataFrame(scaler_eval.transform(X_test), columns=X.columns)

    # Swapped test sets for symmetric prediction
    X_test_swap = _swap_features(X_test)
    X_test_swap_sc = pd.DataFrame(scaler_eval.transform(X_test_swap), columns=X.columns)

    tscv = TimeSeriesSplit(n_splits=5)

    # ═══════════════════════════════════════════════════════════════════════
    #  Optuna tuning — LightGBM, XGBoost, CatBoost
    # ═══════════════════════════════════════════════════════════════════════

    # ── LightGBM ──
    if progress_cb:
        progress_cb(f"  --- Tuning LightGBM ({n_trials} trials) ---")

    def lgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose": -1,
        }
        scores = []
        for tr, val in tscv.split(X_train):
            m = lgb.LGBMClassifier(**p)
            m.fit(X_train.iloc[tr], y_train_raw.iloc[tr])
            scores.append(log_loss(y_train_raw.iloc[val],
                                   m.predict_proba(X_train.iloc[val])[:, 1]))
        return np.mean(scores)

    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[_trial_bar("LightGBM", n_trials, status_var)])
    lp = lgb_study.best_params
    if progress_cb:
        progress_cb(f"  Best CV log-loss: {lgb_study.best_value:.4f}")
        progress_cb("")

    # ── XGBoost ──
    if progress_cb:
        progress_cb(f"  --- Tuning XGBoost ({n_trials} trials) ---")

    def xgb_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "eval_metric": "logloss", "verbosity": 0,
        }
        scores = []
        for tr, val in tscv.split(X_train):
            m = xgb.XGBClassifier(**p)
            m.fit(X_train.iloc[tr], y_train_raw.iloc[tr])
            scores.append(log_loss(y_train_raw.iloc[val],
                                   m.predict_proba(X_train.iloc[val])[:, 1]))
        return np.mean(scores)

    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[_trial_bar("XGBoost ", n_trials, status_var)])
    xp = xgb_study.best_params
    if progress_cb:
        progress_cb(f"  Best CV log-loss: {xgb_study.best_value:.4f}")
        progress_cb("")

    # ── CatBoost ──
    if progress_cb:
        progress_cb(f"  --- Tuning CatBoost ({n_trials} trials) ---")

    def cb_obj(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "verbose": 0, "allow_writing_files": False,
        }
        scores = []
        for tr, val in tscv.split(X_train):
            m = cb.CatBoostClassifier(**p)
            m.fit(X_train.iloc[tr], y_train_raw.iloc[tr], verbose=0)
            scores.append(log_loss(y_train_raw.iloc[val],
                                   m.predict_proba(X_train.iloc[val])[:, 1]))
        return np.mean(scores)

    cb_study = optuna.create_study(direction="minimize")
    cb_study.optimize(cb_obj, n_trials=n_trials, show_progress_bar=False,
                      callbacks=[_trial_bar("CatBoost", n_trials, status_var)])
    cp = cb_study.best_params
    if progress_cb:
        progress_cb(f"  Best CV log-loss: {cb_study.best_value:.4f}")
        progress_cb("")

    # ═══════════════════════════════════════════════════════════════════════
    #  Build model specs — tuned params for the big 3, defaults for the rest
    # ═══════════════════════════════════════════════════════════════════════

    def _make_specs():
        return [
            ("LightGBM", lambda: lgb.LGBMClassifier(
                n_estimators=lp["n_estimators"], learning_rate=lp["lr"],
                max_depth=lp["max_depth"], num_leaves=lp["num_leaves"],
                min_child_samples=lp["min_child_samples"],
                subsample=lp["subsample"], colsample_bytree=lp["colsample_bytree"],
                reg_alpha=lp["reg_alpha"], reg_lambda=lp["reg_lambda"], verbose=-1)),
            ("XGBoost", lambda: xgb.XGBClassifier(
                n_estimators=xp["n_estimators"], learning_rate=xp["lr"],
                max_depth=xp["max_depth"], min_child_weight=xp["min_child_weight"],
                subsample=xp["subsample"], colsample_bytree=xp["colsample_bytree"],
                reg_alpha=xp["reg_alpha"], reg_lambda=xp["reg_lambda"],
                eval_metric="logloss", verbosity=0)),
            ("CatBoost", lambda: cb.CatBoostClassifier(
                iterations=cp["iterations"], learning_rate=cp["lr"],
                depth=cp["depth"], l2_leaf_reg=cp["l2_leaf_reg"],
                random_strength=cp["random_strength"],
                bagging_temperature=cp["bagging_temperature"],
                verbose=0, allow_writing_files=False)),
            ("HistGBM", lambda: HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.05, max_depth=6, random_state=42)),
            ("RandForest", lambda: RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1)),
            ("ExtraTrees", lambda: ExtraTreesClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1)),
            ("MLP", lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=500,
                early_stopping=True, random_state=42, learning_rate="adaptive")),
            ("TabNet", lambda: TabNetClassifier(verbose=0, seed=42)),
            ("SVM", lambda: SVC(
                kernel="rbf", probability=True, C=1.0, random_state=42)),
            ("AdaBoost", lambda: AdaBoostClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42)),
            ("LogReg", lambda: LogisticRegression(max_iter=2000, C=1.0)),
        ]

    # ═══════════════════════════════════════════════════════════════════════
    #  Train all models on train split, evaluate on held-out test set
    # ═══════════════════════════════════════════════════════════════════════
    if progress_cb:
        progress_cb("  --- Training all models on train split ---")

    test_probas = {}
    for name, make_model in _make_specs():
        if progress_cb:
            progress_cb(f"    {name}...")
        model = make_model()
        X_fit = X_train_sc if name in NEEDS_SCALE else X_train
        _fit_model(name, model, X_fit, y_train_raw)
        # Symmetric prediction: average P(A|A=red) and 1-P(A|A=blue)
        X_pred = X_test_sc if name in NEEDS_SCALE else X_test
        X_pred_sw = X_test_swap_sc if name in NEEDS_SCALE else X_test_swap
        p_fwd = _predict_model(name, model, X_pred)
        p_rev = _predict_model(name, model, X_pred_sw)
        test_probas[name] = (p_fwd + (1.0 - p_rev)) / 2.0

    # ── Optimize ensemble weights on test set ──
    weights, ensemble_ll = _optimize_weights(test_probas, y_test_raw)

    # ── Compute weighted ensemble predictions ──
    test_blend = sum(weights[n] * test_probas[n] for n in test_probas)
    ensemble_acc = accuracy_score(y_test_raw, (test_blend >= 0.5).astype(int))

    # ── Results table ──
    if progress_cb:
        progress_cb("")
        progress_cb(f"  --- Test Set Results ({test_size} fights, never seen during tuning) ---")
        progress_cb(f"  {'Model':<14} {'Accuracy':>10} {'Log-Loss':>10} {'Weight':>8}")
        progress_cb(f"  {'-'*44}")
        for name in test_probas:
            acc = accuracy_score(y_test_raw, (test_probas[name] >= 0.5).astype(int))
            ll = log_loss(y_test_raw, test_probas[name])
            progress_cb(f"  {name:<14} {acc:>9.1%} {ll:>10.4f} {weights[name]:>7.0%}")
        progress_cb(f"  {'-'*44}")
        progress_cb(f"  {'Ensemble':<14} {ensemble_acc:>9.1%} {ensemble_ll:>10.4f}")
        progress_cb("")

    # ═══════════════════════════════════════════════════════════════════════
    #  Retrain all models on ALL data for production predictions
    # ═══════════════════════════════════════════════════════════════════════
    if progress_cb:
        progress_cb("  --- Retraining all models on full data ---")

    imputer_full = SimpleImputer(strategy="median")
    X_all_orig = pd.DataFrame(imputer_full.fit_transform(X), columns=X.columns)
    y_all_orig = y.reset_index(drop=True)
    X_all, y_all = _augment_swap(X_all_orig, y_all_orig)
    scaler_full = StandardScaler()
    X_all_sc = pd.DataFrame(scaler_full.fit_transform(X_all), columns=X.columns)

    final_models = {}
    for name, make_model in _make_specs():
        if progress_cb:
            progress_cb(f"    {name}...")
        model = make_model()
        X_fit = X_all_sc if name in NEEDS_SCALE else X_all
        _fit_model(name, model, X_fit, y_all)
        final_models[name] = model

    if progress_cb:
        progress_cb("")
        progress_cb(f"  Model ready. | Test accuracy: {ensemble_acc:.1%} | Test log-loss: {ensemble_ll:.4f}")
        progress_cb("")

    return EnsembleModel(final_models, imputer_full, scaler_full,
                         X.columns.tolist(), weights)


# ─── Fuzzy name matching ──────────────────────────────────────────────────────

def fuzzy_find(name, fighter_history):
    if name in fighter_history and fighter_history[name]:
        return name
    lower = name.lower()
    for key in fighter_history:
        if key.lower() == lower and fighter_history[key]:
            return key
    matches = [k for k in fighter_history if lower in k.lower() and fighter_history[k]]
    if len(matches) == 1:
        return matches[0]
    return None


# ─── Excel export ─────────────────────────────────────────────────────────────

def _auto_width(ws):
    for col_idx in range(1, ws.max_column + 1):
        mx = 0
        for row in ws.iter_rows(min_col=col_idx, max_col=col_idx):
            for cell in row:
                try:
                    mx = max(mx, len(str(cell.value or "")))
                except Exception:
                    pass
        ws.column_dimensions[get_column_letter(col_idx)].width = mx + 3


def export_to_excel(output_path, predictions):
    wb = Workbook()
    hdr_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    hdr_font = Font(bold=True, size=11, color="000000")
    hdr_align = Alignment(horizontal="left", vertical="center")
    thin = Side(border_style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    pct_fmt = "0.0%"

    ws = wb.active
    ws.title = "Predictions"

    headers = ["Red Corner", "Blue Corner", "Weight Class", "Predicted Winner", "Win%"]
    for ci, hdr in enumerate(headers, 1):
        cell = ws.cell(row=1, column=ci, value=hdr)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = hdr_align
        cell.border = border

    for ri, p in enumerate(predictions, 2):
        winner = p["name_a"] if p["prob_a"] >= 0.5 else p["name_b"]
        win_pct = max(p["prob_a"], 1.0 - p["prob_a"])
        data = [
            p["name_a"], p["name_b"], p.get("weight_class", ""),
            winner, win_pct,
        ]
        for ci, val in enumerate(data, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(horizontal="left", vertical="center")
            cell.border = border
            if ci == 5:
                cell.number_format = pct_fmt

    _auto_width(ws)
    wb.save(output_path)


# ─── GUI ──────────────────────────────────────────────────────────────────────

BG = "#1a1a2e"
BG_HEADER = "#16213e"
BG_INPUT = "#0f3460"
FG = "#e0e0e0"
ACCENT = "#e94560"
ACCENT2 = "#533483"
MUTED = "#a0c4ff"
GREEN = "#4ade80"
BAR_A = "#e94560"
BAR_B = "#3b82f6"


class MLPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UFC ML FIGHT PREDICTOR")
        self.root.geometry("960x780")
        self.root.resizable(True, True)
        self.root.configure(bg=BG)

        self.model = None
        self.fighter_history = None
        self.glicko_ratings = None
        self.opp_glicko_list = None
        self._training = False

        self._build_ui()

    # ── UI ──

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=BG_HEADER, pady=12)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame, text="UFC ML FIGHT PREDICTOR",
            font=("Helvetica", 20, "bold"), fg=ACCENT, bg=BG_HEADER,
        ).pack()

        main = tk.Frame(self.root, bg=BG, padx=16, pady=10)
        main.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Enter matchups and click Predict.")
        self.status_label = tk.Label(
            main, textvariable=self.status_var, bg=BG, fg=MUTED,
            font=("Helvetica", 9, "italic"),
        )
        self.status_label.pack(anchor="w", pady=(0, 4))

        tk.Label(
            main,
            text="Enter matchups (Red Corner,BLue Corner,Weight Class,Gender,Rounds):",
            bg=BG, fg=MUTED, font=("Helvetica", 9, "italic"),
        ).pack(anchor="w", pady=(4, 2))

        self.fight_input = tk.Text(
            main, height=10, font=("Courier New", 10),
            bg=BG_INPUT, fg="white", insertbackground="white",
            relief="flat", wrap="word",
        )
        self.fight_input.pack(fill="x", pady=4)

        btn_frame = tk.Frame(main, bg=BG)
        btn_frame.pack(fill="x", pady=6)

        tk.Button(
            btn_frame, text="Load from Fights.csv",
            command=self._load_fights_csv,
            font=("Helvetica", 10, "bold"), bg=ACCENT2, fg="white",
            relief="flat", padx=14, cursor="hand2",
        ).pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="Clear", command=self._clear,
            font=("Helvetica", 10, "bold"), bg="#444466", fg="white",
            relief="flat", padx=14, cursor="hand2",
        ).pack(side="left", padx=4)

        self.predict_btn = tk.Button(
            btn_frame, text="Predict", command=self._predict,
            font=("Helvetica", 11, "bold"), bg=ACCENT, fg="white",
            relief="flat", padx=20, cursor="hand2",
        )
        self.predict_btn.pack(side="right", padx=4)

        results_frame = tk.Frame(main, bg=BG)
        results_frame.pack(fill="both", expand=True, pady=(6, 0))

        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")

        self.results_canvas = tk.Canvas(
            results_frame, bg=BG, highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        self.results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.results_canvas.yview)

        self.results_inner = tk.Frame(self.results_canvas, bg=BG)
        self.results_canvas.create_window((0, 0), window=self.results_inner, anchor="nw")
        self.results_inner.bind(
            "<Configure>",
            lambda _: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")),
        )
        self.results_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.results_canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

    # ── Helpers ──

    def _log(self, msg):
        """Print to terminal AND update GUI status bar."""
        print(msg, flush=True)
        self.status_var.set(msg)

    # ── Actions ──

    def _load_fights_csv(self):
        fights_path = os.path.join(SCRIPT_DIR, "Fights.csv")
        if not os.path.exists(fights_path):
            self._log("Fights.csv not found in script directory.")
            return
        self.fight_input.delete("1.0", tk.END)
        with open(fights_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not row or not row[0].strip():
                    continue
                self.fight_input.insert(tk.END, ",".join(c.strip() for c in row) + "\n")

    def _clear(self):
        self.fight_input.delete("1.0", tk.END)
        for w in self.results_inner.winfo_children():
            w.destroy()

    def _predict(self):
        text = self.fight_input.get("1.0", tk.END).strip()
        if not text:
            self._log("Enter at least one matchup.")
            return
        if self._training:
            return

        self._training = True
        self.predict_btn.config(state="disabled")

        # Clear previous results
        for w in self.results_inner.winfo_children():
            w.destroy()

        def _do():
            try:
                self._log("")
                self._log("  ==========================================")
                self._log("       UFC ML FIGHT PREDICTOR")
                self._log("  ==========================================")
                self._log("")

                # ── Step 1: Build training data ──
                data_path = os.path.join(SCRIPT_DIR, "pure_fight_data.csv")
                if not os.path.exists(data_path):
                    self._log(f"  Error: {data_path} not found")
                    return

                self._log("  Loading data...")
                X, y, fh, gr, og = build_training_data(data_path, self._log)
                self.fighter_history = fh
                self.glicko_ratings = gr
                self.opp_glicko_list = og

                # ── Step 2: Tune and train ──
                self.model = tune_and_train(X, y, n_trials=50, progress_cb=self._log,
                                           status_var=self.status_var)

                # ── Step 3: Predict matchups ──
                self._log("  Predicting matchups...")

                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                warnings_list = []
                predictions = []

                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    a_name, b_name = parts[0], parts[1]
                    weight_class = parts[2] if len(parts) > 2 else ""
                    rounds_sched = int(parts[4]) if len(parts) > 4 else 3
                    is_title = 1 if rounds_sched == 5 else 0

                    a_key = fuzzy_find(a_name, self.fighter_history)
                    b_key = fuzzy_find(b_name, self.fighter_history)

                    if not a_key:
                        warnings_list.append(f"{a_name}: not found, using defaults")
                        self._log(f"    WARNING: {a_name} not found, using default ratings")
                    if not b_key:
                        warnings_list.append(f"{b_name}: not found, using defaults")
                        self._log(f"    WARNING: {b_name} not found, using default ratings")

                    a_display = a_key or a_name
                    b_display = b_key or b_name

                    a_hist = self.fighter_history.get(a_key, []) if a_key else []
                    b_hist = self.fighter_history.get(b_key, []) if b_key else []
                    a_glicko = self.glicko_ratings.get(a_key, (MU_0, PHI_0, SIGMA_0)) if a_key else (MU_0, PHI_0, SIGMA_0)
                    b_glicko = self.glicko_ratings.get(b_key, (MU_0, PHI_0, SIGMA_0)) if b_key else (MU_0, PHI_0, SIGMA_0)
                    a_opp_g = self.opp_glicko_list.get(a_key, []) if a_key else []
                    b_opp_g = self.opp_glicko_list.get(b_key, []) if b_key else []

                    now = datetime.now()
                    a_feats = compute_fighter_features(a_hist, a_glicko, a_opp_g, now)
                    b_feats = compute_fighter_features(b_hist, b_glicko, b_opp_g, now)
                    matchup = compute_matchup_features(a_feats, b_feats, is_title, rounds_sched)

                    prob_a = self.model.predict_proba_single(matchup)
                    prob_b = 1.0 - prob_a

                    predictions.append({
                        "name_a": a_display, "name_b": b_display,
                        "prob_a": prob_a,
                        "weight_class": weight_class, "rounds": rounds_sched,
                    })

                    self._add_matchup_card(a_display, b_display, weight_class, prob_a, prob_b)

                # ── Step 4: Export ──
                output_path = os.path.join(SCRIPT_DIR, "ML_Predictions.xlsx")
                try:
                    export_to_excel(output_path, predictions)
                    export_msg = f"Saved to {os.path.basename(output_path)}"
                except Exception as e:
                    export_msg = f"Export failed: {e}"
                    messagebox.showerror("Export Error", str(e))

                self._log("")
                self._log(f"  {len(predictions)} matchups predicted. {export_msg}")
                if warnings_list:
                    self._log(f"  Warnings: {'; '.join(warnings_list[:3])}")
                self._log("")

            except Exception as e:
                self._log(f"Error: {e}")
                import traceback; traceback.print_exc()
            finally:
                self._training = False
                self.predict_btn.config(state="normal")

        threading.Thread(target=_do, daemon=True).start()

    def _add_matchup_card(self, name_a, name_b, weight_class, prob_a, prob_b):
        card = tk.Frame(self.results_inner, bg=BG_HEADER, pady=8, padx=12)
        card.pack(fill="x", pady=4, padx=4)

        names_frame = tk.Frame(card, bg=BG_HEADER)
        names_frame.pack(fill="x")

        a_is_fav = prob_a >= prob_b
        a_color = GREEN if a_is_fav else FG
        b_color = GREEN if not a_is_fav else FG

        tk.Label(
            names_frame, text=name_a,
            font=("Helvetica", 11, "bold"), fg=a_color, bg=BG_HEADER, anchor="w",
        ).pack(side="left")

        tk.Label(
            names_frame, text="vs",
            font=("Helvetica", 10), fg="#666", bg=BG_HEADER,
        ).pack(side="left", padx=10)

        tk.Label(
            names_frame, text=name_b,
            font=("Helvetica", 11, "bold"), fg=b_color, bg=BG_HEADER, anchor="e",
        ).pack(side="left")

        if weight_class:
            tk.Label(
                names_frame, text=f"  [{weight_class}]",
                font=("Helvetica", 9), fg=MUTED, bg=BG_HEADER,
            ).pack(side="left", padx=(10, 0))

        # Probability bar
        bar_frame = tk.Frame(card, bg=BG_HEADER, pady=4)
        bar_frame.pack(fill="x")

        BAR_W, BAR_H = 500, 24
        bar_c = tk.Canvas(bar_frame, width=BAR_W, height=BAR_H, bg=BG_HEADER, highlightthickness=0)
        bar_c.pack(anchor="w")

        a_w = max(1, int(prob_a * BAR_W))
        bar_c.create_rectangle(0, 0, a_w, BAR_H, fill=BAR_A, outline="")
        bar_c.create_rectangle(a_w, 0, BAR_W, BAR_H, fill=BAR_B, outline="")

        if prob_a >= 0.12:
            bar_c.create_text(a_w // 2, BAR_H // 2, text=f"{prob_a:.1%}",
                              fill="white", font=("Helvetica", 9, "bold"))
        if prob_b >= 0.12:
            bar_c.create_text(a_w + (BAR_W - a_w) // 2, BAR_H // 2, text=f"{prob_b:.1%}",
                              fill="white", font=("Helvetica", 9, "bold"))

        pick = f"{name_a} ({prob_a:.1%})" if a_is_fav else f"{name_b} ({prob_b:.1%})"
        tk.Label(
            card, text=f"Pick: {pick}",
            font=("Helvetica", 10), fg=GREEN, bg=BG_HEADER, anchor="w",
        ).pack(anchor="w")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    MLPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
