"""
Per-fighter offensive/defensive rate library built from pure_fight_data.csv.

Used by Monte_Carlo.py to project prop bet totals (significant strikes, total
strikes, takedowns, knockdowns) for upcoming matchups. Rates are time-decayed
(2-year half-life, matching UFC_Model._time_weight) and Bayesian-shrunk toward
population means so fighters with thin samples don't get extreme projections.

Workflow:
    library = build_fighter_stats('pure_fight_data.csv')
    proj = project_matchup_rates('Aljamain Sterling', 'Youssef Zalal', library)
    # proj['rates_a']['sig_str_landed'] is per-minute rate for A in this matchup
"""

import difflib
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


# Stats we accumulate. Each maps to (red_col, blue_col) in pure_fight_data.csv.
# When red lands X, that's offense for red and "absorbed" (defense allowed) for blue.
STAT_COLS = {
    "sig_str_landed":    ("r_sig_str",     "b_sig_str"),
    "sig_str_attempted": ("r_sig_str_att", "b_sig_str_att"),
    "total_str_landed":  ("r_str",         "b_str"),
    "td_landed":         ("r_td",          "b_td"),
    "td_attempted":      ("r_td_att",      "b_td_att"),
    "kd_landed":         ("r_kd",          "b_kd"),
    "head_landed":       ("r_head",        "b_head"),
    "body_landed":       ("r_body",        "b_body"),
    "leg_landed":        ("r_leg",         "b_leg"),
    "sub_att":           ("r_sub_att",     "b_sub_att"),
    "ctrl_sec":          ("r_ctrl_sec",    "b_ctrl_sec"),
}

# Match UFC_Model.py's _time_weight default.
HALF_LIFE_DAYS = 730
# Bayesian shrinkage: equivalent to "fight-minutes worth" of population prior.
SHRINK_MINUTES = 25.0


def _is_finish(method_str):
    if not method_str:
        return False
    m = str(method_str).lower()
    return "decision" not in m and m != "draw"


def _classify_method(method_str):
    """Map raw method string to one of: 'Decision', 'KO/TKO', 'Submission', or None."""
    if not method_str:
        return None
    m = str(method_str).lower()
    if "decision" in m:
        return "Decision"
    if m == "draw":
        return None
    if "submission" in m:
        return "Submission"
    if "ko" in m or "tko" in m or "doctor" in m or "dq" in m:
        return "KO/TKO"
    return None


def build_fighter_stats(csv_path, as_of_date=None):
    """Return a library mapping fighter name → per-minute offensive & defensive rates,
    plus winning-finish-time pools per method, plus a "__population__" fallback."""
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["event_date", "total_fight_time_sec", "r_name", "b_name"])
    df = df[df["total_fight_time_sec"] > 0]

    as_of = pd.Timestamp(as_of_date or datetime.now().date())

    acc = defaultdict(lambda: {
        "off": defaultdict(float),
        "def_": defaultdict(float),
        "minutes": 0.0,
        "fights": 0,
        # Per-method finish times when this fighter won (list of minutes).
        "finish_times": defaultdict(list),
    })
    pop_landed = defaultdict(lambda: {"total": 0.0, "minutes": 0.0})
    # Population finish pool keyed by (method, total_rounds_scheduled).
    pop_finish = defaultdict(list)

    for row in df.itertuples(index=False):
        days_ago = (as_of - row.event_date).days
        w = 0.5 ** (days_ago / HALF_LIFE_DAYS) if days_ago >= 0 else 0.0
        if w <= 0:
            continue
        minutes = row.total_fight_time_sec / 60.0
        wm = w * minutes

        rounds_val = getattr(row, "total_rounds", None)
        rounds = int(rounds_val) if rounds_val and not pd.isna(rounds_val) else None
        method_class = _classify_method(getattr(row, "method", None))
        winner_corner = getattr(row, "winner", None)

        if rounds and method_class in ("KO/TKO", "Submission"):
            pop_finish[(method_class, rounds)].append(minutes)
            # Attribute the finish time to the winning corner.
            if winner_corner == "Red":
                acc[row.r_name]["finish_times"][method_class].append(minutes)
            elif winner_corner == "Blue":
                acc[row.b_name]["finish_times"][method_class].append(minutes)

        r, b = row.r_name, row.b_name
        for stat, (rcol, bcol) in STAT_COLS.items():
            r_val = getattr(row, rcol, None)
            b_val = getattr(row, bcol, None)
            if r_val is None or b_val is None or pd.isna(r_val) or pd.isna(b_val):
                continue
            r_val, b_val = float(r_val), float(b_val)

            acc[r]["off"][stat] += w * r_val
            acc[b]["def_"][stat] += w * r_val
            acc[b]["off"][stat] += w * b_val
            acc[r]["def_"][stat] += w * b_val

            pop_landed[stat]["total"] += w * (r_val + b_val)
            pop_landed[stat]["minutes"] += wm * 2

        acc[r]["minutes"] += wm
        acc[b]["minutes"] += wm
        acc[r]["fights"] += 1
        acc[b]["fights"] += 1

    pop_rates = {
        stat: (vals["total"] / vals["minutes"]) if vals["minutes"] > 0 else 0.0
        for stat, vals in pop_landed.items()
    }

    library = {}
    for fighter, data in acc.items():
        m = data["minutes"]
        if m <= 0:
            continue
        rates = {"fights": data["fights"], "minutes": m}
        for stat in STAT_COLS:
            prior = pop_rates.get(stat, 0.0)
            rates[f"{stat}_off_pm"] = (data["off"][stat] + prior * SHRINK_MINUTES) / (m + SHRINK_MINUTES)
            rates[f"{stat}_def_pm"] = (data["def_"][stat] + prior * SHRINK_MINUTES) / (m + SHRINK_MINUTES)
        rates["finish_times"] = {
            method: np.array(times) for method, times in data["finish_times"].items() if times
        }
        library[fighter] = rates

    # Population fallback pools: keyed by (method, rounds), plus an all-finishes catch-all.
    all_finishes = [t for ts in pop_finish.values() for t in ts]
    library["__population__"] = {
        "rates": pop_rates,
        "finish_pool_by_method_rounds": {k: np.array(v) for k, v in pop_finish.items()},
        "finish_pool_all": np.array(all_finishes) if all_finishes else np.array([5.0]),
    }
    return library


def fuzzy_lookup(name, library, cache=None):
    """Find a fighter in the library tolerating case + minor spelling variation."""
    if cache is not None and name in cache:
        return cache[name]
    result = None
    if name in library and not name.startswith("__"):
        result = library[name]
    else:
        lower_map = {k.casefold(): k for k in library if not k.startswith("__")}
        key = lower_map.get(name.casefold())
        if key:
            result = library[key]
        else:
            close = difflib.get_close_matches(name.casefold(), lower_map.keys(), n=1, cutoff=0.88)
            if close:
                result = library[lower_map[close[0]]]
    if cache is not None:
        cache[name] = result
    return result


def project_matchup_rates(name_a, name_b, library, lookup_cache=None):
    """Per-stat per-minute landing rates for A and B in this matchup.

    Uses geometric blend: A_landed_pm = sqrt(A_offense × B_defense_allowed).
    Geometric is symmetric and self-correcting if one side has weird outlier rates.
    Falls back to population mean per stat for fighters not found.
    """
    pop = library.get("__population__", {}).get("rates", {})
    a = fuzzy_lookup(name_a, library, lookup_cache)
    b = fuzzy_lookup(name_b, library, lookup_cache)

    rates_a, rates_b = {}, {}
    for stat in STAT_COLS:
        pop_rate = pop.get(stat, 0.0)
        a_off = a.get(f"{stat}_off_pm", pop_rate) if a else pop_rate
        b_def = b.get(f"{stat}_def_pm", pop_rate) if b else pop_rate
        b_off = b.get(f"{stat}_off_pm", pop_rate) if b else pop_rate
        a_def = a.get(f"{stat}_def_pm", pop_rate) if a else pop_rate
        rates_a[stat] = float(np.sqrt(max(a_off, 1e-9) * max(b_def, 1e-9)))
        rates_b[stat] = float(np.sqrt(max(b_off, 1e-9) * max(a_def, 1e-9)))

    return {
        "rates_a": rates_a,
        "rates_b": rates_b,
        "a_found": a is not None,
        "b_found": b is not None,
        "a_minutes": a.get("minutes", 0.0) if a else 0.0,
        "b_minutes": b.get("minutes", 0.0) if b else 0.0,
    }


PERSONAL_POOL_MIN = 3  # require at least this many personal finishes to use them


def _resolve_finish_pool(name, method, rounds, library, lookup_cache):
    """Return the empirical finish-time pool to bootstrap from for (winner, method, rounds).

    Uses the fighter's own finishes when they have ≥ PERSONAL_POOL_MIN of that method,
    blended 50/50 with the population pool for that (method, rounds) bucket. Falls back
    to population alone if personal data is too thin.
    """
    pop = library["__population__"]
    pop_pool = pop["finish_pool_by_method_rounds"].get((method, int(rounds)))
    if pop_pool is None or len(pop_pool) == 0:
        pop_pool = pop["finish_pool_all"]

    fighter = fuzzy_lookup(name, library, lookup_cache)
    personal = None
    if fighter is not None:
        personal = fighter.get("finish_times", {}).get(method)
    if personal is not None and len(personal) >= PERSONAL_POOL_MIN:
        # Blend personal with population to avoid overfitting to a tiny pool.
        return np.concatenate([personal, pop_pool])
    return pop_pool


def sample_durations(card, winners, methods, library, rng, lookup_cache=None):
    """Sample per-trial fight durations (minutes), conditioned on winner + method.

    `winners`: (n_trials, n_fights) int8 with 0=fighter A, 1=fighter B.
    `methods`: (n_trials, n_fights) int8 with 0=Decision, 1=KO/TKO, 2=Submission.
    `card`:   list of dicts with `name_a`, `name_b`, `rounds` per fight.

    Decisions get full duration (rounds × 5). For finishes, bootstraps from the
    winner's personal finish-time pool (for that method) when they have enough
    history, else from the population pool for that (method, rounds) bucket.
    Sampled times are clipped to [10s, rounds × 5 min].
    """
    n_trials, n_fights = methods.shape
    decision_full = np.array([f["rounds"] for f in card], dtype=float) * 5.0
    durations = np.zeros((n_trials, n_fights), dtype=float)

    for fi, f in enumerate(card):
        rounds = f["rounds"]
        max_minutes = rounds * 5.0
        pools = {
            (winner_idx, method_idx): _resolve_finish_pool(
                f["name_a"] if winner_idx == 0 else f["name_b"],
                "KO/TKO" if method_idx == 1 else "Submission",
                rounds, library, lookup_cache,
            )
            for winner_idx in (0, 1) for method_idx in (1, 2)
        }
        # Pre-sample n_trials draws from each (winner, method) pool.
        draws = {k: rng.choice(np.clip(v, 10/60.0, max_minutes), size=n_trials) for k, v in pools.items()}

        fight_methods = methods[:, fi]
        fight_winners = winners[:, fi]
        # Build duration vector: decision → full; otherwise pick the matching draw.
        dur = np.where(fight_methods == 0, decision_full[fi], 0.0)
        for (wi, mi), arr in draws.items():
            mask = (fight_winners == wi) & (fight_methods == mi)
            dur = np.where(mask, arr, dur)
        durations[:, fi] = dur

    return durations


def round_of_finish(durations, methods, rounds_per_fight):
    """Compute which round each finish occurred in (1-indexed). Returns 0 for decisions."""
    rounds_arr = np.array(rounds_per_fight, dtype=int)
    rd = np.ceil(durations / 5.0).astype(int)
    rd = np.minimum(rd, rounds_arr[None, :])
    rd = np.maximum(rd, 1)
    return np.where(methods == 0, 0, rd)
