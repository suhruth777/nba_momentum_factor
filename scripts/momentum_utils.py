"""
Utilities for processing NBA play-by-play into a standardized scoring timeline
and deriving momentum-oriented features at the game level.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplayv2, leaguegamelog


# ----------------------------
# Parsing and processing utils
# ----------------------------

def _time_to_seconds(t: str):
    """Convert 'MM:SS' to seconds; return NaN if malformed."""
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def process_pbp(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process play-by-play into a scoring timeline with OT handling, differential,
    momentum index, scoring runs, and per-event lead swings.

    Returns a tidy event-level DataFrame sorted chronologically.
    """
    df = df_raw.copy()

    # Keep core columns (only those that exist)
    cols_keep = [
        "GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING",
        "HOMEDESCRIPTION", "VISITORDESCRIPTION",
        "SCORE", "EVENTMSGTYPE"
    ]
    df = df.loc[:, [c for c in cols_keep if c in df.columns]].copy()

    # Scoring events only: 1 = made FG, 3 = FT
    df = df[df["EVENTMSGTYPE"].isin([1, 3])].copy()

    # Parse SCORE -> cumulative home/away
    split_ = df["SCORE"].astype(str).str.split("-", n=1, expand=True)
    df["home_score"] = pd.to_numeric(split_[0], errors="coerce")
    df["away_score"] = pd.to_numeric(split_[1], errors="coerce")
    # Forward-fill partial-score rows (e.g., first of two FTs)
    df[["home_score", "away_score"]] = (
        df[["home_score", "away_score"]].ffill().fillna(0).astype(int)
    )

    # Normalize time to absolute seconds from tipoff (reg=720s/period; OT=300s)
    df["seconds_remaining_in_period"] = df["PCTIMESTRING"].astype(str).apply(_time_to_seconds)
    df = df.dropna(subset=["seconds_remaining_in_period"])

    per_len = np.where(df["PERIOD"] <= 4, 720, 300)
    elapsed_in_period = per_len - df["seconds_remaining_in_period"]

    p = df["PERIOD"].astype(int)
    base_offset = 720 * np.minimum(p - 1, 4) + 300 * np.maximum(p - 1 - 4, 0)
    df["total_seconds_elapsed"] = base_offset + elapsed_in_period

    # Chronological order
    if "EVENTNUM" in df.columns:
        df = df.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    else:
        df = df.sort_values(["PERIOD", "total_seconds_elapsed"]).reset_index(drop=True)

    # Differential and smoothed momentum
    df["score_diff"] = df["home_score"] - df["away_score"]
    df["momentum_index"] = df["score_diff"].rolling(window=10, min_periods=1).mean()

    # Who scored on each event (object dtype to avoid str/NaN dtype promotion issues)
    dh = df["home_score"].diff().fillna(0)
    da = df["away_score"].diff().fillna(0)
    df["scoring_team"] = pd.Series(index=df.index, dtype="object")
    df.loc[dh > 0, "scoring_team"] = "home"
    df.loc[da > 0, "scoring_team"] = "away"

    # Run IDs and run points
    df["run_id"] = (df["scoring_team"] != df["scoring_team"].shift()).cumsum()

    df["points_scored"] = 0
    df.loc[dh > 0, "points_scored"] = dh[dh > 0].astype(int)
    df.loc[da > 0, "points_scored"] = da[da > 0].astype(int)

    # Per-event swing magnitude
    df["swing_magnitude"] = df["score_diff"].diff().abs().fillna(0)

    # Final tidy set
    keep = [
        "GAME_ID", "PERIOD", "PCTIMESTRING", "total_seconds_elapsed",
        "HOMEDESCRIPTION", "VISITORDESCRIPTION",
        "home_score", "away_score", "score_diff", "momentum_index",
        "scoring_team", "run_id", "points_scored", "swing_magnitude"
    ]
    return df.loc[:, [c for c in keep if c in df.columns]].copy()


def feature_pack(df: pd.DataFrame) -> dict:
    """
    Derive game-level features used in season-scale analysis.
    Home perspective: positive diff implies home lead.

    Returns a dict of scalar features.
    """
    final_diff = float(df["score_diff"].iloc[-1])
    home_win = 1 if final_diff > 0 else 0
    ot_game = 1 if df["PERIOD"].max() > 4 else 0

    # Time anchors (seconds)
    HT, Q4_START, Q4_END = 1440, 2160, 2880

    def at_or_before(t, col):
        sub = df[df["total_seconds_elapsed"] <= t]
        return float(sub[col].iloc[-1]) if len(sub) else np.nan

    diff_ht = at_or_before(HT, "score_diff")
    diff_q4 = at_or_before(Q4_START, "score_diff")
    diff_l2 = at_or_before(Q4_END - 120, "score_diff")

    # Late windows (fallback to tail if sparse)
    late2 = df[(df["PERIOD"] == 4) & (df["total_seconds_elapsed"] >= Q4_END - 120) & (df["total_seconds_elapsed"] <= Q4_END)]
    if late2.empty:
        late2 = df.tail(10)
    m_idx_l2_mean = float(late2["momentum_index"].mean())

    run_summary = (
        df.groupby("run_id", dropna=True)
        .agg(team=("scoring_team", "first"),
             events=("scoring_team", "size"),
             points=("points_scored", "sum"))
        .reset_index()
    )
    max_run_pts = float(run_summary["points"].max()) if len(run_summary) else np.nan
    avg_run_pts = float(run_summary["points"].mean()) if len(run_summary) else np.nan

    var_diff = float(np.var(df["score_diff"]))
    stability = float("inf") if var_diff == 0 else 1.0 / var_diff

    # Last lead change timing (last sign flip)
    sgn = np.sign(df["score_diff"].replace(0, np.nan)).ffill()
    flips = df.loc[sgn.ne(sgn.shift()), "total_seconds_elapsed"]
    last_flip_time = float(flips.iloc[-1]) if len(flips) else np.nan
    leader_after_last_flip = (
        1 if (len(flips) and df.loc[df["total_seconds_elapsed"] >= last_flip_time, "score_diff"].iloc[0] > 0)
        else (0 if len(flips) else np.nan)
    )
    decisive_flip = (leader_after_last_flip == home_win) if not np.isnan(leader_after_last_flip) else np.nan

    return {
        "game_id": df["GAME_ID"].iloc[0],
        "home_win": home_win,
        "final_margin": abs(final_diff),
        "ot_game": ot_game,
        "diff_halftime": diff_ht,
        "diff_q4start": diff_q4,
        "diff_at_2min": diff_l2,
        "momentum_l2_mean": m_idx_l2_mean,
        "max_run_points": max_run_pts,
        "avg_run_points": avg_run_pts,
        "stability_invvar": stability,
        "last_lead_change_time": last_flip_time,
        "decisive_last_lead_change": decisive_flip,
    }


# ----------------------------
# API helpers for season work
# ----------------------------

def fetch_process_by_game_id(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play via nba_api for a game_id and return the processed event-level DataFrame."""
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    df_raw = pbp.get_data_frames()[0]
    return process_pbp(df_raw)


def get_season_game_ids(season: str, season_type: str = "Regular Season") -> list[str]:
    """Return unique GAME_IDs for a season/season_type (each game appears twice in logs; de-duplicate)."""
    gl = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type).get_data_frames()[0]
    return gl["GAME_ID"].drop_duplicates().tolist()


def build_season_dataset(game_ids: list[str], sample_size: int | None = 200, sleep_s: float = 0.7) -> pd.DataFrame:
    """
    Iterate over game_ids (optionally a sample), fetch PBP, process, and assemble game-level features.
    Adds light rate limiting to respect nba_api endpoint throttling.
    """
    out = []
    taken = game_ids if sample_size is None else game_ids[:sample_size]
    for i, gid in enumerate(taken, 1):
        try:
            df_proc = fetch_process_by_game_id(gid)
            feats = feature_pack(df_proc)
            out.append(feats)
            if i % 25 == 0:
                print(f"Processed {i}/{len(taken)} games")
        except Exception as e:
            print(f"Skipping {gid} due to error: {e}")
        time.sleep(sleep_s)
    return pd.DataFrame(out)


if __name__ == "__main__":
    # Optional: quick smoke
    from pprint import pprint
    try:
        df = fetch_process_by_game_id("0022300001")
        print(df[["home_score","away_score","score_diff","momentum_index"]].head())
        pprint(feature_pack(df))
    except Exception as e:
        print("Smoke test failed:", e)