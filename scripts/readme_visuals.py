#!/usr/bin/env python3
"""
Generates README figures from the season-scale export (1 row = 1 game).
- Reads:   data/tableau_momentum_export.csv
- Writes:  visuals/decisive_share.png
           visuals/late_momentum_alignment.png
           visuals/near_tie_winrate.png
           visuals/auc_comparison.png
           README_momentum_section.md  (markdown block to paste into README)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: compute AUC directly from the export (no need to re-run the notebook)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


# ---------------------------
# Config (tweak if you want)
# ---------------------------
NEAR_TIE_THRESHOLD = 2      # |diff_at_2min| <= 2
DECISIVE_CUTOFF_MIN = 5     # "Decisive" = last flip occurs before 5:00
EXPORT_PATH = Path("data/tableau_momentum_export.csv")   # relative to repo root
VIS_DIR = Path("visuals")
READOUT_MD = Path("README_momentum_section.md")


# ---------------------------
# Helpers
# ---------------------------
def check_required_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {EXPORT_PATH}:\n"
            f"  {missing}\n\n"
            "Regenerate the export in your 02_season_scale_analysis notebook (Step 6.4)."
        )

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2/(2*n)
    adj = z*np.sqrt((p*(1-p) + z**2/(4*n))/n)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return max(0.0, low), min(1.0, high)

def set_pct_axis(ax):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

def annotate_percent(ax, rects):
    for r in rects:
        v = r.get_height()
        ax.text(r.get_x() + r.get_width()/2.0, v + 0.01, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10)


# ---------------------------
# Load data
# ---------------------------
if not EXPORT_PATH.exists():
    raise SystemExit(
        f"Could not find {EXPORT_PATH}. Make sure you ran Step 6.4 to export the CSV."
    )

df = pd.read_csv(EXPORT_PATH)

required = [
    "game_id","home_win","ot_game",
    "last_lead_change_time","decisive_last_lead_change",
    "diff_q4start","diff_at_2min","momentum_l2_mean",
    "max_run_points","stability_invvar",
    # Helpers (if your export included them; otherwise we re-compute)
    # "decided_before_5min","late_momentum_sign","winner_sign","near_tie_flag"
]
check_required_cols(df, required)

# Compute helper columns if not present
if "decided_before_5min" not in df.columns:
    df["decided_before_5min"] = (df["last_lead_change_time"] <= (2880 - DECISIVE_CUTOFF_MIN*60)).astype(int)

if "late_momentum_sign" not in df.columns:
    df["late_momentum_sign"] = np.sign(df["momentum_l2_mean"]).replace(0, np.nan)

if "winner_sign" not in df.columns:
    df["winner_sign"] = np.where(df["home_win"] == 1, 1, -1).astype(int)

if "near_tie_flag" not in df.columns:
    df["near_tie_flag"] = (df["diff_at_2min"].abs() <= NEAR_TIE_THRESHOLD).astype(int)


# ---------------------------
# 1) Decisive share (≤ cutoff vs > cutoff) + CIs
# ---------------------------
mask = df["last_lead_change_time"].notna()
n_decisive = int(df.loc[mask, "decided_before_5min"].sum())
n_total = int(mask.sum())
p_decisive = n_decisive / n_total if n_total else np.nan
p_late = 1.0 - p_decisive if n_total else np.nan
ci_dec_lo, ci_dec_hi = wilson_interval(n_decisive, n_total)
ci_late_lo, ci_late_hi = wilson_interval(n_total - n_decisive, n_total)

VIS_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
labels = ["Decisive ≤ cutoff", "Last flip > cutoff"]
vals = [p_decisive, p_late]
errs_low = [p_decisive - ci_dec_lo, p_late - ci_late_lo]
errs_hi = [ci_dec_hi - p_decisive, ci_late_hi - p_late]
bars = ax.bar(labels, vals, width=0.6, yerr=[errs_low, errs_hi], capsize=6)
set_pct_axis(ax)
ax.set_title(f"When Is the Last Lead Change? (cutoff = {DECISIVE_CUTOFF_MIN}:00)")
annotate_percent(ax, bars)
fig.tight_layout()
fig_path_decisive = VIS_DIR / "decisive_share.png"
fig.savefig(fig_path_decisive, dpi=160)


# ---------------------------
# 2) Late momentum alignment + CI
# ---------------------------
m = df["late_momentum_sign"].notna() & df["winner_sign"].notna()
k_align = int((np.sign(df.loc[m, "late_momentum_sign"]) == df.loc[m, "winner_sign"]).sum())
n_align = int(m.sum())
p_align = k_align / n_align if n_align else np.nan
ci_a_lo, ci_a_hi = wilson_interval(k_align, n_align)

fig, ax = plt.subplots(figsize=(6.0, 4.5))
bars = ax.bar(["Alignment (final 2:00)"], [p_align], width=0.6,
              yerr=[[p_align - ci_a_lo], [ci_a_hi - p_align]], capsize=6)
set_pct_axis(ax)
ax.axhline(0.5, linestyle="--", linewidth=1)  # 50% baseline
ax.text(0.02, 0.51, "50% baseline", transform=ax.get_yaxis_transform(), fontsize=9)
ax.set_title("Late Momentum Alignment with Winner")
annotate_percent(ax, bars)
fig.tight_layout()
fig_path_align = VIS_DIR / "late_momentum_alignment.png"
fig.savefig(fig_path_align, dpi=160)


# ---------------------------
# 3) Near-tie win rates by momentum side + CIs
# ---------------------------
nt = df[(df["near_tie_flag"] == 1) & df["late_momentum_sign"].notna()]

home_m = nt[nt["late_momentum_sign"] > 0]
away_m = nt[nt["late_momentum_sign"] < 0]

k_home = int(home_m["home_win"].sum())
n_home = int(len(home_m))
p_home = k_home / n_home if n_home else np.nan
ci_h_lo, ci_h_hi = wilson_interval(k_home, n_home)

k_away = int((1 - away_m["home_win"]).sum())  # away wins
n_away = int(len(away_m))
p_away = k_away / n_away if n_away else np.nan
ci_aw_lo, ci_aw_hi = wilson_interval(k_away, n_away)

fig, ax = plt.subplots(figsize=(7.0, 4.5))
labels = ["Momentum favors HOME", "Momentum favors AWAY"]
vals = [p_home, p_away]
err_low = [p_home - ci_h_lo, p_away - ci_aw_lo]
err_hi = [ci_h_hi - p_home, ci_aw_hi - p_away]
bars = ax.bar(labels, vals, width=0.6, yerr=[err_low, err_hi], capsize=6)
set_pct_axis(ax)
ax.set_title(f"Near-Tie (|diff at 2:00| ≤ {NEAR_TIE_THRESHOLD}): Win Rate by Late Momentum Side")
annotate_percent(ax, bars)
fig.tight_layout()
fig_path_neartie = VIS_DIR / "near_tie_winrate.png"
fig.savefig(fig_path_neartie, dpi=160)


# ---------------------------
# 4) AUC comparison (baseline vs +momentum) — computed here
# ---------------------------
model_df = df.dropna(subset=[
    "home_win", "diff_q4start", "diff_at_2min", "ot_game",
    "momentum_l2_mean", "max_run_points", "stability_invvar"
]).copy()

baseline_feats = ["diff_q4start", "diff_at_2min", "ot_game"]
momentum_feats = ["momentum_l2_mean", "max_run_points", "stability_invvar"]

X_base = model_df[baseline_feats].values
X_full = model_df[baseline_feats + momentum_feats].values
y = model_df["home_win"].astype(int).values

Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.25, random_state=42, stratify=y)
Xf_tr, Xf_te, _, _   = train_test_split(X_full, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
pipe.fit(Xb_tr, y_tr); auc_base = roc_auc_score(y_te, pipe.predict_proba(Xb_te)[:,1])
pipe.fit(Xf_tr, y_tr); auc_full = roc_auc_score(y_te, pipe.predict_proba(Xf_te)[:,1])
delta_auc = auc_full - auc_base

fig, ax = plt.subplots(figsize=(7.0, 4.5))
labels = ["Baseline", "+Momentum"]
vals = [auc_base, auc_full]
bars = ax.bar(labels, vals, width=0.6)
# zoom in to show the difference more clearly
ymin = max(0.5, min(vals) - 0.05)
ymax = min(1.0, max(vals) + 0.05)
ax.set_ylim(ymin, ymax)
yticks = [round(x, 3) for x in np.linspace(ymin, ymax, 6)]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.3f}" for t in yticks])
ax.set_title("Predictive Discrimination (AUC)")
for r, v in zip(bars, vals):
    ax.text(r.get_x()+r.get_width()/2, v + (ymax-ymin)*0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.annotate(f"Δ = +{delta_auc:.3f}",
            xy=(0.5, (auc_base+auc_full)/2),
            xytext=(1.25, max(vals)-(ymax-ymin)*0.05),
            arrowprops=dict(arrowstyle="->", linewidth=1))
fig.tight_layout()
fig_path_auc = VIS_DIR / "auc_comparison.png"
fig.savefig(fig_path_auc, dpi=160)


# ---------------------------
# README block
# ---------------------------
readme_block = f"""
## Momentum at Scale — Key Findings

**Data & Methods.** Event-level play-by-play is converted into a continuous scoring differential (home − away). A short-horizon **Momentum Index** (rolling average over recent events) summarizes flow. Game-level signals include the **time of the last lead change**, **late momentum** (mean Momentum Index in the final 2:00), **run magnitude**, and **stability** (inverse variance). We evaluate decision relevance and predictive value across a season sample.

### 1) When do games get decided?
![Decisive last lead change share](visuals/decisive_share.png)

The final lead change occurs **before the {DECISIVE_CUTOFF_MIN}:00 mark** in roughly **{p_decisive:.1%}** of games (95% CI {ci_dec_lo:.1%}–{ci_dec_hi:.1%}). Only **{p_late:.1%}** of games see the last flip after the cutoff.

### 2) Does late momentum align with winners?
![Late momentum alignment](visuals/late_momentum_alignment.png)

Across all games with valid late-momentum readings, the sign of the **final-2:00 Momentum Index** aligns with the eventual winner **{p_align:.1%}** of the time (95% CI {ci_a_lo:.1%}–{ci_a_hi:.1%}).

### 3) What happens when the score is basically tied?
![Near-tie win rate by late momentum side](visuals/near_tie_winrate.png)

Conditioning on **near-tie states** at 2:00 (|score diff| ≤ {NEAR_TIE_THRESHOLD}), the team with late momentum wins about **{p_home:.1%}** when momentum favors **home** (95% CI {ci_h_lo:.1%}–{ci_h_hi:.1%}) and **{p_away:.1%}** when momentum favors **away** (95% CI {ci_aw_lo:.1%}–{ci_aw_hi:.1%}). Momentum remains outcome-relevant even after controlling for a tight scoreboard.

### 4) Is momentum predictive beyond score checkpoints?
![AUC comparison](visuals/auc_comparison.png)

A baseline classifier using score checkpoints (start of Q4, score at 2:00) and an OT flag achieves **AUC = {auc_base:.3f}**. Adding momentum features (late Momentum Index, max run points, stability) improves performance to **AUC = {auc_full:.3f}** (**Δ = +{delta_auc:.3f}**). The baseline is already strong; the incremental lift indicates momentum adds predictive signal rather than merely replaying the score.

**Caveats.** Results are associational and depend on window definitions. Robustness checks (alt windows 1–3 minutes, decisive cutoff 3–7 minutes, OT exclusion) are recommended to validate stability. Team strength and opponent effects can be layered in future iterations.
"""
READOUT_MD.write_text(readme_block.strip() + "\n")

print("Saved figures to:", VIS_DIR.resolve())
print(" -", fig_path_decisive)
print(" -", fig_path_align)
print(" -", fig_path_neartie)
print(" -", fig_path_auc)
print("Wrote README block →", READOUT_MD)