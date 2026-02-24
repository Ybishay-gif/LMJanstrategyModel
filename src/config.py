from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DEFAULT_PATHS = {
    "state_strategy": "data/state_strategy.txt",
    "state_data": "data/state_data.csv",
    "state_seg": "data/state_seg_data.csv",
    "channel_group": "data/channel_group_data.csv",
    "channel_price_exp": "data/channel_price_exploration_state.csv",
    "channel_state": "data/channel_group_state.csv",
}
OVERRIDES_PATH = Path("data/manual_overrides.json")
AUTH_USERS_PATH = Path("data/auth_users.json")
AUTH_ALLOWLIST_PATH = Path("data/allowed_emails.txt")
ADMIN_EMAIL = "ybishay@kissterra.com"
SESSION_TTL_DAYS = 7

STRATEGY_SCALE = {
    "Strongest Momentum": 1.00,
    "Moderate Momentum": 0.80,
    "Minimal Growth": 0.55,
    "LTV Constrained": 0.35,
    "Closure Constrained": 0.30,
    "Inactive/Low Spend": 0.10,
}

STRATEGY_COLOR = {
    "Strongest Momentum": "#FFD400",
    "Moderate Momentum": "#74D2D4",
    "LTV Constrained": "#B2ACE6",
    "Closure Constrained": "#1B185A",
    "Minimal Growth": "#B5B5B5",
    "Inactive/Low Spend": "#F3F4F6",
}

PERFORMANCE_GROUP_COLOR = {
    "Top Performance": "#16A34A",
    "Strong Performance": "#4ADE80",
    "Balanced": "#A3E635",
    "Mixed Risk": "#FACC15",
    "Weak Performance": "#F97316",
    "Poor Performance": "#DC2626",
    "Low Sig - Review": "#94A3B8",
}

CONFLICT_PERF_COLOR = {
    "Good | Full Match": "#166534",
    "Good | Small Conflict": "#22C55E",
    "Good | High Conflict": "#86EFAC",
    "OK | Full Match": "#A16207",
    "OK | Small Conflict": "#EAB308",
    "OK | High Conflict": "#FDE047",
    "Poor | Full Match": "#991B1B",
    "Poor | Small Conflict": "#EF4444",
    "Poor | High Conflict": "#FCA5A5",
    "Unknown | Unknown": "#94A3B8",
}

STATE_CENTER = {
    "AL": (32.7, -86.7), "AK": (64.2, -149.5), "AZ": (34.3, -111.7), "AR": (34.9, -92.4),
    "CA": (37.3, -119.7), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
    "FL": (27.8, -81.7), "GA": (32.7, -83.3), "HI": (20.8, -157.5), "ID": (44.1, -114.7),
    "IL": (40.0, -89.2), "IN": (40.0, -86.1), "IA": (42.1, -93.5), "KS": (38.5, -98.0),
    "KY": (37.5, -85.3), "LA": (31.2, -92.3), "ME": (45.2, -69.0), "MD": (39.0, -76.7),
    "MA": (42.3, -71.8), "MI": (44.3, -85.6), "MN": (46.4, -94.6), "MS": (32.7, -89.7),
    "MO": (38.6, -92.6), "MT": (46.9, -110.0), "NE": (41.5, -99.8), "NV": (38.8, -116.4),
    "NH": (43.7, -71.6), "NJ": (40.1, -74.7), "NM": (34.5, -106.1), "NY": (42.9, -75.0),
    "NC": (35.5, -79.4), "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5),
    "OR": (44.0, -120.5), "PA": (41.0, -77.5), "RI": (41.7, -71.6), "SC": (33.8, -80.9),
    "SD": (44.5, -100.2), "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
    "VT": (44.1, -72.7), "VA": (37.5, -78.7), "WA": (47.4, -120.7), "WV": (38.6, -80.6),
    "WI": (44.5, -89.8), "WY": (43.0, -107.6), "DC": (38.9, -77.0),
}

DARK_CSS = """
<style>
:root {
  --bg: #0b1220;
  --panel: #121a2b;
  --panel-soft: #18233a;
  --text: #e6edf9;
  --muted: #97a6c2;
  --accent: #3dd6ff;
  --good: #4ade80;
}
.stApp {
  background: radial-gradient(circle at 20% 0%, #1a2440 0%, #0b1220 48%, #080d18 100%);
  color: var(--text);
}
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #0a1020 100%);
}
.hero-card {
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(61,214,255,0.18), rgba(96,165,250,0.10));
  border: 1px solid rgba(61,214,255,0.3);
  margin-bottom: 12px;
}
.tiny-note {
  color: var(--muted);
  font-size: 0.90rem;
}
.conflict-legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-top:8px; }
.conflict-item { display:inline-flex; align-items:center; gap:8px; color:var(--text); font-size:0.85rem; }
.swatch { width:26px; height:16px; border:1px solid rgba(148,163,184,0.6); border-radius:4px; }
.striped {
  background: repeating-linear-gradient(
    45deg,
    #f3f4f6 0px,
    #f3f4f6 12px,
    #e5e7eb 12px,
    #e5e7eb 24px
  );
}
.dots {
  background-color: #ffffff;
  background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
  background-size: 12px 12px;
  background-position: 0 0;
}
.solid { background: #86efac; }
.alt-card {
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(15,23,42,0.55);
  min-height: 170px;
}
.alt-card h5 {
  margin: 0 0 8px 0;
  font-size: 1rem;
}
.alt-kpi {
  font-size: 0.82rem;
  color: #cbd5e1;
  margin: 2px 0;
}
</style>
"""

LIGHT_CSS = """
<style>
.hero-card {
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(2,132,199,0.12), rgba(14,165,233,0.08));
  border: 1px solid rgba(2,132,199,0.25);
  margin-bottom: 12px;
}
.tiny-note {
  color: #334155;
  font-size: 0.90rem;
}
.conflict-legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-top:8px; }
.conflict-item { display:inline-flex; align-items:center; gap:8px; color:#0f172a; font-size:0.85rem; }
.swatch { width:26px; height:16px; border:1px solid #94a3b8; border-radius:4px; }
.striped {
  background: repeating-linear-gradient(
    45deg,
    #f3f4f6 0px,
    #f3f4f6 12px,
    #e5e7eb 12px,
    #e5e7eb 24px
  );
}
.dots {
  background-color: #ffffff;
  background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
  background-size: 12px 12px;
  background-position: 0 0;
}
.solid { background: #86efac; }
.alt-card {
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(248,250,252,0.9);
  min-height: 170px;
}
.alt-card h5 {
  margin: 0 0 8px 0;
  font-size: 1rem;
}
.alt-kpi {
  font-size: 0.82rem;
  color: #334155;
  margin: 2px 0;
}
</style>
"""


@dataclass
class Settings:
    max_cpc_increase_pct: float
    min_bids_channel_state: int
    cpc_penalty_weight: float
    growth_weight: float
    profit_weight: float
    aggressive_cutoff: float
    controlled_cutoff: float
    maintain_cutoff: float
    min_intent_for_scale: float
    roe_pullback_floor: float
    cr_pullback_ceiling: float
    max_adj_strongest: float
    max_adj_moderate: float
    max_adj_minimal: float
    max_adj_constrained: float
    min_clicks_intent_sig: int
    min_bids_price_sig: int
    min_clicks_price_sig: int
    min_binds_perf_sig: int
    optimization_mode: str
    max_perf_drop: float
    min_new_performance: float


OPTIMIZATION_MODES = [
    "Max Growth",
    "Growth Leaning",
    "Balanced",
    "Cost Leaning",
    "Optimize Cost",
]


def mode_factor(mode: str) -> float:
    m = {
        "Max Growth": 1.0,
        "Growth Leaning": 0.75,
        "Balanced": 0.5,
        "Cost Leaning": 0.25,
        "Optimize Cost": 0.0,
    }
    return m.get(mode, 0.5)


def effective_cpc_cap_pct(settings: Settings) -> float:
    # Cost mode keeps a tight cap; growth mode allows wider CPC expansion,
    # but never above the explicit guardrail slider.
    f = mode_factor(settings.optimization_mode)
    mode_cap = 12.0 + (45.0 - 12.0) * f
    return min(float(settings.max_cpc_increase_pct), mode_cap)


def effective_cpc_penalty(settings: Settings) -> float:
    # Cost mode penalizes CPC harder; growth mode softens the penalty.
    f = mode_factor(settings.optimization_mode)
    mult = 1.6 - 1.25 * f  # 1.6 -> 0.35
    return settings.cpc_penalty_weight * mult


def classify_perf_group(roe: float, combined_ratio: float, performance: float, binds: float, min_binds_sig: int) -> tuple[str, bool]:
    sig = pd.notna(binds) and (binds >= min_binds_sig)
    if not sig:
        return "Low Sig - Review", False

    strong = 0
    weak = 0

    if pd.notna(roe):
        if roe > 0:
            strong += 1
        elif roe < -0.25:
            weak += 1
    if pd.notna(combined_ratio):
        if combined_ratio < 1.00:
            strong += 1
        elif combined_ratio > 1.15:
            weak += 1
    if pd.notna(performance):
        if performance >= 1.00:
            strong += 1
        elif performance < 0.80:
            weak += 1

    if strong == 3:
        return "Top Performance", True
    if weak == 3:
        return "Poor Performance", True
    if strong >= 2 and weak == 0:
        return "Strong Performance", True
    if weak >= 2 and strong == 0:
        return "Weak Performance", True
    if weak >= 1 and strong <= 1:
        return "Mixed Risk", True
    return "Balanced", True
