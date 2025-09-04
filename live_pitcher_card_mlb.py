"""
MLB Pitcher Player Cards — plotting utilities for the Streamlit dashboard.

This module exposes:
  - pitching_dashboard(pitcher_id: int, df: pd.DataFrame) -> matplotlib.figure.Figure
and a set of helper functions it uses internally.

Nothing executes at import time; it is safe to `from live_pitcher_card_mlb import pitching_dashboard`.
"""

from __future__ import annotations

# --- std / third-party imports ---
import math
import re
from io import BytesIO
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
import statsapi
from PIL import Image
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
import cairosvg
import concurrent.futures
from pathlib import Path

# --- local imports ---
from api_scraper import MLB_Scrape

scraper = MLB_Scrape()

from pathlib import Path
import os
import pandas as pd
import numpy as np

# Your local OneDrive folder (fallback for running on your PC)
LEGACY_DATA_DIR = Path(r"C:\Users\jakev\OneDrive\Documents\FlashStats\Statcast")

def resolve_data(filename: str, override: str | Path | None = None) -> Path:
    """
    Find a data file in common locations:
    - explicit override path
    - repo ./data/<filename> (works on Streamlit Cloud)
    - current working dir ./data/<filename> (works in Streamlit run)
    - same folder as this module
    - your local OneDrive Statcast folder (legacy fallback)
    - PITCH_CARDS_DATA_DIR env var (optional)
    """
    # 1) explicit override
    if override:
        p = Path(override)
        if p.exists():
            return p

    # 2) env var directory (optional)
    env_dir = os.getenv("PITCH_CARDS_DATA_DIR")
    # build candidate list
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / filename,
        Path.cwd() / "data" / filename,
        here / filename,
        (Path(env_dir) / filename) if env_dir else None,
        LEGACY_DATA_DIR / filename,
    ]

    for p in candidates:
        if p and p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {filename}. Looked in:\n" +
        "\n".join(str(c) for c in candidates if c)
    )

# replaces the hard-coded C:\ path
STATCAST_CSV = resolve_data("statcast_2025_grouped.csv")
df_statcast_group = pd.read_csv(STATCAST_CSV)

# -------------------- Styling --------------------
font_properties = {"family": "DejaVu Sans", "size": 12}
font_properties_titles = {"family": "DejaVu Sans", "size": 20}
font_properties_axes = {"family": "DejaVu Sans", "size": 16}

sns.set_theme(style="whitegrid", palette="deep", font="DejaVu Sans", font_scale=1.5)
mpl.rcParams["figure.dpi"] = 300

# -------------------- Color / pitch maps --------------------
pitch_colors = {
    # Fastballs
    "FF": {"color": "#C21014", "name": "4-Seam Fastball"},
    "FA": {"color": "#C21014", "name": "Fastball"},
    "SI": {"color": "#F4B400", "name": "Sinker"},
    "FC": {"color": "#993300", "name": "Cutter"},
    # Offspeed
    "CH": {"color": "#00B386", "name": "Changeup"},
    "FS": {"color": "#66CCCC", "name": "Splitter"},
    "SC": {"color": "#33CC99", "name": "Screwball"},
    "FO": {"color": "#339966", "name": "Forkball"},
    # Sliders
    "SL": {"color": "#FFCC00", "name": "Slider"},
    "ST": {"color": "#CCCC66", "name": "Sweeper"},
    "SV": {"color": "#9999FF", "name": "Slurve"},
    # Curves
    "KC": {"color": "#0000CC", "name": "Knuckle Curve"},
    "CU": {"color": "#3399FF", "name": "Curveball"},
    "CS": {"color": "#66CCFF", "name": "Slow Curve"},
    # Knuckleball
    "KN": {"color": "#3333CC", "name": "Knuckleball"},
    # Others
    "EP": {"color": "#999966", "name": "Eephus"},
    "PO": {"color": "#CCCCCC", "name": "Pitchout"},
    "UN": {"color": "#9C8975", "name": "Unknown"},
}
dict_color = {k: v["color"] for k, v in pitch_colors.items()}
dict_pitch = {k: v["name"] for k, v in pitch_colors.items()}

team_colors = {
    "AZ": "#A71930", "ATL": "#CE1141", "BAL": "#DF4601", "BOS": "#BD3039",
    "CHC": "#0E3386", "CWS": "#27251F", "CIN": "#C6011F", "CLE": "#00385D",
    "COL": "#333366", "DET": "#FA4616", "HOU": "#002D62", "KC": "#004687",
    "LAA": "#BA0021", "LAD": "#005A9C", "MIA": "#00A3E0", "MIL": "#ffc52f",
    "MIN": "#002B5C", "NYM": "#FF5910", "NYY": "#003087", "ATH": "#EFB21E",
    "PHI": "#E81828", "PIT": "#FDB827", "SD": "#2F241D", "SF": "#FD5A1E",
    "SEA": "#0C2C56", "STL": "#C41E3A", "TB": "#092C5C", "TEX": "#C0111F",
    "TOR": "#134A8E", "WSH": "#AB0003",
}

# -------------------- wOBA helpers --------------------
W_COEFFS = {
    "Walk": 0.693,
    "Hit By Pitch": 0.724,
    "Single": 0.883,
    "Double": 1.252,
    "Triple": 1.583,
    "Home Run": 2.034,
}
AB_EVENTS = {
    "Strikeout", "Strikeout Double Play", "Groundout", "Flyout", "Lineout",
    "Pop Out", "Forceout", "Grounded Into DP", "Double Play", "Field Error",
    "Fielders Choice", "Fielders Choice Out", "Bunt Groundout", "Bunt Pop Out",
    "Bunt Lineout",
}
SF_EVENTS = {"Sac Fly", "Sac Fly Double Play"}
EXCLUDED_EVENTS = {"Intent Walk", "Sac Bunt", "Catcher Interference"}


def _compute_woba_row(event: str) -> Tuple[float, float]:
    if pd.isna(event):
        return np.nan, 0.0
    num = W_COEFFS.get(event, 0.0)
    if event in AB_EVENTS or event in W_COEFFS or event in SF_EVENTS or event == "Hit By Pitch":
        den = 1.0
    elif event in EXCLUDED_EVENTS:
        den = 0.0
    else:
        return np.nan, 0.0
    return num, den


def add_woba(df: pd.DataFrame) -> pd.DataFrame:
    comps = df["event"].map(_compute_woba_row)
    df["woba_numerator"] = comps.map(lambda x: x[0])
    df["woba_denominator"] = comps.map(lambda x: x[1])
    df["woba_value"] = np.where(
        df["woba_denominator"] > 0, df["woba_numerator"] / df["woba_denominator"], np.nan
    )
    return df


# -------------------- Arm angle & processing --------------------
def _fetch_height(pitcher_id: int) -> Tuple[int, float]:
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}"
        data = requests.get(url, timeout=6).json()
        height_str = data["people"][0]["height"]
        m = re.match(r"(\d+)' ?(\d+)", height_str)
        if m:
            feet, inches = int(m.group(1)), int(m.group(2))
            return pitcher_id, feet * 12 + inches
    except Exception:
        pass
    return pitcher_id, np.nan


def _get_pitcher_heights_parallel(pitcher_ids: List[int]) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_fetch_height, pitcher_ids))
    return dict(results)


def _calculate_arm_angle(row: pd.Series) -> float:
    vertical = row["z0"] - row["shoulder_height_ft"]
    horizontal = row["x0"]
    if row.get("pitcher_hand") == "L":
        horizontal = -horizontal
    return float(np.degrees(np.arctan2(vertical, abs(horizontal))))


def _load_arm_angle_lookup(path: str) -> dict:
    try:
        return pd.read_csv(path).set_index("pitcher_id")["mean_arm_angle"].to_dict()
    except FileNotFoundError:
        return {}


def df_processing(
    df_pyb: pd.DataFrame,
    lookup_csv: str | None = None,   # optional override
) -> pd.DataFrame:
    df = df_pyb.copy()
    df = df.rename(columns={"start_speed": "release_speed", "extension": "release_extension"})
    df["is_hardhit"] = df.get("launch_speed", np.nan) > 95
    df = add_woba(df)

    # find the CSV (repo data first, then your OneDrive fallback)
    csv_path = resolve_data("mlb_arm_angle.csv", override=lookup_csv)
    arm_angle_lookup = pd.read_csv(csv_path).set_index("pitcher_id")["mean_arm_angle"].astype(float).to_dict()
    df["arm_angle"] = df["pitcher_id"].map(arm_angle_lookup)

    missing_ids = df.loc[df["arm_angle"].isna(), "pitcher_id"].dropna().unique().tolist()
    if missing_ids:
        heights = _get_pitcher_heights_parallel([int(i) for i in missing_ids])
        df["height_inches"] = df["pitcher_id"].map(heights).fillna(75)
        df["shoulder_height_ft"] = df["height_inches"] * 0.70 / 12.0
        est_angles = (
            df.loc[df["arm_angle"].isna()]
              .groupby("pitcher_id")
              .apply(lambda g: g.apply(_calculate_arm_angle, axis=1).mean())
        )
        df["arm_angle"] = df["arm_angle"].combine_first(df["pitcher_id"].map(est_angles))
    return df


# -------------------- Small plot helpers --------------------
def player_headshot(pitcher_id: int, ax: plt.Axes) -> None:
    url = (
        f"https://img.mlbstatic.com/mlb-photos/image/"
        f"upload/d_people:generic:headshot:67:current.png"
        f"/w_640,q_auto:best/v1/people/{pitcher_id}/headshot/silo/current.png"
    )
    try:
        img = Image.open(BytesIO(requests.get(url, timeout=6).content))
        ax.set_xlim(0, 1.3); ax.set_ylim(0, 1)
        ax.imshow(img, extent=[0, 1, 0, 1], origin="upper")
    except Exception:
        ax.text(0.5, 0.5, "Headshot unavailable", ha="center", va="center")
    ax.axis("off")


def player_bio(pitcher_id: int, df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw name + bio + 'Pitching Summary' + `Month DD, YYYY vs OPP` from df."""
    try:
        data = requests.get(
            f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam",
            timeout=6,
        ).json()
        p = data["people"][0]
        player_name = p["fullName"]
        pitcher_hand = p["pitchHand"]["code"]
        age = p["currentAge"]
        height = p["height"]
        weight = p["weight"]
    except Exception:
        player_name, pitcher_hand, age, height, weight = "Unknown", "?", "—", "—", "—"

    # pull date/opponent from the data you passed in
    try:
        game_date_display = pd.to_datetime(df["game_date"].iloc[0]).strftime("%B %d, %Y")
        opponent = str(df["batter_team"].iloc[0])
    except Exception:
        game_date_display, opponent = "—", "—"

    team_color = team_colors.get(opponent, "black")
    fontsize = 56 if len(player_name) < 18 else 44

    ax.text(0.5, 1.0, player_name, va="top", ha="center",
            fontsize=fontsize, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(0.5, 0.65, f"{pitcher_hand}HP | Age: {age} | {height} | {weight} lbs",
            va="top", ha="center", fontsize=30, fontfamily="Arial")
    ax.text(0.5, 0.40, "Pitching Summary", va="top", ha="center",
            fontsize=40, fontweight="bold", fontfamily="Georgia")
    ax.text(0.5, 0.15, f"{game_date_display} vs {opponent}",
            va="top", ha="center", fontsize=30, fontstyle="italic", color=team_color)
    ax.axis("off")


def _team_id_from_df(df: pd.DataFrame) -> int | None:
    """Team the PITCHER played for in this game/date."""
    if "pitcher_team_id" not in df.columns:
        return None
    s = pd.to_numeric(df["pitcher_team_id"], errors="coerce").dropna().astype(int)
    if s.empty:
        return None
    # if multiple games/rows, use the most frequent team id
    return int(s.mode().iat[0])

def _fetch_logo_png(team_id: int, size: int = 300) -> bytes:
    svg = requests.get(f"https://www.mlbstatic.com/team-logos/{team_id}.svg", timeout=6).content
    return cairosvg.svg2png(bytestring=svg, output_width=size, output_height=size)

def plot_logo(df_game: pd.DataFrame, ax: plt.Axes, *, size: int = 300) -> None:
    ax.axis("off")
    tid = _team_id_from_df(df_game)
    if tid is None:
        ax.text(0.5, 0.5, "Logo unavailable", ha="center", va="center")
        return
    try:
        png = _fetch_logo_png(tid, size)
        img = Image.open(BytesIO(png))
        ax.imshow(img, interpolation="nearest", origin="upper", aspect="equal")
    except Exception:
        ax.text(0.5, 0.5, f"Logo {tid} unavailable", ha="center", va="center")


# -------------------- Velocity KDE panel --------------------
def velocity_kdes(
    df: pd.DataFrame,
    ax: plt.Axes,
    gs: gridspec.GridSpec,
    gs_x: List[int],
    gs_y: List[int],
    fig: plt.Figure,
    df_statcast_group: pd.DataFrame,
) -> None:
    """Side panel of per-pitch-type velocity distributions."""
    ax.axis("off")
    ax.set_title("Pitch Velocity Distribution", fontdict={"size": 20})

    # guards
    if df is None or df.empty or "pitch_type" not in df.columns:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center")
        return

    items_in_order = (
        df["pitch_type"].value_counts().sort_values(ascending=False).index.tolist()
    )
    if len(items_in_order) == 0:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center")
        return

    inner = gridspec.GridSpecFromSubplotSpec(
        len(items_in_order), 1, subplot_spec=gs[gs_x[0]:gs_x[-1], gs_y[0]:gs_y[-1]]
    )
    ax_top = [fig.add_subplot(s) for s in inner]

    for idx, pt in enumerate(items_in_order):
        this = df[df["pitch_type"] == pt]["release_speed"]
        color = dict_color.get(pt, "gray")

        if this.nunique() <= 1:
            ax_top[idx].plot([this.mean(), this.mean()], [0, 1], linewidth=4, color=color, zorder=20)
        else:
            sns.kdeplot(this, ax=ax_top[idx], fill=True,
                        clip=(float(this.min()), float(this.max())), color=color)

        # current mean
        ax_top[idx].plot([this.mean(), this.mean()],
                         [ax_top[idx].get_ylim()[0], ax_top[idx].get_ylim()[1]],
                         color=color, linestyle="--")
        # league mean for same pitch
        league = df_statcast_group[df_statcast_group["pitch_type"] == pt]["release_speed"]
        if not league.empty and np.isfinite(league.mean()):
            ax_top[idx].plot([league.mean(), league.mean()],
                             [ax_top[idx].get_ylim()[0], ax_top[idx].get_ylim()[1]],
                             color=color, linestyle=":")

        # axes pretties
        try:
            xmin = math.floor(float(df["release_speed"].min()) / 5) * 5
            xmax = math.ceil(float(df["release_speed"].max()) / 5) * 5
        except Exception:
            xmin, xmax = 70, 105
        ax_top[idx].set_xlim(xmin, xmax)
        ax_top[idx].set_xlabel(""); ax_top[idx].set_ylabel("")
        if idx < len(items_in_order) - 1:
            ax_top[idx].spines["top"].set_visible(False)
            ax_top[idx].spines["right"].set_visible(False)
            ax_top[idx].spines["left"].set_visible(False)
            ax_top[idx].tick_params(axis="x", colors="none")
        ax_top[idx].set_yticks([]); ax_top[idx].grid(axis="x", linestyle="--")
        ax_top[idx].text(-0.01, 0.5, pt, transform=ax_top[idx].transAxes,
                         fontsize=14, va="center", ha="right")

    # bottom subplot x labels
    ax_top[-1].set_xlabel("Velocity (mph)")


# -------------------- Rolling velo line --------------------
def plot_velo_by_pitch(df: pd.DataFrame, ax: plt.Axes | None = None, window: int = 3) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center"); return

    df = df.reset_index(drop=True).copy()
    df["Pitch #"] = df.index + 1

    for pt, group in df.groupby("pitch_type"):
        color = dict_color.get(pt, "gray")
        g = group.copy()
        g["Rolling Velo"] = pd.to_numeric(g["release_speed"], errors="coerce").rolling(
            window=window, min_periods=1
        ).mean()
        ax.plot(g["Pitch #"], g["Rolling Velo"], label=f"{pt}", color=color, lw=2)

    ax.set_title(f"Rolling {window}-Pitch Velocity", fontsize=16)
    ax.set_xlabel("Pitch #"); ax.set_ylabel("Velocity (MPH)")
    ax.grid(True)


# -------------------- Break plot --------------------
def _add_ellipse(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    if len(x) < 2:  # covariance needs >= 2
        return
    cov = np.cov(x, y)
    lam, vec = np.linalg.eig(cov)
    lam = np.sqrt(lam)
    angle = np.degrees(np.arctan2(*vec[:, 0][::-1]))
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=lam[0] * 4, height=lam[1] * 4,
        angle=angle, edgecolor=color, facecolor=color, alpha=0.4, lw=1.5, zorder=1
    )
    ax.add_patch(ell)


def break_plot(df: pd.DataFrame, ax: plt.Axes) -> None:
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center"); ax.axis("off"); return

    d = df.copy()
    if d.get("pitcher_hand", pd.Series(["R"])).iloc[0] == "L":
        d["hb"] = -d["hb"]

    sns.scatterplot(ax=ax, x=d["hb"], y=d["ivb"], hue=d["pitch_type"],
                    palette=dict_color, ec="black", alpha=1, zorder=2)
    for pt, g in d.groupby("pitch_type"):
        _add_ellipse(ax, g["hb"], g["ivb"], dict_color.get(pt, "gray"))

    ax.axhline(0, color="#808080", alpha=0.5, linestyle="--", zorder=1)
    ax.axvline(0, color="#808080", alpha=0.5, linestyle="--", zorder=1)
    ax.set_xlabel("Horizontal Break (in)", fontdict=font_properties_axes)
    ax.set_ylabel("Induced Vertical Break (in)", fontdict=font_properties_axes)

    if "arm_angle" in d.columns and not d["arm_angle"].isna().all():
        mean_angle_deg = float(d["arm_angle"].mean())
        title = f"Pitch Breaks – Est. Arm Angle: {mean_angle_deg:.0f}°"
        ax.set_title(title, fontdict=font_properties_titles)
        mean_angle_rad = np.deg2rad(mean_angle_deg)
        length = 35
        x_end = length * np.cos(mean_angle_rad)
        y_end = length * np.sin(mean_angle_rad)
        ax.plot([0, x_end], [0, y_end], linestyle="--", color="black", alpha=0.7)

    ax.get_legend().remove()
    ax.set_xticks(range(-20, 21, 10)); ax.set_yticks(range(-20, 21, 10))
    ax.set_xlim((-25, 25)); ax.set_ylim((-25, 25))
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    hand = d.get("pitcher_hand", pd.Series(["R"])).iloc[0]
    if hand == "R":
        ax.text(-24.2, -24.2, "← Glove Side", fontstyle="italic",
                ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
        ax.text(24.2, -24.2, "Arm Side →", fontstyle="italic",
                ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
    else:
        ax.invert_xaxis()
        ax.text(24.2, -24.2, "← Arm Side", fontstyle="italic",
                ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)
        ax.text(-24.2, -24.2, "Glove Side →", fontstyle="italic",
                ha="right", va="bottom", bbox=dict(facecolor="white", edgecolor="black"), fontsize=10, zorder=3)


# -------------------- Box score mini-table --------------------
def plot_boxscore_pitcher_line(
    pitcher_id: int, df: pd.DataFrame, date: str, ax: plt.Axes, fontsize: int = 16
) -> None:
    game_ids = df.loc[(df["pitcher_id"] == pitcher_id) & (df["game_date"] == date), "game_id"].unique()
    if len(game_ids) == 0:
        ax.text(0.5, 0.5, "No game found for this pitcher on date", ha="center", va="center"); ax.axis("off"); return

    game_id = int(game_ids[0])
    try:
        box = statsapi.boxscore_data(gamePk=game_id)
    except Exception:
        ax.text(0.5, 0.5, "Box score unavailable", ha="center", va="center"); ax.axis("off"); return

    pitcher = None
    for side in ["homePitchers", "awayPitchers"]:
        pitcher = next((p for p in box.get(side, []) if p.get("personId") == pitcher_id), None)
        if pitcher:
            break
    if not pitcher:
        ax.text(0.5, 0.5, "Pitcher not in box score", ha="center", va="center"); ax.axis("off"); return

    columns = ["ip", "h", "r", "er", "bb", "k", "hr", "p", "s", "era"]
    headers = ["IP", "H", "R", "ER", "BB", "K", "HR", "Pitches", "Strikes", "ERA"]
    values = [pitcher.get(col, "—") for col in columns]

    table = ax.table(cellText=[values], colLabels=headers, cellLoc="center", bbox=[0.0, 0.0, 1, 1])
    table.set_fontsize(fontsize)
    for i in range(len(headers)):
        table.get_celld()[(0, i)].get_text().set_fontweight("bold")
    ax.axis("off")


# -------------------- Pitch summary table --------------------
def _sdiv(a, b):
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    return np.where((b == 0) | (~np.isfinite(b)), np.nan, a / b)

pitch_stats_dict = {
    "pitch": {"table_header": "$\\bf{Count}$", "format": ".0f"},
    "release_speed": {"table_header": "$\\bf{Velocity}$", "format": ".1f"},
    "max_speed": {"table_header": "$\\bf{Max}$", "format": ".1f"},
    "ivb": {"table_header": "$\\bf{iVB}$", "format": ".1f"},
    "hb": {"table_header": "$\\bf{HB}$", "format": ".1f"},
    "spin_rate": {"table_header": "$\\bf{Spin}$", "format": ".0f"},
    "release_pos_x": {"table_header": "$\\bf{hRel}$", "format": ".1f"},
    "release_pos_z": {"table_header": "$\\bf{vRel}$", "format": ".1f"},
    "release_extension": {"table_header": "$\\bf{Ext.}$", "format": ".1f"},
    "bip": {"table_header": "$\\bf{BIP}$", "format": ".0f"},
    "pitch_usage": {"table_header": "$\\bf{Pitch\\%}$", "format": ".1%"},
    "whiff_rate": {"table_header": "$\\bf{Whiff\\%}$", "format": ".1%"},
    "in_zone_rate": {"table_header": "$\\bf{Zone\\%}$", "format": ".1%"},
    "chase_rate": {"table_header": "$\\bf{Chase\\%}$", "format": ".1%"},
    "hard_hit_rate": {"table_header": "$\\bf{HH\\%}$", "format": ".1%"},
    "woba": {"table_header": "$\\bf{wOBA}$", "format": ".3f"},
}
table_columns = [
    "pitch_description", "pitch", "pitch_usage", "release_speed", "max_speed", "ivb", "hb", "spin_rate",
    "release_pos_x", "release_pos_z", "release_extension", "bip", "in_zone_rate", "chase_rate",
    "whiff_rate", "hard_hit_rate", "woba",
]

def _fmt_series(s: pd.Series, fmt: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce"); out = s.astype(object)
    mask = s.notna() & np.isfinite(s); out.loc[mask] = out.loc[mask].map(lambda x: format(float(x), fmt))
    out.loc[~mask] = "—"; return out

def _df_grouping(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    for b in ("is_whiff", "is_swing"):
        if b in d.columns: d[b] = d[b].fillna(False)

    d["in_zone"] = ((1 <= d["zone"]) & (d["zone"] <= 9)).astype(int)
    d["out_zone"] = ((11 <= d["zone"]) & (d["zone"] <= 14)).astype(int)
    d["chase"] = (d["is_swing"] & (d["out_zone"] == 1)).astype(int)
    d["type"] = d["play_code"].map({"C": "S", "F": "S", "S": "S", "W": "S",
                                    "B": "B", "*B": "B", "H": "B", "D": "X", "X": "X", "E": "X"})
    bip_mask = d["type"].eq("X")

    g = (d.groupby("pitch_type").agg(
        pitch=("pitch_type", "count"),
        release_speed=("release_speed", "mean"),
        max_speed=("release_speed", "max"),
        ivb=("ivb", "mean"),
        hb=("hb", "mean"),
        spin_rate=("spin_rate", "mean"),
        release_pos_x=("x0", "mean"),
        release_pos_z=("z0", "mean"),
        release_extension=("release_extension", "mean"),
        is_swing=("is_swing", "sum"),
        is_whiff=("is_whiff", "sum"),
        in_zone=("in_zone", "sum"),
        out_zone=("out_zone", "sum"),
        chase=("chase", "sum"),
        hard_hit=("is_hardhit", "sum"),
        bip=("type", lambda x: (x == "X").sum()),
        woba=("woba_value", "mean"),
    ).reset_index())

    g["pitch_description"] = g["pitch_type"].map(dict_pitch)
    g["color"] = g["pitch_type"].map(dict_color)

    tot = g["pitch"].sum()
    g["pitch_usage"] = _sdiv(g["pitch"], tot)
    g["whiff_rate"] = _sdiv(g["is_whiff"], g["is_swing"])
    g["in_zone_rate"] = _sdiv(g["in_zone"], g["pitch"])
    g["chase_rate"] = _sdiv(g["chase"], g["out_zone"])
    g["hard_hit_rate"] = _sdiv(g["hard_hit"], g["bip"])

    g = g.sort_values("pitch_usage", ascending=False)
    stripe_colors = g["color"].tolist()

    all_row = pd.DataFrame([{
        "pitch_type": "All", "pitch_description": "All",
        "pitch": d["pitch_type"].count(), "pitch_usage": 1.0,
        "release_speed": np.nan, "max_speed": np.nan, "ivb": np.nan, "hb": np.nan,
        "spin_rate": np.nan, "release_pos_x": np.nan, "release_pos_z": np.nan,
        "release_extension": pd.to_numeric(d["release_extension"], errors="coerce").mean(),
        "in_zone_rate": d["in_zone"].sum() / max(d["pitch_type"].count(), 1),
        "chase_rate": d["chase"].sum() / max(d["out_zone"].sum(), 1),
        "whiff_rate": d["is_whiff"].sum() / max(d["is_swing"].sum(), 1),
        "hard_hit_rate": pd.to_numeric(d.loc[bip_mask, "is_hardhit"], errors="coerce").mean(),
        "bip": int(bip_mask.sum()), "woba": pd.to_numeric(d["woba_value"], errors="coerce").mean(),
        "color": "#FFFFFF",
    }])
    df_plot = pd.concat([g, all_row], ignore_index=True)
    return df_plot, stripe_colors


def _plot_pitch_format(df_plot: pd.DataFrame) -> pd.DataFrame:
    out = df_plot[table_columns].copy()
    for col, props in pitch_stats_dict.items():
        if col in out.columns:
            out[col] = _fmt_series(out[col], props["format"])
    out["pitch_description"] = out["pitch_description"].fillna("—")
    return out


cmap_sum = mpl.colors.LinearSegmentedColormap.from_list("", ["#648FFF", "#FFFFFF", "#FFB000"])
cmap_sum_r = mpl.colors.LinearSegmentedColormap.from_list("", ["#FFB000", "#FFFFFF", "#648FFF"])
cmap_sum.set_bad("#ffffff"); cmap_sum_r.set_bad("#ffffff")
COLOR_STATS = {"release_speed", "release_extension", "whiff_rate", "in_zone_rate", "chase_rate", "hard_hit_rate", "woba"}


def _mk_norm(vmin, vmax, pad=0.02):
    if np.isnan(vmin) or np.isnan(vmax):
        return None
    if vmin == vmax:
        span = abs(vmin) if vmin != 0 else 1.0
        vmin -= span * pad; vmax += span * pad
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _norm_params(stat, base_mean):
    if not np.isfinite(base_mean):
        return 0.0, 1.0, cmap_sum
    if stat == "release_speed":
        return base_mean * 0.95, base_mean * 1.05, cmap_sum
    if stat == "release_extension":
        return base_mean * 0.80, base_mean * 1.20, cmap_sum
    if stat in {"hard_hit_rate", "woba"}:
        return base_mean * 0.70, base_mean * 1.30, cmap_sum_r
    return base_mean * 0.70, base_mean * 1.30, cmap_sum


def _safe_mean(series):
    if series is None or len(series) == 0:
        return np.nan
    return pd.to_numeric(series, errors="coerce").mean()


def _get_cell_colors(df_group: pd.DataFrame, df_statcast_group: pd.DataFrame) -> List[List[str]]:
    means = df_statcast_group.groupby("pitch_type").mean(numeric_only=True)
    norms = {}
    for pt, row in means.iterrows():
        for stat in COLOR_STATS:
            mu = row.get(stat, np.nan)
            vmin, vmax, _ = _norm_params(stat, mu)
            norms[(pt, stat)] = _mk_norm(vmin, vmax)

    colors = []
    for _, gr in df_group.iterrows():
        pt = gr["pitch_type"]
        row_colors = []
        for col in table_columns:
            if col in COLOR_STATS and pd.api.types.is_number(gr[col]):
                _, _, cmap = _norm_params(col, means.loc[pt][col] if (pt in means.index and col in means.columns) else _safe_mean(df_group[col]))
                norm = norms.get((pt, col))
                row_colors.append("#ffffff" if norm is None else mcolors.to_hex(cmap(norm(float(gr[col])))))
            else:
                row_colors.append("#ffffff")
        colors.append(row_colors)
    return colors


def pitch_table(df: pd.DataFrame, df_statcast_group: pd.DataFrame, ax: plt.Axes, fontsize: int = 15) -> None:
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No pitch data", ha="center", va="center"); ax.axis("off"); return

    df_group, stripe_colors = _df_grouping(df)
    cell_colors = _get_cell_colors(df_group, df_statcast_group)
    df_display = _plot_pitch_format(df_group)

    table = ax.table(
        cellText=df_display.values, colLabels=table_columns, cellLoc="center",
        bbox=[0, -0.1, 1, 1], colWidths=[2.5] + [1] * (len(table_columns) - 1),
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 0.5)

    nice_headers = ["$\\bf{Pitch\\ Name}$"] + [pitch_stats_dict.get(c, {}).get("table_header", "---") for c in table_columns[1:]]
    for i, name in enumerate(nice_headers):
        table.get_celld()[(0, i)].get_text().set_text(name)

    for i in range(1, len(df_display) + 1):
        cell = table.get_celld()[(i, 0)]
        label = cell.get_text().get_text()
        cell.get_text().set_fontweight("bold")
        cell.set_text_props(color="#000000" if label in ["Splitter", "Slider", "Changeup", "Sinker", "Screwball", "Forkball", "Sweeper"] else "#FFFFFF")
        if i - 1 < len(stripe_colors):
            cell.set_facecolor(stripe_colors[i - 1])

    ax.axis("off")


# -------------------- Dashboard --------------------
def pitching_dashboard(pitcher_id: int, df: pd.DataFrame) -> plt.Figure:
    """
    Build the full dashboard figure for a single pitcher on one date.
    - df: pitch-level data (already filtered to the game/day you want)
          must include columns like: game_date, batter_team, pitch_type, release_speed, etc.
    """
    df = df_processing(df)
    fig = plt.figure(figsize=(20, 20))

    gs = gridspec.GridSpec(
        6, 8,
        height_ratios=[2, 20, 9, 36, 36, 7],
        width_ratios=[1, 22, 22, 28, 28, 18, 18, 1],
    )

    ax_headshot = fig.add_subplot(gs[1, 1:3])
    ax_bio = fig.add_subplot(gs[1, 3:5])
    ax_logo = fig.add_subplot(gs[1, 5:7])

    ax_season_table = fig.add_subplot(gs[2, 1:7])

    ax_plot_1 = fig.add_subplot(gs[3, 1:3])
    ax_plot_2 = fig.add_subplot(gs[3, 3:5])
    ax_plot_3 = fig.add_subplot(gs[3, 5:7])

    ax_table = fig.add_subplot(gs[4, 1:7])

    ax_footer = fig.add_subplot(gs[-1, 1:7])
    ax_header = fig.add_subplot(gs[0, 1:7]); ax_header.axis("off")
    fig.add_subplot(gs[:, 0]).axis("off")
    fig.add_subplot(gs[:, -1]).axis("off")

    # top row
    player_headshot(pitcher_id, ax=ax_headshot)
    player_bio(pitcher_id, df, ax=ax_bio)
    plot_logo(df, ax=ax_logo)

    # box score + tables/plots
    date_iso = str(df.loc[df["pitcher_id"] == pitcher_id, "game_date"].iloc[0])
    plot_boxscore_pitcher_line(pitcher_id=pitcher_id, df=df, date=date_iso, ax=ax_season_table, fontsize=20)

    pitch_table(df, ax=ax_table, df_statcast_group=df_statcast_group, fontsize=15)
    velocity_kdes(df=df, ax=ax_plot_1, gs=gs, gs_x=[3, 4], gs_y=[1, 3], fig=fig, df_statcast_group=df_statcast_group)
    break_plot(df=df, ax=ax_plot_2)
    plot_velo_by_pitch(df, ax=ax_plot_3, window=3)

    # footer
    ax_footer.axis("off")
    ax_footer.text(0, 1, "By: Jake Vickroy", ha="left", va="top", fontsize=24)
    ax_footer.text(0.5, 1, "Color Coding Compares to League Average By Pitch", ha="center", va="top", fontsize=16)
    ax_footer.text(1, 1, "Data: MLB", ha="right", va="top", fontsize=24)

    plt.tight_layout()
    return fig


__all__ = ["pitching_dashboard"]
