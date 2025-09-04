import os, io, math, datetime, base64, requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsapi

# --- your imports (must exist in the environment/codebase) ---
import pybaseball as pyb
from live_pitcher_card_mlb import pitching_dashboard   # must return a Matplotlib Figure

st.set_page_config(page_title="MLB Daily Pitching Dashboard", layout="wide")

# -------------------- Data helpers --------------------

# Sport IDs we want to include and their display levels
SPORT_LEVELS = {
    1:  "MLB",
    11: "AAA",
    12: "AA",
    13: "A+",
    14: "A",
    16: "Rookie",
}

TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId={sport_id}&activeStatus=Y"
ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(show_spinner=True)  # keep your caching; you can add ttl=600 later if you want
def load_pitchers_all_levels(max_workers: int = 24) -> pd.DataFrame:
    """
    Fast loader for MLB + MiLB (AAA, AA, A+, A, Rookie) active rosters.
    Returns columns: key_mlbam, full_name, team, team_id, position, team_level
    """
    # 1) Reuse a single session (connection pooling)
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "FlashStats/1.0",
    })

    # 2) Get all teams for all sports (sequential, only ~5 calls)
    team_meta = []
    for sport_id, level_label in SPORT_LEVELS.items():
        url = TEAMS_URL.format(sport_id=sport_id)  # already includes activeStatus=Y
        try:
            # Ask only for what we need to trim payload/parse time
            r = session.get(url, params={"fields": "teams,id,name"}, timeout=10)
            teams = (r.json() or {}).get("teams", []) or []
        except Exception:
            teams = []

        for t in teams:
            tid = t.get("id")
            tname = t.get("name")
            if not tid or not tname:
                continue
            team_meta.append({"id": int(tid), "name": tname, "level": level_label})

    if not team_meta:
        return pd.DataFrame(columns=["key_mlbam","full_name","team","team_id","position","team_level"])

    # 3) Fetch all rosters in parallel (I/O bound -> threads are perfect)
    def _fetch_team_roster(team: dict) -> list[dict]:
        url = ROSTER_URL.format(team_id=team["id"])  # already includes rosterType=active
        try:
            r = session.get(
                url,
                params={"fields": "roster,person,id,fullName,position,abbreviation"},
                timeout=10,
            )
            roster = (r.json() or {}).get("roster", []) or []
        except Exception:
            roster = []

        out = []
        for row in roster:
            pos = ((row.get("position") or {}).get("abbreviation"))
            if pos != "P":
                continue
            person = row.get("person") or {}
            pid = person.get("id")
            pname = person.get("fullName")
            if not pid or not pname:
                continue
            out.append({
                "key_mlbam": int(pid),
                "full_name": pname,
                "team": team["name"],
                "team_id": int(team["id"]),
                "position": "Pitcher",
                "team_level": team["level"],
            })
        return out

    rows: list[dict] = []
    max_workers = max(8, min(max_workers, 32))  # sensible bounds
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for team_rows in ex.map(_fetch_team_roster, team_meta):
            if team_rows:
                rows.extend(team_rows)

    if not rows:
        return pd.DataFrame(columns=["key_mlbam","full_name","team","team_id","position","team_level"])

    # 4) Build the DataFrame exactly like before
    df = pd.DataFrame(rows).drop_duplicates(subset=["key_mlbam", "team_level"])
    level_order = pd.CategoricalDtype(categories=["MLB","AAA","AA","A+","A","Rookie"], ordered=True)
    df["team_level"] = df["team_level"].astype(level_order)
    return df.sort_values(["team_level","team","full_name"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def list_dates():
    start = pd.Timestamp(2025, 3, 18)
    today = pd.Timestamp.today().normalize()
    end = max(start, today)
    return pd.date_range(start=start, end=end)

def _text_as_png_src(text: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

from api_scraper import MLB_Scrape
@st.cache_data(show_spinner=True, ttl=60)
def fetch_statcast(date_str: str) -> pd.DataFrame:
    """Get pitch-level data for all games on a date -> pandas DataFrame."""
    sched = statsapi.schedule(start_date=date_str, end_date=date_str)
    gamepk = [g['game_id'] for g in sched]
    if not gamepk:
        return pd.DataFrame()

    scraper = MLB_Scrape()  # fresh instance each call
    game_data = scraper.get_data(game_list_input=gamepk)
    return scraper.get_data_df(data_list=game_data).to_pandas()

def render_dashboard(pitcher_id: int, date_str: str):
    df_day = fetch_statcast(date_str)
    if df_day.empty:
        st.image(_text_as_png_src("No games played on this date."), width='stretch')
        return
    df_p = df_day[df_day['pitcher_id'] == pitcher_id].reset_index(drop=True)
    if df_p.empty:
        st.image(_text_as_png_src("This pitcher did not pitch on the selected date."), width='stretch')
        return

    # build & render your Matplotlib dashboard
    fig = pitching_dashboard(pitcher_id, df_p)
    st.pyplot(fig, width='stretch')

    # optional download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download PNG", buf.getvalue(), "pitching_dashboard.png", "image/png")

# -------------------- UI --------------------

st.title("MLB Daily Pitching Dashboard")

with st.sidebar:
    st.caption("Filters")

    # Dates: Mar 18, 2025 -> today ; default to today
    date_list = [d.date() for d in list_dates()]
    default_date_index = max(0, len(date_list) - 1)
    date = st.selectbox(
        "Date",
        date_list if date_list else [],
        index=default_date_index if date_list else 0,
        format_func=(lambda d: d.strftime("%B %d, %Y")) if date_list else str,
    )

    # Live pitchers (MLB + MiLB)
    df_pitchers = load_pitchers_all_levels()
    if df_pitchers.empty:
        st.error("Couldn’t load pitchers. Click Refresh.")
        st.stop()

    # Level selector (defaults to MLB when present)
    levels = [lvl for lvl in ["MLB","AAA","AA","A+","A"] if lvl in df_pitchers["team_level"].unique()]
    level_default_idx = levels.index("MLB") if "MLB" in levels else 0
    level = st.selectbox("Level", levels, index=level_default_idx)

    teams = sorted(df_pitchers.loc[df_pitchers["team_level"] == level, "team"].unique().tolist())
    team = st.selectbox("Team", teams) if teams else None
    if not team:
        st.info("No teams available for this level.")
        st.stop()

    # Build pitcher options as (id, name) tuples; avoid None/attribute errors
    pframe = (
        df_pitchers.loc[df_pitchers["team"] == team, ["key_mlbam", "full_name"]]
        .dropna(subset=["key_mlbam", "full_name"])
        .assign(key_mlbam=lambda d: d["key_mlbam"].astype("int64"))
        .sort_values("full_name")
    )
    pitcher_options = list(pframe.apply(lambda r: (int(r["key_mlbam"]), r["full_name"]), axis=1))

    selected_pitcher = st.selectbox(
        "Pitcher",
        options=pitcher_options,
        format_func=(lambda t: t[1]) if pitcher_options else None,
    )
    pitcher_id = selected_pitcher[0] if selected_pitcher else None

    refresh = st.button("Refresh", type="primary")
    if refresh:
        fetch_statcast.clear()
        st.rerun()

# Main panel
if pitcher_id is not None:
    render_dashboard(pitcher_id, date.strftime("%Y-%m-%d"))
else:
    st.info("Choose a pitcher.")
