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
@st.cache_data(show_spinner=True)
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

    # Dates
    date_list = [d.date() for d in list_dates()]
    default_date_index = max(0, len(date_list) - 1)
    date = st.selectbox(
        "Date",
        date_list if date_list else [],
        index=default_date_index if date_list else 0,
        format_func=(lambda d: d.strftime("%B %d, %Y")) if date_list else str,
    )

    # Pull once (cached) and build the set of MLB pitcher_ids for the date
    df_day = fetch_statcast(date.strftime("%Y-%m-%d"))
    pitched_ids_mlb: set[int] = set()
    if not df_day.empty and "pitcher_id" in df_day.columns:
        pitched_ids_mlb = set(
            pd.to_numeric(df_day["pitcher_id"], errors="coerce")
              .dropna()
              .astype("int64")
              .tolist()
        )

    # Live pitchers (MLB + MiLB)
    df_pitchers = load_pitchers_all_levels()
    if df_pitchers.empty:
        st.error("Couldn’t load pitchers. Click Refresh.")
        st.stop()

    # Ensure numeric dtype for joins
    df_pitchers["key_mlbam"] = pd.to_numeric(df_pitchers["key_mlbam"], errors="coerce").astype("Int64")

    # Scope for the UI:
    # - Include ONLY pitchers who pitched that MLB date,
    #   across whichever current level/team they now belong to (MLB or MiLB).
    # - Otherwise, show the full pool.
    if pitched_ids_mlb:
        df_scope = df_pitchers[df_pitchers["key_mlbam"].isin(pitched_ids_mlb)].copy()
    else:
        df_scope = df_pitchers.copy()

    # If the filter yields nothing, inform and stop early
    if df_scope.empty:
        st.info(f"No pitchers in current rosters (any level) recorded an MLB pitch on {date.strftime('%b %d, %Y')}.")
        st.stop()

    # Level selector: built from the filtered scope so only levels with at least one qualifying pitcher appear
    level_order_all = ["MLB", "AAA", "AA", "A+", "A", "Rookie"]
    levels = [lvl for lvl in level_order_all if lvl in df_scope["team_level"].dropna().unique()]
    if not levels:
        st.info("No levels available for the current filter.")
        st.stop()

    level_default_idx = levels.index("MLB") if "MLB" in levels else 0
    level = st.selectbox("Level", levels, index=level_default_idx)

    # Teams limited to the selected level *within* the scope
    teams = sorted(df_scope.loc[df_scope["team_level"] == level, "team"].dropna().unique().tolist())
    team = st.selectbox("Team", teams) if teams else None
    if not team:
        st.info("No teams available for this level (with the current filter).")
        st.stop()

    # Pitchers limited to the selected team *within* the scope
    pframe = (
        df_scope.loc[
            (df_scope["team_level"] == level) & (df_scope["team"] == team),
            ["key_mlbam", "full_name"],
        ]
        .dropna(subset=["key_mlbam", "full_name"])
        .assign(key_mlbam=lambda d: d["key_mlbam"].astype("int64"))
        .sort_values("full_name")
    )
    pitcher_options = list(pframe.apply(lambda r: (int(r["key_mlbam"]), r["full_name"]), axis=1))

    if not pitcher_options:
        st.info(f"No pitchers for {team} match the current filter.")
        pitcher_id = None
    else:
        selected_pitcher = st.selectbox(
            "Pitcher",
            options=pitcher_options,
            format_func=(lambda t: t[1]),
        )
        pitcher_id = selected_pitcher[0] if selected_pitcher else None

    refresh = st.button("Refresh", type="primary")
    if refresh:
        fetch_statcast.clear()
        load_pitchers_all_levels.clear()
        st.rerun()

# Main panel
if pitcher_id is not None:
    render_dashboard(pitcher_id, date.strftime("%Y-%m-%d"))
else:
    st.info("Choose a pitcher.")
