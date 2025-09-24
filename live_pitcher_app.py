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

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(show_spinner=True, ttl=3*60*60)
def load_pitchers_from_date(date_str: str) -> pd.DataFrame:
    """
    From the pitch-level DF for `date_str`, return one row per pitcher with
    columns: pitcher_id, pitcher_name, pitcher_team_id, team_name.
    """
    df_day = fetch_statcast(date_str)
    cols = ["pitcher_id", "pitcher_name", "pitcher_team_id", "team_name"]
    if df_day.empty:
        return pd.DataFrame(columns=cols)

    # Ensure required columns exist
    required = {"pitcher_id", "pitcher_name", "pitcher_team_id"}
    missing = required - set(df_day.columns)
    if missing:
        # graceful fallback: create empties for missing fields
        for c in missing:
            df_day[c] = pd.NA

    # Unique (pitcher_id, pitcher_name, pitcher_team_id)
    base = (
        df_day.loc[:, ["pitcher_id", "pitcher_name", "pitcher_team_id"]]
        .dropna(subset=["pitcher_id", "pitcher_team_id"])
        .assign(
            pitcher_id=lambda d: pd.to_numeric(d["pitcher_id"], errors="coerce"),
            pitcher_team_id=lambda d: pd.to_numeric(d["pitcher_team_id"], errors="coerce"),
        )
        .dropna(subset=["pitcher_id", "pitcher_team_id"])
        .astype({"pitcher_id": "int64", "pitcher_team_id": "int64"})
        .drop_duplicates(subset=["pitcher_id", "pitcher_team_id"])
        .reset_index(drop=True)
    )

    if base.empty:
        return pd.DataFrame(columns=cols)

    # Map team_id -> team_name from schedule of that date
    sched = statsapi.schedule(start_date=date_str, end_date=date_str)
    team_name_by_id: dict[int, str] = {}
    for g in sched:
        for tid_key, tname_key in [("home_id", "home_name"), ("away_id", "away_name")]:
            tid = g.get(tid_key)
            tname = g.get(tname_key)
            if tid and tname:
                team_name_by_id[int(tid)] = tname

    out = (
        base.assign(team_name=lambda d: d["pitcher_team_id"].map(team_name_by_id).fillna("Unknown Team"))
            .loc[:, ["pitcher_id", "pitcher_name", "pitcher_team_id", "team_name"]]
            .sort_values(["team_name", "pitcher_name"])
            .reset_index(drop=True)
    )
    return out

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
@st.cache_data(show_spinner=True, ttl=30) # 30 seconds. Keeps live pitchers fresh to choose from
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

# init once so chart stays until you click Generate again
if "generated" not in st.session_state:
    st.session_state.generated = False
if "last_params" not in st.session_state:
    st.session_state.last_params = None

with st.sidebar:
    st.header("Filters")

    # Date picker with calendar popover
    today = pd.Timestamp.now(tz="America/Los_Angeles").normalize().date()
    date = st.date_input(
        "Date",
        value=today,          # default = today
        min_value=datetime.date(2017, 4, 2),  # MLB season start
        max_value=today,      # no future dates
        format="MM/DD/YYYY",
        help="Click to open calendar.",
    )

    # Pull once (cached) & build the set of MLB pitcher_ids that pitched on this date
    df_day = fetch_statcast(date.isoformat())
    pitched_ids_mlb: set[int] = set()
    if not df_day.empty and "pitcher_id" in df_day.columns:
        pitched_ids_mlb = set(
            pd.to_numeric(df_day["pitcher_id"], errors="coerce")
              .dropna()
              .astype("int64")
              .tolist()
        )

    # --- Build choices strictly from the selected date’s pitch DF (with names) ---
    df_scope = load_pitchers_from_date(date.strftime("%Y-%m-%d"))

    if df_scope.empty:
        st.info(f"No recorded MLB pitches on {date.strftime('%b %d, %Y')}. "
                "If games are underway, click **Generate** to refresh live pitchers.")
        pitcher_id = None
        team_id = None
        team = None
    else:
        # Teams that appear on that date
        team_rows = (
            df_scope.loc[:, ["pitcher_team_id", "team_name"]]
            .drop_duplicates()
            .sort_values("team_name")
            .itertuples(index=False, name=None)   # -> [(team_id, team_name), ...]
        )
        team_options = list(team_rows)

        selected_team = st.selectbox(
            "Team",
            options=team_options,
            format_func=lambda t: t[1] if t else "",
        )
        team_id = selected_team[0] if selected_team else None
        team = selected_team[1] if selected_team else None

        # Pitchers for that team (show names)
        if team_id is not None:
            pframe = (
                df_scope.loc[df_scope["pitcher_team_id"] == team_id, ["pitcher_id", "pitcher_name"]]
                .drop_duplicates()
                .sort_values("pitcher_name")
            )
            pitcher_options = list(pframe.itertuples(index=False, name=None))  # [(pitcher_id, pitcher_name), ...]
        else:
            pitcher_options = []

        if not pitcher_options:
            st.info(f"No pitchers for {team} on {date.strftime('%b %d, %Y')}.")
            pitcher_id = None
        else:
            selected_pitcher = st.selectbox(
                "Pitcher",
                options=pitcher_options,
                format_func=lambda t: t[1],
            )
            pitcher_id = selected_pitcher[0] if selected_pitcher else None

    # Actions
    generate = st.button("Generate", type="primary")

    if generate:
        # 1) Always refresh the selected date’s pitch feed (DON’T touch rosters)
        fetch_statcast.clear()

        date_str = date.strftime("%Y-%m-%d")

        if pitcher_id is None:
            # Use Generate as a "refresh filters" action when nothing is selected
            st.rerun()
        else:
            # Normal generation path with the same pitcher/date
            st.session_state.generated = True
            st.session_state.last_params = {
                "pitcher_id": int(pitcher_id),
                "date_str": date_str,
            }
            st.session_state._trigger_generate = True
            st.rerun()

# -------------------- Main panel --------------------
params = st.session_state.get("last_params")
curr   = st.session_state.get("_current_selection")

selection_changed = False
if curr is not None:
    if params is None:
        selection_changed = True
    else:
        selection_changed = (
            curr.get("pitcher_id") != params.get("pitcher_id")
            or curr.get("date_str")   != params.get("date_str")
        )

# If the user changed date/pitcher, hide the old chart until Generate is clicked
if selection_changed:
    st.session_state.generated = False

if st.session_state.get("_trigger_generate", False) and params:
    # Show spinner ONLY on the click-triggered run
    st.session_state._trigger_generate = False
    with st.spinner("Building pitcher dashboard..."):
        render_dashboard(params["pitcher_id"], params["date_str"])

elif st.session_state.get("generated") and params and not selection_changed:
    # Persist the last dashboard while the selection hasn't changed
    render_dashboard(params["pitcher_id"], params["date_str"])

else:
    st.info("Pick a date → team → pitcher, then click **Generate**.")


