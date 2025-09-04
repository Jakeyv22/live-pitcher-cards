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

@st.cache_data(show_spinner=False)
def load_chadwick():
    import pybaseball as pyb
    df_chadwick = pyb.chadwick_register()
    df = df_chadwick[df_chadwick['mlb_played_last'] == 2025].copy()
    df = df[df['key_mlbam'].notna() & (df['key_mlbam'] > 0)]
    df['full_name'] = df['name_first'].fillna('') + ' ' + df['name_last'].fillna('')
    return df

@st.cache_data(show_spinner=True)
def enrich_chadwick(df: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
    df = df.copy()
    df['team'] = 'Unknown'
    df['position'] = 'Unknown'
    df['team_level'] = 'Unknown'

    valid_ids = df['key_mlbam'].dropna().astype(int).tolist()
    if not valid_ids:
        return df
    n_batches = math.ceil(len(valid_ids) / batch_size)

    team_ids = set()
    person_team_map = {}

    for i in range(n_batches):
        batch_ids = valid_ids[i * batch_size:(i + 1) * batch_size]
        ids_str = ",".join(map(str, batch_ids))
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}&hydrate=currentTeam"
        try:
            data = requests.get(url, timeout=6).json()
            for person in data.get('people', []):
                pid = person['id']
                team = person.get('currentTeam', {})
                position = person.get('primaryPosition', {}).get('name', 'Unknown')
                team_name = team.get('name', 'Unknown')
                team_id = team.get('id', None)
                person_team_map[pid] = {'team': team_name, 'team_id': team_id, 'position': position}
                if team_id:
                    team_ids.add(team_id)
        except Exception:
            continue

    team_level_map = {}
    for team_id in team_ids:
        try:
            team_data = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{team_id}", timeout=5).json()
            if 'teams' in team_data and team_data['teams']:
                sport_name = team_data['teams'][0].get('sport', {}).get('name', 'Unknown')
                team_level_map[team_id] = sport_name
        except Exception:
            team_level_map[team_id] = 'Unknown'

    for idx, row in df.iterrows():
        pid = int(row['key_mlbam'])
        info = person_team_map.get(pid, {})
        team_id = info.get('team_id')
        df.at[idx, 'team'] = info.get('team', 'Unknown')
        df.at[idx, 'position'] = info.get('position', 'Unknown')
        df.at[idx, 'team_level'] = team_level_map.get(team_id, 'Unknown')

    df['team_level'] = df['team_level'].replace({
        'Major League Baseball': 'MLB',
        'Triple-A': 'AAA',
        'Double-A': 'AA',
        'High-A': 'A+',
        'Single-A': 'A',
    })

    return df.sort_values('name_last')

@st.cache_data(show_spinner=False)
def list_dates():
    start = pd.Timestamp(2025, 3, 18)
    today = pd.Timestamp.today().normalize()
    # Only include up to the current calendar date
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
def fetch_day_statcast(date_str: str) -> pd.DataFrame:
    """Get pitch-level data for all games on a date -> pandas DataFrame."""
    sched = statsapi.schedule(start_date=date_str, end_date=date_str)
    gamepk = [g['game_id'] for g in sched]
    if not gamepk:
        return pd.DataFrame()
    
    # fresh instance avoids any internal cache sticking around
    scraper = MLB_Scrape()

    game_data = scraper.get_data(game_list_input=gamepk)
    return scraper.get_data_df(data_list=game_data).to_pandas()

def render_dashboard(pitcher_id: int, date_str: str):
    df_day = fetch_day_statcast(date_str)
    if df_day.empty:
        st.image(_text_as_png_src("No games played on this date."), use_column_width=True)
        return
    df_p = df_day[df_day['pitcher_id'] == pitcher_id].reset_index(drop=True)
    if df_p.empty:
        st.image(_text_as_png_src("This pitcher did not pitch on the selected date."), use_column_width=True)
        return

    # build & render your Matplotlib dashboard
    fig = pitching_dashboard(pitcher_id, df_p)
    st.pyplot(fig, use_container_width=True)

    # optional download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download PNG", buf.getvalue(), "pitching_dashboard.png", "image/png")

# -------------------- UI --------------------

st.title("MLB Daily Pitching Dashboard")

with st.sidebar:
    st.caption("Filters")

    # dates: Mar 18, 2025 -> today ; default to today (last entry)
    date_list = [d.date() for d in list_dates()]
    default_date_index = max(0, len(date_list) - 1)

    date = st.selectbox(
        "Date",
        date_list,
        index=default_date_index,
        format_func=lambda d: d.strftime("%B %d, %Y"),  # e.g., September 03, 2025
    )

    with st.status("Loading player directory…", expanded=False):
        base_df = load_chadwick()
        enriched = enrich_chadwick(base_df)

    # Pitchers only
    df_pitchers = enriched[enriched['position'].str.contains("Pitcher", na=False)].copy()

    # Level defaults to MLB when available
    levels = sorted(df_pitchers['team_level'].dropna().unique())
    level_default_idx = levels.index('MLB') if 'MLB' in levels else 0
    level = st.selectbox("Level", levels, index=level_default_idx)

    teams = sorted(df_pitchers.loc[df_pitchers['team_level'] == level, 'team'].dropna().unique())
    team = st.selectbox("Team", teams)

    pframe = df_pitchers.loc[df_pitchers['team'] == team, ['key_mlbam', 'full_name']].dropna()
    pid_display = st.selectbox(
        "Pitcher",
        pframe.sort_values('full_name').itertuples(index=False),
        format_func=lambda r: r.full_name if hasattr(r, 'full_name') else str(r),
    )
    pitcher_id = int(pid_display.key_mlbam) if hasattr(pid_display, 'key_mlbam') else int(pid_display)

    refresh = st.button("Refresh", type="primary")
    if refresh:
        fetch_day_statcast.clear()  # clear just this function’s cache
        st.rerun()                  # rerun immediately


# Main panel
if pitcher_id:
    render_dashboard(pitcher_id, date.strftime("%Y-%m-%d"))
else:
    st.info("Choose a pitcher.")
