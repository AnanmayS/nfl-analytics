"""NFL predictions & player types dashboard. Run with `make app`."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import duckdb
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.archetype_report import LABELS  # noqa: E402
from src.analysis.fan_view import (  # noqa: E402
    confidence_check,
    confidence_label,
    describe_archetype,
    game_reasons,
    league_picks,
    metric_tiers,
    percentiles,
    pick_summary,
    plain_label,
    spread_text,
    strength_word,
    team_games,
    team_identity,
    team_pcts,
    vegas_favorite_text,
)
from src.archetypes.pipeline import SPECS  # noqa: E402
from src.config import DATA, DB_PATH, ROOT  # noqa: E402
from src.export import APP_DB  # noqa: E402

st.set_page_config(page_title="NFL Game Predictions", page_icon="🏈", layout="wide")
PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Dark2
POS_NAMES = {"QB": "Quarterbacks", "RB": "Running backs", "WR": "Wide receivers",
             "TE": "Tight ends", "DEF": "Team defenses"}

st.markdown("""
<style>
.card {border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: 14px 16px;
       margin-bottom: 12px;}
.card .row {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
.card img {width: 34px; height: 34px;}
.card .big {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.card .muted {opacity: .7; font-size: .9rem;}
.bar {display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: 8px 0 4px;}
.bar div {height: 100%;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem;
       font-weight: 600; border: 1px solid rgba(128,128,128,.45);}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
# Without a local pipeline (e.g. on Streamlit Community Cloud) the app reads the slim database
# that the scheduled GitHub Actions refresh publishes (.github/workflows/refresh.yml).
HOSTED = not DB_PATH.exists()
HOSTED_DB_URL = os.environ.get(
    "NFL_APP_DB_URL",
    "https://github.com/AnanmayS/nfl-analytics/releases/download/app-data/app.duckdb")
DB = APP_DB if HOSTED else DB_PATH
REFRESH_PID, REFRESH_LOG = DATA / "refresh.pid", DATA / "refresh.log"


def refresh_running() -> bool:
    try:
        pid = int(REFRESH_PID.read_text())
    except (FileNotFoundError, ValueError):
        return False
    try:  # our own child: reap it when done, so a finished run isn't left as a zombie
        if os.waitpid(pid, os.WNOHANG)[0] == 0:
            return True
    except ChildProcessError:  # started by an earlier app session: just check it's alive
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            pass
    REFRESH_PID.unlink(missing_ok=True)
    return False


def start_refresh() -> None:
    """Run the weekly refresh in the background (same as `make refresh`)."""
    log = open(REFRESH_LOG, "w")  # noqa: SIM115 - handed to the child process
    proc = subprocess.Popen([sys.executable, "-m", "src.cli", "refresh"], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    REFRESH_PID.write_text(str(proc.pid))


if refresh_running():
    st.title("🏈 Getting the latest data…")
    st.info("Pulling new game results, injury reports and betting lines, then re-running "
            "every pick. This takes a couple of minutes; the dashboard comes back on its own "
            "when it's done.")

    @st.fragment(run_every="10s")
    def _poll() -> None:  # re-checks every 10s; reloads the whole page once the run is done
        if not refresh_running():
            st.rerun(scope="app")
        tail = REFRESH_LOG.read_text().splitlines()[-1:] if REFRESH_LOG.exists() else []
        if tail:
            st.caption(f"Latest step: {tail[0][:160]}")

    _poll()
    st.stop()


@st.cache_resource(ttl="15m", show_spinner="Loading the latest picks…")
def fetch_hosted_db() -> None:
    """Download the published database; replace the local copy only when it changed, so the
    query cache (keyed on the file's mtime) resets only on new data."""
    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = APP_DB.with_suffix(".download")
    try:
        urllib.request.urlretrieve(HOSTED_DB_URL, tmp)
    except OSError:
        if not APP_DB.exists():
            raise
        return  # keep serving the copy we have
    if APP_DB.exists() and tmp.read_bytes() == APP_DB.read_bytes():
        tmp.unlink()
    else:
        tmp.replace(APP_DB)


if HOSTED:
    try:
        fetch_hosted_db()
    except OSError as e:
        st.error(f"Couldn't download the latest data ({e}).")
        st.stop()


@st.cache_data(show_spinner=False)
def _q(sql: str, version: float) -> pl.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(sql).pl()
    finally:
        con.close()


def q(sql: str) -> pl.DataFrame:
    """Cached query; the cache resets whenever the database file changes."""
    return _q(sql, DB.stat().st_mtime)


if not DB.exists():
    st.error("No data yet. Run `make all` first.")
    st.stop()
have = set(q("SELECT table_name FROM information_schema.tables")["table_name"].to_list())
if "game_predictions" not in have or "raw_teams" not in have:
    st.error("Predictions haven't been built yet. Run `make all`.")
    st.stop()

preds = q("SELECT * FROM game_predictions")
CURRENT = int(preds["season"].max())
HISTORY = [CURRENT, CURRENT - 1]  # how far back the track record goes
teams_raw = q("SELECT team_abbr, team_name, team_nick, team_color, team_color2, team_logo_espn "
              "FROM raw_teams")
active = set(preds.filter(pl.col("season") == CURRENT)["home_team"].to_list())
TEAMS = {r["team_abbr"]: r for r in teams_raw.iter_rows(named=True) if r["team_abbr"] in active}


def name(abbr: str) -> str:
    return TEAMS.get(abbr, {}).get("team_name", abbr)


def nick(abbr: str) -> str:
    return TEAMS.get(abbr, {}).get("team_nick", abbr)


def logo(abbr: str) -> str:
    return TEAMS.get(abbr, {}).get("team_logo_espn", "")


def color(abbr: str, alt: bool = False) -> str:
    t = TEAMS.get(abbr, {})
    return (t.get("team_color2") if alt else t.get("team_color")) or "#888"


def win_bar(left: str, right: str, p_left: float) -> str:
    c2 = color(right) if color(right).lower() != color(left).lower() else color(right, True)
    return (f"<div class='bar'><div style='width:{p_left*100:.0f}%;background:{color(left)}'>"
            f"</div><div style='width:{(1-p_left)*100:.0f}%;background:{c2}'></div></div>")


def fmt_date(d) -> str:
    return d.strftime("%a %b %-d") if d is not None else ""


def stat(col, label: str, value: str, sub: str) -> None:
    """A big number with a neutral grey subtitle (no up/down arrow)."""
    col.metric(label, value, sub, delta_color="off", delta_arrow="off")


def opponent_label(is_home: bool, opp: str) -> str:
    return f"{'vs' if is_home else 'at'} {name(opp)}"


@st.cache_data(show_spinner=False)
def _tiers(version: float) -> pl.DataFrame:
    return metric_tiers(preds.filter(pl.col("season").is_in(HISTORY)))


def tiers() -> pl.DataFrame:
    return _tiers(DB.stat().st_mtime)


def injury_html(game: dict, first: str) -> str:
    """Injury-report lines for both teams of a game_predictions row, `first` team on top."""
    side = {game["home_team"]: "home", game["away_team"]: "away"}
    rows = []
    for t in sorted(side, key=lambda x: x != first):
        sd = side[t]
        if not game.get(f"{sd}_report_available"):
            txt = "<span class='muted'>report not out yet</span>"
        elif game.get(f"{sd}_key_absences"):
            txt = game[f"{sd}_key_absences"]
            if (game.get(f"{sd}_qb_out") or 0) >= 0.5:
                txt = f"⚠️ {txt}"
        else:
            txt = "no key players listed out"
        rows.append(f"{nick(t)}: {txt}")
    return "<b>Injury report</b><br>" + "<br>".join(rows)


def identity_for(team: str) -> dict:
    """Team identity from its most recent pre-game form, play-calling and key players."""
    g = team_games(preds, team).filter(pl.col("season") == CURRENT)
    up = g.filter(pl.col("status") == "upcoming")
    ref = up.row(0, named=True) if up.height else g.row(-1, named=True)
    pcts = team_pcts(tiers(), CURRENT, ref["week"], team)
    prof = q("SELECT season, team, pass_rate_pct FROM team_profile").filter(
        (pl.col("team") == team) & pl.col("season").is_in(HISTORY)).sort("season")
    prr = prof["pass_rate_pct"][-1] if prof.height else None
    kp = q("SELECT * FROM team_key_players").filter(pl.col("team") == team)
    kp = kp.filter(pl.col("season") == kp["season"].max()).sort("pos", "rk")
    dtype = None
    if "archetype_def" in have:
        d = q("SELECT team, season, archetype FROM archetype_def").filter(
            (pl.col("team") == team) & (pl.col("season") <= CURRENT)).sort("season")
        dtype = d["archetype"][-1] if d.height else None
    return team_identity(pcts, prr, kp.to_dicts(), dtype)


def defense_text(team: str, short: bool = False) -> str:
    """Defense style (this season's stats so far) and recent form (rolling, vs. league)."""
    d = identity_for(team)["defense"]
    if short:
        style = f"{d['type']} · " if d["type"] else ""
        return f"{style}{d['overall']} on recent form"
    style = f"Style this season: {d['type']}. " if d["type"] else ""
    best = f", best at {d['best_at']}" if d["best_at"] else ""
    worst = f", weakest at {d['worst_at']}" if d["worst_at"] else ""
    return f"{style}Recent form: {d['overall']} overall{best}{worst}."


def reasons_html(game: dict) -> str:
    """The betting line (when posted), then why our stats model leans the way it does."""
    home, away, p = game["home_team"], game["away_team"], game["p_logit"]
    pick, other = (home, away) if p >= 0.5 else (away, home)
    spread = game.get("spread_line")
    if spread is not None and game.get("p_blend") is not None:
        fav = home if spread > 0 else away
        line_txt = ("Pick'em" if spread == 0 else f"{nick(fav)} favored by {abs(spread):g}")
        head = (f"<b>The betting line:</b> {line_txt}. Our final number blends the line with "
                f"our stats model, which on its own gives the {nick(pick)} "
                f"{max(p, 1 - p):.0%}.<br><br>")
    else:
        head = ("<b>No betting line yet</b>, so this pick is our stats model alone. It will "
                "also factor in the line once one is posted.<br><br>")
    rs = game_reasons(game, tiers(), {t: nick(t) for t in TEAMS})
    pro = [r for r in rs if r["favors"] == pick and r["push"] >= 0.05][:3]
    con = [r for r in rs if r["favors"] == other and r["push"] >= 0.05][:2]

    def items(rows):
        return "".join(f"<li><span class='pill'>{strength_word(r['push'])}</span> "
                       f"{r['text']}</li>" for r in rows)

    html = head + f"<b>Why our stats model leans {nick(pick)}</b><ul style='margin:4px 0 8px'>"
    html += items(pro) or "<li>No single big factor; it's close across the board</li>"
    html += "</ul>"
    if con:
        html += (f"<b>What the {nick(other)} have going for them</b>"
                 f"<ul style='margin:4px 0 8px'>{items(con)}</ul>")
    if game.get("neutral"):
        html += "<div class='muted'>Neutral-site game, so neither team gets home field.</div>"
    return html


# ------------------------------------------------------------------ sidebar
abbrs = sorted(TEAMS, key=name)
default = st.query_params.get("team", "KC")
last_played = preds.filter(pl.col("result").is_not_null())["game_date"].max()
with st.sidebar:
    team = st.selectbox("Pick a team", abbrs, format_func=name,
                        index=abbrs.index(default) if default in abbrs else 0)
    st.query_params["team"] = team
    st.image(logo(team), width=90)
    page = st.radio("Go to", ["My team", "This week", "Track record", "Player types",
                              "Under the hood"], label_visibility="collapsed",
                    format_func=lambda x: "Player & defense types" if x == "Player types" else x)
    info = q("SELECT * FROM refresh_info").row(0, named=True) if "refresh_info" in have else None
    if info:
        st.caption(f"Data updated {info['refreshed_at']:%a %b %-d, %-I:%M %p}. Results through "
                   f"{info['results_through']:%b %-d}; injury reports through week "
                   f"{info['injury_report_week']}.")
    else:
        st.caption(f"Results through {last_played:%b %-d, %Y}.")
    if not HOSTED and st.button("Get latest data", width="stretch",
                 help="Pulls new results, injury reports and betting lines, then re-runs "
                      "every pick (a couple of minutes). Best before games: injury reports are "
                      "final two days before kickoff."):
        start_refresh()
        st.rerun()
    st.caption("Every prediction comes from a model trained only on information available "
               "before that game.")


# ------------------------------------------------------------------ my team
def page_team(team: str) -> None:
    g = team_games(preds, team)
    season = g.filter(pl.col("season") == CURRENT)
    done = season.filter(pl.col("won").is_not_null())
    wins, losses = int(done["won"].sum()), int((~done["won"]).sum())
    ties = season.filter((pl.col("status") == "final") & pl.col("won").is_null()).height
    record = f"{wins}–{losses}" + (f"–{ties}" if ties else "")
    c1, c2 = st.columns([1, 9])
    c1.image(logo(team), width=72)
    c2.title(name(team))
    c2.caption(f"{CURRENT} record: {record}")

    ident = identity_for(team)
    best = " and ".join(ident["strengths"]) or "nothing that stands out yet"
    weak = f" Weak spot: {ident['weakness'][0]}." if ident["weakness"] else ""
    st.markdown(f"""
<div class='card'><div class='muted'>Team identity</div>
  <div style='font-size:1.25rem;font-weight:700;margin:2px 0 6px'>{ident['label']}</div>
  <div>Best at: {best}.{weak}</div>
  <div style='margin-top:6px'><b>Defense:</b> {defense_text(team)}</div>
  <div class='muted' style='margin-top:6px'>Key players: {' · '.join(ident['players'])}</div>
</div>""", unsafe_allow_html=True)
    st.caption("Based on recent form compared with the rest of the league, how often they "
               "pass on early downs in close games, their defensive style, and the player "
               "types of their most-used players.")

    upcoming = season.filter(pl.col("status") == "upcoming")
    if upcoming.height:
        nxt = upcoming.row(0, named=True)
        p, opp = nxt["win_prob"], nxt["opponent"]
        where = "vs" if nxt["is_home"] else "at"
        margin = abs(nxt["projected_margin"])
        if margin < 1:
            verdict = "Projected to be a coin flip"
        elif p >= 0.5:
            verdict = f"Projected to win by about {margin:.0f} points"
        else:
            verdict = f"Projected to lose by about {margin:.0f} points"
        game_row = preds.filter(pl.col("game_id") == nxt["game_id"]).row(0, named=True)
        st.subheader("Next game")
        st.markdown(f"""
<div class='card'>
  <div class='row'><img src='{logo(team)}'><b>{nick(team)}</b>
     <span class='muted'>{where}</span><img src='{logo(opp)}'><b>{nick(opp)}</b>
     <span class='muted'>· Week {nxt['week']} · {fmt_date(nxt['game_date'])}</span></div>
  <div style='margin-top:10px' class='big'>{p:.0%} chance to win</div>
  {win_bar(team, opp, p)}
  <div class='muted'>{nick(team)} {p:.0%} · {nick(opp)} {1-p:.0%} ·
     {"stats model + betting line" if nxt['vegas_spread'] is not None
      else "stats model only (no line yet)"}</div>
  <div style='margin-top:8px'><span class='pill'>{confidence_label(p)}</span>
     &nbsp;{verdict} · Vegas: {spread_text(nxt['vegas_spread'], nick(team))}</div>
  <div class='muted' style='margin-top:6px'>The {nick(opp)}: {identity_for(opp)['label']}</div>
  <div style='margin-top:6px'><b>Defenses</b><br>{nick(team)}: {defense_text(team, True)}<br>
     {nick(opp)}: {defense_text(opp, True)}</div>
  <div style='margin-top:6px'>{injury_html(game_row, team)}</div>
  <hr style='margin:10px 0;opacity:.3'>
  {reasons_html(game_row)}
</div>""", unsafe_allow_html=True)

        if upcoming.height > 1:
            st.subheader("Rest of the schedule")
            rest = upcoming.slice(1).select(
                pl.col("week").alias("Week"),
                pl.col("game_date").map_elements(fmt_date, return_dtype=pl.String)
                .alias("Date"),
                pl.struct("is_home", "opponent").map_elements(
                    lambda r: opponent_label(r["is_home"], r["opponent"]),
                    return_dtype=pl.String).alias("Opponent"),
                (pl.col("win_prob") * 100).round(0).alias("Win chance"),
                pl.col("win_prob").map_elements(confidence_label, return_dtype=pl.String)
                .alias("Outlook"),
                pl.col("vegas_spread").map_elements(
                    lambda s: spread_text(s, nick(team)), return_dtype=pl.String,
                    skip_nulls=False).alias("Vegas line"),
            )
            st.dataframe(rest, hide_index=True, width="stretch", column_config={
                "Win chance": st.column_config.ProgressColumn(
                    "Win chance", min_value=0, max_value=100, format="%d%%"),
            })
            st.caption("Games further out use the team's form as of today, so they'll shift "
                       "as the season goes on. Vegas usually posts lines about a week ahead.")
    else:
        st.info(f"No upcoming games for the {nick(team)} this season.")

    st.subheader("How have our picks done?")
    yr = st.segmented_control("Season", HISTORY, default=CURRENT, key="hist_season") or CURRENT
    hist = g.filter((pl.col("season") == yr) & (pl.col("status") == "final"))
    if hist.height == 0:
        st.info("No finished games yet.")
    else:
        s = pick_summary(hist)
        m1, m2 = st.columns(2)
        stat(m1, "Our model picked the winner", f"{s['model_right']} of {s['games']} games",
             f"{s['model_right'] / max(s['games'], 1):.0%} correct")
        stat(m2, "Vegas favorite won", f"{s['vegas_right']} of {s['vegas_games']} games",
             f"{s['vegas_right'] / max(s['vegas_games'], 1):.0%} correct")

        def pick(r) -> str:
            p = r["win_prob"]
            return (f"{nick(team)} ({p:.0%})" if p >= 0.5
                    else f"{nick(r['opponent'])} ({1 - p:.0%})")

        def final(r) -> str:
            res = "W" if r["won"] else ("L" if r["won"] is False else "T")
            return f"{res} {r['points_for']:.0f}–{r['points_against']:.0f}"

        def mark(v) -> str:
            return "–" if v is None else ("✅" if v else "❌")

        table = hist.sort("game_date", descending=True).select(
            pl.col("week").alias("Week"),
            pl.struct("is_home", "opponent").map_elements(
                lambda r: opponent_label(r["is_home"], r["opponent"]),
                return_dtype=pl.String).alias("Opponent"),
            pl.struct("win_prob", "opponent").map_elements(pick, return_dtype=pl.String)
            .alias("Our pick"),
            pl.struct("won", "points_for", "points_against").map_elements(
                final, return_dtype=pl.String).alias("Final"),
            pl.col("model_correct").map_elements(mark, return_dtype=pl.String,
                                                 skip_nulls=False).alias("We got it?"),
            pl.col("vegas_correct").map_elements(mark, return_dtype=pl.String,
                                                 skip_nulls=False).alias("Vegas got it?"),
        )
        st.dataframe(table, hide_index=True, width="stretch")

    roster_types(team)


def roster_types(team: str) -> None:
    """Key players and their player type, from the most recent season with data."""
    rows = []
    for pos, n in [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 2)]:
        t = f"archetype_{pos.lower()}"
        if t not in have:
            continue
        vol = SPECS[pos].volume_col
        df = q(f"SELECT player_name, season, archetype, {vol} AS vol FROM {t} "
               f"WHERE team = '{team}' ORDER BY season DESC, vol DESC")
        if df.height == 0:
            continue
        latest = df["season"].max()
        for r in df.filter(pl.col("season") == latest).head(n).iter_rows(named=True):
            rows.append({"Position": POS_NAMES[pos][:-1], "Player": r["player_name"],
                         "Player type": r["archetype"], "Season": str(r["season"])})
    if rows:
        st.subheader("Key players and their style")
        st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("See 'Player types' in the sidebar for what each style means. This "
                   "season's types are based on games so far.")


# ------------------------------------------------------------------ this week
def page_week() -> None:
    season = preds.filter(pl.col("season") == CURRENT)
    upcoming_weeks = season.filter(pl.col("result").is_null())["week"]
    weeks = sorted(season["week"].unique().to_list())
    start = int(upcoming_weeks.min()) if upcoming_weeks.len() else weeks[-1]
    week = st.select_slider("Week", options=weeks, value=start)
    st.title(f"Week {week} predictions")
    games = season.filter(pl.col("week") == week).sort("game_date", "gametime")
    cols = st.columns(2)
    for i, r in enumerate(games.iter_rows(named=True)):
        home, away, p = r["home_team"], r["away_team"], r["p_final"]
        fav, fp = (home, p) if p >= 0.5 else (away, 1 - p)
        result = ""
        changes = [f"{nick(t)} start {r[f'{sd}_expected_qb']} (usually "
                   f"{r[f'{sd}_recent_qb']})"
                   for t, sd in ((away, "away"), (home, "home"))
                   if (r.get(f"{sd}_qb_change") or 0) > 0 and r.get(f"{sd}_expected_qb")]
        qb_flag = (f"<div style='margin-top:6px'>⚠️ QB change: {'; '.join(changes)}</div>"
                   if changes else "")
        if r["result"] is not None:
            right = None if r["result"] == 0 else (p > 0.5) == (r["result"] > 0)
            tag = {True: "✅ our pick won", False: "❌ our pick lost", None: "tie"}[right]
            result = (f"<div style='margin-top:6px'><b>Final:</b> {nick(away)} "
                      f"{r['away_score']:.0f}, {nick(home)} {r['home_score']:.0f} · {tag}</div>")
        cols[i % 2].markdown(f"""
<div class='card'>
  <div class='muted'>{fmt_date(r['game_date'])}</div>
  <div class='row' style='margin-top:4px'><img src='{logo(away)}'>{nick(away)}
    <span class='muted'>at</span><img src='{logo(home)}'>{nick(home)}</div>
  {win_bar(away, home, 1 - p)}
  <div class='muted'>{nick(away)} {1-p:.0%} · {nick(home)} {p:.0%}</div>
  <div style='margin-top:6px'>Our pick: <b>{nick(fav)}</b> ({fp:.0%},
    {confidence_label(fp).lower()})<br>
    Vegas: {vegas_favorite_text(r['spread_line'], nick(home), nick(away))}</div>
  {qb_flag}{result}
</div>""", unsafe_allow_html=True)
        with cols[i % 2].expander("Why this pick?"):
            st.markdown(f"""
<div class='muted'>{nick(away)}: {identity_for(away)['label']}<br>
{nick(home)}: {identity_for(home)['label']}</div>
<div style='margin-top:6px'><b>Defenses</b><br>{nick(away)}: {defense_text(away, True)}<br>
{nick(home)}: {defense_text(home, True)}</div>
<div style='margin-top:6px'>{injury_html(r, away)}</div><div style='margin-top:8px'></div>
{reasons_html(r)}""", unsafe_allow_html=True)


# ------------------------------------------------------------------ track record
def page_track() -> None:
    st.title("Track record")
    st.write(f"How our picks did against real results over the last two seasons "
             f"({CURRENT - 1} and {CURRENT} so far), next to the Vegas favorite.")
    picks = league_picks(preds.filter(pl.col("season").is_in(HISTORY)))
    cols = st.columns(len(HISTORY))
    for c, yr in zip(cols, sorted(HISTORY), strict=True):
        p = picks.filter(pl.col("season") == yr)
        v = p.filter(pl.col("vegas_correct").is_not_null())
        c.markdown(f"#### {yr}{' (so far)' if yr == CURRENT else ''}")
        stat(c, "Our pick won", f"{p['model_correct'].mean():.0%}",
             f"{int(p['model_correct'].sum())} of {p.height} games")
        stat(c, "Vegas favorite won", f"{v['vegas_correct'].mean():.0%}",
             f"{int(v['vegas_correct'].sum())} of {v.height} games")
        stat(c, "Our stats model alone", f"{p['stats_correct'].mean():.0%}",
             f"{int(p['stats_correct'].sum())} of {p.height} games")

    st.subheader("When we're confident, are we right?")
    cc = confidence_check(picks)
    fig = go.Figure([
        go.Bar(name="What we predicted", x=cc["bucket"], y=cc["expected"],
               marker_color="#9ecae1", text=[f"{x:.0%}" for x in cc["expected"]]),
        go.Bar(name="How often our pick won", x=cc["bucket"], y=cc["actual"],
               marker_color="#3182bd", text=[f"{x:.0%}" for x in cc["actual"]]),
    ])
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                      xaxis_title="How sure we were about the pick", yaxis_title="",
                      height=380, legend={"orientation": "h", "y": 1.12}, margin={"t": 30})
    st.plotly_chart(fig, width="stretch")
    st.caption("If the two bars in a group match, our percentages mean what they say. "
               "Games per group: " + ", ".join(
                   f"{b}: {n}" for b, n in zip(cc["bucket"], cc["games"], strict=True)))
    st.info("Bottom line: our pick blends our stats model with the betting line, so it picks "
            "winners as often as Vegas does, and its percentages have been slightly more "
            "accurate than the line's own. The stats model alone (no betting info) is shown "
            "for comparison: since 2018 it trails Vegas by about a point, more in the last "
            "two seasons.")

    if "early_week_eval" in have:
        tuesday_section()


def tuesday_section() -> None:
    """Early-week picks scored against the opening line (the fair 'beat Vegas' test)."""
    ev = q("SELECT * FROM early_week_eval")
    get = {r["metric"]: r for r in ev.iter_rows(named=True)}
    st.subheader("Tuesday picks vs. the opening line")
    st.write("The line at kickoff already knows everything: injuries, starters, news. The "
             "fair test is earlier in the week: our pick made with only what's known on the "
             "Tuesday of game week, against the line Vegas opened with.")
    n = get["Tuesday pick: picks right"]["games"]
    c1, c2, c3 = st.columns(3)
    stat(c1, "Our Tuesday pick won", f"{get['Tuesday pick: picks right']['value']:.1%}",
         f"{n:,} games since 2018")
    stat(c2, "Opening-line favorite won", f"{get['Opening line: picks right']['value']:.1%}",
         "the line when betting opened")
    stat(c3, "Closing-line favorite won", f"{get['Closing line: picks right']['value']:.1%}",
         "the line at kickoff")
    only_us = get["Games where only the Tuesday pick was right"]
    only_open = get["Games where only the opening line was right"]
    move = get["Line moved toward our Tuesday number (disagree by 3+ pts)"]
    cover = get["Our side covered the opening spread (disagree by 3+ pts)"]
    st.markdown(
        f"- In the **{only_us['games']:.0f} games** where our Tuesday pick and the opening line "
        f"picked different winners, we were right **{only_us['value']:.0f}** times and the "
        f"opening line **{only_open['value']:.0f}** (very unlikely to be luck).\n"
        f"- When our Tuesday numbers disagreed with the opening line by 3+ points, the line "
        f"then **moved our way {move['value']:.0%}** of the time ({move['games']:.0f} games). "
        f"Our model sees where the market is heading.\n"
        f"- Against the opening point spread in those games our side won "
        f"**{cover['value']:.1%}**, a bit above the 52.4% needed to break even at standard "
        f"odds, but that's still within luck for {cover['games']:.0f} games, so it isn't "
        f"proven.")
    bys = q("SELECT * FROM early_week_by_season")
    fig = go.Figure([
        go.Bar(name="Our Tuesday pick", x=bys["season"], y=bys["tuesday_pick"],
               marker_color="#3182bd"),
        go.Bar(name="Opening-line favorite", x=bys["season"], y=bys["opening_line"],
               marker_color="#bdbdbd"),
    ])
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_range=[0.5, 0.75],
                      height=340, legend={"orientation": "h", "y": 1.15}, margin={"t": 30},
                      yaxis_title="Picked the winner")
    st.plotly_chart(fig, width="stretch")
    st.caption("Opening lines come from the Australia Sports Betting historical file "
               "(aussportsbetting.com/data). Re-download it each week to add opening lines "
               "for upcoming games. Its terms are personal use only.")


# ------------------------------------------------------------------ player types
def page_players() -> None:
    st.title("Player & defense types")
    st.write("We grouped players by *how* they play: where they get the ball, how they move "
             "the offense and how efficient they are. Team defenses are grouped the same way "
             "by how they rush the passer, cover and stop the run. Pick one to see the types.")
    pos = st.radio("Position", list(POS_NAMES), format_func=POS_NAMES.get, horizontal=True,
                   label_visibility="collapsed")
    spec = SPECS[pos]
    df = q(f"SELECT * FROM archetype_{pos.lower()}")
    cent = q(f"SELECT * FROM archetype_centroids_{pos.lower()} ORDER BY n DESC")
    vol = spec.volume_col
    cols = st.columns(2)
    for i, c in enumerate(cent.iter_rows(named=True)):
        recent = (df.filter(pl.col("archetype") == c["archetype"])
                  .sort(["season", vol], descending=True)
                  .unique("player_name", keep="first", maintain_order=True).head(4))
        ex = ", ".join(f"{n} ({s})" for n, s in
                       zip(recent["player_name"], recent["season"], strict=True))
        traits = "".join(f"<li>{x}</li>" for x in describe_archetype(c, spec.features, pos))
        cols[i % 2].markdown(f"""
<div class='card'><div style='font-size:1.15rem;font-weight:700'>{c['archetype']}</div>
<ul style='margin:6px 0'>{traits}</ul>
<div class='muted'>Recent examples: {ex}</div></div>""", unsafe_allow_html=True)

    st.subheader("Look up a player")
    base = df.filter(~pl.col("projected"))
    ranked = percentiles(base, spec.features)
    cur = df.filter(pl.col("projected"))
    if cur.height:  # current-season players ranked against the same historical pool
        cur = cur.with_columns(
            pl.Series(f"pct_{f}", [None if v is None else float((base[f] <= v).mean() * 100)
                                   for v in cur[f]], dtype=pl.Float64)
            for f in spec.features)
        ranked = pl.concat([ranked, cur.select(ranked.columns)])
    ranked = ranked.with_columns(key=pl.format("{} — {} {}", "player_name", "season", "team"))
    ordered = ranked.sort(["season", vol], descending=True)
    options = ordered["key"].to_list()
    mine = ordered.filter(pl.col("team") == team)["key"].to_list()  # selected team first
    chosen = st.multiselect("Choose one or two to compare", options,
                            default=(mine or options)[:1], max_selections=2)
    if not chosen:
        return
    long = []
    for k in chosen:
        r = ranked.filter(pl.col("key") == k).row(0, named=True)
        st.markdown(f"**{k}** → <span class='pill'>{r['archetype']}</span>",
                    unsafe_allow_html=True)
        long += [{"player": k, "stat": plain_label(f, pos), "pct": r[f"pct_{f}"]}
                 for f in spec.features]
    fig = px.bar(pl.DataFrame(long), y="stat", x="pct", color="player", barmode="group",
                 orientation="h", range_x=[0, 100], height=110 + 46 * len(spec.features),
                 color_discrete_sequence=["#3182bd", "#fd8d3c"],
                 labels={"pct": f"Percentile vs. other {POS_NAMES[pos].lower()}", "stat": ""})
    fig.add_vline(x=50, line_dash="dot", line_color="grey", annotation_text="average",
                  annotation_position="top")
    fig.update_layout(legend={"orientation": "h", "y": -0.15, "title": ""},
                      yaxis={"autorange": "reversed"}, margin={"t": 30})
    st.plotly_chart(fig, width="stretch")
    st.caption("A bar at 80 means higher than 80% of the comparison group since 2016. "
               "Higher isn't always better (e.g. sacks taken, big plays allowed).")


# ------------------------------------------------------------------ under the hood
def page_advanced() -> None:
    st.title("Under the hood")
    st.caption("Technical views for analysts: the clustering map, model diagnostics and the "
               "archetype-mix tests.")
    with st.expander("Archetype map (UMAP)", expanded=True):
        pos = st.selectbox("Position", list(SPECS), key="adv_pos")
        spec = SPECS[pos]
        df = q(f"SELECT * FROM archetype_{pos.lower()}").with_columns(
            label=pl.format("{} {}", "player_name", "season"),
            sample=pl.when("projected").then(pl.lit("current season (projected)"))
            .otherwise(pl.lit("completed seasons")))
        cent = q(f"SELECT * FROM archetype_centroids_{pos.lower()} ORDER BY n DESC")
        fig = px.scatter(df, x="umap_x", y="umap_y", color="archetype", symbol="sample",
                         hover_name="label", color_discrete_sequence=PALETTE, height=560,
                         category_orders={"archetype": cent["archetype"].to_list()},
                         hover_data={"team": True, spec.volume_col: True,
                                     "membership_prob": ":.2f", "umap_x": False,
                                     "umap_y": False, "sample": False})
        fig.update_traces(marker={"size": 6})
        fig.update_layout(xaxis_title="", yaxis_title="", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        zc = [c for c in cent.columns if c.startswith("z_")]
        heat = go.Figure(go.Heatmap(
            z=cent.select(zc).to_numpy(), x=[LABELS.get(c[2:], c[2:]) for c in zc],
            y=cent["archetype"].to_list(), colorscale="RdBu_r", zmid=0, zmin=-1.5, zmax=1.5,
            text=cent.select(zc).to_numpy().round(2), texttemplate="%{text}"))
        heat.update_layout(title="Centroid z-scores", height=360,
                           yaxis={"autorange": "reversed"})
        st.plotly_chart(heat, width="stretch")
        st.dataframe(q(f"SELECT * FROM archetype_selection_{pos.lower()}"),
                     width="stretch")
    with st.expander("Game model metrics (2018–present, walk-forward)"):
        st.dataframe(q("SELECT * FROM game_metrics"), hide_index=True, width="stretch")
        cal = q("SELECT * FROM game_calibration")
        cfig = px.line(cal, x="predicted", y="observed", color="model", markers=True,
                       height=420, title="Calibration")
        cfig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line={"dash": "dot", "color": "grey"})
        st.plotly_chart(cfig, width="stretch")
        st.dataframe(q("SELECT * FROM game_vs_vegas"), hide_index=True, width="stretch")
    with st.expander("Archetype mix → team offense"):
        st.dataframe(q("SELECT * FROM archetype_mix_tests"), hide_index=True,
                     width="stretch")
        st.dataframe(q("SELECT * FROM game_model_lift"), hide_index=True,
                     width="stretch")
        st.dataframe(q("SELECT * FROM qb_wr_pairings ORDER BY n DESC"), hide_index=True,
                     width="stretch")


PAGES = {"My team": lambda: page_team(team), "This week": page_week,
         "Track record": page_track, "Player types": page_players,
         "Under the hood": page_advanced}
PAGES[page]()
