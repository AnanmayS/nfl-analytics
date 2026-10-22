# NFL archetypes × game outcomes

Three connected analyses on nflverse data (2016–present):

1. **Player archetypes** (unsupervised): WR, TE, RB and QB player-seasons clustered into named roles.
2. **Game outcome model** (supervised): pre-kickoff home win probability and point margin, validated walk-forward and benchmarked against the closing line.
3. **Archetype mix → team efficiency**: whether roster archetype composition (and QB × WR pairings) explains offense beyond individual talent, and whether it improves the game model.

```bash
make install      # uv sync
make ingest       # nflreadpy -> data/raw/*.parquet -> data/nfl.duckdb (raw_* tables)
make features     # feat_receiving / feat_rushing / feat_passing
make archetypes   # archetype_<pos> tables + data/out/archetypes_<pos>.html reports
make train        # team features, injury availability, archetype mix, walk-forward models
make evaluate     # train + comparison models (refreshes the lift tables; ~3x slower)
make refresh      # weekly update: latest results, injury reports and lines -> new picks
make bets         # paper bets: place on games starting within the hour, settle finished ones
make schedule     # run refresh automatically Tue + Sat 8 AM (macOS launchd); make unschedule
make app          # Streamlit dashboard
make export       # data/app.duckdb: the dashboard's tables only, for the hosted app
make test
```

Stack: `nflreadpy` for data, parquet cache, DuckDB for transforms, polars (no pandas), scikit-learn / umap-learn / hdbscan / xgboost, Streamlit + Plotly. Managed with `uv`.

## Repo layout

| Path | Contents |
|---|---|
| `src/ingest/nflverse.py` | Per-season parquet cache (completed seasons cached; current season re-pulled), loads `raw_*` tables |
| `src/features/receiving.py`, `rushing.py`, `passing.py` | Player-season features (DuckDB SQL) |
| `src/features/team.py` | Team-game stats and leak-free rolling pre-kickoff features |
| `src/archetypes/pipeline.py`, `naming.py` | Standardize → UMAP → GMM/HDBSCAN → named archetypes |
| `src/models/game.py`, `train.py` | Models, walk-forward loop, scoring, lift and Vegas tests |
| `src/analysis/archetype_mix.py` | Part 3: snap-weighted mix, nested tests, pre-kickoff mix features |
| `src/analysis/archetype_report.py` | Standalone HTML archetype reports |
| `src/analysis/fan_view.py` | Plain-language views for the dashboard (team-perspective predictions, pick records, trait descriptions) |
| `src/bets.py` | Paper bets: placed an hour before kickoff, settled from final scores |
| `app/streamlit_app.py` | Dashboard |
| `tests/` | Feature unit tests and leakage tests |

## Keeping picks current

nflverse updates play-by-play nightly and posts each week's official injury reports, so new information arrives all week. **`make refresh`** pulls all of it and re-runs everything; so does the dashboard's **Get latest data** button. Completed seasons stay cached, and only the current season is re-downloaded. It takes a couple of minutes. Run it:

- after Monday night, for the latest results and form;
- again on Friday or Saturday, when final injury reports (Out / Doubtful / Questionable) and fresh betting lines are posted.

The sidebar shows when the data was last updated, which date results run through, and which week's injury reports are loaded.

**Automatic refresh (macOS).** `make schedule` installs a launchd job (`scripts/schedule_refresh.sh`) that runs the refresh every Tuesday and Saturday at 8:00 AM. It needs only the Mac, not the dashboard or a terminal. If the Mac is asleep then, the job runs on wake; if it's shut down, that run is skipped. Output goes to `data/refresh_scheduled.log`. Other commands:

- `scripts/schedule_refresh.sh status` checks whether the job is installed and how its last run ended;
- `scripts/schedule_refresh.sh run-now` triggers a run immediately;
- `make unschedule` removes the job.

While any refresh is running, an open dashboard shows a "getting the latest data" screen and reloads by itself when the refresh is done.

## Hosted dashboard

The dashboard can run online so the picks are there every week without the Mac:

- **GitHub Actions** (`.github/workflows/refresh.yml`) runs the refresh every Tuesday and Saturday at 13:00 UTC, the same days as the launchd job, and can be started by hand from the Actions tab. The parquet cache and database are kept in the Actions cache between runs, so a normal run only re-downloads the current season. If the cache is gone (first run, or evicted after 7 days unused), it runs `make all` instead, which takes longer.
- Each run writes **`data/app.duckdb`** (`make export`, `src/export.py`). It holds only the ~26 tables the dashboard reads, about 10 MB instead of ~800 MB, and is uploaded to the `app-data` release.
- **Streamlit Community Cloud** serves `app/streamlit_app.py`. With no local `data/nfl.duckdb`, the app downloads that release file, checks for a new one every 15 minutes, and hides the "Get latest data" button. `app/requirements.txt` lists the dashboard's own dependencies. It leaves out the clustering libraries and uses the CPU-only XGBoost build, so the app installs and starts quickly. Set `NFL_APP_DB_URL` to read the file from somewhere else.

The runner doesn't have `data/external/nfl.xlsx`. The export also drops every opening-line column and the Tuesday-pick tables, so nothing from that personal-use file is published. The Tuesday section of Track record therefore shows only on the local dashboard.

A second workflow, **Place bets** (`.github/workflows/bets.yml`), runs every 15 minutes and keeps the paper-bet ledger (below) as `bets.parquet` on the same release.

One-time setup: at [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose **Create app**, pick this repo, branch `main` and file `app/streamlit_app.py`, and set Python 3.12 under Advanced settings. A Community Cloud app sleeps after several days without visitors and wakes in about 30 seconds when opened.

## Dashboard

`make app` opens a team-first dashboard. Pick a team in the sidebar; the choice is kept in the URL (`?team=KC`), so it can be bookmarked.

- **My team.** A team-identity card, then a card for the next game: win chance, a plain outlook ("Favored", "Toss-up"), projected margin, the Vegas line and both teams' injury reports. Below it, the rest of the schedule with win-chance bars, then how our picks did in each of that team's games this season and last (✅/❌ next to Vegas), and the team's key players with their player type.
- **This week.** A card for every game in the selected week, with our pick, the Vegas favorite and, once played, the final score.
- **Bets.** Season profit for each paper-betting strategy, a running total by week, bets waiting on results, the coming week's games, and every settled bet (see "Paper bets").
- **Track record.** League-wide pick accuracy against Vegas for the last two seasons, and a "when we're confident, are we right?" chart comparing predicted with actual win rates.
- **Player types.** Each archetype described in words, with recent examples. A player lookup shows percentile bars against others at the position; it starts on the selected team's top player.
- **Under the hood.** The technical views: UMAP map, centroid heatmap, model metrics, calibration and the Part 3 tests.

Fans see the **final pick**: the stats model blended with the betting line when one is posted, and the stats model alone otherwise (see "Closing the gap to Vegas"). The projected margin is derived from that win probability (σ = 13.45), so the two always agree. Explanations name the line first, then list what the stats model sees.

**Team identity** (`team_identity` in `src/analysis/fan_view.py`; inputs from `src/analysis/team_profile.py`) combines three things:

- **Play-calling:** pass-first, balanced or run-first, from the early-down pass rate in close games (win probability 20–80%, outside the two-minute drill) ranked against the league.
- **Which side carries the team:** the average league percentile of the core offensive metrics compared with the core defensive ones.
- **Standouts:** the two best and the single worst rolling metrics, and the archetypes of the most-used QB, RB, WRs and TE.

**Why we lean a team.** The logistic model is a weighted sum, so each prediction splits exactly into per-factor pushes (`contrib_*` columns in `game_predictions`, taken from that week's walk-forward fit). The biggest pushes are shown as sentences that compare the two teams' league standing on that factor, e.g. "Patriots' passing offense (one of the league's best) vs. Jets' (below average)". A factor is shown only when the favored team really is better on it, and home field is left out for neutral-site games. All of a unit's factors are summed into one line, e.g. "Jets' defense as a whole (one of the league's best) vs. Patriots' (below average); biggest gap: run defense". This uses EPA per play allowed on every snap as the whole-unit rating, and the line appears only when that rating agrees with the direction of the push. Otherwise the individual factors are listed. These reasons describe the model's math; player archetypes appear in the identity card but don't drive this model.

**Defense in the identity card** shows the defense's archetype, built from this season's stats so far, separately from its recent-form rating (rolling, blended with last season). It also names its best and weakest areas. Every game card compares the two defenses.

## Paper bets

To see what betting our picks would actually win or lose, `src/bets.py` "places" a flat $100 bet on every game about an hour before kickoff and settles it from the final score. No money is involved.

**When.** The *Place bets* workflow runs every 15 minutes from 12:00 to 05:59 UTC. That covers every kickoff from 9:30 AM ET London games through Monday night, plus settling the late games. Each run pulls the current season's schedule from nflverse for the latest lines and scores, then bets every game that kicks off within the next 60 minutes (65 to allow for a late run) and doesn't have a bet yet. GitHub sometimes starts scheduled runs late. If a game has already kicked off it is skipped, never bet late.

**The pick.** The stats model comes from the last refresh (Tuesday or Saturday). It's blended with the line **as it stands at bet time**, using that week's fitted blend coefficients (`blend_coef_*` in `game_predictions`), so a line that moved since Saturday still counts. Games with no line get no bet.

**Three strategies**, each $100 a bet:

| Strategy | Bet |
|---|---|
| Moneyline on our pick | Every game: our pick to win, at its moneyline |
| Spread on our side | Every game: whichever side our projected margin (σ = 13.45, as on the dashboard) says beats the spread, at its spread odds (−110 if none is posted) |
| Moneyline value bets | Only when our win chance beats the moneyline's break-even (vig included) by 3+ points, on either side |

**Settling.** A moneyline bet wins if the team wins, and a tie is a push. A spread bet wins if the team's margin plus its line is positive, and exactly zero is a push. A win pays at the American odds taken, so $100 at −150 wins $66.67 and at +150 wins $150.

**The ledger** (`data/bets.parquet`, one row per bet) only grows. Odds, line, our win chance and the break-even are fixed when a bet is placed. Each row also stores its status, scores and profit. When a new ledger is started, this season's finished games are **backfilled** from the closing line and marked `backfill`, so the season total starts at week 1; every later bet is `live`. `python -m src.cli bets --backfill` fills games a live run missed the same way.

On GitHub the ledger lives on the `app-data` release and is uploaded only when it changes. The hosted dashboard downloads it every 15 minutes. Locally, `make bets` does one run against `data/bets.parquet`, and the local dashboard reads that file if it exists, otherwise the published ledger. Bets need predictions built with the blend coefficients, so after upgrading, run `make refresh` (or the Refresh picks workflow) once. Until then the bet step logs a warning and places nothing.

**What to expect.** The final pick leans almost entirely on the betting line (see "Closing the gap to Vegas"). Betting it every game should come out near break-even minus the bookmaker's cut. Backfilling 2026 weeks 1–6 gave −5.8% on moneylines (58-34-1) and −13.9% on spreads (39-48-6). The value strategy bets rarely and is too small a sample to judge.

## Data

`raw_pbp`, `raw_player_stats`, `raw_rosters`, `raw_snap_counts`, `raw_participation`, `raw_ngs_{passing,receiving,rushing}`, `raw_schedules` (closing spread, total and moneylines) and `raw_players` (a pfr ↔ gsis ID crosswalk; with it, ~99.9% of snaps map to a player). Participation data isn't published for 2026 yet. Schedules, snap counts and rosters use pre-relocation team codes (OAK, SD); these are normalized to LV/LAC where joined.

## Part 1 — Archetypes

**Features** (regular season; two-point tries and sacks excluded):

- **WR/TE** (≥50 / ≥40 targets): aDOT, target share and air-yards share (denominators use only games the player appeared in), YAC over expected (pbp `xyac`), NGS average separation, middle-of-field target rate, red-zone target share, catch rate over expected (`complete − cp`), EPA/target.
- **RB** (≥100 touches): rush EPA/att, NGS RYOE/att, target share, route participation, explosive run rate (10+ yds), goal-line carry share (inside the 5).
- **Team defenses** (whole team-seasons, `src/features/defense.py`): pass and run EPA allowed, pressure rate, blitz rate, stacked-box rate (vs. that season's league average), throw depth allowed, explosive plays allowed, takeaway rate, third-down rate allowed, completion % over expected allowed. 2026 defenses are projected after 3+ games.
- **QB** (≥200 dropbacks): aDOT, NGS time to throw, scramble rate, CPOE, sack rate, EPA/dropback (`qb_epa`), deep-pass rate (20+ air yards).

**Pipeline:**

1. Median-impute and z-score using completed seasons only.
2. Embed with UMAP: 2-D (`n_neighbors=15`, `min_dist=0.1`) for plots, 5-D (`n_neighbors=30`, `min_dist=0`) for clustering.
3. Fit candidate GMMs (k = 3–9) and HDBSCAN (three `min_cluster_size` values).
4. **Choose the model.** BIC on a UMAP embedding keeps falling as k grows (UMAP manufactures local density), so it can't pick k. Instead k is the most bootstrap-stable GMM with k ≥ 4: mean ARI of 10 refits on 80% subsamples, preferring the smaller k when stability is within 0.02. HDBSCAN replaces the GMM only if its silhouette in the original z-space beats it by >0.05 with ≤25% noise. It never did; it labelled 7–31% of players as noise.
5. **Name clusters.** Each archetype is named from its centroid z-scores. Every feature has a word for a high centroid and/or a low one; the two most extreme features are used, ordered adjective → noun, with a third added only to break a tie.
6. **Project the in-progress season** (2026) onto the fitted scaler, UMAP and GMM, with the volume threshold prorated by weeks played.

Output: `archetype_<pos>` (player, season, team, archetype, membership probability, UMAP coordinates, z-scores), `archetype_centroids_<pos>` and `archetype_selection_<pos>`.

**Results** (fit on 2016–2025; RB on 2018–2025):

| Pos | k | Stability (ARI) | Silhouette (z-space) | Archetypes (n) — examples |
|---|---|---|---|---|
| WR | 4 | 0.92 | 0.13 | Deep Low-Catch (235) — Evans '18, McLaurin '23 · Sure-Handed Efficient (228) — Hill '22, Landry '17 · Red-Zone Alpha (200) — Kupp '21, M. Thomas '19, Chase '25 · Underneath Complement (187) — Godwin '22, Tate '17 |
| TE | 7 | 0.69 | 0.09 | Red-Zone Efficient Alpha (47) — McBride '25, Bowers '24 · Red-Zone Middle-Field Alpha (48) — Ertz '19, Kelce '24 · Sure-Handed Efficient (63) · Separator Complement (64) · Middle-Field Complement (49) · Deep Contested (45) — Graham '18, Gesicki '19 · Underneath Perimeter (34) |
| RB | 7 | 0.89 | 0.14 | Every-Down Pass-Catcher (80) — McCaffrey '19/'25 · Goal-Line Back (82) — Henry '22, Jacobs '24 · Elusive Efficient (68) — Henry '20, Barkley '24 · Explosive Pass-Catcher (40) — Cook '23, Hall '25 · Two-Down Back (59) — Peterson '18 · Inefficient Pass-Catcher (66) · Inefficient Two-Down Back (36) |
| DEF | 4 | 0.84 | 0.11 | Lockdown Sticky-Coverage Defense (80) · Keep-It-In-Front No-Big-Plays Defense (73) · Tight-Underneath Big-Play-Prone Defense (94) · Pass-Vulnerable Third-Down-Leaky Defense (73) |
| QB | 5 | 0.71 | 0.12 | Efficient Sack-Avoiding (58) — Mahomes '21/'22, Allen '21 · Conservative Accurate (83) — Brady '21/'22, Burrow '24 · Hold-the-Ball Scrambler (67) — Wilson '20, C. Williams '24 · Quick-Release Pocket Passer (98) — Cousins '22 · Inefficient Erratic (57) — Howell '23, Young '23 |

Membership is confident (mean max-posterior 0.97–0.99). Silhouettes in the original feature space are low (0.09–0.14), though. **Player roles form a continuum.** The archetypes are a useful, reproducible partition of that continuum, not natural species.

**Design note:** efficiency features (CROE, EPA/target, rush EPA, CPOE, EPA/dropback) are *included* in the clustering, by choice. So archetypes mix role and quality, e.g. "Inefficient Erratic" or "Sure-Handed Efficient". To cluster on style only, move those features to `PositionSpec.profile` in `src/archetypes/pipeline.py`. They stay z-scored and reported, just not clustered on. An earlier style-only WR run gave 6 role-pure clusters (Underneath Separator, Red-Zone Alpha, Perimeter Complement, Deep Contested, Deep Perimeter, Underneath Alpha).

## Part 2 — Game model

**Team-game stats** (all games, from pbp): offensive and defensive EPA/play split by pass and run, success rate, early-down EPA, turnover rate, and pressure rate (sack or QB hit per dropback). Participation pressure data stops in 2025, so this proxy keeps one definition across seasons.

**Whole-defense stats** (what each defense allowed or did):

- Overall EPA/play allowed.
- Explosive plays allowed (runs of 10+ yards, passes of 20+).
- Third-down conversion rate allowed.
- Completion % over expected allowed.
- Opponents' average throw depth.
- Blitz rate (5+ pass rushers).
- Stacked-box rate (8+ defenders in the box).

Blitz and box counts come from participation data through 2025 and FTN charting for 2026; the two sources are identical for 2023–25. The stacked-box rate's league level halves in 2024 when the source changes. It is therefore used only for display, where teams are compared within the same week, and for defense archetypes, as a deviation from that season's league average.

**Rolling, strictly pre-kickoff features** (`src/features/team.py`):

- Within a season, each metric is an exponentially weighted mean of prior games (half-life 5 games).
- The **prior** is last season's team mean regressed 50% toward last season's league mean. It enters as 4 pseudo-games that decay like any other game, so week 1 is pure prior and the prior fades as games accrue.
- A game's feature is the as-of state from the team's last game **strictly before** its kickoff date. Unplayed future games get the latest state.

**Model inputs:** home-minus-away differences of the 12 core metrics, the 6 whole-defense stats (everything above except the stacked-box rate), the starting-QB change and quality swing (below), rest difference (clipped ±7), and neutral-site and division-game flags. Home field is the intercept.

The whole-defense stats were tested before being added: decided on 2018–21 and confirmed on 2022–26, with results stored in `game_defense_lift`. They leave the logistic model unchanged (log loss 0.6413 → 0.6413, CI ±0.004) and nudge XGBoost (0.6524 → 0.6492, Δ −0.003, CI −0.008 to +0.002). Since they cost nothing in accuracy, they're kept so the explanations can cite defense as a whole.

**This week's injury report** (`src/features/availability.py`, `game_availability`). For every scheduled game, including upcoming ones, each player's recent role is his average snap share over the team's last 3 games before kickoff. His injury-report status is turned into a chance he sits: Out 1.0, Doubtful 0.9, Questionable 0.2. A player who already missed those 3 games has no recent role, so a long absence that the team's form already reflects isn't counted twice. Three candidate features came out of this:

- the starting QB being out;
- offensive starters missing, weighted by snap share;
- defensive starters missing, weighted the same way.

All three were tested walk-forward, decided on 2018–21 and confirmed on 2022–26. Only the **starting-QB flag** helped in both periods (log loss −0.0009 and −0.0017; the full set was *worse* on 2018–21), so it was added to the model, improving log loss from 0.6413 to 0.6400. It has since been replaced by the starting-QB features below, which subsume it. The missing-starter counts and named absences are still shown on the dashboard.

**Who's starting at QB** (`src/features/quarterbacks.py`, `game_qb`). A review of 2025–26 misses found that most of the model's own errors were QB changes the injury report never shows:

- players moved to injured reserve drop off the report (e.g. Joe Burrow in 2025 week 3);
- benchings and rookies taking over (Quinn Ewers for Miami in 2025 week 16);
- offseason moves, since week-1 form comes from last season's QB (Aaron Rodgers to Pittsburgh).

Each game's **expected starter** is the schedule's listed starting QB (nflverse fills this for played games and usually a week or two ahead). Failing that, if the recent starter isn't active on that week's roster (injured reserve, etc.) or is listed Out or Doubtful, it's the next QB by recent snaps. Otherwise it's the recent starter. Each QB gets a **rating**: EPA per dropback from games before kickoff only, with earlier seasons weighted 0.6 per season back and 300 dropbacks of replacement-level play (−0.10) mixed in, so rookies and seldom-used backups rate near replacement. Features: `qb_change` (the expected starter isn't the recent one) and `qb_delta` (rating swing vs. the recent starter).

Four variants were compared walk-forward against the injury-only QB flag. **qb_change + qb_delta** was best on 2018–21 (−0.0041) and on 2022–26 (−0.0049), and is now in the model. Over 2018–26, log loss improves from 0.6400 to 0.6355 (Δ −0.0045, 95% CI −0.0091 to −0.0001); this is the first addition that clears noise. Against a model with no QB information at all (`game_qb_lift`), logistic goes 0.6413 → 0.6354 (95% CI −0.0108 to −0.0007) and XGBoost 0.6492 → 0.6441 (−0.0089 to −0.0013). The pick rate is essentially unchanged: 63.4% → 63.1% over 2,298 games, and 60.9% → 61.7% for 2025–26. Of the 21 games in 2025–26 where the model was confidently wrong and Vegas was right, 8 involved a QB change; the new features get 4 of those 8 right. One caveat: for past games, the listed starter is whoever actually started, which was nearly always announced before kickoff; for upcoming games, it's nflverse's projection.

**Models:** logistic regression (C = 0.01, chosen on 2018–23) and an XGBoost classifier for win probability; ridge and an XGBoost regressor for margin.

**Validation:** walk-forward by week, 2018 → 2026 week 6. Each week's games are predicted by models fit on every game that kicked off before that week began. 2016 is burn-in (no prior) and 2017 is the first training season.

**Vegas benchmark:** closing spread → P(home win) = Φ(spread / 13.45). The de-vigged moneyline is shown too.

**Results** (2,298 scored games, 2018–2026; ties and games without a moneyline excluded):

| | Log loss | Brier | Accuracy | Margin MAE |
|---|---|---|---|---|
| **Final pick (stats + line)** | **0.6115** | **0.212** | **66.1%** | — |
| Stats model (logistic) | 0.627 | 0.219 | 65.1% | — |
| XGBoost | 0.640 | 0.224 | 64.0% | — |
| Ridge margin | — | — | 64.7% | 10.19 |
| **Closing spread** | **0.612** | **0.212** | **66.1%** | **9.85** |
| De-vigged moneyline | 0.610 | 0.211 | 66.3% | — |

The stats-only models are well calibrated but trail the market. In 2024 the stats model matched Vegas's pick rate (70.5%) and beat its log loss, but over 2018–26 it trails by about a point. A likelihood-ratio test of `y ~ logit(p_vegas) + logit(p_stats)` finds no meaningful information beyond the closing spread (p = 0.74). The fitted Vegas coefficient of ~1.15 means the σ = 13.45 conversion is slightly *under*-confident for this era; a σ near 11 fits better.

### Closing the gap to Vegas

The goal was to match the Vegas favorite's pick rate (66.1% over 2018–26). To avoid tuning to the test set, every idea was tried on **2018–23 only**; **2024–26 was held out** and scored once at the end. Changes in pick rate of about ±1 point (≈23 games) are within noise, so log loss decided close calls.

| Step (dev 2018–23: Vegas 65.5% / 0.613) | Dev picks | Dev log loss |
|---|---|---|
| Starting model | 62.97% | 0.6425 |
| + opponent-adjusted ratings (200-day memory) | 64.79% | 0.6388 |
| + stronger regularization (C 0.1 → 0.01) | 65.03% | 0.6328 |
| + success-rate rating, margins capped at ±21 | 65.09% | 0.6316 |
| Tried and dropped: shorter rating memory (50–75 days), lean feature sets, home-field trend terms, a second short-memory rating, ridge/XGBoost blends | ≤ 64.8% | ≥ 0.634 |

**Opponent-adjusted ratings** (`src/features/ratings.py`, `team_ratings`). Each week, weighted ridge regressions are refit on games before that week, with older games decaying over a 200-day half-life:

- a **margin rating**: schedule-adjusted point differential, like SRS, with margins capped at ±21;
- **offense and defense ratings** from EPA per play;
- **offense and defense ratings** from success rate.

Raw rolling stats count a big game against a weak opponent the same as one against a strong opponent; these ratings don't. On their own they improve log loss from 0.6320 to 0.6270 (95% CI −0.0092 to −0.0004; `game_ratings_lift`).

**Held out, 2024–26 (648 games):**

| | Picks | Log loss |
|---|---|---|
| Starting model | 63.89% | 0.6175 |
| **Stats model (all of the above)** | **64.97%** | **0.6152** |
| Vegas | 67.59% | 0.6084 |

The stats model improved on held-out seasons but stayed behind Vegas, which was unusually sharp in 2024–26. The last step adds the market itself. **Stats + line blend** (`market_blend` in `src/models/game.py`): each week, a two-input logistic regression on logit(stats model) and logit(Vegas) is fit on earlier games only, using the stats model's own out-of-sample predictions.

| | Picks, 2018–26 | Log loss, 2018–26 | Picks, holdout | Log loss, holdout |
|---|---|---|---|---|
| Stats model | 65.06% | 0.6270 | 64.97% | 0.6152 |
| **Final pick: stats + line** | **66.10%** | **0.6115** | **67.59%** | **0.6052** |
| Vegas favorite / spread | 66.10% | 0.6117 | 67.59% | 0.6084 |

The final pick **ties Vegas's pick rate** and is **slightly better calibrated**, clearly so on the holdout (0.6052 vs. 0.6084). Be clear about why: the learned weights are about 1.19 on the line and 0.04 on the stats model. The blend mostly corrects the spread-to-probability conversion, which is a little timid for this era (σ = 13.45), rather than adding new information. With no line posted (games more than a week or two out), the pick is the stats model alone.

### Tuesday picks vs. the opening line

The closing line already contains everything known by kickoff, so it's the hardest possible benchmark. The fairer test of "beating Vegas" is earlier in the week (`src/models/early.py`): a model using only what's known on the **Tuesday of game week**, blended with the **opening** line and scored against that same opening line. Tuesday information means:

- every earlier game (rolling form, opponent-adjusted ratings) and rest days;
- the QB who has been starting, unless last week's roster had him on injured reserve.

It excludes injury reports (they start Wednesday), announced starters and the closing line. Nothing was re-tuned for this test: it uses the same model settings as above.

Opening lines come from the [Australia Sports Betting](https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/) historical NFL file. It is sourced from Pinnacle (2014–Feb 2018), bet365 (Sep 2018–Feb 2025) and betr (Sep 2025 on), and matched to all 2,860 completed games from 2016–2025 (`src/ingest/odds.py`, `game_lines`). The site's bot check means you download it yourself to `data/external/nfl.xlsx`. Its terms allow personal use only and forbid republishing, so the file and everything derived from it stay in the git-ignored `data/` folder.

Results, 2018–26 (2,283 games; `early_week_eval`, `early_week_by_season`):

| | Picks | Log loss |
|---|---|---|
| Tuesday stats model (no line) | 64.4% | 0.632 |
| **Tuesday pick (stats + opening line)** | **66.2%** | **0.620** |
| Opening line | 65.4% | 0.621 |
| Closing line (kickoff) | 66.3% | 0.610 |

- **Beats the opening line on picks.** In the 46 games where the Tuesday pick and the opening favorite differed, the Tuesday pick was right 32 times to 14 (exact test p = 0.011). It was ahead in 7 of 9 seasons. The log-loss gain over the opener (−0.0013) is not significant (CI −0.0046 to +0.0021).
- **Predicts where the line goes.** When the Tuesday model disagrees with the opener by 3+ points, the line moves toward it 67% of the time (644 games, p ≈ 10⁻¹⁹); at 4.5+ points, 70%. The model carries information the opening line doesn't, which the market then prices in by kickoff.
- **Against the opening spread, not proven.** In those games the model's side covers the opening spread 54.2% (3+ points, 742 games) and 55.4% (4.5+ points, 368 games). That's above the 52.4% break-even at standard odds, but within luck (p = 0.18 and 0.13). It would take several more seasons to tell.

For upcoming games this needs a fresh copy of the file each week; without one, upcoming games have no opening line.

## Part 3 — Archetype mix → team efficiency

**Composition:** each player-game's offensive snaps (`raw_snap_counts`, mapped to gsis IDs) are tagged with that player-season's archetype. Players below the volume threshold are "Unlabeled". Shares are computed within each position group per team-season (`team_archetype_mix`).

**Test** (`archetype_mix_tests`): nested OLS models of team offensive EPA/play on 256 team-seasons (2018–2025). Each step gets an F-test and 10×10-fold ridge CV R².

1. Talent controls only.
2. Add archetype main effects: primary-QB archetype dummies, plus RB, WR and TE share vectors.
3. Add QB-archetype × WR-share interactions (the "pairings").

There are two versions of "talent":

| Talent control | Step | Added params | F-test p | CV R² before → after |
|---|---|---|---|---|
| Same-season individual efficiency (QB EPA/db, receiver EPA/tgt, rush EPA) | + archetype mix | 22 | 0.003 | 0.966 → 0.966 |
| | + QB × WR pairings | 16 | 0.62 | 0.966 → 0.963 |
| Prior-season team EPA | + archetype mix | 22 | <1e-28 | 0.11 → 0.51 |
| | + QB × WR pairings | 16 | 0.85 | 0.51 → 0.45 |

**Reading this:**

- **Pairings explain nothing beyond individual talent**, under either control. The QB × WR residual table (`qb_wr_pairings`) has cell means within ±0.008 EPA/play, about one standard error.
- Against same-season efficiency, archetype main effects are in-sample significant but add **nothing out of sample**.
- Against prior-season EPA, they look powerful. That's mostly circular: the archetypes were clustered partly on same-season efficiency, so they partly *are* the outcome.

**Game-model lift** (`game_model_lift`). Pre-kickoff mix features are home-minus-away archetype snap shares, built so they can't leak:

- Snaps come only from the team's earlier games this season; a team's first game uses last season's snaps.
- Each player's archetype comes from his most recent season *before* the current one.

Paired bootstrap over games (95% CI):

| Model | Base | + mix | Δ (95% CI) |
|---|---|---|---|
| Logistic, log loss | 0.635 | 0.652 | +0.017 (+0.006, +0.027), worse |
| XGBoost, log loss | 0.644 | 0.642 | −0.002 (−0.007, +0.003), n.s. |
| Ridge, margin MAE | 10.24 | 10.48 | +0.24 (+0.13, +0.36), worse |
| XGBoost, margin MAE | 10.33 | 10.28 | −0.05 (−0.13, +0.03), n.s. |

The 46 mix columns overfit the linear models. XGBoost picks up a sliver, but the best model still trails the base logistic and the market, and no variant adds information beyond the spread (p ≥ 0.26).

## Leakage safeguards and tests

`make test` runs 53 tests:

- **Feature definitions** (`test_receiving_features.py`): sacks, two-point tries and postseason are excluded; share denominators use only games the player appeared in; CROE and YACOE match hand calculations; the NGS weekly fallback works.
- **Archetype helpers** (`test_archetypes.py`): naming order and uniqueness; standardization uses fit-set statistics only.
- **Rolling features** (`test_team_features.py`):
  - perturbing game *g* and all later games leaves the feature for game *g* unchanged;
  - week 1 equals the regressed prior;
  - a same-day state is never used;
  - the prior doesn't depend on the current season;
  - the exponential weights match a hand calculation.
- **Injury availability** (`test_availability.py`): roles come only from earlier games, statuses map to sit chances, a long absence already reflected in recent snaps isn't counted again, and relocated team codes are normalized.
- **Starting QB** (`test_quarterbacks.py`): ratings use only earlier games and decay old seasons; a new or unknown QB rates at replacement level; a listed starter wins; injured reserve or an Out listing falls back to the backup.
- **Ratings and blend** (`test_ratings_blend.py`): ratings recover the true team order, don't move when games on or after the week start are changed, and the blend produces nothing for games without a line.
- **Opening lines and Tuesday picks** (`test_odds_early.py`): old team names, ±1-day dates, duplicate team codes and swapped neutral-site listings match the right game; the evaluation counts disagreements and line moves correctly.
- **Paper bets** (`test_bets.py`): bets go in only in the hour before kickoff, never after it and never twice; backfill uses only finished games; Eastern kickoff times convert across daylight saving; moneyline, spread and push settlement and payouts match hand calculations; value bets can take the underdog. The stored blend coefficients reproduce the blend exactly (`test_ratings_blend.py`).
- **Integration** (`test_game_features_integration.py`): on the real DB, features for games on a date are identical (to 1e-12) whether or not that date's and later games exist in the input.

The walk-forward loop trains only on `game_date < week_start`. Archetype projections for 2026 use models fit on 2016–2025. Pre-kickoff mix features use only earlier snaps and earlier-season archetypes.

## Limitations

- **Scheme effects on archetypes.** Middle-of-field rate stands in for slot rate (true alignment isn't public). Target share, route participation and aDOT reflect coordinators and teammates as much as the player, and the same player changes archetype with a new team (e.g. JuJu Smith-Schuster '18 vs '20).
- **Small samples.** ~35 qualifying QBs and ~55 RBs per season; TE clustering (n = 350) is the least stable (ARI 0.69). Partial-season (2026) projections rest on ~6 games.
- **Coverage gaps.** RB archetypes start in 2018 because NGS rush yards over expected does. Route participation for 2026 is imputed because participation data isn't out yet.
- **Proxies.** Pressure is sack-or-hit; route participation is "on field for a dropback".
- **Defensive scheme data.** Man/zone coverage exists only for 2018–25, so it's left out of the defense archetypes. Blitz and box counts switch sources in 2023 and 2026 (participation → FTN charting).
- **Archetypes include efficiency** (by choice), which muddies "beyond talent" tests. The style-only toggle exists for a cleaner Part 3.
- **Injury reports** use fixed sit chances (Out 1.0, Doubtful 0.9, Questionable 0.2) and can't see late scratches or players who aren't on the report. Reports pulled mid-week are incomplete, so refresh again after Friday's final report. The model uses only the QB features. "Starters missing" counts are shown on the dashboard but didn't improve predictions.
- **QB ratings** are a single number (shrunken EPA per dropback). They don't capture scheme fit, the supporting cast, or how a backup has looked in practice. Games more than a week or two out have no listed starter yet and fall back to roster and injury status.
- **The game model omits** weather and travel. Its week-1 rosters are last season's (no free agency or draft). Hyperparameters are untuned defaults.
- **Vegas efficiency.** The closing line already prices everything here and more. The final pick matches Vegas's pick rate by leaning on the line; the stats model alone is about a point behind over 2018–26. Beating Vegas on picks would take information the market lacks, not a better summary of public play-by-play.
- **The blend uses the line available at refresh time.** For past games that's the closing spread; for upcoming games it's the current line, which can still move.
