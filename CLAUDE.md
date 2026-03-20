# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NBA prop betting pipeline that parses sportsbook boards, fetches live player data, runs a 15-layer analysis pipeline, and builds parlays via NEXUS v4 (38-agent multi-agent consensus, primary) with Parlay Engine v1 as backup. Primary outputs: NEXUS SAFE 3-leg + NEXUS AGGRESSIVE 5-8 leg. Shadow lab: 100 diverse Engine strategies + 22 NEXUS strategies for leaderboard tracking. All Python 3, no framework.

## Key Commands

```bash
# Full automated pipeline (research + analyze + parlays)
python3 predictions/run_board_v5.py 2026-03-13 /path/to/board.json

# Grade yesterday's predictions against actual box scores
python3 predictions/grade_results.py 2026-03-12

# Parse raw sportsbook text into structured JSON
python3 predictions/parse_board.py < raw_board.txt
```

## Architecture

**Data flow:** Raw board text → `parse_board.py` → `run_board_v5.py` orchestrator → parallel `game_researcher.py` agents (injuries, spreads, B2B, news) → `analyze_v3.py` per-line 15-layer pipeline → `correlations.py` enrichment → `xgb_model.py` XGBoost scoring (136 features) → `mlp_model.py` MLP scoring → ensemble_prob (60/40 blend) → `pregame_check.py` availability filter → `parlay_nexus.py` NEXUS v4 primary (38-agent consensus) → `parlay_engine.py` backup parlays (ensemble + UNDER bias) → output JSON.

### Core Modules (all in `predictions/`)

- **`run_board_v5.py`** — Main orchestrator. Resolves game context, spawns parallel research agents, pre-fetches player logs into cache, runs analysis, feeds NEXUS v4 (primary) + Parlay Engine v1 (backup). Worker counts auto-scale via `_calc_workers()`.
- **`parlay_engine.py`** — Parlay Engine v1.2 (BACKUP): data-driven builder using ensemble_prob (XGB 60% + MLP 40%) + strong UNDER bias (0.30 bonus). Hardened filters: GTD/Questionable -0.15 penalty in both `_primary_score()` and `_floor_score()`, ensemble_prob >= 0.50 floor for SAFE, D/F tier blocked from AGG survival. Consensus tracking on 100 shadow strategies. v11: Total coverage scoring — 8 new factor groups. Shadow: 102 diverse 3-leg strategies (32 curated incl `mlp_top`/`ensemble_top` + 70 grid-generated). 3-level fallback cascade guarantees zero nulls.
- **`analyze_v3.py`** — Per-prop market-calibrated scoring (v13). Core projection now market-anchored: dynamic blend of statistical avg + sportsbook line (15-40% line weight scaled by hit rate — coin-flip HR=40% weight, strong HR=15%). Hit rate direction calibration: L10/season HR adjusts projection (HR<40% → -3%, HR>80% → +3%). 2.5% empirical UNDER correction (cross-day data: UNDER 60-65%, OVER 35-45%). Thin gap UNDER flip: gaps 0-1.0 with HR<60% → UNDER. Backtested +7pp improvement (46%→53%) across 4,000 predictions. Plus: rate-based opponent defense, pace, WITH/WITHOUT teammate splits (v10.1: diminishing returns, ±20% cap), foul trouble, streaks, distance-scaled B2B, blowout risk, tier grading (S/A/B/C/D/F). v11 live travel computation.
- **`pregame_check.py`** — Pre-game availability filter. Checks NBA schedule for postponed games, filters OUT players from candidate pool. Runs before NEXUS.
- **`parlay_nexus.py`** — NEXUS v4 38-agent parlay builder (PRIMARY): 5-tier system (3 Scouts → 4 Evaluators → 11 Constructors → 15 Devils → 5 Judges). Soft screen (CORE/FLEX/REACH/KILL) replaces binary hard screen. Cascade fallback guarantees output (full v4 → relaxed → survival). Borda count consensus across 5 judges. Outputs 4 parlays: 1x LOCK 3-leg + 1x Main 5-leg + 1x Value 4-leg + 1x Aggressive 6-8 leg. Scout_venue enhanced with travel_distance. Promoted to primary builder (Mar 19) — NEXUS SAFE 3/3 on Mar 17 + Mar 19 vs Engine 0 cashed parlays.
- **`pbp_client.py`** — PBPStats.com API client. Free REST API (no auth). Fetches pace/efficiency, WOWY splits, on/off impact. Cache at `predictions/cache/pbp/`. 1.0s rate limit. `TEAM_ABR_TO_ID` for all 30 teams. `prefetch_teams()` warms cache in 1 call.
- **`ref_model.py`** — Referee tendency model. Extracts officials per game from `BoxScoreSummaryV2`. Builds ref database with avg fouls, total points, over rate per crew. `enrich_with_ref_features()` adds 4 features. `build_ref_database()` for historical backfill (~14min for full season).
- **`coach_model.py`** — Coach profiling model. Derives rotation_depth, star_minutes_share, blowout_bench_rate, pace_tendency from cached game logs. Zero new API calls for data (only coach name lookup). `enrich_with_coach_features()` adds 5 features.
- **`sim_model.py`** — Monte Carlo possession simulation. KDE/normal distribution fitted to L10 game logs. 5000 sims per prop, vectorized numpy (~0.5ms/prop). `enrich_with_sim()` adds sim_prob, sim_mean, sim_std. Context-adjusted variant with pace/fatigue/matchup factors.
- **`nn_embedder.py`** — Deep NN embedding extractor. MLPClassifier(256, 128, 64, 32) trained on same data. Extracts 32-dim penultimate layer activations via manual forward pass through coefs_. `enrich_with_embeddings()` adds `nn_emb_0..nn_emb_31` to each prop. XGBoost then sees learned nonlinear interaction patterns.
- **`meta_learner.py`** — Stacked meta-learner. LogisticRegression with isotonic calibration learns optimal blend of xgb_prob, mlp_prob, sim_prob + 9 interaction/context features. Replaces hardcoded `0.6*xgb + 0.4*mlp` ensemble. Trains from graded daily predictions (genuine out-of-sample). Falls back to 60/40 if not trained.
- **`regression_model.py`** — XGBRegressor predicting actual stat values (not hit/miss). MAE 5.93. Margin predictions: `reg_margin` = predicted - line. Picks with |margin| >= 3 hit at 95%+. Derives `reg_over_prob` via normal CDF. Per-stat residual stds (BLK 1.53, PTS 10.62).
- **`parlay_optimizer.py`** — Correlation-aware parlay builder. Computes pairwise independence, selects legs that maximize adjusted parlay probability (not just individual prob). Backtested Mar 16-19: optimized 57.1% leg HR vs greedy 38.9%. Runs as shadow strategy `corr_optimized`.
- **`train_current_lines.py`** — Focused training data generator. Takes today's sportsbook board (real lines), applies them retroactively to each player's full season game logs. 81 players × ~51 games = ~39K focused samples with REAL labels. Trains a focused XGBoost model. **AUC 0.589 vs baseline 0.554 (+0.035)**. CLI: `python3 train_current_lines.py --board <path>`.
- **`regression_model.py`** — XGBRegressor predicting actual stat values. MAE 5.93. Margin predictions: `reg_margin` = predicted - line. Picks with |margin| >= 3 hit at 95%+. Derives `reg_over_prob` via normal CDF.
- **`parlay_optimizer.py`** — Correlation-aware parlay builder. Selects legs that maximize adjusted parlay probability via independence scoring. Backtested: **57.1% leg HR vs greedy 38.9%**. Shadow strategy `corr_optimized`.
- **`measure_features.py`** — Feature measurement. v8/v9 features are 100% NaN in training data. 92-feature model is still best (AUC 0.5964).
- **`venue_data.py`** — Static arena altitude/timezone/lat/lng map for all 30 NBA teams. Used by scout_venue() for altitude and travel fatigue adjustments. `haversine_miles()` and `get_travel_distance()` for city-to-city distance computation. Zero API calls.
- **`game_researcher.py`** — Parallel research via `ThreadPoolExecutor`. 8 research types per team. `ScoreboardCache` eliminates ~340 redundant ESPN calls (pre-warms ~15 dates in ~3s). `_calc_workers()` auto-scales thread pools. Outputs `GAMES` dict consumed by the runner.
- **`nba_fetcher.py`** — `NBAFetcher` class wrapping `nba_api`. Pulls game logs, team rankings, splits. Rate-limited at 0.6s between calls. Caches game logs 4hrs, team rankings 12hrs. `prefetch_player_logs()` pre-loads all unique players before analysis. Extracts PLUS_MINUS, PF from cached DataFrames for v4 scouts. NEW: `get_usage_metrics()` computes dynamic usage rate from cached game logs. `get_without_stats()` computes WITH/WITHOUT teammate splits. `get_team_rankings()` now returns `league_avg` for rate-based defense.
- **`parse_board.py`** — Parses raw sportsbook text into structured JSON. Three formats: (1) ParlayPlay web copy-paste (`parse_board_web` — detects `athlete or team avatar` markers), (2) tab-delimited TSV, (3) ParlayPlay multi-line. Auto-detects format. Deduplicates by picking line closest to 1.87x multiplier.
- **`self_heal.py`** — Post-grading bias detection: stat bias, matchup bias, combo penalty, streak false positives. Generates `corrections.json`.
- **`grade_results.py`** — Compares predictions to actual box scores via nba_api. Usage: `python3 grade_results.py YYYY-MM-DD`.
- **`sgo_client.py`** — Sports Game Odds API client. Fetches real sportsbook lines (FanDuel, DraftKings, BetMGM, Caesars, ESPN BET). Caches daily props to `predictions/cache/sgo/`. `fetch_all_historical()` fetches 3,188 completed events with spreads/totals/scores. Usage: `python3 sgo_client.py --cache` (daily props), `--history` (historical events), `--probe` (metadata), `--test-completed YYYY-MM-DD`.
- **`backfill_sgo_box_scores.py`** — SGO box score backfill. Reconstructs 4.5M prop records from 342K SGO box scores (764 players, 462 dates). Pre-sampled to 200K with recency+tier weighting. Cross-source diversity vs nba_api backfill. Has real plus_minus, spreads (97.4%), matchup adjustments (87.1%). Output: `predictions/cache/sgo_backfill_training_data.json`. Usage: `python3 predictions/backfill_sgo_box_scores.py`.
- **`grade_shadow_parlays.py`** — Grades 22 shadow parlays (strategy backtesting lab) against actuals. Updates cumulative tracker/leaderboard at `predictions/shadow_parlay_tracker.json`. Usage: `python3 grade_shadow_parlays.py YYYY-MM-DD`.
- **`backfill_training_data.py`** — Reconstructs ~164K real labeled prop records from 237 cached nba_api game logs. Rolling stats from prior games only (no data leakage). Computes real context features: matchup, travel, usage, plus_minus. Output: `predictions/cache/backfill_training_data.json`. Sampled to 30K in XGBoost training (weighted: recent + higher tiers preferred).

### Data Sources

- **Live:** `nba_api` (stats.nba.com) for game logs, team rankings, box scores
- **SGO API:** `sgo_client.py` — Sports Game Odds API (real sportsbook lines from FanDuel/DraftKings/BetMGM/Caesars/ESPN BET). Player props only available pre-game; cached daily to `predictions/cache/sgo/`. API key expires March 19, 2026. Historical events (3,188 games with spreads/totals) cached to `predictions/cache/sgo/historical_events.json`.
- **Static:** `NBA Database (1947 - Present)/PlayerStatistics.csv` (historical fallback), `predictions/venue_data.py` (arena altitude/timezone)
- **Generated:** `predictions/injury_impacts.json` (88 WITH/WITHOUT pairs), `predictions/team_rankings.json` (cached), `predictions/corrections.json` (self-heal output), `predictions/cache/backfill_training_data.json` (164K reconstructed props from nba_api), `predictions/cache/sgo_backfill_training_data.json` (200K reconstructed props from SGO box scores), `predictions/cache/sgo/historical_events.json` (3,188 events with spreads/totals)
- **Daily outputs:** `predictions/YYYY-MM-DD/` folders with board, research, predictions, parlays, grading

### Key Constants in `analyze_v3.py`

- Tier thresholds: S(4+), A(3-4), B(2-3), C(1.5-2), D(1-1.5), F(0-1)
- Combo stats (`pra`, `pr`, `pa`, `ra`) get 0.5 gap penalty before tier grading
- UNDER penalty removed in v5 (UNDERs keep earned tier)
- Defense adjustment divisor: 75 (~20% max impact)
- Foul trouble adjustment (NEW v4): L5 PF avg >= 4.0 + OVER = -5% projection
- **v13 Market calibration:** Dynamic line blend (15-40% weight based on hit rate), hit rate direction adjustment (HR<50% → projection reduced, HR>70% → boosted), 2.5% systematic UNDER correction, thin gap (0-1.0) UNDER flip when HR<60%

### NEXUS v4 Soft Screen (`parlay_nexus.py`)

**Replaces binary hard screen with tiered penalty system:**
- **CORE (1.0x):** Passes all filters
- **FLEX (0.85x):** Fails exactly 1 soft filter
- **REACH (0.70x):** Fails exactly 2 soft filters
- **KILL:** Fails 3+ soft filters OR any hard kill

**Hard kills (always reject):** Tier D/F, injury OUT/Doubtful, mins < 40%, L10 HR < 40%
**Soft filters (1-2 OK):** mins 50-59%, L10 HR 55-59%, L5 HR 30-39%, gap 1.0-1.49, miss_count=3, GTD/Questionable

### NEXUS v4 Agent Architecture (`parlay_nexus.py`)

| Tier | Count | Agents |
|------|-------|--------|
| Scouts | 3 | efficiency (PLUS_MINUS, PF), venue (altitude, timezone), context (rest, clinch) |
| Evaluators | 4 | statistician (pure numbers), matchup_hunter (context), floor_master (worst-case), momentum (trends) |
| Constructors | 11 | nexus_score, hit_rate, floor_safety, game_spread, stat_diversity, under_heavy, anti_blowout, streak_aligned, home_focused, matchup_exploit, xgb_prob |
| Devils | 15 | blowout, fatigue, floor_test, combo_killer, minutes_risk, injury_cascade, opponent_history, thin_margin, correlation_leak, recent_miss, trend_reversal, gtd_cascade, usage_conflict, line_trap, consensus |
| Judges | 5 | safety (merged conservative+risk+floor), edge (merged aggressive), context (merged matchup+momentum), historical, consensus |
| **Total** | **38** | |

**Guaranteed output cascade:** Full v4 → Relaxed v4 (no evaluators/devils) → Survival build (top 3 by profile_score)

### XGBoost ML Layer (`xgb_model.py`)

XGBoost v5 binary classifier: 5-source training (graded 25x + backfill 10x + sgo_backfill 8x + 10yr 1x + legacy 1x). ~3.4K graded + ~30K backfill (nba_api) + ~50K sgo_backfill (SGO box scores) + ~50K 10yr + ~15K legacy = ~148K total. 136 features (84 base + 8 v7 enrichment + 12 v8 ref/coach/sim + 32 v9 nn_embeddings). v8: referee crew tendencies, coach profiles, Monte Carlo sim_prob. v9: 32-dim learned embeddings from deep NN (256→128→64→32). Meta-learner replaces hardcoded 60/40 ensemble blend. Walk-forward CV: pooled AUC=0.605, top-decile=65.3%. Top features: miss_streak, game_total_signal, l10_avg_pf, directional_gap, is_b2b, travel_distance.

```bash
python3 predictions/backfill_training_data.py   # Generate backfill data from cached game logs (~164K records)
python3 predictions/xgb_model.py train          # Train XGBoost on all data (graded + backfill + historical)
python3 predictions/xgb_model.py eval           # Walk-forward CV only
python3 predictions/xgb_model.py score <file>   # Score a board/results file
python3 predictions/xgb_model.py importance     # Feature importance report
python3 predictions/mlp_model.py train          # Train MLP neural network (same features as XGBoost)
python3 predictions/mlp_model.py eval           # MLP walk-forward CV
```

### MLP Neural Network (`mlp_model.py`)

scikit-learn `MLPClassifier(128, 64)` ensemble partner for XGBoost. Same 136 features via shared `engineer_features()`. StandardScaler + median imputation (MLPs can't handle NaN). Walk-forward CV matches XGBoost folds. Ensemble: `ensemble_prob = 0.6 * xgb_prob + 0.4 * mlp_prob`. Shadow strategies `mlp_top` and `ensemble_top` compete on leaderboard. Daily pipeline retrains both models.

### Backfill Training Data (`backfill_training_data.py`)

Reconstructs ~164K real labeled prop records from 237 cached nba_api game logs. For each player-game (starting from game 11): computes rolling stats from prior games only (no data leakage), generates realistic prop lines from L10 avg, labels from actual box scores. Computes REAL context features: home/away splits, matchup adjustments (team rankings), travel distance (venue_data.py), usage rate, plus/minus, PF. Output: `predictions/cache/backfill_training_data.json`. Sampled to 30K in training (weighted: recent dates + higher tiers preferred).

### Daily Pipeline (`daily_pipeline.py`)

One-command daily workflow: grade yesterday (primary + shadow parlays) → retrain XGBoost + MLP → run today's board. Usage: `python3 predictions/daily_pipeline.py` (auto-detects dates) or `--grade-only`, `--retrain-only`, `--run` for individual steps.

### Primary Parlay Grader (`grade_primary_parlays.py`)

Grades the actual SAFE + AGGRESSIVE parlays against box scores and maintains cumulative W/L tracker at `predictions/primary_parlay_tracker.json`. Usage: `python3 predictions/grade_primary_parlays.py YYYY-MM-DD`.

### Parlay Engine v1 (`parlay_engine.py`)

**Backup parlay builder (NEXUS v4 is now primary).** Data-driven scoring from 2,328-prop analysis. Half-Kelly bet sizing based on calibrated xgb_prob. Hardened filters (v14): GTD -0.15 penalty, ensemble_prob >= 0.50 SAFE floor, D/F blocked from AGG, consensus tracking on shadows.
- UNDER bias: 68.1% vs OVER 47.1%
- COLD+UNDER: 75.3%, BLK/STL UNDER: 73.3%, HOT streak trap: 49.2%

**Primary outputs:**
- SAFE 3-leg: tier S/A/B, mins>=60%, L10 HR>=60%, no combos, sorted by composite score
- AGGRESSIVE 8-leg: UNDER-heavy (5+ UNDERs), broader filters, sorted by composite score

**Composite scoring (v11):** Base: `xgb_prob + dir_bonus(0.30 UNDER) + streak_adj(-0.08 HOT / +0.12 COLD+UNDER) + gap_bonus(capped 0.10) + stat_bonus(0.05 BLK/STL UNDER) + combo_pen(-0.10) + hr_bonus(0.10) + mins_bonus(0.05) + blowout_pen(spread≥12: -0.02 to -0.10) + miss_pen(-0.05 if miss≥5)` + 8 new factor groups: **(1) Opponent Intelligence** — opp_matchup_delta (±0.08 via `_stat_scale`), team_vs_opp_delta (±0.03), opp_off_pressure (±0.03). **(2) Usage & Role** — usage_rate (0.03 high+OVER / 0.02 low+UNDER), usage_trend (±0.03), dynamic_without_delta (0.04), usage_boost (capped 0.04). **(3) Efficiency** — plus_minus (±0.02), efficiency_trend (0.02). **(4) Defense Signal** — opp_stat_allowed_vs_league_avg (0.03). **(5) Foul Trouble** — l10_avg_pf ≥4.0 (±0.03). **(6) Travel** — travel_miles_7day >5000mi (-0.03 OVER), tz_shifts ≥2 (-0.02). **(7) Game Total** — high/low total ±0.03.

**Shadow system:** 100 strategies (30 curated + 70 grid-generated) via parametric param dicts. One universal `build_parlay_from_params()` interprets any strategy. Player trio uniqueness enforced (no two shadows share same 3 players). 3-level fallback cascade guarantees zero nulls.

### Shadow Parlay Backtesting Lab

~122 independent 3-leg parlays built per day: 100 from Parlay Engine v1 (30 curated + 70 grid, with consensus tracking) + ~22 from NEXUS v4. Saved to `predictions/YYYY-MM-DD/shadow_parlays.json`. Graded daily to build a cumulative leaderboard.

- **Engine strategies 1-30:** Curated data-driven (under_cold_*, under_gap2_*, under_blkstl_*, under_sab_*, xgb_top_*, anti_hot_*, old_pipeline, hybrid, floor_first, xgb_only, under_pure)
- **Engine strategies 31-100:** Grid-generated via deterministic parameter sampling (direction x sort x tier x stat x streak)
- **NEXUS strategies:** 22 existing (11 constructors + 11 shadow-only)
- **Tracker:** `predictions/shadow_parlay_tracker.json` — cumulative W/L by strategy with leaderboard
- **Grade:** `python3 predictions/grade_shadow_parlays.py YYYY-MM-DD`
