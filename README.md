# NBA Prop Betting Pipeline

AI-powered NBA player prop prediction system. Parses sportsbook boards, fetches live data, runs a 15-layer analysis pipeline, and builds parlays via multi-agent consensus.

## Quick Start

```bash
# Parse a sportsbook board
python3 predictions/parse_board.py < board.txt > /tmp/board.json

# Train focused model on today's lines (2 min)
python3 predictions/train_current_lines.py --board /tmp/board.json

# Run full pipeline (research + analyze + parlays)
python3 predictions/run_board_v5.py 2026-03-20 /tmp/board.json

# Grade yesterday's predictions
python3 predictions/grade_results.py 2026-03-19
```

## Architecture

```
Board Text → parse_board.py → Parsed Props (JSON)
                                    ↓
                          run_board_v5.py (orchestrator)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            game_researcher.py  nba_fetcher.py  venue_data.py
            (parallel agents)   (NBA API data)  (arena/travel)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                          analyze_v3.py (15-layer pipeline)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            correlations.py   matchup_scanner.py  game_flow.py
            (enrichment)      (defense-first)     (blowout risk)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              xgb_model.py    mlp_model.py    model_arena.py
              (XGBoost)       (MLP NN)        (10-model ensemble)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                          ensemble_prob (calibrated)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            parlay_nexus.py   parlay_engine.py  parlay_optimizer.py
            (NEXUS v4 primary) (Engine backup)  (correlation-aware)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                          Output: predictions/YYYY-MM-DD/
```

## Module Reference

### Core Pipeline
| Module | Purpose |
|--------|---------|
| `run_board_v5.py` | Main orchestrator — runs everything end-to-end |
| `parse_board.py` | Parses raw sportsbook text into structured JSON |
| `analyze_v3.py` | 15-layer per-prop analysis (v13 market-calibrated) |
| `daily_pipeline.py` | One-command: grade yesterday + retrain + run today |

### Data Sources
| Module | Purpose |
|--------|---------|
| `nba_fetcher.py` | NBA API wrapper — game logs, team rankings, box scores |
| `game_researcher.py` | Parallel research agents — injuries, spreads, B2B, news |
| `venue_data.py` | Static arena altitude/timezone/lat-lng for all 30 teams |
| `sgo_client.py` | Sports Game Odds API — multi-book sportsbook lines |
| `pbp_client.py` | PBPStats.com API — pace, WOWY, on/off splits |

### ML Models
| Module | Purpose |
|--------|---------|
| `xgb_model.py` | XGBoost classifier — 143 features, primary scorer |
| `mlp_model.py` | MLP neural network — ensemble partner for XGBoost |
| `model_arena.py` | 10-model arena — trains diverse models, keeps top 5 |
| `regression_model.py` | XGBRegressor — predicts actual stat values, not just hit/miss |
| `train_current_lines.py` | Focused training — uses today's real sportsbook lines |
| `nn_embedder.py` | Deep NN — extracts 32-dim learned embeddings |
| `meta_learner.py` | Stacked meta-learner — learned model blending |
| `sim_model.py` | Monte Carlo simulation — KDE on L10 game logs |

### Parlay Builders
| Module | Purpose |
|--------|---------|
| `parlay_nexus.py` | NEXUS v4 — 38-agent consensus builder (PRIMARY) |
| `parlay_engine.py` | Engine v1.2 — data-driven UNDER bias builder (BACKUP) |
| `parlay_optimizer.py` | Correlation-aware — selects uncorrelated legs |
| `ev_optimizer.py` | Expected value — optimizes for +EV not just accuracy |

### Enrichment & Analysis
| Module | Purpose |
|--------|---------|
| `matchup_scanner.py` | Defense-first scanner — finds top 3-5 matchup exploits |
| `correlations.py` | Teammate/opponent correlations, usage redistribution |
| `game_flow.py` | Game script modeling — blowout risk, minutes impact |
| `line_movement.py` | Multi-book line disagreement analysis |
| `ref_model.py` | Referee crew tendency features |
| `coach_model.py` | Coach rotation/pace profiling |
| `pregame_check.py` | Pre-game availability filter (OUT/postponed) |
| `self_heal.py` | Post-grading bias detection |
| `market_signal.py` | SGO fair odds + consensus enrichment |

### Grading & Tracking
| Module | Purpose |
|--------|---------|
| `grade_results.py` | Grade predictions vs actual box scores |
| `grade_primary_parlays.py` | Grade NEXUS primary + Engine backup parlays |
| `grade_shadow_parlays.py` | Grade 122 shadow strategies for leaderboard |
| `measure_features.py` | Feature importance measurement + pruning |

### Training Data
| Module | Purpose |
|--------|---------|
| `backfill_training_data.py` | Reconstructs ~164K props from cached game logs |
| `backfill_sgo_box_scores.py` | Reconstructs ~200K props from SGO box scores |
| `backfill_historical_csv.py` | Historical CSV data loader |

## Key Design Decisions

- **NEXUS v4 is primary** — 38-agent multi-agent consensus beats single-scorer Engine (75% vs 25% leg HR)
- **Defense-first selection** — Matchup scanner (85.7% backtest) starts with "which defense is weak?" not "which player is good?"
- **Calibration > accuracy** — Brier score is primary metric. Well-calibrated probabilities matter more for parlays than raw accuracy
- **Focused training on real lines** — Today's sportsbook lines applied to historical games outperforms synthetic backfill data
- **1/5th Kelly sizing** — Research shows 1/5 Kelly earned 98% ROI vs full Kelly crash
- **143 features** — Including EWMA, median (skewness signal), per-minute production rate, and 32 NN embeddings

## Requirements

```
python >= 3.9
nba_api
xgboost
scikit-learn
numpy
pandas
scipy
```

Optional (enhanced models):
```
catboost
lightgbm
```

## Output Structure

```
predictions/
├── YYYY-MM-DD/
│   ├── YYYY-MM-DD_full_board.json    # All analyzed props
│   ├── YYYY-MM-DD_game_research.json # Research per game
│   ├── primary_parlays.json          # NEXUS primary picks
│   ├── engine_parlays.json           # Engine backup picks
│   ├── nexus_parlays.json            # Raw NEXUS output
│   ├── shadow_parlays.json           # 122 shadow strategies
│   ├── matchup_scanner.json          # Defense-first exploits
│   ├── game_locks.json               # Best single pick per game
│   ├── pregame_report.json           # Availability check
│   └── summary.json                  # Pipeline metadata
├── cache/                            # Cached data (not in git)
├── primary_parlay_tracker.json       # Cumulative W/L record
└── shadow_parlay_tracker.json        # Shadow strategy leaderboard
```
