#!/usr/bin/env python3
"""
Daily Pipeline — Grade yesterday + retrain XGBoost + run today's board.

One command to do everything:
    python3 predictions/daily_pipeline.py                     # Auto-detect dates
    python3 predictions/daily_pipeline.py --grade-only 2026-03-15
    python3 predictions/daily_pipeline.py --retrain-only
    python3 predictions/daily_pipeline.py --run 2026-03-16 /path/to/board.json

Steps:
1. Grade yesterday's predictions (primary + shadow parlays)
2. Retrain XGBoost with new graded data
3. Run today's board (if provided)
"""
import os
import sys
import subprocess
from datetime import datetime, timedelta

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, desc):
    """Run a command and print status."""
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] Command exited with code {result.returncode}")
    return result.returncode


def grade_yesterday(yesterday_str):
    """Grade yesterday's results."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: GRADE {yesterday_str}")
    print(f"{'='*60}")

    # Grade full board predictions
    grade_file = os.path.join(PREDICTIONS_DIR, yesterday_str, f'{yesterday_str}_full_board.json')
    if os.path.exists(grade_file):
        run_cmd(
            f"python3 {PREDICTIONS_DIR}/grade_results.py {yesterday_str}",
            f"Grading predictions for {yesterday_str}"
        )
    else:
        print(f"  No full board found for {yesterday_str}, skipping prediction grading")

    # Grade primary parlays
    primary_file = os.path.join(PREDICTIONS_DIR, yesterday_str, 'primary_parlays.json')
    if os.path.exists(primary_file):
        run_cmd(
            f"python3 {PREDICTIONS_DIR}/grade_primary_parlays.py {yesterday_str}",
            f"Grading primary parlays for {yesterday_str}"
        )
    else:
        print(f"  No primary parlays found for {yesterday_str}")

    # Grade shadow parlays
    shadow_file = os.path.join(PREDICTIONS_DIR, yesterday_str, 'shadow_parlays.json')
    if os.path.exists(shadow_file):
        run_cmd(
            f"python3 {PREDICTIONS_DIR}/grade_shadow_parlays.py {yesterday_str}",
            f"Grading shadow parlays for {yesterday_str}"
        )
    else:
        print(f"  No shadow parlays found for {yesterday_str}")


def retrain_model():
    """Retrain XGBoost + MLP with all available graded data."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: RETRAIN ML MODELS")
    print(f"{'='*60}")

    run_cmd(
        f"python3 {PREDICTIONS_DIR}/xgb_model.py train",
        "Retraining XGBoost on graded + backfill + historical data"
    )

    run_cmd(
        f"python3 {PREDICTIONS_DIR}/mlp_model.py train",
        "Retraining MLP neural network on graded + backfill + historical data"
    )


def run_board(date_str, board_path):
    """Run the full pipeline for today's board."""
    print(f"\n{'='*60}")
    print(f"  STEP 3: RUN PIPELINE FOR {date_str}")
    print(f"{'='*60}")

    # Train focused model on today's real sportsbook lines (AUC 0.589 vs 0.554 baseline)
    focused_script = os.path.join(PREDICTIONS_DIR, 'train_current_lines.py')
    if os.path.exists(focused_script):
        run_cmd(
            f"python3 {focused_script} --board {board_path}",
            f"Training focused XGBoost on today's board lines"
        )

    run_cmd(
        f"python3 {PREDICTIONS_DIR}/run_board_v5.py {date_str} {board_path}",
        f"Running full pipeline for {date_str}"
    )


def main():
    args = sys.argv[1:]

    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    yesterday_str = (today - timedelta(days=1)).strftime('%Y-%m-%d')

    if '--grade-only' in args:
        idx = args.index('--grade-only')
        date = args[idx + 1] if idx + 1 < len(args) else yesterday_str
        grade_yesterday(date)
        return

    if '--retrain-only' in args:
        retrain_model()
        return

    if '--run' in args:
        idx = args.index('--run')
        date = args[idx + 1] if idx + 1 < len(args) else today_str
        board = args[idx + 2] if idx + 2 < len(args) else None
        if not board:
            print("Usage: daily_pipeline.py --run YYYY-MM-DD /path/to/board.json")
            sys.exit(1)
        run_board(date, board)
        return

    # Full daily pipeline: grade yesterday → retrain → optionally run today
    print(f"{'='*60}")
    print(f"  DAILY PIPELINE")
    print(f"  Yesterday: {yesterday_str}")
    print(f"  Today:     {today_str}")
    print(f"{'='*60}")

    # Step 1: Grade yesterday
    grade_yesterday(yesterday_str)

    # Step 2: Retrain
    retrain_model()

    # Step 3: Run today if board exists
    board_candidates = [
        f'/tmp/parsed_board.json',
        os.path.join(PREDICTIONS_DIR, today_str, f'{today_str}_parsed_board.json'),
    ]
    board_path = None
    for c in board_candidates:
        if os.path.exists(c):
            board_path = c
            break

    if board_path:
        run_board(today_str, board_path)
    else:
        print(f"\n  No board found for {today_str}. Upload board and run:")
        print(f"    python3 predictions/run_board_v5.py {today_str} /path/to/board.json")

    print(f"\n{'='*60}")
    print(f"  DAILY PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
