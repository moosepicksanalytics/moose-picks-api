"""
Daily automation script for training models and generating predictions.
Can be run manually or scheduled via cron/task scheduler.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_all import train_all_models
from scripts.export_predictions import export_predictions_for_date
from app.espn_client.fetcher import fetch_games_for_date
from app.espn_client.parser import parse_and_store_games
from app.prediction.settling import settle_predictions
from app.odds_api.client import fetch_and_update_game_odds


def fetch_todays_games(sports: list = ["NFL", "NHL", "NBA", "MLB"]):
    """Fetch and store today's games from ESPN."""
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Fetching Today's Games ({date_str})")
    print(f"{'='*60}\n")
    
    total_games = 0
    for sport in sports:
        print(f"Fetching {sport} games...")
        try:
            games = fetch_games_for_date(sport, date_str)
            if games:
                print(f"  Found {len(games)} {sport} games from ESPN")
                stored = parse_and_store_games(sport, games, only_final=False)  # Store all games (including scheduled)
                total_games += stored
                if stored > 0:
                    print(f"  ✓ Stored {stored} {sport} games in database")
                else:
                    print(f"  ⚠️  Failed to store {sport} games (check logs for errors)")
            else:
                print(f"  No {sport} games found for today from ESPN")
        except Exception as e:
            print(f"  ✗ Error fetching {sport} games: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Total games fetched and stored: {total_games}")
    return total_games


def daily_workflow(
    sports: list = ["NFL", "NHL", "NBA", "MLB"],
    train: bool = True,
    predict: bool = True,
    predict_date: str = None,
    min_edge: float = 0.05,
    config_path: str = "config.yaml"
):
    """
    Complete daily workflow: fetch games, train models, generate predictions.
    
    Args:
        sports: List of sports to process
        train: Whether to retrain models
        predict: Whether to generate predictions
        predict_date: Date to predict for (default: today)
        min_edge: Minimum edge threshold for predictions
        config_path: Path to config.yaml
    """
    print("=" * 70)
    print("Moose Picks ML - Daily Automation")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sports: {', '.join(sports)}")
    print("=" * 70)
    
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    # Step 1: Settle yesterday's predictions
    print("\n[1/5] Settling yesterday's predictions...")
    try:
        for sport in sports:
            print(f"Settling {sport} predictions for {yesterday_str}...")
            settle_predictions(yesterday_str, sport=sport)
    except Exception as e:
        print(f"✗ Error settling predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 2: Fetch today's games
    print("\n[2/5] Fetching today's games...")
    fetch_todays_games(sports)
    
    # Step 3: Fetch and update odds from The Odds API
    print("\n[3/5] Fetching odds from The Odds API...")
    try:
        for sport in sports:
            print(f"Fetching {sport} odds...")
            updated = fetch_and_update_game_odds(sport, today_str)
            if updated > 0:
                print(f"  ✓ Updated odds for {updated} {sport} games")
            else:
                print(f"  No {sport} games found to update")
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch odds: {e}")
        print("  Continuing without odds update...")
        import traceback
        traceback.print_exc()
    
    # Step 4: Train models (if requested)
    if train:
        print("\n[4/5] Training models...")
        try:
            train_all_models(config_path)
            print("✓ Model training completed")
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[4/5] Skipping model training (--no-train flag)")
    
    # Step 5: Generate predictions (if requested)
    if predict:
        print("\n[5/5] Generating predictions...")
        if predict_date is None:
            predict_date = today_str
        
        # Since games are stored in UTC, also check tomorrow in case games are scheduled
        # for "tomorrow" in UTC but "today" in local timezone
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_check = [predict_date]
        if predict_date == today_str:
            dates_to_check.append(tomorrow)  # Also check tomorrow for UTC games
        
        for sport in sports:
            try:
                for check_date in dates_to_check:
                    try:
                        print(f"\nGenerating predictions for {sport} on {check_date}...")
                        export_predictions_for_date(
                            sport=sport,
                            date_str=check_date,
                            config_path=config_path,
                            output_dir="exports",
                            min_edge=min_edge
                        )
                    except Exception as e:
                        print(f"✗ Prediction generation failed for {sport} on {check_date}: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"✗ Prediction generation failed for {sport}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n[5/5] Skipping prediction generation (--no-predict flag)")
    
    print("\n" + "=" * 70)
    print("Daily workflow complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily automation for training and predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (fetch, train, predict)
  python scripts/daily_automation.py
  
  # Only generate predictions (no training)
  python scripts/daily_automation.py --no-train
  
  # Only train models (no predictions)
  python scripts/daily_automation.py --no-predict
  
  # Predict for a specific date
  python scripts/daily_automation.py --predict-date 2024-12-15
  
  # Only NFL
  python scripts/daily_automation.py --sports NFL
        """
    )
    
    parser.add_argument(
        "--sports",
        type=str,
        nargs="+",
        default=["NFL", "NHL", "NBA", "MLB"],
        help="Sports to process (default: NFL NHL NBA MLB)"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Skip prediction generation"
    )
    parser.add_argument(
        "--predict-date",
        type=str,
        help="Date to generate predictions for (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge threshold for predictions (default: 0.05)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    daily_workflow(
        sports=args.sports,
        train=not args.no_train,
        predict=not args.no_predict,
        predict_date=args.predict_date,
        min_edge=args.min_edge,
        config_path=args.config
    )
