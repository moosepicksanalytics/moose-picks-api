"""
Backfill historical odds data for games in the database.

Note: The Odds API typically only provides current/future odds, not historical odds.
This script will attempt to backfill, but many historical games may not have odds available.

The 100% accuracy issue is NOT caused by missing odds - it's caused by data leakage
(features that encode the target variable). We've already fixed that in the code.
Missing odds would actually LOWER accuracy, not raise it to 100%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from app.database import SessionLocal
from app.models.db_models import Game
from app.odds_api.client import fetch_and_update_game_odds
from sqlalchemy import func, or_


def get_games_missing_odds(sport: str, start_date: str = None, end_date: str = None) -> list:
    """
    Get games that are missing odds data.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        List of Game objects missing odds
    """
    db = SessionLocal()
    try:
        query = db.query(Game).filter(
            Game.sport == sport,
            Game.home_score.isnot(None),  # Only games with final scores
            Game.away_score.isnot(None)
        )
        
        # Filter by date range if provided
        if start_date:
            query = query.filter(func.date(Game.date) >= start_date)
        if end_date:
            query = query.filter(func.date(Game.date) <= end_date)
        
        # Find games missing odds (missing spread OR moneyline OR over_under)
        missing_odds = query.filter(
            or_(
                Game.spread.is_(None),
                Game.home_moneyline.is_(None),
                Game.over_under.is_(None)
            )
        ).all()
        
        return missing_odds
    finally:
        db.close()


def backfill_odds_for_date_range(
    sport: str,
    start_date: str,
    end_date: str,
    dry_run: bool = False
) -> dict:
    """
    Attempt to backfill odds for games in a date range.
    
    Note: The Odds API typically doesn't provide historical odds, so this may
    not find much data. This is mainly for documentation/debugging purposes.
    
    Args:
        sport: Sport code
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        dry_run: If True, only report what would be done
    
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*60}")
    print(f"Backfilling odds for {sport} from {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Get games missing odds
    missing_games = get_games_missing_odds(sport, start_date, end_date)
    print(f"Found {len(missing_games)} games missing odds data")
    
    if len(missing_games) == 0:
        print("✓ All games already have odds data")
        return {
            "sport": sport,
            "total_games": 0,
            "missing_odds": 0,
            "attempted": 0,
            "updated": 0
        }
    
    # Group by date
    games_by_date = {}
    for game in missing_games:
        game_date = game.date.date() if hasattr(game.date, 'date') else game.date
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, "%Y-%m-%d").date()
        
        if game_date not in games_by_date:
            games_by_date[game_date] = []
        games_by_date[game_date].append(game)
    
    print(f"Games missing odds by date:")
    for date, games in sorted(games_by_date.items()):
        print(f"  {date}: {len(games)} games")
    
    if dry_run:
        print("\n[DRY RUN] Would attempt to fetch odds for these dates")
        return {
            "sport": sport,
            "total_games": len(missing_games),
            "missing_odds": len(missing_games),
            "attempted": 0,
            "updated": 0
        }
    
    # Attempt to fetch odds for each date
    # Note: The Odds API typically only has current/future odds, not historical
    updated_count = 0
    attempted_dates = set()
    
    for date, games in sorted(games_by_date.items()):
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
        
        # Skip if we've already tried this date
        if date_str in attempted_dates:
            continue
        
        # Skip future dates (odds API won't have them)
        if date > datetime.now().date():
            print(f"  Skipping future date: {date_str}")
            continue
        
        # Skip dates too far in the past (odds API won't have them)
        days_ago = (datetime.now().date() - date).days
        if days_ago > 7:
            print(f"  Skipping old date (>{days_ago} days ago): {date_str}")
            continue
        
        print(f"\n  Attempting to fetch odds for {date_str}...")
        attempted_dates.add(date_str)
        
        try:
            updated = fetch_and_update_game_odds(sport, date_str)
            updated_count += updated
            print(f"    ✓ Updated {updated} games")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total games missing odds: {len(missing_games)}")
    print(f"  Dates attempted: {len(attempted_dates)}")
    print(f"  Games updated: {updated_count}")
    print(f"{'='*60}\n")
    
    return {
        "sport": sport,
        "total_games": len(missing_games),
        "missing_odds": len(missing_games),
        "attempted": len(attempted_dates),
        "updated": updated_count
    }


def backfill_all_sports(
    start_date: str = None,
    end_date: str = None,
    dry_run: bool = False
):
    """
    Backfill odds for all sports.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        dry_run: If True, only report what would be done
    """
    sports = ["NFL", "NHL", "NBA", "MLB"]
    results = []
    
    for sport in sports:
        result = backfill_odds_for_date_range(sport, start_date, end_date, dry_run)
        results.append(result)
    
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result['sport']}: {result['missing_odds']} missing, {result['updated']} updated")
    print("="*60)
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. The Odds API typically only provides current/future odds, not historical odds")
    print("2. Missing odds data does NOT cause 100% accuracy - that's from data leakage")
    print("3. We've already fixed the data leakage issues in the code")
    print("4. Missing odds would actually LOWER accuracy, not raise it")
    print("5. This backfill is mainly for documentation - most historical games won't have odds")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill historical odds data")
    parser.add_argument("--sport", choices=["NFL", "NHL", "NBA", "MLB"], help="Sport to backfill")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't fetch")
    
    args = parser.parse_args()
    
    if args.sport:
        backfill_odds_for_date_range(
            args.sport,
            args.start_date,
            args.end_date,
            args.dry_run
        )
    else:
        backfill_all_sports(
            args.start_date,
            args.end_date,
            args.dry_run
        )
