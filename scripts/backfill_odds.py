"""
Backfill historical odds data for games in the database using The Odds API historical endpoint.

⚠️  IMPORTANT: Historical odds cost 10x more than current odds (10 credits per region per market).
For 3 markets (h2h, spreads, totals) with 1 region, each date costs 30 credits.
With 20k credits/month, you can backfill ~666 dates (about 2 years of daily data).

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
from app.odds_api.client import fetch_and_update_game_odds_historical, fetch_and_update_game_odds
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
        # Parse string dates to date objects for proper comparison
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            query = query.filter(func.date(Game.date) >= start_date_obj)
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            query = query.filter(func.date(Game.date) <= end_date_obj)
        
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
    dry_run: bool = False,
    use_historical: bool = True
) -> dict:
    """
    Backfill odds for games in a date range using The Odds API historical endpoint.
    
    ⚠️  COST WARNING: Historical odds cost 10 credits per region per market.
    For 3 markets (h2h, spreads, totals) with 1 region = 30 credits per date.
    
    Args:
        sport: Sport code
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        dry_run: If True, only report what would be done
        use_historical: If True, use historical endpoint (costs 10x more but works for past dates)
    
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*60}")
    print(f"Backfilling odds for {sport} from {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    if use_historical:
        print("⚠️  Using HISTORICAL odds endpoint (costs 10x more: 30 credits per date)")
        print("   Each date with 3 markets (h2h, spreads, totals) = 30 credits")
        print("   With 20k credits/month, you can backfill ~666 dates\n")
    
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
        
        # For historical endpoint, we can go back to June 6, 2020
        # For current endpoint, skip dates too far in the past
        if not use_historical:
            days_ago = (datetime.now().date() - date).days
            if days_ago > 7:
                print(f"  Skipping old date (>{days_ago} days ago): {date_str}")
                print(f"    Use --use-historical flag to backfill older dates")
                continue
        else:
            # Historical endpoint available from June 6, 2020
            min_date = datetime(2020, 6, 6).date()
            if date < min_date:
                print(f"  Skipping date before historical data available: {date_str}")
                print(f"    Historical odds available from {min_date}")
                continue
        
        print(f"\n  Attempting to fetch odds for {date_str}...")
        attempted_dates.add(date_str)
        
        try:
            if use_historical:
                updated = fetch_and_update_game_odds_historical(sport, date_str)
            else:
                from app.odds_api.client import fetch_and_update_game_odds
                updated = fetch_and_update_game_odds(sport, date_str)
            updated_count += updated
            print(f"    ✓ Updated {updated} games")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
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
    dry_run: bool = False,
    use_historical: bool = True
):
    """
    Backfill odds for all sports.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        dry_run: If True, only report what would be done
        use_historical: If True, use historical endpoint (costs 10x more)
    """
    sports = ["NFL", "NHL", "NBA", "MLB"]
    results = []
    
    for sport in sports:
        result = backfill_odds_for_date_range(sport, start_date, end_date, dry_run, use_historical)
        results.append(result)
    
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result['sport']}: {result['missing_odds']} missing, {result['updated']} updated")
    print("="*60)
    
    if use_historical:
        total_dates = len(set([r.get('attempted', 0) for r in results]))
        estimated_cost = total_dates * 30  # 3 markets * 1 region * 10 credits
        print(f"\n💰 Estimated API cost: ~{estimated_cost} credits ({total_dates} dates × 30 credits)")
        print(f"   Remaining credits after backfill: Check API response headers")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. Historical odds are available from June 6, 2020 (snapshots at 5-10 min intervals)")
    print("2. Historical odds cost 10x more than current odds (30 credits per date)")
    print("3. Missing odds data does NOT cause 100% accuracy - that's from data leakage")
    print("4. We've already fixed the data leakage issues in the code")
    print("5. Historical odds require a paid API plan")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Backfill historical odds data using The Odds API historical endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be backfilled
  python scripts/backfill_odds.py --sport NFL --start-date 2024-01-01 --end-date 2024-01-31 --dry-run
  
  # Backfill last 30 days for NFL (costs ~900 credits)
  python scripts/backfill_odds.py --sport NFL --start-date 2024-11-01 --end-date 2024-11-30
  
  # Backfill all sports for a date range
  python scripts/backfill_odds.py --start-date 2024-10-01 --end-date 2024-10-31

Cost: Historical odds cost 30 credits per date (3 markets × 1 region × 10 credits).
      With 20k credits/month, you can backfill ~666 dates.
        """
    )
    parser.add_argument("--sport", choices=["NFL", "NHL", "NBA", "MLB"], help="Sport to backfill")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't fetch")
    parser.add_argument("--use-historical", action="store_true", default=True, 
                       help="Use historical endpoint (default: True, costs 10x more)")
    parser.add_argument("--no-historical", dest="use_historical", action="store_false",
                       help="Use current odds endpoint (only works for recent dates)")
    
    args = parser.parse_args()
    
    if args.sport:
        backfill_odds_for_date_range(
            args.sport,
            args.start_date,
            args.end_date,
            args.dry_run,
            args.use_historical
        )
    else:
        backfill_all_sports(
            args.start_date,
            args.end_date,
            args.dry_run,
            args.use_historical
        )
