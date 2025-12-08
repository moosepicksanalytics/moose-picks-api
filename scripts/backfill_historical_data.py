"""
Backfill historical game data for training models.
Fetches past 5 seasons of data for NFL, NHL, NBA, MLB.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.espn_client.fetcher import fetch_games_for_date
from app.espn_client.parser import parse_and_store_games
from app.data_loader import get_season_date_range


def get_season_dates(sport: str, season_year: int):
    """Get start and end dates for a season."""
    return get_season_date_range(sport, season_year)


def backfill_season(sport: str, season_year: int, delay: float = 0.1):
    """
    Backfill all games for a single season.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        season_year: Year the season starts
        delay: Delay between API calls (seconds)
    
    Returns:
        Number of games stored
    """
    start_date, end_date = get_season_dates(sport, season_year)
    
    print(f"\n{'='*60}")
    print(f"Backfilling {sport} {season_year} Season")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    total_games = 0
    current_date = start
    days_processed = 0
    
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        
        try:
            # Fetch games for this date
            games = fetch_games_for_date(sport, date_str)
            
            if games:
                # Parse and store games (only stores games with final scores for training)
                stored = parse_and_store_games(sport, games, only_final=True)
                total_games += stored
                
                if stored > 0:
                    print(f"  {date_str}: Stored {stored} games")
            
            # Small delay to avoid rate limiting
            time.sleep(delay)
            
        except Exception as e:
            print(f"  {date_str}: Error - {e}")
        
        current_date += timedelta(days=1)
        days_processed += 1
        
        # Progress update every 30 days
        if days_processed % 30 == 0:
            print(f"  Progress: {days_processed} days processed, {total_games} games stored...")
    
    print(f"\n✓ {sport} {season_year}: {total_games} games stored")
    return total_games


def backfill_sport(sport: str, seasons: list, delay: float = 0.1):
    """
    Backfill multiple seasons for a sport.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        seasons: List of season years
        delay: Delay between API calls (seconds)
    
    Returns:
        Total number of games stored
    """
    print(f"\n{'='*70}")
    print(f"Backfilling {sport} - {len(seasons)} seasons")
    print(f"{'='*70}")
    
    total_games = 0
    for season in seasons:
        try:
            games = backfill_season(sport, season, delay=delay)
            total_games += games
        except Exception as e:
            print(f"✗ Error backfilling {sport} {season}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ {sport} Total: {total_games} games stored")
    return total_games


def backfill_all_sports(seasons: list = None, delay: float = 0.1):
    """
    Backfill historical data for all 4 sports.
    
    Args:
        seasons: List of season years (default: past 5 seasons)
        delay: Delay between API calls (seconds)
    """
    if seasons is None:
        # Default: past 5 seasons (2020-2024)
        current_year = datetime.now().year
        seasons = list(range(current_year - 4, current_year + 1))
    
    sports = ["NFL", "NHL", "NBA", "MLB"]
    
    print("=" * 70)
    print("Moose Picks ML - Historical Data Backfill")
    print("=" * 70)
    print(f"Seasons: {', '.join(map(str, seasons))}")
    print(f"Sports: {', '.join(sports)}")
    print(f"Delay between API calls: {delay}s")
    print("=" * 70)
    print("\n⚠️  This will take a while! Fetching games day-by-day...")
    print("   Progress will be shown as games are stored.\n")
    
    total_all = 0
    results = {}
    
    for sport in sports:
        try:
            games = backfill_sport(sport, seasons, delay=delay)
            results[sport] = games
            total_all += games
        except Exception as e:
            print(f"✗ Error backfilling {sport}: {e}")
            results[sport] = 0
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("Backfill Summary")
    print("=" * 70)
    for sport in sports:
        print(f"{sport}: {results.get(sport, 0)} games")
    print(f"\nTotal: {total_all} games stored")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical game data for training models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill all 4 sports, past 5 seasons (default)
  python scripts/backfill_historical_data.py
  
  # Backfill specific sports
  python scripts/backfill_historical_data.py --sports NFL NHL
  
  # Backfill specific seasons
  python scripts/backfill_historical_data.py --seasons 2022 2023 2024
  
  # Backfill with faster API calls (lower delay)
  python scripts/backfill_historical_data.py --delay 0.05
  
  # Backfill single sport
  python scripts/backfill_historical_data.py --sports NHL --seasons 2023 2024
        """
    )
    
    parser.add_argument(
        "--sports",
        type=str,
        nargs="+",
        default=["NFL", "NHL", "NBA", "MLB"],
        help="Sports to backfill (default: all 4)"
    )
    
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Season years to backfill (default: past 5 seasons: 2020-2024)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    if len(args.sports) == 1:
        # Single sport
        sport = args.sports[0]
        seasons = args.seasons or list(range(datetime.now().year - 4, datetime.now().year + 1))
        backfill_sport(sport, seasons, delay=args.delay)
    elif len(args.sports) == 4 and args.sports == ["NFL", "NHL", "NBA", "MLB"]:
        # All sports
        backfill_all_sports(seasons=args.seasons, delay=args.delay)
    else:
        # Multiple specific sports
        seasons = args.seasons or list(range(datetime.now().year - 4, datetime.now().year + 1))
        total = 0
        for sport in args.sports:
            games = backfill_sport(sport, seasons, delay=args.delay)
            total += games
        print(f"\n✓ Total across all sports: {total} games")
