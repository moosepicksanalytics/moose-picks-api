"""
Backfill Over/Under (O/U) data for historical games.
Calculates actual_total and ou_result from existing scores and over_under values.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.database import SessionLocal
from app.models.db_models import Game
from app.utils.ou_calculator import OUCalculator
from datetime import datetime, timedelta
from sqlalchemy import func, or_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_ou_for_sport(sport: str, start_date: datetime = None, end_date: datetime = None):
    """
    Backfill O/U data for completed games in date range.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        start_date: Start date for backfill (defaults to 2021-01-01)
        end_date: End date for backfill (defaults to today)
    """
    db = SessionLocal()
    
    try:
        if start_date is None:
            start_date = datetime(2021, 1, 1)
        if end_date is None:
            end_date = datetime.utcnow()
        
        logger.info(f"Backfilling {sport} O/U data from {start_date.date()} to {end_date.date()}")
        
        # Find all final games with scores (use case-insensitive status matching)
        games = db.query(Game).filter(
            Game.sport == sport,
            or_(
                Game.status.ilike("%final%"),
                Game.status == "final",
                Game.status == "STATUS_FINAL"
            ),
            Game.home_score.isnot(None),
            Game.away_score.isnot(None),
            Game.date >= start_date,
            Game.date <= end_date
        ).order_by(Game.date).all()
        
        total_games = len(games)
        updated_count = 0
        skipped_count = 0
        no_over_under_count = 0
        games_with_over_under = 0
        
        logger.info(f"  Found {total_games} final games with scores to process")
        
        for game in games:
            try:
                # Track if game has over_under value
                has_over_under = game.over_under is not None
                if has_over_under:
                    games_with_over_under += 1
                else:
                    no_over_under_count += 1
                
                # Calculate O/U data from existing scores and over_under
                ou_data = OUCalculator.process_game_from_scores(
                    home_score=game.home_score,
                    away_score=game.away_score,
                    over_under=game.over_under
                )
                
                # Track if we're actually updating anything
                needs_update = False
                updates_made = []
                
                # Always update actual_total if we can calculate it (even if over_under is missing)
                if ou_data['actual_total'] is not None:
                    if game.actual_total != ou_data['actual_total']:
                        game.actual_total = ou_data['actual_total']
                        needs_update = True
                        updates_made.append("actual_total")
                
                # Update closing_total if we have it (from over_under)
                if ou_data['closing_total'] is not None:
                    if game.closing_total != ou_data['closing_total']:
                        game.closing_total = ou_data['closing_total']
                        needs_update = True
                        updates_made.append("closing_total")
                
                # Update ou_result if we can calculate it (requires both actual_total and closing_total)
                # Force update if current value is None, empty string, or different
                if ou_data['ou_result'] is not None:
                    current_result = str(game.ou_result).strip() if game.ou_result else ""
                    new_result = str(ou_data['ou_result']).strip()
                    
                    # Update if different, or if current is empty/None
                    if current_result != new_result or not current_result:
                        game.ou_result = ou_data['ou_result']
                        needs_update = True
                        updates_made.append("ou_result")
                
                # Only skip if we didn't need to update anything
                if not needs_update:
                    skipped_count += 1
                    continue
                
                updated_count += 1
                
                # Log first few updates for debugging
                if updated_count <= 5:
                    logger.debug(f"  Updated game {game.id}: {', '.join(updates_made)}")
                
                if updated_count % 100 == 0:
                    db.commit()
                    logger.info(f"  Processed {updated_count}/{total_games} games...")
                    
            except Exception as e:
                logger.error(f"Error processing game {game.id}: {e}")
                continue
        
        db.commit()
        logger.info(f"✓ Updated {updated_count} {sport} games with O/U data")
        logger.info(f"  Skipped {skipped_count} games (already had complete O/U data)")
        logger.info(f"  Games with over_under: {games_with_over_under} (can get ou_result)")
        logger.info(f"  Games without over_under: {no_over_under_count} (got actual_total but NOT ou_result - need odds backfill)")
        logger.info(f"  Total processed: {total_games}")
        
        if no_over_under_count > 0:
            logger.warning(f"  ⚠️  {no_over_under_count} games don't have over_under values!")
            logger.warning(f"  To get ou_result for these games, backfill odds first:")
            logger.warning(f"  POST /api/backfill-odds?sport={sport}&start_date={start_date.date()}&end_date={end_date.date()}")
            logger.warning(f"  Then run this O/U backfill again to calculate ou_result.")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error backfilling {sport} O/U data: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return 0
    finally:
        db.close()


def backfill_all_sports():
    """Backfill all sports with default date ranges"""
    backfill_ranges = {
        'NFL': (datetime(2021, 9, 1), datetime.utcnow()),
        'NHL': (datetime(2021, 10, 1), datetime.utcnow()),
        'NBA': (datetime(2021, 10, 1), datetime.utcnow()),
        'MLB': (datetime(2022, 4, 1), datetime.utcnow())
    }
    
    total_updated = 0
    for sport, (start_date, end_date) in backfill_ranges.items():
        updated = backfill_ou_for_sport(sport, start_date, end_date)
        total_updated += updated
    
    logger.info(f"\n✓ Backfill complete! Total games updated: {total_updated}")
    return total_updated


def validate_ou_coverage(sport: str, min_games: int = 100):
    """
    Check if we have sufficient O/U data for a sport.
    
    Args:
        sport: Sport code
        min_games: Minimum number of games with O/U data needed
        
    Returns:
        Dictionary with coverage statistics
    """
    db = SessionLocal()
    
    try:
        # Count total final games - use same status matching as backfill
        status_filter = or_(
            Game.status.ilike("%final%"),
            Game.status == "final",
            Game.status == "STATUS_FINAL"
        )
        
        total_games = db.query(func.count(Game.id)).filter(
            Game.sport == sport,
            status_filter
        ).scalar() or 0
        
        # Count games with O/U result - use same status matching
        games_with_ou = db.query(func.count(Game.id)).filter(
            Game.sport == sport,
            status_filter,
            Game.ou_result.isnot(None)
        ).scalar() or 0
        
        # Get distribution - query for all non-null, non-empty ou_result values
        # Use same status matching as backfill to ensure consistency
        distribution_query = db.query(
            Game.ou_result,
            func.count(Game.id).label('count')
        ).filter(
            Game.sport == sport,
            status_filter,
            Game.ou_result.isnot(None)
        ).group_by(Game.ou_result).all()
        
        # Build distribution dictionary, normalizing values
        distribution = {}
        for row in distribution_query:
            ou_result = row[0]
            count = int(row[1]) if row[1] is not None else 0
            
            # Skip None, empty strings, and whitespace-only values
            if not ou_result:
                continue
            
            ou_result_str = str(ou_result).strip()
            if not ou_result_str or ou_result_str.lower() == 'none':
                continue
            
            # Normalize to uppercase for consistency
            ou_result_normalized = ou_result_str.upper()
            
            # Aggregate counts for same normalized value (case-insensitive)
            if ou_result_normalized in distribution:
                distribution[ou_result_normalized] += count
            else:
                distribution[ou_result_normalized] = count
        
        # Debug: If we have games with O/U data but no distribution, investigate
        if games_with_ou > 0 and len(distribution) == 0:
            # Check what ou_result values actually exist
            sample_results = db.query(Game.ou_result).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result.isnot(None)
            ).limit(10).all()
            sample_values = [repr(r[0]) for r in sample_results]  # Use repr to see exact values
            logger.warning(f"  ⚠️  Warning: {games_with_ou} games have ou_result but distribution is empty.")
            logger.warning(f"  Sample ou_result values (first 10): {sample_values}")
            
            # Check for empty strings
            empty_string_count = db.query(func.count(Game.id)).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result == ''
            ).scalar()
            if empty_string_count > 0:
                logger.warning(f"  Found {empty_string_count} games with empty string ou_result")
            
            # Check total count vs what we're grouping
            total_with_result = db.query(func.count(Game.id)).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result.isnot(None),
                Game.ou_result != ''
            ).scalar()
            logger.warning(f"  Games with non-empty ou_result: {total_with_result}")
        
        coverage = {
            'sport': sport,
            'total_completed': int(total_games) if total_games else 0,
            'with_ou_data': int(games_with_ou) if games_with_ou else 0,
            'coverage_pct': round((games_with_ou / total_games * 100) if total_games > 0 else 0, 2),
            'distribution': distribution,
            'can_train': games_with_ou >= min_games if games_with_ou else False
        }
        
        logger.info(f"O/U Coverage {sport}: {coverage['coverage_pct']:.1f}% ({games_with_ou}/{total_games})")
        
        # Log distribution with more detail
        if distribution:
            logger.info(f"  Distribution: {distribution}")
            for result_type, count in distribution.items():
                logger.info(f"    {result_type}: {count}")
        else:
            logger.warning(f"  ⚠️  Distribution is empty despite {games_with_ou} games with ou_result")
            # Try a direct count to verify
            over_count = db.query(func.count(Game.id)).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result == "OVER"
            ).scalar()
            under_count = db.query(func.count(Game.id)).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result == "UNDER"
            ).scalar()
            push_count = db.query(func.count(Game.id)).filter(
                Game.sport == sport,
                or_(
                    Game.status.ilike("%final%"),
                    Game.status == "final",
                    Game.status == "STATUS_FINAL"
                ),
                Game.ou_result == "PUSH"
            ).scalar()
            logger.info(f"  Direct counts - OVER: {over_count}, UNDER: {under_count}, PUSH: {push_count}")
        
        logger.info(f"  Can train: {coverage['can_train']}")
        
        return coverage
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill Over/Under data for historical games')
    parser.add_argument('--sport', type=str, choices=['NFL', 'NHL', 'NBA', 'MLB', 'ALL'],
                       default='ALL', help='Sport to backfill (default: ALL)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--validate', action='store_true', help='Only validate coverage, do not backfill')
    
    args = parser.parse_args()
    
    if args.validate:
        if args.sport == 'ALL':
            for sport in ['NFL', 'NHL', 'NBA', 'MLB']:
                validate_ou_coverage(sport)
        else:
            validate_ou_coverage(args.sport)
    else:
        if args.sport == 'ALL':
            backfill_all_sports()
        else:
            start_date = datetime.fromisoformat(args.start) if args.start else None
            end_date = datetime.fromisoformat(args.end) if args.end else None
            backfill_ou_for_sport(args.sport, start_date, end_date)
            validate_ou_coverage(args.sport)
