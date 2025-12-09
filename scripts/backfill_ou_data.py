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
from sqlalchemy import func
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
        
        # Find all final games with scores but missing O/U data
        games = db.query(Game).filter(
            Game.sport == sport,
            Game.status == "final",
            Game.home_score.isnot(None),
            Game.away_score.isnot(None),
            Game.date >= start_date,
            Game.date <= end_date
        ).all()
        
        total_games = len(games)
        updated_count = 0
        skipped_count = 0
        
        for game in games:
            try:
                # Skip if already has complete O/U data
                if game.actual_total is not None and game.ou_result is not None:
                    skipped_count += 1
                    continue
                
                # Calculate O/U data from existing scores and over_under
                ou_data = OUCalculator.process_game_from_scores(
                    home_score=game.home_score,
                    away_score=game.away_score,
                    over_under=game.over_under
                )
                
                # Update game with O/U data
                if ou_data['closing_total'] is not None:
                    game.closing_total = ou_data['closing_total']
                if ou_data['actual_total'] is not None:
                    game.actual_total = ou_data['actual_total']
                if ou_data['ou_result'] is not None:
                    game.ou_result = ou_data['ou_result']
                
                updated_count += 1
                
                if updated_count % 100 == 0:
                    db.commit()
                    logger.info(f"  Processed {updated_count}/{total_games} games...")
                    
            except Exception as e:
                logger.error(f"Error processing game {game.id}: {e}")
                continue
        
        db.commit()
        logger.info(f"✓ Updated {updated_count} {sport} games with O/U data")
        logger.info(f"  Skipped {skipped_count} games (already had O/U data)")
        logger.info(f"  Total processed: {total_games}")
        
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
        # Count total final games
        total_games = db.query(func.count(Game.id)).filter(
            Game.sport == sport,
            Game.status == "final"
        ).scalar()
        
        # Count games with O/U result
        games_with_ou = db.query(func.count(Game.id)).filter(
            Game.sport == sport,
            Game.status == "final",
            Game.ou_result.isnot(None)
        ).scalar()
        
        # Get distribution - also check for empty strings
        distribution_query = db.query(
            Game.ou_result,
            func.count(Game.id).label('count')
        ).filter(
            Game.sport == sport,
            Game.status == "final",
            Game.ou_result.isnot(None),
            Game.ou_result != ''  # Exclude empty strings
        ).group_by(Game.ou_result).all()
        
        # Build distribution dictionary, ensuring all values are properly converted
        distribution = {}
        for row in distribution_query:
            ou_result = row[0]
            count = int(row[1]) if row[1] is not None else 0
            if ou_result and str(ou_result).strip():  # Ensure not None and not empty
                distribution[str(ou_result).strip()] = count
        
        # Debug: Log if we have games with O/U data but no distribution
        if games_with_ou > 0 and len(distribution) == 0:
            # Check what ou_result values actually exist
            sample_results = db.query(Game.ou_result).filter(
                Game.sport == sport,
                Game.status == "final",
                Game.ou_result.isnot(None)
            ).limit(5).all()
            logger.warning(f"  ⚠️  Warning: {games_with_ou} games have ou_result but distribution is empty. Sample values: {[r[0] for r in sample_results]}")
        
        coverage = {
            'sport': sport,
            'total_completed': int(total_games) if total_games else 0,
            'with_ou_data': int(games_with_ou) if games_with_ou else 0,
            'coverage_pct': round((games_with_ou / total_games * 100) if total_games > 0 else 0, 2),
            'distribution': distribution,
            'can_train': games_with_ou >= min_games if games_with_ou else False
        }
        
        logger.info(f"O/U Coverage {sport}: {coverage['coverage_pct']:.1f}% ({games_with_ou}/{total_games})")
        logger.info(f"  Distribution: {coverage['distribution']}")
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
