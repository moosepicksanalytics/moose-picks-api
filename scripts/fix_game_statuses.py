"""
Fix game statuses in database - correct games marked as 'final' that shouldn't be.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Game


def fix_game_statuses(sport: str = None, dry_run: bool = True):
    """
    Fix games that are incorrectly marked as 'final'.
    
    A game should only be 'final' if:
    1. Status is 'final' AND
    2. Has scores AND
    3. Game time is in the past
    
    Args:
        sport: Sport to fix (None = all sports)
        dry_run: If True, only show what would be changed
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        
        # Query for games marked as final
        query = db.query(Game).filter(Game.status == "final")
        if sport:
            query = query.filter(Game.sport == sport)
        
        games = query.all()
        
        fixed_count = 0
        would_fix = 0
        
        for game in games:
            game_time = game.date
            if game_time.tzinfo is None:
                game_time = game_time.replace(tzinfo=timezone.utc)
            
            # Check if game should actually be final
            should_be_final = (
                game.home_score is not None and
                game.away_score is not None and
                game_time < now
            )
            
            if not should_be_final:
                would_fix += 1
                # Game is incorrectly marked as final
                if game_time > now:
                    new_status = "scheduled"
                    reason = "game is in the future"
                elif game.home_score is None or game.away_score is None:
                    new_status = "scheduled"
                    reason = "game has no scores"
                else:
                    new_status = "scheduled"
                    reason = "unknown"
                
                print(f"Game {game.id} ({game.sport}): {game.away_team} @ {game.home_team}")
                print(f"  Current: status='{game.status}', date={game.date}, scores={game.home_score}-{game.away_score}")
                print(f"  Should be: status='{new_status}' ({reason})")
                
                if not dry_run:
                    game.status = new_status
                    fixed_count += 1
                else:
                    print(f"  [DRY RUN] Would change to '{new_status}'")
                print()
        
        if not dry_run:
            db.commit()
            print(f"✓ Fixed {fixed_count} game statuses")
        else:
            print(f"[DRY RUN] Would fix {would_fix} game statuses")
            print("Run with --no-dry-run to apply changes")
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix incorrectly marked game statuses")
    parser.add_argument("--sport", type=str, help="Sport to fix (NFL, NHL, NBA, MLB)")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually apply changes (default: dry run)")
    
    args = parser.parse_args()
    
    fix_game_statuses(sport=args.sport, dry_run=not args.no_dry_run)
