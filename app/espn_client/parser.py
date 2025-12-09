from app.database import SessionLocal
from app.models.db_models import Game
from app.utils.ou_calculator import OUCalculator
from datetime import datetime


def parse_and_store_games(sport: str, games_data: list, only_final: bool = False):
    """
    Parse ESPN JSON events and store in DB.
    
    Args:
        sport: Sport code
        games_data: List of ESPN event dicts
        only_final: If True, only store games with final scores (for training)
    
    Returns:
        Number of games stored
    """
    db = SessionLocal()
    stored_count = 0
    
    try:
        for event in games_data:
            game_id = event.get("id")
            
            # Extract teams
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            
            if len(competitors) < 2:
                continue
            
            home = competitors[0]
            away = competitors[1]
            
            # Extract status and normalize
            status_raw = event.get("status", {}).get("type", {}).get("name", "")
            status_raw_upper = status_raw.upper() if status_raw else ""
            
            # Get scores first (needed to determine if game is actually final)
            home_score = int(home.get("score", 0)) if home.get("score") is not None else None
            away_score = int(away.get("score", 0)) if away.get("score") is not None else None
            
            # Normalize ESPN status values
            # Only mark as "final" if explicitly final AND has scores
            if status_raw_upper == "STATUS_FINAL" or status_raw_upper == "FINAL":
                # Only mark as final if game has scores
                if home_score is not None and away_score is not None:
                    status = "final"
                else:
                    # Has final status but no scores - might be data issue, treat as scheduled
                    status = "scheduled"
                    if stored_count < 3:  # Only log first few for debugging
                        print(f"    Warning: Game {game_id} marked FINAL but has no scores, treating as scheduled")
            elif "SCHEDULED" in status_raw_upper or status_raw_upper == "":
                status = "scheduled"
            elif "IN_PROGRESS" in status_raw_upper or "LIVE" in status_raw_upper or "IN PROGRESS" in status_raw_upper:
                status = "in_progress"
            elif "POSTPONED" in status_raw_upper or "DELAYED" in status_raw_upper:
                status = "scheduled"  # Treat postponed/delayed as scheduled
            else:
                # Unknown status - default to scheduled if no scores, final if has scores
                if home_score is not None and away_score is not None:
                    status = "final"  # Has scores, probably finished
                else:
                    status = "scheduled"  # No scores, probably not started
                if stored_count < 3:  # Only log first few for debugging
                    print(f"    Info: Game {game_id} has unknown status '{status_raw}', defaulting to '{status}' (scores: {home_score}-{away_score})")
            
            # If only_final is True, skip games without final scores
            if only_final and (status != "final" or home_score is None or away_score is None):
                continue
            
            # Extract odds - try different paths
            home_moneyline = None
            away_moneyline = None
            over_under = None
            spread = None
            
            # Try to extract from odds array
            odds_array = comp.get("odds", [])
            if odds_array and len(odds_array) > 0:
                odds = odds_array[0]
                home_moneyline = float(odds.get("homeMoneyLine") or 0) or None
                away_moneyline = float(odds.get("awayMoneyLine") or 0) or None
                over_under = float(odds.get("overUnder") or 0) or None
                
                if odds.get("spread"):
                    spread = float(odds.get("spread"))
            
            # Extract and calculate O/U data
            ou_data = OUCalculator.process_game_ou_data(event)
            closing_total = ou_data.get('closing_total')
            actual_total = ou_data.get('actual_total')
            ou_result = ou_data.get('ou_result')
            
            # Fallback: if OU calculator didn't find closing_total but we have over_under, use it
            if closing_total is None and over_under is not None:
                closing_total = over_under
                # Recalculate with scores if available
                if home_score is not None and away_score is not None:
                    actual_total = home_score + away_score
                    ou_result = OUCalculator.determine_ou_result(actual_total, closing_total)
            
            # Create game record
            game = Game(
                id=game_id,
                sport=sport,
                league=comp.get("league", {}).get("name", ""),
                date=datetime.fromisoformat(event.get("date", "").replace("Z", "+00:00")),
                home_team=home.get("team", {}).get("displayName", ""),
                away_team=away.get("team", {}).get("displayName", ""),
                status=status,
                home_moneyline=home_moneyline,
                away_moneyline=away_moneyline,
                spread=spread,
                over_under=over_under,  # Keep for backward compatibility
                closing_total=closing_total,
                actual_total=actual_total,
                ou_result=ou_result,
                home_score=home_score,
                away_score=away_score,
                espn_data=event,
            )
            
            # Upsert - check if game already exists
            existing = db.query(Game).filter(Game.id == game_id).first()
            if existing:
                # Update existing game
                existing.sport = game.sport
                existing.league = game.league
                existing.date = game.date
                existing.home_team = game.home_team
                existing.away_team = game.away_team
                existing.status = game.status
                existing.home_moneyline = game.home_moneyline
                existing.away_moneyline = game.away_moneyline
                existing.spread = game.spread
                existing.over_under = game.over_under
                existing.closing_total = game.closing_total
                existing.actual_total = game.actual_total
                existing.ou_result = game.ou_result
                existing.home_score = game.home_score
                existing.away_score = game.away_score
                existing.espn_data = game.espn_data
                existing.updated_at = datetime.utcnow()
            else:
                # Add new game
                db.add(game)
            
            stored_count += 1
        
        db.commit()
        if stored_count > 0:
            print(f"  ✓ Committed {stored_count} games to database")
        return stored_count
    except Exception as e:
        print(f"✗ Error parsing/storing games: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return 0
    finally:
        db.close()
