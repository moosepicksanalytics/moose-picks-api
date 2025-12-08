from app.database import SessionLocal
from app.models.db_models import Game
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
            # Normalize ESPN status values: "STATUS_FINAL" -> "final", etc.
            if "FINAL" in status_raw.upper():
                status = "final"
            elif "SCHEDULED" in status_raw.upper():
                status = "scheduled"
            elif "IN_PROGRESS" in status_raw.upper() or "LIVE" in status_raw.upper():
                status = "in_progress"
            else:
                status = status_raw.lower() if status_raw else "scheduled"
            
            # Get scores
            home_score = int(home.get("score", 0)) if home.get("score") is not None else None
            away_score = int(away.get("score", 0)) if away.get("score") is not None else None
            
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
                over_under=over_under,
                home_score=home_score,
                away_score=away_score,
                espn_data=event,
            )
            
            # Upsert
            existing = db.query(Game).filter(Game.id == game_id).first()
            if existing:
                for key, val in vars(game).items():
                    if not key.startswith("_"):
                        setattr(existing, key, val)
            else:
                db.add(game)
            
            stored_count += 1
        
        db.commit()
        return stored_count
    except Exception as e:
        print(f"✗ Error parsing/storing games: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return 0
    finally:
        db.close()
