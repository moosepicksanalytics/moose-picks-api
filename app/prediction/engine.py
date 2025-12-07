from app.database import SessionLocal
from app.models.db_models import Prediction
from datetime import datetime
import uuid


def predict_for_game(sport: str, market: str, game_id: str):
    """
    Return prediction with current odds from the-odds-api.
    Uses live odds + simple heuristics until ML models are trained.
    """
    from app.odds_client.fetcher import fetch_odds_for_sport, extract_moneyline_odds
    
    try:
        # Map sport names
        sport_map = {
            "NFL": "americanfootball_nfl",
            "NBA": "basketball_nba",
            "NHL": "hockey_nhl",
            "MLB": "baseball_mlb"
        }
        
        odds_sport = sport_map.get(sport, sport.lower())
        
        # Fetch live odds
        odds_data = fetch_odds_for_sport(odds_sport)
        
        if not odds_data:
            return {"error": f"No odds data for {sport}"}
        
        # Find the game by ID or teams
        game_event = None
        for event in odds_data:
            if event.get("id") == game_id:
                game_event = event
                break
        
        if not game_event:
            return {"error": f"Game {game_id} not found"}
        
        # Extract odds based on market
        home_team = game_event.get("home_team", "")
        away_team = game_event.get("away_team", "")
        
        if market == "moneyline" or market == "h2h":
            home_odds, away_odds = extract_moneyline_odds(game_event)
            
            # Simple heuristic: convert odds to implied probability
            if home_odds and away_odds:
                # American odds to probability conversion
                home_prob = calculate_probability_from_odds(home_odds)
                away_prob = calculate_probability_from_odds(away_odds)
                
                # Normalize
                total = home_prob + away_prob
                home_prob = home_prob / total if total > 0 else 0.5
                away_prob = away_prob / total if total > 0 else 0.5
            else:
                home_prob = 0.50
                away_prob = 0.50
            
            return {
                "game_id": game_id,
                "sport": sport,
                "market": market,
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_probability": round(home_prob, 3),
                "away_probability": round(away_prob, 3),
                "recommended_pick": "home" if home_prob > away_prob else "away",
                "confidence": round(abs(home_prob - away_prob), 3),
                "kelly_fraction": calculate_kelly(home_prob if home_prob > away_prob else away_prob, 
                                                  home_odds if home_prob > away_prob else away_odds),
                "recommended_unit_size": 1.0,
                "model_version": "live_odds_heuristic"
            }
        
        else:
            return {"error": f"Market {market} not yet supported"}
    
    except Exception as e:
        print(f"Error in predict_for_game: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def calculate_probability_from_odds(american_odds):
    """
    Convert American odds to probability.
    Positive odds: prob = 100 / (odds + 100)
    Negative odds: prob = -odds / (-odds + 100)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)


def calculate_kelly(win_prob, odds):
    """
    Calculate Kelly Criterion fraction.
    kelly = (bp - q) / b
    where b = odds/100, p = win_prob, q = 1-p
    """
    if not odds or odds == 0:
        return 0.0
    
    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / -odds) + 1
    
    b = decimal_odds - 1  # Profit multiple
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b if b > 0 else 0
    
    # Cap kelly at 25% for safety
    kelly = min(max(kelly, 0), 0.25)
    
    return round(kelly, 3)


def get_daily_picks(date_str: str, sport: str = "NFL"):
    """
    Get all picks for a date and sport.
    date_str: "YYYY-MM-DD"
    """
    from app.espn_client.fetcher import fetch_games_for_date
    
    try:
        # Convert date to YYYYMMDD format
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_formatted = date_obj.strftime("%Y%m%d")
        
        # Fetch games
        games = fetch_games_for_date(sport, date_formatted)
        
        if not games:
            return {"error": f"No games found for {date_str}"}
        
        picks = []
        for game in games:
            game_id = game.get("id")
            pred = predict_for_game(sport, "moneyline", game_id)
            
            if "error" not in pred:
                picks.append(pred)
        
        return {
            "date": date_str,
            "sport": sport,
            "total_games": len(games),
            "picks": picks
        }
    
    except Exception as e:
        print(f"Error getting daily picks: {e}")
        return {"error": str(e)}
