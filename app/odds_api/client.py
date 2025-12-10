"""
The Odds API client for fetching real-time betting odds.
Supports NFL, NHL, NBA, MLB with 20k calls/month limit.
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.config import settings


# The Odds API sport keys
ODDS_API_SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NHL": "icehockey_nhl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

# The Odds API region (us, uk, au, etc.)
ODDS_API_REGION = "us"

# The Odds API markets
ODDS_API_MARKETS = {
    "moneyline": "h2h",  # Head-to-head (moneyline)
    "spread": "spreads",
    "totals": "totals",
    "over_under": "totals",
}


def fetch_odds_for_sport(
    sport: str,
    date: Optional[str] = None,
    markets: Optional[List[str]] = None
) -> List[Dict]:
    """
    Fetch odds from The Odds API for a sport and date.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        date: Date in YYYY-MM-DD format (optional, defaults to today)
        markets: List of markets to fetch (default: ["h2h", "spreads", "totals"])
    
    Returns:
        List of game odds data
    """
    if not settings.ODDS_API_KEY:
        print("Warning: ODDS_API_KEY not set. Cannot fetch odds from The Odds API.")
        return []
    
    sport_key = ODDS_API_SPORT_KEYS.get(sport)
    if not sport_key:
        print(f"Warning: Sport {sport} not supported by The Odds API")
        return []
    
    # Default markets
    if markets is None:
        markets = ["h2h", "spreads", "totals"]
    
    # Format date (ISO format: YYYYMMDD or ISO8601)
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_param = date_obj.strftime("%Y-%m-%dT12:00:00Z")  # ISO8601 format
        except ValueError:
            print(f"Invalid date format: {date}. Use YYYY-MM-DD")
            return []
    else:
        # Default to today
        date_param = datetime.now().strftime("%Y-%m-%dT12:00:00Z")
    
    # The Odds API endpoint
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    
    params = {
        "apiKey": settings.ODDS_API_KEY,
        "regions": ODDS_API_REGION,
        "markets": ",".join(markets),
        "dateFormat": "iso",
    }
    
    # Add date if specified
    if date:
        params["commenceTimeFrom"] = date_param
        # Add 24 hours for end time
        end_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        params["commenceTimeTo"] = end_date.strftime("%Y-%m-%dT12:00:00Z")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Log API usage
        remaining = response.headers.get("x-requests-remaining", "unknown")
        used = response.headers.get("x-requests-used", "unknown")
        print(f"The Odds API: {used} used, {remaining} remaining this month")
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds from The Odds API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return []


def fetch_historical_odds_for_sport(
    sport: str,
    date: str,
    markets: Optional[List[str]] = None
) -> List[Dict]:
    """
    Fetch historical odds from The Odds API for a sport and date.
    
    Historical odds are available from June 6th 2020, with snapshots at 5-10 minute intervals.
    This endpoint costs 10 credits per region per market (vs 1 for current odds).
    Only available on paid usage plans.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        date: Date in YYYY-MM-DD format, or ISO8601 timestamp (e.g., "2021-10-18T12:00:00Z")
        markets: List of markets to fetch (default: ["h2h", "spreads", "totals"])
    
    Returns:
        List of game odds data (from the "data" field in the response)
    """
    if not settings.ODDS_API_KEY:
        print("Warning: ODDS_API_KEY not set. Cannot fetch historical odds from The Odds API.")
        return []
    
    sport_key = ODDS_API_SPORT_KEYS.get(sport)
    if not sport_key:
        print(f"Warning: Sport {sport} not supported by The Odds API")
        return []
    
    # Default markets
    if markets is None:
        markets = ["h2h", "spreads", "totals"]
    
    # Format date as ISO8601 timestamp
    # If date is YYYY-MM-DD, use noon UTC for that date
    # If date is already ISO8601, use it as-is
    try:
        if "T" in date:
            # Already ISO8601 format
            date_param = date
        else:
            # YYYY-MM-DD format, convert to ISO8601 at noon UTC
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_param = date_obj.strftime("%Y-%m-%dT12:00:00Z")
    except ValueError:
        print(f"Invalid date format: {date}. Use YYYY-MM-DD or ISO8601 (e.g., 2021-10-18T12:00:00Z)")
        return []
    
    # The Odds API historical endpoint
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"
    
    params = {
        "apiKey": settings.ODDS_API_KEY,
        "regions": ODDS_API_REGION,
        "markets": ",".join(markets),
        "date": date_param,
        "dateFormat": "iso",
        "oddsFormat": "american",  # Use American odds format
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Historical API returns data wrapped in a structure with timestamp info
        # Extract the actual data array
        if isinstance(result, dict) and "data" in result:
            data = result["data"]
            timestamp = result.get("timestamp", "unknown")
            print(f"  Historical snapshot timestamp: {timestamp}")
        else:
            # Fallback: assume it's already the data array
            data = result
        
        # Log API usage (historical costs 10x more)
        remaining = response.headers.get("x-requests-remaining", "unknown")
        used = response.headers.get("x-requests-used", "unknown")
        last_cost = response.headers.get("x-requests-last", "unknown")
        print(f"  The Odds API (Historical): {used} used, {remaining} remaining, last call cost: {last_cost}")
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical odds from The Odds API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
            if e.response.status_code == 402:
                print("  ⚠️  Historical odds require a paid plan. Upgrade your API plan to use this feature.")
        return []


def parse_odds_data(odds_data: List[Dict], sport: str) -> List[Dict]:
    """
    Parse The Odds API response into our format.
    
    Args:
        odds_data: Raw response from The Odds API
        sport: Sport code
    
    Returns:
        List of parsed odds dicts
    """
    parsed = []
    
    for game in odds_data:
        game_id = game.get("id", "")
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        
        # Extract odds from bookmakers
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        
        # Use first bookmaker (or best odds - you can enhance this)
        bookmaker = bookmakers[0]
        markets_data = bookmaker.get("markets", [])
        
        odds_dict = {
            "game_id": game_id,
            "sport": sport,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": bookmaker.get("title", "unknown"),
        }
        
        # PRIORITY 1: Check for Pinnacle odds in betGrades.edgeData (most accurate)
        # This handles cached_odds table structure where correct odds are nested
        bet_grades = game.get("betGrades", {})
        edge_data = bet_grades.get("edgeData", {}) if isinstance(bet_grades, dict) else {}
        
        if edge_data.get("pinnacleHomeML") is not None and edge_data.get("pinnacleAwayML") is not None:
            pinnacle_home = edge_data.get("pinnacleHomeML")
            pinnacle_away = edge_data.get("pinnacleAwayML")
            
            # Validate: For moneyline, one must be negative (favorite) and one positive (underdog)
            # Exception: Both can be negative if both are heavy favorites (rare but possible)
            # Reject if both are positive (invalid for moneyline)
            is_valid = True
            if pinnacle_home > 0 and pinnacle_away > 0:
                # Invalid: both positive
                is_valid = False
            elif pinnacle_home < 0 and pinnacle_away < 0:
                # Both negative - check if they're reasonable (both heavy favorites)
                # This is rare but can happen. Accept if both are < -100
                if abs(pinnacle_home) < 100 or abs(pinnacle_away) < 100:
                    # Both are small negatives - likely invalid
                    is_valid = False
            
            if is_valid:
                # Use Pinnacle odds from edgeData (these are the correct odds)
                odds_dict["home_moneyline"] = pinnacle_home
                odds_dict["away_moneyline"] = pinnacle_away
                odds_dict["bookmaker"] = "Pinnacle (from edgeData)"
            else:
                # Invalid Pinnacle odds - fall through to other parsing
                pass
        else:
            # PRIORITY 2: Parse moneyline from standard bookmakers structure (h2h)
            h2h_market = next((m for m in markets_data if m.get("key") == "h2h"), None)
            if h2h_market:
                outcomes = h2h_market.get("outcomes", [])
                h2h_home_ml = None
                h2h_away_ml = None
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        h2h_home_ml = outcome.get("price")
                    elif outcome.get("name") == away_team:
                        h2h_away_ml = outcome.get("price")
                
                # Validate h2h odds before using
                if h2h_home_ml is not None and h2h_away_ml is not None:
                    if h2h_home_ml > 0 and h2h_away_ml > 0:
                        # Invalid: both positive - skip
                        pass
                    else:
                        odds_dict["home_moneyline"] = h2h_home_ml
                        odds_dict["away_moneyline"] = h2h_away_ml
            
            # PRIORITY 3: Fallback - check for Pinnacle in bookmakers list
            if "home_moneyline" not in odds_dict or "away_moneyline" not in odds_dict:
                pinnacle_bookmaker = next((b for b in bookmakers if b.get("title", "").lower() == "pinnacle"), None)
                if pinnacle_bookmaker:
                    pinnacle_markets = pinnacle_bookmaker.get("markets", [])
                    pinnacle_h2h = next((m for m in pinnacle_markets if m.get("key") == "h2h"), None)
                    if pinnacle_h2h:
                        outcomes = pinnacle_h2h.get("outcomes", [])
                        pinnacle_home_ml = None
                        pinnacle_away_ml = None
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                pinnacle_home_ml = outcome.get("price")
                            elif outcome.get("name") == away_team:
                                pinnacle_away_ml = outcome.get("price")
                        
                        # Validate Pinnacle odds before using
                        if pinnacle_home_ml is not None and pinnacle_away_ml is not None:
                            if pinnacle_home_ml > 0 and pinnacle_away_ml > 0:
                                # Invalid: both positive - skip
                                pass
                            else:
                                odds_dict["home_moneyline"] = pinnacle_home_ml
                                odds_dict["away_moneyline"] = pinnacle_away_ml
                                odds_dict["bookmaker"] = "Pinnacle"
            
            # PRIORITY 4: Try other reputable bookmakers if still no odds
            if "home_moneyline" not in odds_dict or "away_moneyline" not in odds_dict:
                # Try other bookmakers (DraftKings, FanDuel, etc.)
                for bookmaker in bookmakers:
                    if bookmaker.get("title", "").lower() in ["draftkings", "fanduel", "betmgm", "caesars"]:
                        bm_markets = bookmaker.get("markets", [])
                        bm_h2h = next((m for m in bm_markets if m.get("key") == "h2h"), None)
                        if bm_h2h:
                            outcomes = bm_h2h.get("outcomes", [])
                            bm_home_ml = None
                            bm_away_ml = None
                            for outcome in outcomes:
                                if outcome.get("name") == home_team:
                                    bm_home_ml = outcome.get("price")
                                elif outcome.get("name") == away_team:
                                    bm_away_ml = outcome.get("price")
                            
                            # Validate before using
                            if bm_home_ml is not None and bm_away_ml is not None:
                                if bm_home_ml > 0 and bm_away_ml > 0:
                                    # Invalid - skip
                                    continue
                                else:
                                    odds_dict["home_moneyline"] = bm_home_ml
                                    odds_dict["away_moneyline"] = bm_away_ml
                                    odds_dict["bookmaker"] = bookmaker.get("title", "unknown")
                                    break  # Found valid odds, stop searching
        
        # Parse spreads
        spreads_market = next((m for m in markets_data if m.get("key") == "spreads"), None)
        if spreads_market:
            outcomes = spreads_market.get("outcomes", [])
            for outcome in outcomes:
                if outcome.get("name") == home_team:
                    odds_dict["home_spread"] = outcome.get("point")
                    odds_dict["home_spread_odds"] = outcome.get("price")
                elif outcome.get("name") == away_team:
                    odds_dict["away_spread"] = outcome.get("point")
                    odds_dict["away_spread_odds"] = outcome.get("price")
            
            # Set the spread line (home team's spread)
            if "home_spread" in odds_dict:
                odds_dict["spread"] = odds_dict["home_spread"]
        
        # Parse totals (over/under)
        totals_market = next((m for m in markets_data if m.get("key") == "totals"), None)
        if totals_market:
            outcomes = totals_market.get("outcomes", [])
            for outcome in outcomes:
                if outcome.get("name") == "Over":
                    odds_dict["over_under"] = outcome.get("point")
                    odds_dict["over_odds"] = outcome.get("price")
                elif outcome.get("name") == "Under":
                    odds_dict["under_odds"] = outcome.get("price")
        
        parsed.append(odds_dict)
    
    return parsed


def fetch_and_update_game_odds_historical(
    sport: str,
    date: str
) -> int:
    """
    Fetch historical odds from The Odds API and update games in database.
    
    Args:
        sport: Sport code
        date: Date in YYYY-MM-DD format or ISO8601 timestamp
    
    Returns:
        Number of games updated
    """
    from app.database import SessionLocal
    from app.models.db_models import Game
    
    # Fetch historical odds
    odds_data = fetch_historical_odds_for_sport(sport, date)
    if not odds_data:
        return 0
    
    # Parse odds (same parser works for historical data)
    parsed_odds = parse_odds_data(odds_data, sport)
    
    # Update database (same matching logic)
    db = SessionLocal()
    updated_count = 0
    matched_games = []
    unmatched_odds = []
    
    try:
        print(f"  Attempting to match {len(parsed_odds)} historical odds entries to games...")
        
        for odds in parsed_odds:
            # Try to match by team names and date
            game_date = None
            if odds.get("commence_time"):
                try:
                    game_date = datetime.fromisoformat(odds["commence_time"].replace("Z", "+00:00"))
                except:
                    pass
            
            # Find matching game
            query = db.query(Game).filter(Game.sport == sport)
            
            if game_date:
                # Match by date (same day) - use date-only comparison
                from sqlalchemy import func
                date_only = game_date.date()
                query = query.filter(func.date(Game.date) == date_only)
            
            # Match by team names (fuzzy match - try multiple strategies)
            home_team = odds.get("home_team", "").strip()
            away_team = odds.get("away_team", "").strip()
            
            # Strategy 1: Exact match (case insensitive)
            game = query.filter(
                (Game.home_team.ilike(home_team)) &
                (Game.away_team.ilike(away_team))
            ).first()
            
            # Strategy 2: Contains match
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{home_team}%")) &
                    (Game.away_team.ilike(f"%{away_team}%"))
                ).first()
            
            # Strategy 3: Reverse match (in case teams are swapped)
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{away_team}%")) &
                    (Game.away_team.ilike(f"%{home_team}%"))
                ).first()
            
            if game:
                # Update odds
                updated = False
                if "home_moneyline" in odds and odds["home_moneyline"]:
                    game.home_moneyline = odds["home_moneyline"]
                    updated = True
                if "away_moneyline" in odds and odds["away_moneyline"]:
                    game.away_moneyline = odds["away_moneyline"]
                    updated = True
                if "spread" in odds and odds["spread"]:
                    game.spread = odds["spread"]
                    updated = True
                if "over_under" in odds and odds["over_under"]:
                    game.over_under = odds["over_under"]
                    updated = True
                
                if updated:
                    updated_count += 1
                    matched_games.append(f"{away_team} @ {home_team}")
            else:
                unmatched_odds.append(f"{away_team} @ {home_team}")
        
        db.commit()
        
        if updated_count > 0:
            print(f"  ✓ Updated odds for {updated_count} {sport} games")
            if matched_games:
                print(f"    Matched games: {', '.join(matched_games[:3])}")
        else:
            print(f"  ⚠️  No games matched for odds update")
            if unmatched_odds:
                print(f"    Unmatched odds: {', '.join(unmatched_odds[:3])}")
        
    except Exception as e:
        print(f"Error updating historical odds: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
    
    return updated_count


def fetch_and_update_game_odds(
    sport: str,
    date: Optional[str] = None
) -> int:
    """
    Fetch odds from The Odds API and update games in database.
    
    Args:
        sport: Sport code
        date: Date in YYYY-MM-DD format (optional)
    
    Returns:
        Number of games updated
    """
    from app.database import SessionLocal
    from app.models.db_models import Game
    
    # Fetch odds
    odds_data = fetch_odds_for_sport(sport, date)
    if not odds_data:
        return 0
    
    # Parse odds
    parsed_odds = parse_odds_data(odds_data, sport)
    
    # Update database
    db = SessionLocal()
    updated_count = 0
    matched_games = []
    unmatched_odds = []
    
    try:
        print(f"  Attempting to match {len(parsed_odds)} odds entries to games...")
        
        for odds in parsed_odds:
            # Try to match by team names and date
            game_date = None
            if odds.get("commence_time"):
                try:
                    game_date = datetime.fromisoformat(odds["commence_time"].replace("Z", "+00:00"))
                except:
                    pass
            
            # Find matching game
            query = db.query(Game).filter(Game.sport == sport)
            
            if game_date:
                # Match by date (same day) - use date-only comparison
                from sqlalchemy import func
                date_only = game_date.date()
                query = query.filter(func.date(Game.date) == date_only)
            
            # Match by team names (fuzzy match - try multiple strategies)
            home_team = odds.get("home_team", "").strip()
            away_team = odds.get("away_team", "").strip()
            
            # Strategy 1: Exact match (case insensitive)
            game = query.filter(
                (Game.home_team.ilike(home_team)) &
                (Game.away_team.ilike(away_team))
            ).first()
            
            # Strategy 2: Contains match
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{home_team}%")) &
                    (Game.away_team.ilike(f"%{away_team}%"))
                ).first()
            
            # Strategy 3: Reverse match (in case teams are swapped)
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{away_team}%")) &
                    (Game.away_team.ilike(f"%{home_team}%"))
                ).first()
            
            if game:
                # Update odds with validation
                updated = False
                
                # Validate moneyline odds: one must be negative (favorite), one positive (underdog)
                # Reject if both are positive (invalid for moneyline)
                home_ml = odds.get("home_moneyline")
                away_ml = odds.get("away_moneyline")
                
                if home_ml is not None and away_ml is not None:
                    # Validate odds before updating
                    # Reject if both are positive (invalid for moneyline)
                    # Accept if one is positive and one negative (normal case)
                    # Accept if both are negative (both heavy favorites - rare but valid)
                    if home_ml > 0 and away_ml > 0:
                        print(f"  ⚠️  Skipping invalid odds for {away_team} @ {home_team}: both positive (home={home_ml}, away={away_ml})")
                    elif home_ml < 0 and away_ml < 0 and (abs(home_ml) < 100 or abs(away_ml) < 100):
                        # Both negative but small - likely invalid
                        print(f"  ⚠️  Skipping suspicious odds for {away_team} @ {home_team}: both small negatives (home={home_ml}, away={away_ml})")
                    else:
                        # Valid odds - update
                        game.home_moneyline = home_ml
                        game.away_moneyline = away_ml
                        updated = True
                elif home_ml is not None:
                    # Only home odds available
                    game.home_moneyline = home_ml
                    updated = True
                elif away_ml is not None:
                    # Only away odds available
                    game.away_moneyline = away_ml
                    updated = True
                
                if "spread" in odds and odds["spread"]:
                    game.spread = odds["spread"]
                    updated = True
                if "over_under" in odds and odds["over_under"]:
                    game.over_under = odds["over_under"]
                    updated = True
                
                if updated:
                    updated_count += 1
                    matched_games.append(f"{away_team} @ {home_team}")
            else:
                unmatched_odds.append(f"{away_team} @ {home_team}")
        
        db.commit()
        
        if updated_count > 0:
            print(f"  ✓ Updated odds for {updated_count} {sport} games")
            if matched_games:
                print(f"    Matched games: {', '.join(matched_games[:3])}")
        else:
            print(f"  ⚠️  No games matched for odds update")
            if unmatched_odds:
                print(f"    Unmatched odds: {', '.join(unmatched_odds[:3])}")
                # Debug: Show what games exist in DB
                existing_games = db.query(Game).filter(
                    Game.sport == sport,
                    func.date(Game.date) == datetime.now().date()
                ).limit(5).all()
                if existing_games:
                    print(f"    Existing games in DB: {[f'{g.away_team} @ {g.home_team}' for g in existing_games]}")
        
    except Exception as e:
        print(f"Error updating odds: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
    
    return updated_count
