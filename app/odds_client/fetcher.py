import requests
from app.config import settings

BASE_URL = "https://api.the-odds-api.com/v4"


def fetch_odds_for_sport(sport: str, markets: list = None):
    """
    Fetch live odds from the-odds-api.
    sport: "americanfootball_nfl", "basketball_nba", etc.
    markets: ["h2h", "spreads", "totals"]
    Returns: list of events with odds
    """
    if markets is None:
        markets = ["h2h", "spreads", "totals"]
    
    try:
        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": settings.ODDS_API_KEY,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american"
        }
        
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()  # Returns a list of events
    except Exception as e:
        print(f"Error fetching odds for {sport}: {e}")
        return []


def extract_moneyline_odds(event: dict, bookmaker: str = "draftkings"):
    """
    Extract moneyline odds from an event for a specific bookmaker.
    Returns: (home_odds, away_odds) tuple or (None, None)
    """
    try:
        bookmakers = event.get("bookmakers", [])
        for bm in bookmakers:
            if bm.get("key") == bookmaker:
                markets = bm.get("markets", [])
                for market in markets:
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        if len(outcomes) >= 2:
                            home_odds = None
                            away_odds = None
                            
                            for outcome in outcomes:
                                if outcome.get("name") == event.get("home_team"):
                                    home_odds = outcome.get("price")
                                elif outcome.get("name") == event.get("away_team"):
                                    away_odds = outcome.get("price")
                            
                            return home_odds, away_odds
        
        return None, None
    except Exception as e:
        print(f"Error extracting odds: {e}")
        return None, None
