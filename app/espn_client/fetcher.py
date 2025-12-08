"""
ESPN API client for fetching game data.
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime


def fetch_games_for_date(sport: str, date_str: str) -> List[Dict]:
    """
    Fetch games for a specific date from ESPN API.
    
    Args:
        sport: Sport code (NFL, NHL, etc.)
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        List of game data dicts
    """
    # ESPN API endpoint
    sport_map = {
        "NFL": "football",
        "NHL": "hockey",
        "NBA": "basketball",
        "MLB": "baseball",
    }
    
    league = sport_map.get(sport, sport.lower())
    
    # ESPN API URL
    url = f"https://site.api.espn.com/apis/site/v2/sports/{league}/nfl/scoreboard"
    
    if sport == "NHL":
        url = f"https://site.api.espn.com/apis/site/v2/sports/{league}/nhl/scoreboard"
    
    try:
        response = requests.get(url, params={"dates": date_str.replace("-", "")})
        response.raise_for_status()
        data = response.json()
        
        events = data.get("events", [])
        return events
    except Exception as e:
        print(f"Error fetching games from ESPN: {e}")
        return []
