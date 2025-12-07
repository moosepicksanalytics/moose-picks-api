import requests
from datetime import datetime, timedelta
from app.espn_client.endpoints import ENDPOINTS


def fetch_games_for_date(sport: str, date_str: str):
    """
    Fetch all games for a sport on a given date.
    date_str: "YYYY-MM-DD"
    """
    try:
        url = ENDPOINTS[sport]["scoreboard"]
        resp = requests.get(url, params={"dates": date_str}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])
    except Exception as e:
        print(f"Error fetching {sport} games for {date_str}: {e}")
        return []


def fetch_game_detail(sport: str, game_id: str):
    """Fetch detailed box score + odds for a single game."""
    try:
        url = ENDPOINTS[sport]["event"]
        resp = requests.get(url, params={"event": game_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching {sport} game {game_id}: {e}")
        return {}


def fetch_season_games(sport: str, start_date: str, end_date: str):
    """Backfill: fetch all games in a date range."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_games = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            games = fetch_games_for_date(sport, date_str)
            all_games.extend(games)
            current += timedelta(days=1)
        
        return all_games
    except Exception as e:
        print(f"Error fetching season games: {e}")
        return []
