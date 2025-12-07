from typing import Dict


def extract_features_nfl_moneyline(game_dict: Dict) -> Dict:
    """
    Extract features for NFL moneyline prediction.
    """
    try:
        comp = game_dict.get("competitions", [{}])[0]
        home = comp.get("competitors", [{}])[0]
        away = comp.get("competitors", [{}])[1] if len(comp.get("competitors", [])) > 1 else {}
        
        features = {
            "home_team_id": home.get("team", {}).get("id", ""),
            "away_team_id": away.get("team", {}).get("id", ""),
            "home_moneyline": float(comp.get("odds", [{}])[0].get("homeMoneyLine") or 0),
            "is_home": 1,
        }
        
        return features
    except Exception as e:
        print(f"Error extracting NFL moneyline features: {e}")
        return {}


def extract_features_nba_spread(game_dict: Dict) -> Dict:
    """Extract features for NBA spread prediction."""
    try:
        comp = game_dict.get("competitions", [{}])[0]
        features = {
            "spread": float(comp.get("spread", 0)),
            "home_team_id": comp.get("competitors", [{}])[0].get("team", {}).get("id", ""),
        }
        return features
    except Exception as e:
        print(f"Error extracting NBA spread features: {e}")
        return {}


def extract_features_nhl_totals(game_dict: Dict) -> Dict:
    """Extract features for NHL totals prediction."""
    try:
        comp = game_dict.get("competitions", [{}])[0]
        odds = (comp.get("odds") or [{}])[0]
        features = {
            "over_under": float(odds.get("overUnder") or 0),
        }
        return features
    except Exception as e:
        print(f"Error extracting NHL totals features: {e}")
        return {}


def extract_features_mlb_moneyline(game_dict: Dict) -> Dict:
    """Extract features for MLB moneyline prediction."""
    try:
        comp = game_dict.get("competitions", [{}])[0]
        features = {
            "home_team_id": comp.get("competitors", [{}])[0].get("team", {}).get("id", ""),
        }
        return features
    except Exception as e:
        print(f"Error extracting MLB moneyline features: {e}")
        return {}
