from app.odds_client.fetcher import fetch_odds_for_sport, extract_moneyline_odds

odds = fetch_odds_for_sport('americanfootball_nfl')
print(f'Fetched {len(odds)} games')

if odds:
    event = odds[0]
    print(f'Game: {event.get("home_team")} vs {event.get("away_team")}')
    home_odds, away_odds = extract_moneyline_odds(event)
    print(f'Home odds: {home_odds}, Away odds: {away_odds}')
