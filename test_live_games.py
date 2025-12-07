from app.odds_client.fetcher import fetch_odds_for_sport

odds = fetch_odds_for_sport('americanfootball_nfl')
print(f"Found {len(odds)} NFL games\n")

if odds:
    # Get first 3 games
    for i, game in enumerate(odds[:3]):
        print(f"{i+1}. {game['away_team']} @ {game['home_team']}")
        print(f"   ID: {game['id']}")
        print(f"   Time: {game['commence_time']}\n")
