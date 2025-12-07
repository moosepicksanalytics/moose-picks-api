import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import Game
from app.training.pipeline import train_model_for_market
from app.config import settings

def train_all():
    """Train models for all sports/markets"""
    
    db = SessionLocal()
    
    # Check what we have
    games = db.query(Game).all()
    print(f"Total games in DB: {len(games)}")
    
    if len(games) < 50:
        print("⚠️  Not enough games to train (need 50+)")
        print("Fetching more data first...")
        
        from app.espn_client.fetcher import fetch_games_for_date
        from app.espn_client.parser import parse_and_store_games
        
        # Fetch from 7 different game days
        dates = ['20241208', '20241201', '20241124', '20241117', '20241110', '20241103', '20241027']
        total = 0
        for d in dates:
            games_data = fetch_games_for_date('NFL', d)
            if games_data:
                parse_and_store_games('NFL', games_data)
                total += len(games_data)
                print(f'  {d}: {len(games_data)} games')
        
        print(f'✓ Fetched {total} new games')
    
    # Train models
    sports = ["NFL"]
    markets = ["moneyline", "spread", "over_under"]
    
    for sport in sports:
        for market in markets:
            print(f"\nTraining {sport} {market}...")
            result = train_model_for_market(sport, market)
            if result:
                print(f"✓ {sport} {market} trained!")
            else:
                print(f"✗ {sport} {market} failed")

if __name__ == "__main__":
    train_all()
