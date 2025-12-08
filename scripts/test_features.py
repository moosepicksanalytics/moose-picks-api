"""
Quick test script to verify feature engineering works correctly.
Run this before training models to catch any issues.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure output is flushed immediately
sys.stdout.reconfigure(line_buffering=True)

from app.data_loader import load_games_for_sport
from app.training.features import build_features, get_feature_columns
from app.training.pipeline import load_config


def test_features(sport: str = "NFL", market: str = "moneyline"):
    """
    Test feature engineering with a small sample of data.
    
    Args:
        sport: Sport to test (NFL or NHL)
        market: Market type to test
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Testing Feature Engineering for {sport} - {market}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Ensure database tables exist
    from app.database import engine
    from app.models.db_models import Base
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables initialized", flush=True)
    except Exception as e:
        print(f"⚠️  Warning: Could not create tables: {e}", flush=True)
    
    # Quick check: do we have any data at all?
    from app.database import SessionLocal
    from app.models.db_models import Game
    db = SessionLocal()
    try:
        total_games = db.query(Game).filter(Game.sport == sport).count()
        print(f"Total {sport} games in database: {total_games}", flush=True)
        if total_games == 0:
            print(f"\n⚠️  No {sport} games found in database!", flush=True)
            print("   You need to fetch some games first.", flush=True)
            print("   See NEXT_STEPS.md for instructions.", flush=True)
            print("\n   Quick fix - fetch recent games:", flush=True)
            print(f"   python -c \"from datetime import datetime, timedelta; from app.espn_client.fetcher import fetch_games_for_date; from app.espn_client.parser import parse_and_store_games; date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'); games = fetch_games_for_date('{sport}', date); parse_and_store_games('{sport}', games) if games else print('No games found')\"", flush=True)
            return False
    except Exception as e:
        print(f"❌ Database error: {e}", flush=True)
        print("   Make sure your database is set up correctly.", flush=True)
        return False
    finally:
        db.close()
    
    # Load config
    config = load_config()
    
    # Load a small sample of data (just one season)
    training_seasons = config.get("training_seasons", {}).get(sport, [2023])
    if not training_seasons:
        print(f"❌ No training seasons configured for {sport}")
        return False
    
    # Load one season
    season = training_seasons[-1]  # Use most recent season
    print(f"Loading {sport} data for season {season}...")
    
    try:
        from app.training.pipeline import get_season_date_range
        start_date, end_date = get_season_date_range(sport, season)
        print(f"  Date range: {start_date} to {end_date}")
    except Exception as e:
        print(f"❌ Error getting season date range: {e}")
        # Try loading without date range
        start_date, end_date = None, None
    
    df = load_games_for_sport(
        sport,
        start_date=start_date,
        end_date=end_date,
        min_games_per_team=5  # Lower threshold for testing
    )
    
    if df.empty:
        print(f"❌ No games found for {sport} season {season}")
        print("   Make sure you have data in your database.")
        print("   You can fetch games using:")
        print(f"   python -c \"from app.espn_client.fetcher import fetch_games_for_date; from app.espn_client.parser import parse_and_store_games; from datetime import datetime; date = datetime.now().strftime('%Y-%m-%d'); games = fetch_games_for_date('{sport}', date); parse_and_store_games('{sport}', games) if games else print('No games')\"")
        return False
    
    print(f"✓ Loaded {len(df)} games")
    
    # Build features
    rolling_window = config.get("features", {}).get("rolling_window_games", 10)
    include_rest_days = config.get("features", {}).get("include_rest_days", True)
    include_h2h = config.get("features", {}).get("include_head_to_head", True)
    
    print(f"\nBuilding features...")
    print(f"  - Rolling window: {rolling_window}")
    print(f"  - Rest days: {include_rest_days}")
    print(f"  - Head-to-head: {include_h2h}")
    
    try:
        df_features = build_features(
            df,
            sport=sport,
            market=market,
            rolling_window=rolling_window,
            include_rest_days=include_rest_days,
            include_h2h=include_h2h
        )
        
        print(f"✓ Features built successfully")
        print(f"  - Original columns: {len(df.columns)}")
        print(f"  - Total columns after features: {len(df_features.columns)}")
        
        # Get expected feature columns
        expected_features = get_feature_columns(sport, market)
        available_features = [f for f in expected_features if f in df_features.columns]
        missing_features = [f for f in expected_features if f not in df_features.columns]
        
        print(f"\nFeature Summary:")
        print(f"  - Expected features: {len(expected_features)}")
        print(f"  - Available features: {len(available_features)}")
        print(f"  - Missing features: {len(missing_features)}")
        
        if missing_features:
            print(f"\n⚠️  Missing features (first 10):")
            for feat in missing_features[:10]:
                print(f"     - {feat}")
            if len(missing_features) > 10:
                print(f"     ... and {len(missing_features) - 10} more")
        
        # Check for NaN values
        numeric_cols = df_features.select_dtypes(include=[pd.np.number]).columns
        nan_counts = df_features[numeric_cols].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        
        if len(cols_with_nans) > 0:
            print(f"\n⚠️  Columns with NaN values: {len(cols_with_nans)}")
            print(f"   (This is normal for early games in the dataset)")
        else:
            print(f"\n✓ No NaN values in numeric columns")
        
        # Show sample features
        print(f"\nSample feature values (first game):")
        sample_features = available_features[:10]
        for feat in sample_features:
            val = df_features[feat].iloc[0] if len(df_features) > 0 else None
            print(f"  - {feat}: {val}")
        
        print(f"\n{'='*60}")
        print(f"✅ Feature engineering test PASSED")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error building features:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test both sports and a few markets
    sports = ["NFL", "NHL"]
    markets = ["moneyline", "spread", "totals"]
    
    results = {}
    for sport in sports:
        for market in markets:
            key = f"{sport}_{market}"
            results[key] = test_features(sport, market)
            print("\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for key, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {key}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✅ All tests passed! You're ready to train models.")
    else:
        print(f"\n⚠️  Some tests failed. Review errors above before training.")
