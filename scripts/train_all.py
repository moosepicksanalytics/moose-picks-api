"""
Orchestration script to train all models for all sports and markets.
Loads data, builds features, trains models, evaluates, and saves.
"""
import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.pipeline import train_model_for_market, train_score_projection_models, load_config
from app.config import settings


def train_all_models(config_path: str = "config.yaml"):
    """
    Train all models for all configured sports and markets.
    
    Args:
        config_path: Path to config.yaml
    """
    config = load_config(config_path)
    
    # Get sports from config or settings
    sports = list(config.get("training_seasons", {}).keys())
    if not sports:
        sports = ["NFL", "NHL"]  # Default
    
    markets = ["moneyline", "spread", "totals"]
    
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Moose Picks ML - Training All Models")
    print("=" * 60)
    print(f"Model Version: {model_version}")
    print(f"Sports: {', '.join(sports)}")
    print(f"Markets: {', '.join(markets)}")
    print("=" * 60)
    print()
    
    results = {}
    
    # Train classification models
    for sport in sports:
        print(f"\n{'='*60}")
        print(f"Training {sport} Models")
        print(f"{'='*60}\n")
        
        results[sport] = {}
        
        for market in markets:
            print(f"\n--- Training {sport} {market} ---")
            try:
                result = train_model_for_market(
                    sport=sport,
                    market=market,
                    config=config,
                    model_version=model_version
                )
                
                if result.get("success"):
                    print(f"✓ {sport} {market} trained successfully")
                    print(f"  Train samples: {result.get('n_train_samples')}")
                    print(f"  Val samples: {result.get('n_val_samples')}")
                    
                    eval_metrics = result.get("eval_metrics", {})
                    if "log_loss" in eval_metrics:
                        print(f"  Log Loss: {eval_metrics['log_loss']:.4f}")
                    if "brier_score" in eval_metrics:
                        print(f"  Brier Score: {eval_metrics['brier_score']:.4f}")
                    if "accuracy" in eval_metrics:
                        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                    if "rmse" in eval_metrics:
                        print(f"  RMSE: {eval_metrics['rmse']:.4f}")
                    
                    results[sport][market] = result
                else:
                    print(f"✗ {sport} {market} failed: {result.get('error')}")
                    results[sport][market] = result
                    
            except Exception as e:
                print(f"✗ Error training {sport} {market}: {e}")
                import traceback
                traceback.print_exc()
                results[sport][market] = {"success": False, "error": str(e)}
        
        # Train score projection models
        print(f"\n--- Training {sport} score projection ---")
        try:
            score_result = train_score_projection_models(
                sport=sport,
                config=config,
                model_version=model_version
            )
            
            if score_result.get("success"):
                print(f"✓ {sport} score projection trained successfully")
                results[sport]["score_projection"] = score_result
            else:
                print(f"✗ {sport} score projection failed: {score_result.get('error')}")
                results[sport]["score_projection"] = score_result
                
        except Exception as e:
            print(f"✗ Error training {sport} score projection: {e}")
            import traceback
            traceback.print_exc()
            results[sport]["score_projection"] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    for sport in sports:
        print(f"\n{sport}:")
        for market in markets + ["score_projection"]:
            if market in results.get(sport, {}):
                result = results[sport][market]
                status = "✓" if result.get("success") else "✗"
                print(f"  {status} {market}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml"
    )
    
    args = parser.parse_args()
    train_all_models(args.config)
