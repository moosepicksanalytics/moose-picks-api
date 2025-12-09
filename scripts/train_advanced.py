"""
Advanced training script using the new pipeline with ensemble and Optuna.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.advanced_pipeline import train_advanced_model


def train_all_sports_advanced(
    sports: list = ["NFL", "NHL", "NBA", "MLB"],
    markets: list = ["moneyline", "spread", "totals"],
    use_ensemble: bool = True,
    use_optuna: bool = True,
    n_trials: int = 100
):
    """
    Train advanced models for all sports and markets.
    
    Args:
        sports: List of sports to train
        markets: List of markets to train
        use_ensemble: Whether to use stacking ensemble
        use_optuna: Whether to use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials per algorithm
    """
    results = {}
    
    for sport in sports:
        print(f"\n{'='*60}")
        print(f"Training {sport} Models (Advanced)")
        print(f"{'='*60}\n")
        
        results[sport] = {}
        
        for market in markets:
            print(f"\n--- Training {sport} {market} ---")
            try:
                result = train_advanced_model(
                    sport=sport,
                    market=market,
                    algorithms=["xgb", "catboost", "lightgbm"],
                    use_ensemble=use_ensemble,
                    use_optuna=use_optuna,
                    n_trials=n_trials
                )
                
                if result.get("success"):
                    print(f"✓ {sport} {market} trained successfully")
                    print(f"  Train samples: {result.get('n_train_samples')}")
                    print(f"  Val samples: {result.get('n_val_samples')}")
                    
                    eval_metrics = result.get("eval_metrics", {})
                    
                    # Priority metrics
                    if "roi" in eval_metrics and eval_metrics["roi"]:
                        roi = eval_metrics["roi"]
                        print(f"  ROI: {roi.get('roi', 'N/A'):.2f}%")
                        print(f"  Value bets: {roi.get('value_bets', 0)}")
                        print(f"  Win rate: {roi.get('win_rate', 0):.2%}")
                    
                    if "ece" in eval_metrics:
                        print(f"  ECE (Calibration): {eval_metrics['ece']:.4f}")
                    
                    if "value_bet_accuracy" in eval_metrics:
                        print(f"  Value bet accuracy: {eval_metrics['value_bet_accuracy']:.4f}")
                    
                    if "accuracy" in eval_metrics:
                        print(f"  Overall accuracy: {eval_metrics['accuracy']:.4f}")
                    
                    if "log_loss" in eval_metrics:
                        print(f"  Log Loss: {eval_metrics['log_loss']:.4f}")
                    
                    results[sport][market] = result
                else:
                    print(f"✗ {sport} {market} failed: {result.get('error')}")
                    results[sport][market] = result
                    
            except Exception as e:
                print(f"✗ Error training {sport} {market}: {e}")
                import traceback
                traceback.print_exc()
                results[sport][market] = {"success": False, "error": str(e)}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train advanced ML models")
    parser.add_argument("--sports", nargs="+", default=["NFL", "NHL", "NBA", "MLB"],
                        help="Sports to train")
    parser.add_argument("--markets", nargs="+", default=["moneyline", "spread", "totals"],
                        help="Markets to train")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable ensemble (train single best model)")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Disable Optuna optimization (use default hyperparameters)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials per algorithm")
    
    args = parser.parse_args()
    
    results = train_all_sports_advanced(
        sports=args.sports,
        markets=args.markets,
        use_ensemble=not args.no_ensemble,
        use_optuna=not args.no_optuna,
        n_trials=args.n_trials
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Summary
    for sport in args.sports:
        if sport in results:
            print(f"\n{sport}:")
            for market in args.markets:
                if market in results[sport]:
                    result = results[sport][market]
                    if result.get("success"):
                        metrics = result.get("eval_metrics", {})
                        roi = metrics.get("roi", {})
                        if roi:
                            print(f"  {market}: ROI={roi.get('roi', 0):.2f}%, "
                                  f"ECE={metrics.get('ece', 0):.4f}, "
                                  f"Accuracy={metrics.get('accuracy', 0):.4f}")
