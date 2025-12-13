from app.database import SessionLocal
from app.models.db_models import Prediction, Game
from datetime import datetime, timedelta


def settle_moneyline_prediction(pred: Prediction, game: Game) -> tuple:
    """
    Settle a moneyline prediction.
    
    Returns:
        (result, pnl) tuple
    """
    if pred.home_win_prob is None:
        return "push", 0.0
    
    home_win = game.home_score > game.away_score
    predicted_home_win = pred.home_win_prob > 0.5
    
    if home_win == predicted_home_win:
        result = "win"
        # Calculate PnL based on odds (simplified: assume -110 odds = 0.909 implied prob)
        # For win: +0.909 units, for loss: -1.0 units
        pnl = 0.909  # Approximate for -110 odds
    else:
        result = "loss"
        pnl = -1.0
    
    return result, pnl


def settle_spread_prediction(pred: Prediction, game: Game) -> tuple:
    """
    Settle a spread prediction.
    
    Returns:
        (result, pnl) tuple
    """
    if pred.spread_cover_prob is None or game.spread is None:
        return "push", 0.0
    
    # Calculate actual margin
    actual_margin = game.home_score - game.away_score
    
    # Spread is from home team's perspective (negative means home is underdog)
    # If spread is -3.5, home needs to win by 4+ to cover
    # If spread is +3.5, home can lose by 3 or less to cover
    
    if game.spread < 0:
        # Home is favorite (spread is negative, e.g., -3.5)
        # Home covers if they win by more than |spread|
        home_covers = actual_margin > abs(game.spread)
    else:
        # Home is underdog (spread is positive, e.g., +3.5)
        # Home covers if they lose by less than spread, or win
        home_covers = actual_margin > -game.spread
    
    predicted_home_covers = pred.spread_cover_prob > 0.5
    
    if home_covers == predicted_home_covers:
        result = "win"
        pnl = 0.909  # Approximate for -110 odds
    elif actual_margin == game.spread or abs(actual_margin - game.spread) < 0.1:
        result = "push"
        pnl = 0.0
    else:
        result = "loss"
        pnl = -1.0
    
    return result, pnl


def settle_totals_prediction(pred: Prediction, game: Game) -> tuple:
    """
    Settle a totals (over/under) prediction.
    
    Returns:
        (result, pnl) tuple
    """
    if pred.over_prob is None or game.over_under is None:
        return "push", 0.0
    
    total_score = game.home_score + game.away_score
    predicted_over = pred.over_prob > 0.5
    actual_over = total_score > game.over_under
    
    if actual_over == predicted_over:
        result = "win"
        pnl = 0.909  # Approximate for -110 odds
    elif abs(total_score - game.over_under) < 0.1:
        result = "push"
        pnl = 0.0
    else:
        result = "loss"
        pnl = -1.0
    
    return result, pnl


def settle_predictions(date_str: str, sport: str = None):
    """
    Compare predictions from date_str against final scores and settle them.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        sport: Optional sport filter (NFL, NHL, etc.)
    """
    db = SessionLocal()
    
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Query unsettled predictions from the target date
        # Use timedelta to properly handle month/year boundaries
        start_datetime = datetime.combine(target_date, datetime.min.time())
        end_datetime = start_datetime + timedelta(days=1)
        
        query = db.query(Prediction).filter(
            Prediction.settled == False,
            Prediction.predicted_at >= start_datetime,
            Prediction.predicted_at < end_datetime
        )
        
        if sport:
            query = query.filter(Prediction.sport == sport)
        
        predictions = query.all()
        
        if not predictions:
            print(f"No unsettled predictions found for {date_str}")
            return
        
        settled_count = 0
        win_count = 0
        loss_count = 0
        push_count = 0
        
        for pred in predictions:
            game = db.query(Game).filter(Game.id == pred.game_id).first()
            if not game or game.home_score is None or game.away_score is None:
                continue
            
            # Determine result based on market type
            if pred.market == "moneyline":
                result, pnl = settle_moneyline_prediction(pred, game)
            elif pred.market == "spread":
                result, pnl = settle_spread_prediction(pred, game)
            elif pred.market == "totals":
                result, pnl = settle_totals_prediction(pred, game)
            else:
                # Unknown market type
                continue
            
            # Update prediction
            pred.settled = True
            pred.result = result  # Keep for backward compatibility
            pred.settled_result = result  # New column
            pred.pnl = pnl
            pred.settled_at = datetime.now()
            
            settled_count += 1
            if result == "win":
                win_count += 1
            elif result == "loss":
                loss_count += 1
            else:
                push_count += 1
        
        db.commit()
        
        print(f"✓ Settled {settled_count} predictions for {date_str}")
        print(f"  Wins: {win_count}, Losses: {loss_count}, Pushes: {push_count}")
        
        if settled_count > 0:
            total_pnl = sum(p.pnl for p in predictions if p.settled and p.pnl is not None)
            print(f"  Total PnL: {total_pnl:.2f} units")
        
    except Exception as e:
        print(f"Error settling predictions: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
