# Moose Picks ML - Sports Betting Model Pipeline

A complete machine learning pipeline for training, evaluating, and deploying sports betting models for NFL and NHL. The system predicts win probabilities, spread/puck line covers, totals, and projected scores, then calculates edges against market odds.

## Features

- **Comprehensive Feature Engineering**: Team strength metrics, rolling averages, rest days, home/away splits, head-to-head records
- **Multiple Market Types**: Moneyline, spread/puck line, totals, and score projection models
- **Edge Calculation**: Converts odds to implied probabilities and computes betting edges
- **Model Evaluation**: Log loss, Brier score, calibration, ROI by edge bucket
- **Export Pipeline**: Standardized CSV/JSON output for integration with Lovable
- **Versioned Models**: Models are saved with timestamps and can be retrained easily
- **Prediction Tracking**: Automatically stores and settles predictions in database

## Project Structure

```
moose-picks-api/
├── app/
│   ├── data_loader.py          # Historical data loading with season/week splits
│   ├── training/
│   │   ├── features.py         # Feature engineering (NFL/NHL)
│   │   ├── pipeline.py         # Model training pipeline
│   │   └── evaluate.py         # Evaluation metrics
│   ├── prediction/
│   │   ├── storage.py          # Store predictions in database
│   │   ├── settling.py         # Settle predictions against results
│   │   └── engine.py           # Prediction engine
│   ├── utils/
│   │   ├── odds.py             # Odds conversion and edge calculation
│   │   └── export.py           # Export pipeline for Lovable
│   ├── models/                 # Database models
│   ├── espn_client/            # ESPN API client
│   └── main.py                 # FastAPI application
├── scripts/
│   ├── train_all.py            # Orchestration script for training
│   ├── export_predictions.py   # Generate predictions for upcoming games
│   └── daily_automation.py    # Daily workflow automation
├── tests/                      # Unit tests
├── config.yaml                 # Training configuration
└── requirements.txt            # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (optional, uses defaults if not present):

```env
DATABASE_URL=sqlite:///./moose.db
MODEL_DIR=models/
ODDS_API_KEY=your_odds_api_key_here
```

### 3. Configure Training

Edit `config.yaml` to specify:
- Training seasons for each sport
- Train/validation split strategy
- Model hyperparameters
- Feature engineering options
- Edge thresholds

## Usage

### Loading Historical Data

The system uses data stored in the SQLite database. To populate the database with historical games, use the provided script:

**Load past 3 seasons + current season for NFL and NHL:**
```bash
python scripts/load_historical_data.py
```

**Load for specific sports:**
```bash
python scripts/load_historical_data.py --sports NFL NHL
```

**Load for a single sport:**
```bash
python scripts/load_historical_data.py --sport NFL
```

**Load specific seasons:**
```bash
python scripts/load_historical_data.py --sports NFL --seasons 2021 2022 2023 2024
```

The script will:
- Automatically determine the current season
- Fetch past 3 seasons + current season
- Only store games with final scores (completed games)
- Handle duplicates (upserts existing games)

**Note:** This may take a while as it fetches games day-by-day. The script prints progress as it goes.

### Training Models

Train all models for all configured sports and markets:

```bash
python scripts/train_all.py --config config.yaml
```

This will:
1. Load historical data for configured seasons
2. Build features for each game
3. Split into train/validation sets
4. Train models for:
   - Moneyline (classification)
   - Spread/Puck Line (classification)
   - Totals (classification)
   - Score Projection (regression, separate models for home/away)
5. Evaluate models and print metrics
6. Save models to `models/` directory with version tags

**Output:**
- Models saved as `{sport}_{market}_{timestamp}.pkl`
- Score models saved as `{sport}_score_{home|away}_{timestamp}.pkl`
- Training metrics printed to console

### Generating Predictions

Generate predictions for upcoming games and export to Lovable format:

```bash
python scripts/export_predictions.py \
    --sport NFL \
    --date 2024-12-15 \
    --config config.yaml \
    --output-dir exports \
    --min-edge 0.05
```

This will:
1. Load games from database for the specified date
2. Load latest trained models
3. Build features and generate predictions
4. Calculate edges for all markets
5. Store predictions in database
6. Export to CSV and JSON in standardized format

**Output Format:**

The exported CSV/JSON contains one row per market/side combination with:
- `game_id`, `league`, `season`, `date`
- `home_team`, `away_team`
- `market_type` (moneyline, spread, totals)
- `side` (home, away, over, under, favorite, underdog)
- `line`, `price` (odds)
- `model_prob`, `implied_prob`, `edge`
- `proj_home_score`, `proj_away_score`

### Daily Automation

Run the complete daily workflow (settle predictions, fetch games, train, predict):

```bash
python scripts/daily_automation.py
```

Options:
- `--no-train` - Skip model training
- `--no-predict` - Skip prediction generation
- `--sports NFL NHL` - Specify sports
- `--min-edge 0.05` - Minimum edge threshold

### Running Tests

```bash
pytest tests/
```

Tests cover:
- Odds conversion (American odds ↔ implied probability)
- Edge calculation
- End-to-end pipeline with sample data

## Configuration

### config.yaml

Key configuration options:

```yaml
# Seasons to use for training
training_seasons:
  NFL: [2020, 2021, 2022, 2023, 2024]
  NHL: [2020, 2021, 2022, 2023, 2024]

# Train/validation split
split_strategy: "season"  # or "week" or "random"
validation_seasons: 1

# Minimum edge to surface a play
min_edge_threshold: 0.05

# Model hyperparameters
models:
  moneyline:
    algorithm: "xGBoost"  # or "gradient_boosting"
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1

# Feature engineering
features:
  rolling_window_games: 10
  include_rest_days: true
  include_home_away_splits: true
  include_head_to_head: true
```

## Model Architecture

### Feature Engineering

Features include:
- Team strength metrics (win rate, point differential)
- Rolling averages (last N games)
- Rest days between games
- Home/away splits
- Head-to-head records
- Opponent strength adjustments

### Algorithms

Supports:
- **XGBoost** (default, specified in config)
- **Gradient Boosting** (sklearn fallback)

The algorithm is selected from `config.yaml` - the `algorithm` setting is now properly used.

## Prediction Tracking

The system automatically:
1. **Stores predictions** when generated
2. **Settles predictions** when games complete
3. **Tracks performance** (wins, losses, pushes, PnL)

All predictions are stored in the database with:
- Model probabilities
- Model version used
- Settlement status and results
- Profit/loss calculations

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions to Railway with Lovable integration.

## License

MIT
