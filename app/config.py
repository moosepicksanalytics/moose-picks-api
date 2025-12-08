from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./moose.db"
    MODEL_DIR: str = "models/"
    ODDS_API_KEY: str = ""

    
    SPORTS_CONFIG: dict = {
        "NFL": {
            "league_id": "football",
            "markets": ["moneyline", "spread", "over_under"],
            "min_training_games": 100,
        },
        "NBA": {
            "league_id": "basketball",
            "markets": ["moneyline", "spread", "over_under"],
            "min_training_games": 200,
        },
        "NHL": {
            "league_id": "hockey",
            "markets": ["moneyline", "spread", "over_under"],
            "min_training_games": 100,
        },
        "MLB": {
            "league_id": "baseball",
            "markets": ["moneyline", "run_line"],
            "min_training_games": 150,
        },
    }

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
