from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./moose.db"
    MODEL_DIR: str = "models/"
    ODDS_API_KEY: str = ""
    
    # Security settings
    API_KEYS: str = ""  # Comma-separated list of API keys for authentication
    ALLOWED_ORIGINS: str = "*"  # Comma-separated list of allowed CORS origins, or "*" for all
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60  # Requests per minute per IP
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS into a list."""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]
    
    @property
    def api_keys_list(self) -> List[str]:
        """Parse API_KEYS into a list."""
        if not self.API_KEYS:
            return []
        return [key.strip() for key in self.API_KEYS.split(",") if key.strip()]

    
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
