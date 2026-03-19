from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = "./kalshi-key.pem"
    kalshi_base_url: str = "https://api.elections.kalshi.com"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_origin_regex: str = ""

    db_path: str = "data/kalshi.db"

    # Wizard lookup scoring settings
    live_lookup_path: str = "prediction-market-analysis/output/conditional_lookup.parquet"
    live_lookup_min_samples: int = 500
    live_lookup_min_markets: int = 20
    live_lookup_fee_cents: float = 7.0

    @property
    def db_url(self) -> str:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{path}"

    @property
    def kalshi_api_path_prefix(self) -> str:
        return "/trade-api/v2"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def live_lookup_abs_path(self) -> str:
        path = Path(self.live_lookup_path)
        if path.is_absolute():
            return str(path)
        return str((Path(__file__).resolve().parents[2] / path).resolve())


settings = Settings()
