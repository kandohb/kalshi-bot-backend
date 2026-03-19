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

    db_path: str = "data/kalshi.db"

    max_position_per_market: int = 5
    max_open_orders: int = 20
    max_capital_cents: int = 2000  # $20 hard cap

    strategy_scheduler_enabled: bool = False
    strategy_interval_minutes: int = 15
    dry_run: bool = True

    # Only consider markets expiring within this many days (for short-term scalping)
    max_expiry_days: int = 7

    exclude_categories: list[str] = ["Finance"]
    require_liquidity: bool = True

    exclude_ticker_prefixes: list[str] = [
        "KXMVECROSS",
        "KXMVESPORTS",
        "KXMVEENTERTAINMENT",
        "KXMVE",
    ]

    # High-gap categories from the paper (maker-taker gap in pp):
    # Entertainment 4.79, Media 7.28, World Events 7.32, Weather 2.57,
    # Crypto 2.69, Sports 2.23. Finance 0.17 (excluded above).
    high_gap_categories: list[str] = [
        "Entertainment", "Media", "World Events",
        "Weather", "Crypto", "Sports",
    ]

    # ── Live Scalper bot settings ──────────────────────────────────────
    live_max_expiry_hours: int = 12
    live_min_volume: int = 50
    live_max_spread: int = 5
    live_max_orders_per_cycle: int = 5
    live_max_capital_per_cycle: int = 500   # cents ($5)
    live_max_capital_per_event: int = 200   # cents ($2)
    live_cooldown_minutes: int = 30
    live_scheduler_interval_minutes: int = 10
    live_scheduler_enabled: bool = False
    live_dry_run: bool = True
    live_exclude_crypto: bool = True
    live_preferred_categories: list[str] = ["Sports"]
    live_prioritize_in_play: bool = True
    live_use_lookup_scoring: bool = True
    live_lookup_path: str = "prediction-market-analysis/output/conditional_lookup.parquet"
    live_lookup_min_samples: int = 500
    live_lookup_min_markets: int = 20
    live_lookup_fee_cents: float = 7.0
    live_lookup_min_edge_cents: float = 1.0
    live_lookup_min_prob_edge: float = 0.10
    live_lookup_allowed_hours_buckets: list[str] = ["1-3h", "3-6h"]
    live_lookup_focus_categories: list[str] = []
    live_market_fetch_target: int = 400
    live_market_fetch_backoff_base_ms: int = 300

    @property
    def db_url(self) -> str:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{path}"

    @property
    def kalshi_api_path_prefix(self) -> str:
        return "/trade-api/v2"

    @property
    def live_lookup_abs_path(self) -> str:
        path = Path(self.live_lookup_path)
        if path.is_absolute():
            return str(path)
        return str((Path(__file__).resolve().parents[2] / path).resolve())


settings = Settings()
