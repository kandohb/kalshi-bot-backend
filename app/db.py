"""SQLAlchemy models and async engine for local persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String, default="bias_scalper", index=True)
    order_id = Column(String, unique=True, index=True)
    client_order_id = Column(String, unique=True)
    ticker = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    action = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Fill(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String, index=True)
    ticker = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)
    filled_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class StrategyRun(Base):
    __tablename__ = "strategy_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String, default="bias_scalper", index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    markets_scanned = Column(Integer, default=0)
    orders_placed = Column(Integer, default=0)
    errors = Column(Text, default="")
    status = Column(String, default="running")
    status_detail = Column(String, default="")
    status_timeline = Column(Text, default="[]")


class Candidate(Base):
    """Trade suggestion persisted between scan and user approval."""
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String, default="bias_scalper", index=True)
    scan_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String, nullable=False)
    title = Column(String, default="")
    side = Column(String, nullable=False)     # "yes" or "no"
    action = Column(String, default="buy")
    price = Column(Integer, nullable=False)   # yes_price for the API
    cost = Column(Integer, nullable=False)    # actual cost in cents
    count = Column(Integer, default=1)
    bias_type = Column(String, nullable=False)  # "longshot_no", "favorite_yes", "asymmetry"
    category = Column(String, default="")
    expiry_time = Column(String, default="")
    reason = Column(String, default="")
    ai_explanation = Column(Text, default="")
    status = Column(String, default="pending")  # pending, approved, rejected, placed
    yes_bid = Column(Integer, default=0)
    yes_ask = Column(Integer, default=0)
    volume = Column(Integer, default=0)
    predicted_win_rate = Column(Float, nullable=True)
    implied_probability = Column(Float, nullable=True)
    predicted_edge = Column(Float, nullable=True)
    net_edge_cents = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=True)
    lookup_key = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_strategy_run_columns(conn)
        await _migrate_candidate_columns(conn)


async def _migrate_strategy_run_columns(conn) -> None:
    """Add new strategy run columns for existing SQLite DBs."""
    result = await conn.execute(text("PRAGMA table_info(strategy_runs)"))
    existing = {row[1] for row in result.fetchall()}
    needed: list[tuple[str, str]] = [
        ("status_detail", "TEXT DEFAULT ''"),
        ("status_timeline", "TEXT DEFAULT '[]'"),
    ]
    for name, sql_type in needed:
        if name in existing:
            continue
        await conn.execute(text(f"ALTER TABLE strategy_runs ADD COLUMN {name} {sql_type}"))


async def _migrate_candidate_columns(conn) -> None:
    """Add new candidate telemetry columns for existing SQLite DBs."""
    result = await conn.execute(text("PRAGMA table_info(candidates)"))
    existing = {row[1] for row in result.fetchall()}
    needed: list[tuple[str, str]] = [
        ("predicted_win_rate", "FLOAT"),
        ("implied_probability", "FLOAT"),
        ("predicted_edge", "FLOAT"),
        ("net_edge_cents", "FLOAT"),
        ("sample_size", "INTEGER"),
        ("lookup_key", "TEXT DEFAULT ''"),
    ]
    for name, sql_type in needed:
        if name in existing:
            continue
        await conn.execute(text(f"ALTER TABLE candidates ADD COLUMN {name} {sql_type}"))
