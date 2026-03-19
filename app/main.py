"""FastAPI application for the Kalshi bias-exploiter bot."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.routes import health, orders, fills, pnl, markets, fund
from app.routes import strategy as strategy_routes
from app.routes import live_scalper as live_scalper_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database")
    await init_db()

    if settings.strategy_scheduler_enabled:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.strategy import scan_markets

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            scan_markets,
            "interval",
            minutes=settings.strategy_interval_minutes,
        )
        scheduler.start()
        logger.info("Strategy scheduler started (every %d min)", settings.strategy_interval_minutes)

    if settings.live_scheduler_enabled:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.bots.live_scalper import auto_run

        live_sched = AsyncIOScheduler()
        live_sched.add_job(
            auto_run,
            "interval",
            minutes=settings.live_scheduler_interval_minutes,
        )
        live_sched.start()
        logger.info("Live scalper scheduler started (every %d min)", settings.live_scheduler_interval_minutes)

    yield


app = FastAPI(
    title="Kalshi Bias Exploiter",
    description="Maker-only bias strategy bot for Kalshi prediction markets",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(orders.router)
app.include_router(fills.router)
app.include_router(pnl.router)
app.include_router(strategy_routes.router)
app.include_router(markets.router)
app.include_router(live_scalper_routes.router)
app.include_router(fund.router)
