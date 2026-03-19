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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database")
    await init_db()

    yield


app = FastAPI(
    title="Kalshi Bias Exploiter",
    description="Maker-only bias strategy bot for Kalshi prediction markets",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=settings.cors_origin_regex or None,
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
app.include_router(fund.router)
