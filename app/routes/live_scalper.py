"""Routes for the Live Scalper bot."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.config import settings
from app.db import StrategyRun, Candidate, async_session
from app.bots.live_scalper import auto_run, BOT_ID, reload_lookup_scorer, get_trade_proposal_context
from app.kalshi_client import kalshi

router = APIRouter(prefix="/api/live-scalper")
logger = logging.getLogger(__name__)

_scheduler_ref: Any = None


MUTABLE_SETTINGS: dict[str, type] = {
    "live_dry_run": bool,
    "live_max_expiry_hours": int,
    "live_min_volume": int,
    "live_max_spread": int,
    "live_max_orders_per_cycle": int,
    "live_max_capital_per_cycle": int,
    "live_max_capital_per_event": int,
    "live_cooldown_minutes": int,
    "live_scheduler_interval_minutes": int,
    "live_use_lookup_scoring": bool,
    "live_lookup_min_samples": int,
    "live_lookup_min_markets": int,
    "live_lookup_fee_cents": float,
    "live_lookup_min_edge_cents": float,
    "live_lookup_min_prob_edge": float,
}

MUTABLE_LIST_SETTINGS: dict[str, type] = {
    "live_lookup_allowed_hours_buckets": str,
    "live_lookup_focus_categories": str,
}


def _config_snapshot() -> dict[str, Any]:
    return {
        "max_expiry_hours": settings.live_max_expiry_hours,
        "min_volume": settings.live_min_volume,
        "max_spread": settings.live_max_spread,
        "max_orders_per_cycle": settings.live_max_orders_per_cycle,
        "max_capital_per_cycle": settings.live_max_capital_per_cycle,
        "max_capital_per_event": settings.live_max_capital_per_event,
        "cooldown_minutes": settings.live_cooldown_minutes,
        "interval_minutes": settings.live_scheduler_interval_minutes,
        "use_lookup_scoring": settings.live_use_lookup_scoring,
        "lookup_path": settings.live_lookup_path,
        "lookup_min_samples": settings.live_lookup_min_samples,
        "lookup_min_markets": settings.live_lookup_min_markets,
        "lookup_fee_cents": settings.live_lookup_fee_cents,
        "lookup_min_edge_cents": settings.live_lookup_min_edge_cents,
        "lookup_min_prob_edge": settings.live_lookup_min_prob_edge,
        "lookup_allowed_hours_buckets": settings.live_lookup_allowed_hours_buckets,
        "lookup_focus_categories": settings.live_lookup_focus_categories,
    }


def _timeline(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


@router.post("/run")
async def trigger_run():
    """Execute one live scalper cycle (scan → rank → place)."""
    run_id, log = await auto_run()
    return {"run_id": run_id, "log": log.to_dict()}


@router.get("/status")
async def get_status():
    """Latest run info and scheduler state."""
    async with async_session() as session:
        result = await session.execute(
            select(StrategyRun)
            .where(StrategyRun.bot_id == BOT_ID)
            .order_by(StrategyRun.id.desc())
            .limit(1)
        )
        run = result.scalar_one_or_none()

    return {
        "bot_id": BOT_ID,
        "scheduler_running": _scheduler_ref is not None,
        "dry_run": settings.live_dry_run,
        "config": _config_snapshot(),
        "last_run": {
            "id": run.id,
            "status": run.status,
            "status_detail": run.status_detail or "",
            "status_timeline": _timeline(run.status_timeline),
            "markets_scanned": run.markets_scanned,
            "orders_placed": run.orders_placed,
            "errors": run.errors or None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        } if run else None,
    }


@router.get("/runs")
async def list_runs(limit: int = 20):
    """Recent run history."""
    async with async_session() as session:
        result = await session.execute(
            select(StrategyRun)
            .where(StrategyRun.bot_id == BOT_ID)
            .order_by(StrategyRun.id.desc())
            .limit(limit)
        )
        rows = list(result.scalars().all())

    return {
        "runs": [
            {
                "id": r.id,
                "status": r.status,
                "status_detail": r.status_detail or "",
                "status_timeline": _timeline(r.status_timeline),
                "markets_scanned": r.markets_scanned,
                "orders_placed": r.orders_placed,
                "errors": r.errors or None,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            }
            for r in rows
        ]
    }


@router.get("/candidates/{run_id}")
async def get_candidates(run_id: int):
    """Get candidates for a specific run."""
    async with async_session() as session:
        result = await session.execute(
            select(Candidate).where(
                Candidate.scan_id == run_id,
                Candidate.bot_id == BOT_ID,
            ).order_by(Candidate.id)
        )
        rows = list(result.scalars().all())

    return {
        "candidates": [
            {
                "id": c.id,
                "ticker": c.ticker,
                "title": c.title,
                "side": c.side,
                "price": c.price,
                "cost": c.cost,
                "bias_type": c.bias_type,
                "category": c.category,
                "expiry_time": c.expiry_time,
                "reason": c.reason,
                "status": c.status,
                "yes_bid": c.yes_bid,
                "yes_ask": c.yes_ask,
                "volume": c.volume,
                "predicted_win_rate": c.predicted_win_rate,
                "implied_probability": c.implied_probability,
                "predicted_edge": c.predicted_edge,
                "net_edge_cents": c.net_edge_cents,
                "sample_size": c.sample_size,
                "lookup_key": c.lookup_key,
            }
            for c in rows
        ]
    }


@router.patch("/config")
async def update_config(payload: dict[str, Any]):
    """Update runtime live scalper config (in-memory only)."""
    if not payload:
        raise HTTPException(status_code=400, detail="No config values provided")

    lookup_related_changed = False
    errors: list[str] = []

    for key, value in payload.items():
        if key in MUTABLE_SETTINGS:
            expected = MUTABLE_SETTINGS[key]
            if expected is float and isinstance(value, int):
                value = float(value)
            if not isinstance(value, expected):
                errors.append(f"{key} must be {expected.__name__}")
                continue
            setattr(settings, key, value)
            if key.startswith("live_lookup_"):
                lookup_related_changed = True
            continue

        if key in MUTABLE_LIST_SETTINGS:
            item_type = MUTABLE_LIST_SETTINGS[key]
            if not isinstance(value, list) or any(not isinstance(v, item_type) for v in value):
                errors.append(f"{key} must be list[{item_type.__name__}]")
                continue
            setattr(settings, key, value)
            lookup_related_changed = True
            continue

        errors.append(f"{key} is not mutable")

    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    if lookup_related_changed:
        reload_lookup_scorer()

    return {
        "status": "updated",
        "config": _config_snapshot(),
    }


@router.post("/propose-trade")
async def propose_trade(payload: dict[str, Any]):
    """Get enriched market context for AI trade proposal generation."""
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        context = await get_trade_proposal_context(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to collect proposal context: {e}") from e
    return context


@router.post("/execute-proposal")
async def execute_proposal(payload: dict[str, Any]):
    """Accept an AI proposal by simulating or placing an order."""
    ticker = str(payload.get("ticker", "")).strip().upper()
    side = str(payload.get("side", "")).strip().lower()
    run_id = int(payload.get("run_id") or 0)
    count = int(payload.get("count") or 1)
    yes_price = int(payload.get("yes_price") or 0)
    force_override = bool(payload.get("force_override", False))
    reason = str(payload.get("reason", "AI proposal")).strip()[:500]

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    if side not in {"yes", "no"}:
        raise HTTPException(status_code=400, detail="side must be yes or no")
    if not (1 <= yes_price <= 99):
        raise HTTPException(status_code=400, detail="yes_price must be between 1 and 99")
    if count <= 0:
        raise HTTPException(status_code=400, detail="count must be >= 1")
    if run_id <= 0:
        raise HTTPException(status_code=400, detail="run_id is required")

    cost_cents = yes_price if side == "yes" else 100 - yes_price
    risk_warnings: list[str] = []
    if cost_cents > settings.live_max_capital_per_event:
        risk_warnings.append(
            f"cost {cost_cents}c exceeds live_max_capital_per_event {settings.live_max_capital_per_event}c"
        )

    if risk_warnings and not force_override:
        raise HTTPException(
            status_code=400,
            detail="; ".join(risk_warnings) + " (set force_override=true to proceed)",
        )

    proposal_status = "ai_accepted_dry_run" if settings.live_dry_run else "ai_accepted"
    async with async_session() as session:
        session.add(
            Candidate(
                bot_id=BOT_ID,
                scan_id=run_id,
                ticker=ticker,
                title=str(payload.get("title", ticker))[:120],
                side=side,
                action="buy",
                price=yes_price,
                cost=cost_cents,
                count=count,
                bias_type="ai_proposal",
                category=str(payload.get("category", ""))[:120],
                expiry_time=str(payload.get("expiry_time", "")),
                reason=reason,
                yes_bid=int(payload.get("yes_bid") or 0),
                yes_ask=int(payload.get("yes_ask") or 0),
                volume=int(payload.get("volume") or 0),
                status=proposal_status,
            )
        )
        await session.commit()

    if settings.live_dry_run:
        return {
            "status": "simulated",
            "dry_run": True,
            "ticker": ticker,
            "side": side,
            "yes_price": yes_price,
            "count": count,
            "risk_warnings": risk_warnings,
        }

    try:
        resp = await kalshi.place_order(
            ticker=ticker,
            action="buy",
            side=side,
            count=count,
            yes_price=yes_price,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to place order: {e}") from e

    return {
        "status": "placed",
        "dry_run": False,
        "ticker": ticker,
        "side": side,
        "yes_price": yes_price,
        "count": count,
        "risk_warnings": risk_warnings,
        "order": resp.get("order", resp),
    }


@router.post("/scheduler/start")
async def start_scheduler():
    """Start the live scalper auto-run scheduler."""
    global _scheduler_ref

    if _scheduler_ref is not None:
        return {"status": "already_running"}

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        auto_run,
        "interval",
        minutes=settings.live_scheduler_interval_minutes,
    )
    scheduler.start()
    _scheduler_ref = scheduler
    logger.info("Live scalper scheduler started (every %d min)", settings.live_scheduler_interval_minutes)
    return {"status": "started", "interval_minutes": settings.live_scheduler_interval_minutes}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the live scalper auto-run scheduler."""
    global _scheduler_ref

    if _scheduler_ref is None:
        return {"status": "not_running"}

    _scheduler_ref.shutdown(wait=False)
    _scheduler_ref = None
    logger.info("Live scalper scheduler stopped")
    return {"status": "stopped"}
