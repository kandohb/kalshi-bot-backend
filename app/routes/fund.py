"""Fund-level overview: aggregates data across all bots."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select, func

from app.db import StrategyRun, Order, Candidate, async_session

router = APIRouter(prefix="/api/fund")

BOT_IDS = ["bias_scalper", "live_scalper"]


@router.get("/overview")
async def fund_overview():
    """Return per-bot summary for the fund overview page."""
    bots = []
    async with async_session() as session:
        for bot_id in BOT_IDS:
            last_run_q = await session.execute(
                select(StrategyRun)
                .where(StrategyRun.bot_id == bot_id)
                .order_by(StrategyRun.id.desc())
                .limit(1)
            )
            last_run = last_run_q.scalar_one_or_none()

            open_orders_q = await session.execute(
                select(func.count(Order.id)).where(
                    Order.bot_id == bot_id,
                    Order.status.in_(["pending", "resting"]),
                )
            )
            open_orders = open_orders_q.scalar() or 0

            total_placed_q = await session.execute(
                select(func.count(Order.id)).where(Order.bot_id == bot_id)
            )
            total_placed = total_placed_q.scalar() or 0

            bots.append({
                "bot_id": bot_id,
                "open_orders": open_orders,
                "total_orders": total_placed,
                "last_run": {
                    "id": last_run.id,
                    "status": last_run.status,
                    "markets_scanned": last_run.markets_scanned,
                    "orders_placed": last_run.orders_placed,
                    "started_at": last_run.started_at.isoformat() if last_run.started_at else None,
                    "ended_at": last_run.ended_at.isoformat() if last_run.ended_at else None,
                } if last_run else None,
            })

    return {"bots": bots}
