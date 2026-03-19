from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import Order, async_session
from app.kalshi_client import kalshi

router = APIRouter(prefix="/api")

_market_cache: dict[str, dict] = {}


@router.get("/orders")
async def list_orders(status: str | None = None, limit: int = 50, bot_id: str | None = None):
    """Return orders from local DB, optionally filtered by status and bot_id."""
    async with async_session() as session:
        q = select(Order).order_by(Order.created_at.desc()).limit(limit)
        if status:
            q = q.where(Order.status == status)
        if bot_id:
            q = q.where(Order.bot_id == bot_id)
        result = await session.execute(q)
        rows = result.scalars().all()
    return {
        "orders": [
            {
                "id": o.id,
                "order_id": o.order_id,
                "ticker": o.ticker,
                "side": o.side,
                "action": o.action,
                "price": o.price,
                "count": o.count,
                "status": o.status,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in rows
        ]
    }


@router.get("/market/{ticker}")
async def get_market_detail(ticker: str):
    """Fetch market info from Kalshi (cached in memory for the session)."""
    if ticker in _market_cache:
        return _market_cache[ticker]
    try:
        data = await kalshi.get_market(ticker)
        market = data.get("market", data)
        result = {
            "ticker": market.get("ticker", ticker),
            "title": market.get("title", ""),
            "subtitle": market.get("subtitle", ""),
            "event_ticker": market.get("event_ticker", ""),
            "category": market.get("category", ""),
            "status": market.get("status", ""),
            "yes_bid": market.get("yes_bid"),
            "yes_ask": market.get("yes_ask"),
            "last_price": market.get("last_price"),
            "volume": market.get("volume"),
            "open_interest": market.get("open_interest"),
            "close_time": market.get("close_time", ""),
            "expiration_time": market.get("expiration_time", ""),
            "result": market.get("result", ""),
            "rules_primary": market.get("rules_primary", ""),
            "yes_sub_title": market.get("yes_sub_title", ""),
            "no_sub_title": market.get("no_sub_title", ""),
        }
        _market_cache[ticker] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/balance")
async def get_balance():
    """Proxy Kalshi balance endpoint."""
    try:
        data = await kalshi.get_balance()
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order on Kalshi and update local DB."""
    try:
        resp = await kalshi.cancel_order(order_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    async with async_session() as session:
        result = await session.execute(
            select(Order).where(Order.order_id == order_id)
        )
        order = result.scalar_one_or_none()
        if order:
            order.status = "cancelled"
            await session.commit()

    return {"status": "cancelled", "order_id": order_id, "kalshi": resp}
