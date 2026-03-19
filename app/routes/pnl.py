"""Portfolio & P&L routes — fetches live data from the Kalshi API."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.kalshi_client import kalshi

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.get("/portfolio")
async def get_portfolio():
    """Fetch complete portfolio snapshot from Kalshi: balance, positions, and recent fills."""
    try:
        balance_data, positions_data, fills_data, resting_orders = await asyncio.gather(
            kalshi.get_balance(),
            kalshi.get_positions(limit=200, settlement_status="unsettled"),
            kalshi.get_fills(limit=50),
            kalshi.get_orders(status="resting", limit=100),
        )
    except Exception as e:
        logger.error("Failed to fetch portfolio from Kalshi: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    positions = positions_data.get("market_positions") or positions_data.get("positions") or []
    fills = fills_data.get("fills", [])
    orders = resting_orders.get("orders", [])

    logger.info("Portfolio: %d positions, %d resting, %d fills", len(positions), len(orders), len(fills))

    # Gather unique tickers to fetch market details for titles
    tickers = set()
    for p in positions:
        tickers.add(p.get("ticker", ""))
    for o in orders:
        tickers.add(o.get("ticker", ""))

    market_titles: dict[str, dict] = {}
    for ticker in tickers:
        if not ticker:
            continue
        try:
            data = await kalshi.get_market(ticker)
            m = data.get("market", data)
            market_titles[ticker] = {
                "title": m.get("title", ""),
                "event_ticker": m.get("event_ticker", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "last_price": m.get("last_price"),
                "close_time": m.get("close_time", ""),
                "result": m.get("result", ""),
                "status": m.get("status", ""),
            }
        except Exception:
            market_titles[ticker] = {"title": ticker}
        await asyncio.sleep(0.05)

    enriched_positions = []
    total_cost = 0
    total_market_value = 0
    total_payout_if_right = 0
    total_realized_pnl = 0
    total_fees = 0

    for p in positions:
        ticker = p.get("ticker", "")
        raw_position = p.get("position", 0) or 0
        side = "yes" if raw_position > 0 else "no"
        contracts = abs(raw_position)
        if contracts == 0:
            continue

        cost = p.get("market_exposure", 0) or 0
        avg_price = round(cost / contracts) if contracts else 0
        realized_pnl = p.get("realized_pnl", 0) or 0
        fees = p.get("fees_paid", 0) or 0

        market_info = market_titles.get(ticker, {})
        last_price = market_info.get("last_price") or market_info.get("yes_bid") or 0

        if side == "yes":
            market_value = last_price * contracts
        else:
            market_value = (100 - last_price) * contracts

        payout_if_right = 100 * contracts
        total_return_cents = market_value - cost

        total_cost += cost
        total_market_value += market_value
        total_payout_if_right += payout_if_right
        total_realized_pnl += realized_pnl
        total_fees += fees

        enriched_positions.append({
            "ticker": ticker,
            "title": market_info.get("title", ticker),
            "subtitle": market_info.get("subtitle", ""),
            "event_ticker": market_info.get("event_ticker", ""),
            "side": side,
            "contracts": contracts,
            "avg_price": avg_price,
            "cost": cost,
            "last_price": last_price,
            "market_value": market_value,
            "payout_if_right": payout_if_right,
            "total_return": total_return_cents,
            "total_return_pct": round((total_return_cents / cost * 100), 1) if cost else 0,
            "realized_pnl": realized_pnl,
            "fees": fees,
            "close_time": market_info.get("close_time", ""),
            "status": market_info.get("status", ""),
            "result": market_info.get("result", ""),
        })

    enriched_positions.sort(key=lambda p: abs(p["total_return"]), reverse=True)

    enriched_orders = []
    for o in orders:
        ticker = o.get("ticker", "")
        market_info = market_titles.get(ticker, {})
        enriched_orders.append({
            "order_id": o.get("order_id", ""),
            "ticker": ticker,
            "title": market_info.get("title", ticker),
            "side": o.get("side", ""),
            "action": o.get("action", ""),
            "price": o.get("yes_price", 0),
            "remaining_count": o.get("remaining_count", 0),
            "created_time": o.get("created_time", ""),
        })

    recent_fills = []
    for f in fills[:20]:
        ticker = f.get("ticker", "")
        recent_fills.append({
            "ticker": ticker,
            "title": market_titles.get(ticker, {}).get("title", ticker),
            "side": f.get("side", ""),
            "action": f.get("action", ""),
            "price": f.get("yes_price", 0),
            "count": f.get("count", 0),
            "created_time": f.get("created_time", ""),
        })

    return {
        "balance": balance_data.get("balance", 0),
        "portfolio_value": balance_data.get("portfolio_value", 0),
        "positions": enriched_positions,
        "resting_orders": enriched_orders,
        "recent_fills": recent_fills,
        "summary": {
            "total_cost": total_cost,
            "total_market_value": total_market_value,
            "total_payout_if_right": total_payout_if_right,
            "unrealized_pnl": total_market_value - total_cost,
            "realized_pnl": total_realized_pnl,
            "total_fees": total_fees,
            "position_count": len(enriched_positions),
            "resting_order_count": len(enriched_orders),
        },
    }


@router.get("/pnl")
async def get_pnl():
    """Legacy P&L endpoint — redirects to portfolio data."""
    portfolio = await get_portfolio()
    s = portfolio["summary"]
    return {
        "by_side": {},
        "total_cost_cents": s["total_cost"],
        "total_contracts": sum(p["contracts"] for p in portfolio["positions"]),
        "note": "Use the Portfolio page for full position and P&L data.",
    }
