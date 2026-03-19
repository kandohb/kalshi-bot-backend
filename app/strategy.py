"""Bias-isolated strategy: scan markets, apply bias-specific filters, generate candidates."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, func

from app.config import settings
from app.db import Order, Fill, StrategyRun, Candidate, async_session
from app.kalshi_client import kalshi

logger = logging.getLogger(__name__)

FINANCE_PREFIXES = {
    "INX", "NASDAQ", "FED", "CPI", "GDP", "PAYROLLS", "TNOTE",
    "SP500", "RUSSELL", "DOW", "FOMC",
}


def _is_finance(event_ticker: str | None, category: str | None) -> bool:
    if category and "finance" in category.lower():
        return True
    if not event_ticker:
        return False
    upper = event_ticker.upper()
    return any(upper.startswith(p) for p in FINANCE_PREFIXES)


def _is_exotic(ticker: str, event_ticker: str | None) -> bool:
    upper_ticker = ticker.upper()
    upper_event = (event_ticker or "").upper()
    for prefix in settings.exclude_ticker_prefixes:
        p = prefix.upper()
        if upper_ticker.startswith(p) or upper_event.startswith(p):
            return True
    return False


def _expires_soon(market: dict[str, Any], max_days: int) -> bool:
    """Return True if market expires within max_days from now."""
    close_str = market.get("close_time") or market.get("expiration_time") or ""
    if not close_str:
        return False
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        cutoff = datetime.now(timezone.utc) + timedelta(days=max_days)
        return close_dt <= cutoff
    except (ValueError, TypeError):
        return False


@dataclass
class CandidateResult:
    ticker: str
    title: str
    action: str
    side: str
    price: int      # yes_price for the API
    cost: int        # actual cents risked
    count: int
    bias_type: str   # "longshot_no", "favorite_yes", "asymmetry"
    category: str
    expiry_time: str
    reason: str
    yes_bid: int
    yes_ask: int
    volume: int


@dataclass
class ScanLog:
    markets_fetched: int = 0
    events_found: int = 0
    events_exotic_skipped: int = 0
    pages_fetched: int = 0
    skipped_no_liquidity: int = 0
    skipped_finance: int = 0
    skipped_exotic: int = 0
    skipped_expiry: int = 0
    evaluated: int = 0
    category_filter: str = ""
    series_filter: str = ""
    candidates: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sample_markets: list[dict] = field(default_factory=list)
    price_distribution: dict[str, int] = field(default_factory=dict)
    bias_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "markets_fetched": self.markets_fetched,
            "events_found": self.events_found,
            "events_exotic_skipped": self.events_exotic_skipped,
            "pages_fetched": self.pages_fetched,
            "skipped_no_liquidity": self.skipped_no_liquidity,
            "skipped_finance": self.skipped_finance,
            "skipped_exotic": self.skipped_exotic,
            "skipped_expiry": self.skipped_expiry,
            "evaluated": self.evaluated,
            "category_filter": self.category_filter,
            "series_filter": self.series_filter,
            "candidates_found": len(self.candidates),
            "candidates": self.candidates[:50],
            "errors": self.errors,
            "sample_markets": self.sample_markets[:8],
            "price_distribution": self.price_distribution,
            "bias_breakdown": self.bias_breakdown,
        }


# ── Bias Strategy Modules ────────────────────────────────────────────────


def _longshot_no(market: dict, yes_mid: int, yes_bid: int, yes_ask: int) -> CandidateResult | None:
    """YES/NO Asymmetry: Buy NO at longshot prices (1-20c NO = YES 80-99c).

    From the paper: NO at 1c has +23% EV, YES at 1c has -41% EV.
    Takers disproportionately buy YES at longshot prices; we take the
    other side by placing a maker limit order for NO.
    """
    if yes_mid < 80 or yes_mid > 99:
        return None

    no_price = 100 - yes_mid
    no_bid_price = (100 - yes_ask) if yes_ask else no_price
    buy_no_at = max(no_bid_price - 1, 1)
    yes_price_for_api = 100 - buy_no_at

    if buy_no_at < 1 or buy_no_at > 20:
        return None

    return CandidateResult(
        ticker=market.get("ticker", ""),
        title=(market.get("title") or "")[:120],
        action="buy", side="no",
        price=yes_price_for_api, cost=buy_no_at, count=1,
        bias_type="longshot_no",
        category=market.get("category") or market.get("event_ticker") or "",
        expiry_time=market.get("close_time") or market.get("expiration_time") or "",
        reason=f"YES/NO asymmetry: YES at {yes_mid}c, NO at {no_price}c. "
               f"Historical NO EV is positive at this level. Buy NO at {buy_no_at}c as maker.",
        yes_bid=yes_bid, yes_ask=yes_ask,
        volume=market.get("volume") or 0,
    )


def _favorite_yes(market: dict, yes_mid: int, yes_bid: int, yes_ask: int) -> CandidateResult | None:
    """Longshot Bias (reverse): Buy YES at 85-99c where favorites are underpriced.

    From the paper: contracts at 95c win 95.83% of the time (above implied 95%).
    Tighter band than before to focus on strongest edge.
    """
    if yes_mid < 85 or yes_mid > 99:
        return None

    buy_price = max(yes_bid - 1, 85)
    if buy_price < 85:
        return None

    return CandidateResult(
        ticker=market.get("ticker", ""),
        title=(market.get("title") or "")[:120],
        action="buy", side="yes",
        price=buy_price, cost=buy_price, count=1,
        bias_type="favorite_yes",
        category=market.get("category") or market.get("event_ticker") or "",
        expiry_time=market.get("close_time") or market.get("expiration_time") or "",
        reason=f"Favorite underpriced: YES at {yes_mid}c wins more than implied. "
               f"Buy YES at {buy_price}c as maker.",
        yes_bid=yes_bid, yes_ask=yes_ask,
        volume=market.get("volume") or 0,
    )


def _asymmetry_exploiter(market: dict, yes_mid: int, yes_bid: int, yes_ask: int) -> CandidateResult | None:
    """Maker-Taker Gap: At mid-range prices where spread exists, place NO maker orders.

    From the paper: makers earn +1.12% excess; NO outperforms YES at 69/99 levels.
    This targets prices 21-79c where the other strategies don't apply, but only
    when there's a meaningful spread to capture and we favor the NO side.
    """
    if yes_mid < 60 or yes_mid >= 80:
        return None

    no_price = 100 - yes_mid
    if no_price < 21 or no_price > 40:
        return None

    no_bid_price = (100 - yes_ask) if yes_ask else no_price
    buy_no_at = max(no_bid_price - 1, 21)
    yes_price_for_api = 100 - buy_no_at

    spread = (yes_ask - yes_bid) if (yes_ask and yes_bid) else 0
    if spread < 2:
        return None

    return CandidateResult(
        ticker=market.get("ticker", ""),
        title=(market.get("title") or "")[:120],
        action="buy", side="no",
        price=yes_price_for_api, cost=buy_no_at, count=1,
        bias_type="asymmetry",
        category=market.get("category") or market.get("event_ticker") or "",
        expiry_time=market.get("close_time") or market.get("expiration_time") or "",
        reason=f"Maker-taker gap: YES {yes_mid}c (spread {spread}c). "
               f"NO historically outperforms. Buy NO at {buy_no_at}c as maker.",
        yes_bid=yes_bid, yes_ask=yes_ask,
        volume=market.get("volume") or 0,
    )


BIAS_STRATEGIES = [_longshot_no, _favorite_yes, _asymmetry_exploiter]


# ── Market Scanner ────────────────────────────────────────────────────────


@dataclass
class ScanParams:
    max_expiry_days: int = 7
    biases: list[str] = field(default_factory=lambda: ["longshot_no", "favorite_yes", "asymmetry"])
    category: str = ""          # e.g. "Sports", "Crypto", "Weather" — empty = all
    series_ticker: str = ""     # e.g. "KXNBA" — empty = all
    league_series: list[str] = field(default_factory=list)  # multiple series for a league


BIAS_FN_MAP: dict[str, Any] = {
    "longshot_no": _longshot_no,
    "favorite_yes": _favorite_yes,
    "asymmetry": _asymmetry_exploiter,
}


def _evaluate_market(market: dict[str, Any], log: ScanLog, params: ScanParams) -> list[CandidateResult]:
    """Run selected bias strategies on a single market. Returns matching candidates."""
    ticker = market.get("ticker", "")
    event_ticker = market.get("event_ticker", "")
    category = market.get("category", "") or ""

    if _is_finance(event_ticker, category):
        log.skipped_finance += 1
        return []

    if _is_exotic(ticker, event_ticker):
        log.skipped_exotic += 1
        return []

    if not _expires_soon(market, params.max_expiry_days):
        log.skipped_expiry += 1
        return []

    yes_bid = market.get("yes_bid") or 0
    yes_ask = market.get("yes_ask") or 0

    if settings.require_liquidity and not yes_bid and not yes_ask:
        log.skipped_no_liquidity += 1
        return []

    log.evaluated += 1
    yes_mid = (yes_bid + yes_ask) // 2 if (yes_bid and yes_ask) else (yes_bid or yes_ask)

    if yes_mid <= 10:
        bucket = "1-10"
    elif yes_mid <= 20:
        bucket = "11-20"
    elif yes_mid <= 35:
        bucket = "21-35"
    elif yes_mid <= 50:
        bucket = "36-50"
    elif yes_mid <= 65:
        bucket = "51-65"
    elif yes_mid <= 80:
        bucket = "66-80"
    elif yes_mid <= 90:
        bucket = "81-90"
    else:
        bucket = "91-99"
    log.price_distribution[bucket] = log.price_distribution.get(bucket, 0) + 1

    active_fns = [BIAS_FN_MAP[b] for b in params.biases if b in BIAS_FN_MAP]

    results: list[CandidateResult] = []
    for strategy_fn in active_fns:
        candidate = strategy_fn(market, yes_mid, yes_bid, yes_ask)
        if candidate:
            results.append(candidate)
            log.bias_breakdown[candidate.bias_type] = log.bias_breakdown.get(candidate.bias_type, 0) + 1

    return results


async def _fetch_markets_direct(slog: ScanLog, params: ScanParams) -> list[dict]:
    """Fetch markets directly via paginated GET /markets with optional series filter.

    Used when a category or series_ticker is specified — much faster and more
    targeted than the events-first approach.
    """
    all_markets: list[dict] = []
    cursor: str | None = None

    for page in range(50):
        kwargs: dict[str, Any] = {"status": "open", "limit": 200}
        if params.series_ticker:
            kwargs["series_ticker"] = params.series_ticker
        if cursor:
            kwargs["cursor"] = cursor
        try:
            data = await kalshi.get_markets(**kwargs)
        except Exception as e:
            logger.warning("Market fetch page %d failed: %s", page, e)
            break
        markets = data.get("markets", [])
        if not markets:
            break
        slog.pages_fetched += 1
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or len(all_markets) >= 10000:
            break
        await asyncio.sleep(0.05)

    logger.info("Direct fetch: %d markets (series=%s)", len(all_markets), params.series_ticker or "all")
    return all_markets


async def _fetch_all_markets(slog: ScanLog, params: ScanParams) -> list[dict]:
    """Fetch markets. Uses direct fetch if a series/category is given, otherwise events-first."""

    # If user picked a specific series, go straight to the markets API (fastest)
    if params.series_ticker:
        return await _fetch_markets_direct(slog, params)

    # If user picked a league (multiple series), fetch markets for each series
    if params.league_series:
        all_markets: list[dict] = []
        for st in params.league_series:
            if len(all_markets) >= 10000:
                break
            sub_params = ScanParams(series_ticker=st)
            batch = await _fetch_markets_direct(slog, sub_params)
            all_markets.extend(batch)
            await asyncio.sleep(0.05)
        logger.info("League fetch: %d markets from %d series", len(all_markets), len(params.league_series))
        return all_markets

    # Events-first approach for broad scan (filters exotics + optional category)
    # Sort events by close time so near-term (live games, etc.) get processed first
    good_events: list[tuple[str, str]] = []  # (close_time, event_ticker)
    skipped_events = 0
    cursor: str | None = None
    max_expiry_cutoff = (
        datetime.now(timezone.utc) + timedelta(days=params.max_expiry_days)
    ).isoformat()

    for page in range(20):
        data = await kalshi.get_events(status="open", limit=200, cursor=cursor)
        events = data.get("events", [])
        if not events:
            break
        slog.pages_fetched += 1

        for ev in events:
            et = ev.get("event_ticker", "")
            cat = ev.get("category", "")
            close = ev.get("close_time") or ev.get("expiration_time") or ev.get("expected_expiration_time") or ""

            if _is_exotic("", et):
                skipped_events += 1
                continue

            if params.category and cat.lower() != params.category.lower():
                continue

            # Pre-filter: skip events that definitely expire beyond our window
            if close and close > max_expiry_cutoff:
                continue

            good_events.append((close or "9999", et))

        cursor = data.get("cursor")
        if not cursor:
            break
        await asyncio.sleep(0.1)

    # Sort by close time — nearest events first (live games, today's markets)
    good_events.sort(key=lambda x: x[0])
    good_event_tickers = [et for _, et in good_events]

    logger.info("Events: %d within %dd window, %d exotic skipped (category=%s)",
                len(good_event_tickers), params.max_expiry_days, skipped_events,
                params.category or "all")

    all_markets: list[dict] = []
    for et in good_event_tickers:
        if len(all_markets) >= 10000:
            break
        try:
            data = await kalshi.get_markets(status="open", limit=100, event_ticker=et)
            markets = data.get("markets", [])
            all_markets.extend(markets)
        except Exception as e:
            logger.warning("Failed to fetch markets for event %s: %s", et, e)
        await asyncio.sleep(0.1)

    slog.events_found = len(good_event_tickers)
    slog.events_exotic_skipped = skipped_events
    logger.info("Fetched %d markets from %d near-term events", len(all_markets), len(good_event_tickers))
    return all_markets


BOT_ID = "bias_scalper"


async def scan_markets(params: ScanParams | None = None) -> tuple[int, ScanLog]:
    """Scan markets and persist candidates to DB. Returns (scan_id, log)."""
    if params is None:
        params = ScanParams()
    slog = ScanLog(
        category_filter=params.category,
        series_filter=params.series_ticker,
    )

    run = StrategyRun(bot_id=BOT_ID, started_at=datetime.now(timezone.utc))
    async with async_session() as session:
        session.add(run)
        await session.commit()
        scan_id = run.id

    try:
        markets = await _fetch_all_markets(slog, params)
        slog.markets_fetched = len(markets)
        logger.info("Fetched %d open markets for scan", len(markets))

        for m in markets[:8]:
            slog.sample_markets.append({
                "ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"),
                "title": (m.get("title") or "")[:80],
                "category": m.get("category", ""),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "volume": m.get("volume"),
                "close_time": m.get("close_time", ""),
            })

        all_candidates: list[CandidateResult] = []
        for market in markets:
            candidates = _evaluate_market(market, slog, params)
            all_candidates.extend(candidates)

        # Persist candidates to DB
        async with async_session() as session:
            for c in all_candidates:
                session.add(Candidate(
                    bot_id=BOT_ID,
                    scan_id=scan_id,
                    ticker=c.ticker,
                    title=c.title,
                    side=c.side,
                    action=c.action,
                    price=c.price,
                    cost=c.cost,
                    count=c.count,
                    bias_type=c.bias_type,
                    category=c.category,
                    expiry_time=c.expiry_time,
                    reason=c.reason,
                    yes_bid=c.yes_bid,
                    yes_ask=c.yes_ask,
                    volume=c.volume,
                    status="pending",
                ))
                slog.candidates.append({
                    "ticker": c.ticker,
                    "title": c.title,
                    "side": c.side,
                    "price": c.price,
                    "cost": c.cost,
                    "bias_type": c.bias_type,
                    "category": c.category,
                    "expiry_time": c.expiry_time,
                    "reason": c.reason,
                    "yes_bid": c.yes_bid,
                    "yes_ask": c.yes_ask,
                    "volume": c.volume,
                })
            await session.commit()

    except Exception as e:
        msg = f"Scan error: {e}"
        logger.error(msg)
        slog.errors.append(msg)

    async with async_session() as session:
        run = await session.get(StrategyRun, scan_id)
        if run:
            run.ended_at = datetime.now(timezone.utc)
            run.markets_scanned = slog.markets_fetched
            run.orders_placed = 0
            run.errors = "\n".join(slog.errors) if slog.errors else ""
            run.status = "scanned"
            await session.commit()

    logger.info("Scan done: %d candidates from %d markets", len(slog.candidates), slog.markets_fetched)
    return scan_id, slog


async def place_approved(candidate_ids: list[int]) -> dict:
    """Place orders for approved candidates. Returns summary."""
    placed = []
    errors = []

    async with async_session() as session:
        result = await session.execute(
            select(Candidate).where(
                Candidate.id.in_(candidate_ids),
                Candidate.status == "pending",
            )
        )
        candidates = list(result.scalars().all())

    capital_used = 0
    for cand in candidates:
        if capital_used + cand.cost > settings.max_capital_cents:
            errors.append(f"Capital cap reached at {capital_used}c, skipping {cand.ticker}")
            break

        try:
            resp = await kalshi.place_order(
                ticker=cand.ticker,
                action=cand.action,
                side=cand.side,
                count=cand.count,
                yes_price=cand.price,
            )
            order_data = resp.get("order", {})

            async with async_session() as session:
                session.add(Order(
                    bot_id=BOT_ID,
                    order_id=order_data.get("order_id", ""),
                    client_order_id=order_data.get("client_order_id", ""),
                    ticker=cand.ticker,
                    side=cand.side,
                    action=cand.action,
                    price=cand.price,
                    count=cand.count,
                    status=order_data.get("status", "pending"),
                ))
                await session.commit()

            # Mark candidate as placed
            async with async_session() as session:
                c = await session.get(Candidate, cand.id)
                if c:
                    c.status = "placed"
                    await session.commit()

            placed.append({
                "id": cand.id,
                "ticker": cand.ticker,
                "side": cand.side,
                "price": cand.price,
                "cost": cand.cost,
                "order_id": order_data.get("order_id", ""),
            })
            capital_used += cand.cost

            await asyncio.sleep(0.15)

        except Exception as e:
            msg = f"Order failed for {cand.ticker}: {e}"
            logger.warning(msg)
            errors.append(msg)

    return {"placed": placed, "errors": errors, "capital_used_cents": capital_used}
