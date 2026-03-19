"""Live Scalper bot: auto-executes on markets expiring within 12h.

Reuses bias functions from the original strategy module. Adds deterministic
scoring, liquidity gates, cooldown tracking, and a single auto_run() entry
point that does scan → rank → cap → place in one cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, func

from app.config import settings
from app.db import Order, StrategyRun, Candidate, async_session
from app.kalshi_client import kalshi
from app.lookup_scorer import LookupScorer
from app.strategy import (
    _favorite_yes,
    _asymmetry_exploiter,
    _is_finance,
    _is_exotic,
    CandidateResult,
)

logger = logging.getLogger(__name__)

BOT_ID = "live_scalper"


def _build_lookup_scorer() -> LookupScorer:
    return LookupScorer(
        lookup_path=settings.live_lookup_abs_path,
        min_samples=settings.live_lookup_min_samples,
        min_markets=settings.live_lookup_min_markets,
        fee_cents=settings.live_lookup_fee_cents,
        min_edge_cents=settings.live_lookup_min_edge_cents,
        min_prob_edge=settings.live_lookup_min_prob_edge,
        allowed_hours_buckets=settings.live_lookup_allowed_hours_buckets,
    )


_LOOKUP_SCORER = _build_lookup_scorer()


def reload_lookup_scorer() -> None:
    """Rebuild lookup scorer after runtime config changes."""
    global _LOOKUP_SCORER
    _LOOKUP_SCORER = _build_lookup_scorer()


MAX_BUCKET_ITEMS = 50


def _market_summary(m: dict[str, Any]) -> dict[str, Any]:
    """Compact snapshot of a market for bucket drill-down."""
    return {
        "ticker": m.get("ticker", ""),
        "title": (m.get("title") or "")[:120],
        "subtitle": (m.get("subtitle") or m.get("yes_sub_title") or "")[:200],
        "event_ticker": m.get("event_ticker", ""),
        "category": m.get("category", ""),
        "yes_bid": m.get("yes_bid"),
        "yes_ask": m.get("yes_ask"),
        "volume": m.get("volume"),
        "close_time": m.get("close_time") or m.get("expiration_time") or "",
    }


@dataclass
class LiveRunLog:
    events_scanned: int = 0
    live_events: int = 0
    markets_fetched: int = 0
    live_markets: int = 0
    skipped_not_live: int = 0
    skipped_finance: int = 0
    skipped_exotic: int = 0
    skipped_no_liquidity: int = 0
    skipped_low_volume: int = 0
    skipped_wide_spread: int = 0
    skipped_cooldown: int = 0
    evaluated: int = 0
    candidates_found: int = 0
    auto_selected: int = 0
    auto_placed: int = 0
    dry_run: bool = True
    errors: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    stage_history: list[dict[str, str]] = field(default_factory=list)
    buckets: dict[str, list[dict]] = field(default_factory=lambda: {
        "no_liquidity": [],
        "low_volume": [],
        "wide_spread": [],
        "cooldown": [],
        "evaluated": [],
        "lookup_no_match": [],
        "finance": [],
        "exotic": [],
    })

    def _bucket_append(self, key: str, market: dict[str, Any]) -> None:
        lst = self.buckets[key]
        if len(lst) < MAX_BUCKET_ITEMS:
            lst.append(_market_summary(market))

    def to_dict(self) -> dict:
        return {
            "events_scanned": self.events_scanned,
            "live_events": self.live_events,
            "markets_fetched": self.markets_fetched,
            "live_markets": self.live_markets,
            "skipped_not_live": self.skipped_not_live,
            "skipped_finance": self.skipped_finance,
            "skipped_exotic": self.skipped_exotic,
            "skipped_no_liquidity": self.skipped_no_liquidity,
            "skipped_low_volume": self.skipped_low_volume,
            "skipped_wide_spread": self.skipped_wide_spread,
            "skipped_cooldown": self.skipped_cooldown,
            "evaluated": self.evaluated,
            "candidates_found": self.candidates_found,
            "auto_selected": self.auto_selected,
            "auto_placed": self.auto_placed,
            "dry_run": self.dry_run,
            "errors": self.errors,
            "candidates": self.candidates[:30],
            "stage_history": self.stage_history,
            "buckets": self.buckets,
        }


BIAS_STRATEGIES = [_favorite_yes, _asymmetry_exploiter]

IN_PLAY_TITLE_HINTS = (
    "live",
    "halftime",
    "reg time",
    "1h",
    "2h",
    "3h",
    "quarter",
    "period",
    "set",
    "match",
)

SPORTS_HINTS = (
    "sports",
    "ncaa",
    "tennis",
    "soccer",
    "epl",
    "la liga",
    "bundesliga",
    "serie a",
    "ufc",
    "mma",
    "golf",
)

SPORTS_EVENT_PREFIXES = (
    "KXNBA",
    "KXNFL",
    "KXMLB",
    "KXNHL",
    "KXNCAA",
    "KXATP",
    "KXWTA",
    "KXEPL",
    "KXUFC",
)

CRYPTO_HINTS = (
    "crypto",
    "bitcoin",
    "ethereum",
    "dogecoin",
    "xrp",
    "solana",
    " sol ",
    " btc",
    " eth",
    " doge",
    "kxbtc",
    "kxeth",
    "kxxrp",
    "kxdoge",
    "kxsol",
    "btc",
    "eth",
    "doge",
)


def _normalize_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalise Kalshi API v2 dollar-string fields to cent-integer fields.

    The API returns prices like ``yes_bid_dollars: "0.6200"`` but all
    downstream code expects ``yes_bid: 62`` (int cents).
    """
    def _dollars_to_cents(key: str) -> int | None:
        raw = m.get(key)
        if raw is None:
            return None
        try:
            return int(round(float(raw) * 100))
        except (ValueError, TypeError):
            return None

    m["yes_bid"] = _dollars_to_cents("yes_bid_dollars") or m.get("yes_bid")
    m["yes_ask"] = _dollars_to_cents("yes_ask_dollars") or m.get("yes_ask")
    m["no_bid"] = _dollars_to_cents("no_bid_dollars") or m.get("no_bid")
    m["no_ask"] = _dollars_to_cents("no_ask_dollars") or m.get("no_ask")
    m["last_price"] = _dollars_to_cents("last_price_dollars") or m.get("last_price")

    for vol_key, src_key in [("volume", "volume_fp"), ("volume_24h", "volume_24h_fp")]:
        raw = m.get(src_key)
        if raw is not None:
            try:
                m[vol_key] = int(float(raw))
            except (ValueError, TypeError):
                pass

    return m


def _is_crypto_market(market: dict[str, Any]) -> bool:
    category = (market.get("category") or "").lower()
    title = (market.get("title") or "").lower()
    ticker = (market.get("ticker") or "").lower()
    event_ticker = (market.get("event_ticker") or "").lower()
    blob = " ".join([category, title, ticker, event_ticker])
    return any(hint in blob for hint in CRYPTO_HINTS)


def _is_priority_market(market: dict[str, Any]) -> bool:
    """Priority bucket for live sports / in-play style markets."""
    return _is_preferred_category(market) or _looks_sports_market(market) or (
        settings.live_prioritize_in_play and _is_in_play_market(market)
    )


def _is_preferred_category(market: dict[str, Any]) -> bool:
    category = (market.get("category") or "").lower()
    if not category:
        return False
    preferred = [c.lower() for c in settings.live_preferred_categories]
    return any(c in category for c in preferred)


def _is_lookup_focus_market(market: dict[str, Any]) -> bool:
    focuses = [c.lower() for c in settings.live_lookup_focus_categories]
    if not focuses:
        return True
    blob = " ".join(
        [
            (market.get("category") or "").lower(),
            (market.get("event_ticker") or "").lower(),
            (market.get("title") or "").lower(),
        ]
    )
    return any(token in blob for token in focuses)


def _looks_sports_market(market: dict[str, Any]) -> bool:
    category = (market.get("category") or "").lower()
    title = (market.get("title") or "").lower()
    event_ticker = (market.get("event_ticker") or "")
    upper_event = event_ticker.upper()
    if any(upper_event.startswith(prefix) for prefix in SPORTS_EVENT_PREFIXES):
        return True
    blob = " ".join([category, title, event_ticker.lower()])
    return any(hint in blob for hint in SPORTS_HINTS)


def _is_in_play_market(market: dict[str, Any]) -> bool:
    title = (market.get("title") or "").lower()
    subtitle = (market.get("subtitle") or "").lower()
    event_ticker = (market.get("event_ticker") or "").lower()
    blob = " ".join([title, subtitle, event_ticker])
    return any(hint in blob for hint in IN_PLAY_TITLE_HINTS)


def _is_live(market: dict[str, Any], max_hours: int) -> bool:
    """Return True if market expires within max_hours from now."""
    close_str = market.get("close_time") or market.get("expiration_time") or ""
    if not close_str:
        return False
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        cutoff = datetime.now(timezone.utc) + timedelta(hours=max_hours)
        return close_dt <= cutoff and close_dt > datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False


def _hours_until_close(market: dict[str, Any]) -> float:
    close_str = market.get("close_time") or market.get("expiration_time") or ""
    if not close_str:
        return 999.0
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        delta = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        return max(delta, 0.0)
    except (ValueError, TypeError):
        return 999.0


def _score_candidate(c: CandidateResult, market: dict[str, Any]) -> float:
    """Deterministic score with sports/in-play priority."""
    spread = (c.yes_ask - c.yes_bid) if (c.yes_ask and c.yes_bid) else 10
    spread_score = max(0, 100 - spread * 10)
    volume_score = min(c.volume / 10, 100)
    hours_left = _hours_until_close(market)
    expiry_score = max(0, 100 - hours_left * 8)
    preference_bonus = 0
    if _is_preferred_category(market):
        preference_bonus += 50
    if settings.live_prioritize_in_play and _is_in_play_market(market):
        preference_bonus += 30
    return spread_score + volume_score + expiry_score + preference_bonus


async def _fetch_live_markets(log: LiveRunLog) -> list[dict]:
    """Fetch markets closing in our time window using server-side filters."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    max_close_ts = now_ts + (settings.live_max_expiry_hours * 3600)

    live: list[dict] = []
    seen_tickers: set[str] = set()
    cursor: str | None = None
    fetched_total = 0

    # max_close_ts/min_close_ts are only compatible with empty status filter.
    # We fetch by close window first, then keep only active/open markets locally.
    target = max(50, settings.live_market_fetch_target)
    for page in range(25):
        try:
            data = await kalshi.get_markets_auth(
                status=None,
                limit=1000,
                cursor=cursor,
                min_close_ts=now_ts,
                max_close_ts=max_close_ts,
                mve_filter="exclude",
            )
        except Exception as e:
            logger.warning("Live scalper market fetch page %d failed: %s", page, e)
            break

        markets = data.get("markets", [])
        if not markets:
            break
        fetched_total += len(markets)

        for m in markets:
            ticker = m.get("ticker", "")
            if not ticker or ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            status = (m.get("status") or "").lower()
            if status not in {"active", "open"}:
                continue
            _normalize_market(m)
            if not _is_lookup_focus_market(m):
                continue
            if _is_live(m, settings.live_max_expiry_hours):
                live.append(m)
            else:
                log.skipped_not_live += 1

        cursor = data.get("cursor")
        if not cursor:
            break
        if len(live) >= target:
            break
        await asyncio.sleep(0.03)

    log.events_scanned = 0
    log.live_events = 0
    log.markets_fetched = fetched_total
    log.live_markets = len(live)
    logger.info(
        "Live scalper: %d live markets from %d fetched (close-window query)",
        len(live), fetched_total,
    )
    return live


async def _get_recent_tickers() -> set[str]:
    """Return tickers this bot has ordered within the cooldown window."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.live_cooldown_minutes)
    async with async_session() as session:
        result = await session.execute(
            select(Order.ticker).where(
                Order.bot_id == BOT_ID,
                Order.created_at >= cutoff,
            )
        )
        return {row[0] for row in result.all()}


def _safe_parse_timeline(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


async def _update_run_stage(run_id: int, detail: str, log: LiveRunLog | None = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    if log is not None:
        log.stage_history.append({"at": now_iso, "detail": detail})

    async with async_session() as session:
        run = await session.get(StrategyRun, run_id)
        if not run:
            return
        run.status_detail = detail
        timeline = _safe_parse_timeline(run.status_timeline)
        timeline.append({"at": now_iso, "detail": detail})
        run.status_timeline = json.dumps(timeline)
        await session.commit()


def _evaluate_live_market(
    market: dict[str, Any],
    log: LiveRunLog,
    cooldown_tickers: set[str],
) -> list[tuple[CandidateResult, float, dict[str, Any]]]:
    """Run bias strategies on a single live market. Returns scored candidates."""
    ticker = market.get("ticker", "")
    event_ticker = market.get("event_ticker", "")
    category = market.get("category", "") or ""

    if _is_finance(event_ticker, category):
        log.skipped_finance += 1
        log._bucket_append("finance", market)
        return []

    if _is_exotic(ticker, event_ticker):
        log.skipped_exotic += 1
        log._bucket_append("exotic", market)
        return []

    if settings.live_exclude_crypto and _is_crypto_market(market):
        log.skipped_exotic += 1
        log._bucket_append("exotic", market)
        return []

    yes_bid = market.get("yes_bid") or 0
    yes_ask = market.get("yes_ask") or 0

    if not yes_bid and not yes_ask:
        log.skipped_no_liquidity += 1
        log._bucket_append("no_liquidity", market)
        return []

    is_priority = _is_priority_market(market)
    min_volume = settings.live_min_volume if not is_priority else max(20, settings.live_min_volume // 2)
    max_spread = settings.live_max_spread if not is_priority else settings.live_max_spread + 3

    volume = market.get("volume") or 0
    if volume < min_volume:
        log.skipped_low_volume += 1
        log._bucket_append("low_volume", market)
        return []

    spread = (yes_ask - yes_bid) if (yes_ask and yes_bid) else 99
    if spread > max_spread:
        log.skipped_wide_spread += 1
        log._bucket_append("wide_spread", market)
        return []

    if ticker in cooldown_tickers:
        log.skipped_cooldown += 1
        log._bucket_append("cooldown", market)
        return []

    log.evaluated += 1
    log._bucket_append("evaluated", market)
    yes_mid = (yes_bid + yes_ask) // 2 if (yes_bid and yes_ask) else (yes_bid or yes_ask)

    results: list[tuple[CandidateResult, float, dict[str, Any]]] = []
    if settings.live_use_lookup_scoring:
        decision = _LOOKUP_SCORER.score_market(market)
        if not decision:
            log._bucket_append("lookup_no_match", market)
            return []
        candidate = CandidateResult(
            ticker=market.get("ticker", ""),
            title=(market.get("title") or "")[:120],
            action="buy",
            side=decision.side,
            price=decision.yes_price,
            cost=decision.cost_cents,
            count=1,
            bias_type="lookup_edge",
            category=market.get("category") or market.get("event_ticker") or "",
            expiry_time=market.get("close_time") or market.get("expiration_time") or "",
            reason=decision.reason,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=volume,
        )
        score = max(0.0, decision.net_edge_cents) * 100 + min(decision.n_contracts / 1000, 100)
        results.append(
            (
                candidate,
                score,
                {
                    "predicted_win_rate": decision.predicted_win_rate,
                    "implied_probability": decision.implied_probability,
                    "predicted_edge": decision.edge,
                    "net_edge_cents": decision.net_edge_cents,
                    "sample_size": decision.n_contracts,
                    "lookup_key": f"{decision.price_bucket}|{decision.side}|{decision.hours_bucket}|{decision.category_prefix}",
                },
            )
        )
        return results

    for strategy_fn in BIAS_STRATEGIES:
        candidate = strategy_fn(market, yes_mid, yes_bid, yes_ask)
        if candidate:
            score = _score_candidate(candidate, market)
            results.append((candidate, score, {}))

    return results


async def auto_run() -> tuple[int, LiveRunLog]:
    """Execute one full live scalper cycle: scan → rank → cap → place.

    Returns (run_id, log).
    """
    log = LiveRunLog(dry_run=settings.live_dry_run)

    run = StrategyRun(bot_id=BOT_ID, started_at=datetime.now(timezone.utc))
    async with async_session() as session:
        session.add(run)
        await session.commit()
        run_id = run.id

    try:
        await _update_run_stage(run_id, "Fetching live markets...", log)
        markets = await _fetch_live_markets(log)
        await _update_run_stage(run_id, f"Evaluating {len(markets)} markets...", log)
        cooldown_tickers = await _get_recent_tickers()

        market_by_ticker: dict[str, dict[str, Any]] = {m.get("ticker", ""): m for m in markets}
        scored: list[tuple[CandidateResult, float, dict[str, Any]]] = []
        for m in markets:
            scored.extend(_evaluate_live_market(m, log, cooldown_tickers))

        log.candidates_found = len(scored)
        await _update_run_stage(run_id, f"Ranking {log.candidates_found} candidates...", log)

        scored.sort(key=lambda x: x[1], reverse=True)
        priority_scored = [
            (c, s, meta)
            for (c, s, meta) in scored
            if _is_priority_market(market_by_ticker.get(c.ticker, {}))
        ]
        ranked_pool = priority_scored if priority_scored else scored

        selected: list[CandidateResult] = []
        capital_used = 0
        event_capital: dict[str, int] = {}

        selected_meta: dict[str, dict[str, Any]] = {}
        for candidate, score, meta in ranked_pool:
            if len(selected) >= settings.live_max_orders_per_cycle:
                break
            if capital_used + candidate.cost > settings.live_max_capital_per_cycle:
                break
            event_key = candidate.category or candidate.ticker
            if event_capital.get(event_key, 0) + candidate.cost > settings.live_max_capital_per_event:
                continue
            selected.append(candidate)
            selected_meta[candidate.ticker] = meta
            capital_used += candidate.cost
            event_capital[event_key] = event_capital.get(event_key, 0) + candidate.cost

        log.auto_selected = len(selected)
        await _update_run_stage(run_id, f"Persisting {log.auto_selected} selected candidates...", log)

        # Persist candidates
        async with async_session() as session:
            for c in selected:
                session.add(Candidate(
                    bot_id=BOT_ID,
                    scan_id=run_id,
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
                    predicted_win_rate=selected_meta.get(c.ticker, {}).get("predicted_win_rate"),
                    implied_probability=selected_meta.get(c.ticker, {}).get("implied_probability"),
                    predicted_edge=selected_meta.get(c.ticker, {}).get("predicted_edge"),
                    net_edge_cents=selected_meta.get(c.ticker, {}).get("net_edge_cents"),
                    sample_size=selected_meta.get(c.ticker, {}).get("sample_size"),
                    lookup_key=selected_meta.get(c.ticker, {}).get("lookup_key", ""),
                    status="auto_selected" if not settings.live_dry_run else "dry_run",
                ))
                log.candidates.append({
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
                    "predicted_win_rate": selected_meta.get(c.ticker, {}).get("predicted_win_rate"),
                    "implied_probability": selected_meta.get(c.ticker, {}).get("implied_probability"),
                    "predicted_edge": selected_meta.get(c.ticker, {}).get("predicted_edge"),
                    "net_edge_cents": selected_meta.get(c.ticker, {}).get("net_edge_cents"),
                    "sample_size": selected_meta.get(c.ticker, {}).get("sample_size"),
                    "lookup_key": selected_meta.get(c.ticker, {}).get("lookup_key", ""),
                })
            await session.commit()

        if not settings.live_dry_run:
            await _update_run_stage(run_id, f"Placing {len(selected)} orders...", log)
            for c in selected:
                try:
                    resp = await kalshi.place_order(
                        ticker=c.ticker,
                        action=c.action,
                        side=c.side,
                        count=c.count,
                        yes_price=c.price,
                    )
                    order_data = resp.get("order", {})

                    async with async_session() as session:
                        session.add(Order(
                            bot_id=BOT_ID,
                            order_id=order_data.get("order_id", ""),
                            client_order_id=order_data.get("client_order_id", ""),
                            ticker=c.ticker,
                            side=c.side,
                            action=c.action,
                            price=c.price,
                            count=c.count,
                            status=order_data.get("status", "pending"),
                        ))
                        await session.commit()

                    log.auto_placed += 1
                    await asyncio.sleep(0.15)

                except Exception as e:
                    msg = f"Live scalper order failed for {c.ticker}: {e}"
                    logger.warning(msg)
                    log.errors.append(msg)
        else:
            await _update_run_stage(run_id, f"Dry run complete: {len(selected)} orders simulated", log)
            logger.info("Live scalper dry run: would place %d orders", len(selected))

    except Exception as e:
        msg = f"Live scalper run error: {e}"
        logger.error(msg)
        log.errors.append(msg)
        await _update_run_stage(run_id, f"Failed: {e}", log)

    async with async_session() as session:
        r = await session.get(StrategyRun, run_id)
        if r:
            r.ended_at = datetime.now(timezone.utc)
            r.markets_scanned = log.markets_fetched
            r.orders_placed = log.auto_placed
            r.errors = "\n".join(log.errors) if log.errors else ""
            r.status = "failed" if log.errors else "completed"
            if r.status == "completed":
                r.status_detail = f"Completed: {log.auto_placed} placed from {log.markets_fetched} markets"
            elif r.status_detail.startswith("Failed:"):
                pass
            else:
                r.status_detail = f"Completed with errors: {len(log.errors)}"
            now_iso = datetime.now(timezone.utc).isoformat()
            timeline = _safe_parse_timeline(r.status_timeline)
            timeline.append({"at": now_iso, "detail": r.status_detail})
            r.status_timeline = json.dumps(timeline)
            log.stage_history.append({"at": now_iso, "detail": r.status_detail})
            await session.commit()

    logger.info(
        "Live scalper done: %d selected, %d placed (dry=%s) from %d live markets",
        log.auto_selected, log.auto_placed, log.dry_run, log.live_markets,
    )
    return run_id, log


async def get_trade_proposal_context(ticker: str) -> dict[str, Any]:
    """Collect live + lookup context for AI trade proposals."""
    data = await kalshi.get_market(ticker)
    market = data.get("market", data)
    _normalize_market(market)

    yes_bid = market.get("yes_bid") or 0
    yes_ask = market.get("yes_ask") or 0
    mid = (yes_bid + yes_ask) / 2.0 if (yes_bid and yes_ask) else float(yes_bid or yes_ask or 0)
    spread = (yes_ask - yes_bid) if (yes_ask and yes_bid) else 99
    spread_cost = max(0.5, spread / 2.0) if spread >= 0 else 99.0

    category_prefix = _LOOKUP_SCORER._category_prefix(market)
    hours_left = _LOOKUP_SCORER._hours_left(market)
    hours_bucket = _LOOKUP_SCORER._hours_bucket(hours_left)

    yes_price = int(max(1, min(99, round(mid)))) if mid else 0
    no_price = 100 - yes_price if yes_price else 0
    yes_bucket = _LOOKUP_SCORER._price_bucket(yes_price) if yes_price else ""
    no_bucket = _LOOKUP_SCORER._price_bucket(no_price) if no_price else ""

    decision = _LOOKUP_SCORER.score_market(market)
    category_stats = _LOOKUP_SCORER.get_category_stats(category_prefix)

    return {
        "market": {
            "ticker": market.get("ticker", ticker),
            "title": market.get("title", ""),
            "subtitle": market.get("subtitle", ""),
            "event_ticker": market.get("event_ticker", ""),
            "category": market.get("category", ""),
            "status": market.get("status", ""),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": market.get("no_bid"),
            "no_ask": market.get("no_ask"),
            "volume": market.get("volume") or 0,
            "close_time": market.get("close_time") or market.get("expiration_time") or "",
            "rules_primary": market.get("rules_primary", ""),
            "rules_secondary": market.get("rules_secondary", ""),
        },
        "market_shape": {
            "category_prefix": category_prefix,
            "hours_left": round(hours_left, 3),
            "hours_bucket": hours_bucket,
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_price_bucket": yes_bucket,
            "no_price_bucket": no_bucket,
            "spread_cents": spread,
            "spread_cost_cents": spread_cost,
        },
        "lookup_decision": {
            "side": decision.side,
            "yes_price": decision.yes_price,
            "cost_cents": decision.cost_cents,
            "predicted_win_rate": decision.predicted_win_rate,
            "implied_probability": decision.implied_probability,
            "edge": decision.edge,
            "avg_ev_cents": decision.avg_ev_cents,
            "net_edge_cents": decision.net_edge_cents,
            "n_contracts": decision.n_contracts,
            "hours_bucket": decision.hours_bucket,
            "category_prefix": decision.category_prefix,
            "price_bucket": decision.price_bucket,
            "reason": decision.reason,
        } if decision else None,
        "category_stats": category_stats,
        "risk_limits": {
            "dry_run": settings.live_dry_run,
            "max_orders_per_cycle": settings.live_max_orders_per_cycle,
            "max_capital_per_cycle": settings.live_max_capital_per_cycle,
            "max_capital_per_event": settings.live_max_capital_per_event,
            "max_spread": settings.live_max_spread,
            "min_volume": settings.live_min_volume,
        },
    }
