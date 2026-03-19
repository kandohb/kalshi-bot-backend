from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.config import settings
from app.db import StrategyRun, Candidate, async_session
from app.kalshi_client import kalshi
from app.lookup_scorer import LookupScorer
from app.strategy import scan_markets, place_approved, ScanParams, BOT_ID

router = APIRouter(prefix="/api/strategy")

WIZARD_BOT_ID = "wizard_runner"
_LOOKUP = LookupScorer(
    lookup_path=settings.live_lookup_abs_path,
    min_samples=max(100, settings.live_lookup_min_samples // 5),
    min_markets=max(10, settings.live_lookup_min_markets // 2),
    fee_cents=settings.live_lookup_fee_cents,
    min_edge_cents=0.0,
    min_prob_edge=0.0,
    allowed_hours_buckets=[],
)


@dataclass
class WizardCandidate:
    ticker: str
    title: str
    category: str
    side: str
    yes_price: int
    cost_cents: int
    strategy: str
    reason: str
    yes_bid: int
    yes_ask: int
    volume: int
    close_time: str
    parquet_score: float
    ai_score: float
    execution_score: float
    final_score: float
    predicted_win_rate: float | None
    implied_probability: float | None
    predicted_edge: float | None
    net_edge_cents: float | None
    sample_size: int | None
    lookup_key: str
    ai_summary: str = ""
    alloc_cents: int = 0
    count: int = 0


class WizardAiOverride(BaseModel):
    ticker: str
    strategy: str
    side: str
    ai_score: float
    summary: str = ""


class WizardRunRequest(BaseModel):
    risk_budget_cents: int = 500
    market_scopes: list[str] = []
    strategies: list[str] = [
        "sure_thing_scrape",
        "longshot_no",
        "nasdaq_momentum",
        "nasdaq_mean_reversion",
    ]
    max_hours_to_close: int = 12
    max_candidates: int = 25
    min_volume: int = 20
    max_spread: int = 10
    dry_run: bool = True
    ai_enabled: bool = True
    relax_if_empty: bool = True
    preview_only: bool = False
    ai_overrides: list[WizardAiOverride] = []


def _normalize_market(m: dict[str, Any]) -> dict[str, Any]:
    def _to_cents(key: str) -> int | None:
        raw = m.get(key)
        if raw is None:
            return None
        try:
            return int(round(float(raw) * 100))
        except (TypeError, ValueError):
            return None

    m["yes_bid"] = _to_cents("yes_bid_dollars") or m.get("yes_bid")
    m["yes_ask"] = _to_cents("yes_ask_dollars") or m.get("yes_ask")
    m["no_bid"] = _to_cents("no_bid_dollars") or m.get("no_bid")
    m["no_ask"] = _to_cents("no_ask_dollars") or m.get("no_ask")
    for k in ("volume", "volume_24h"):
        src = f"{k}_fp"
        if m.get(src) is not None:
            try:
                m[k] = int(float(m[src]))
            except (TypeError, ValueError):
                pass
    return m


def _blob(market: dict[str, Any]) -> str:
    return " ".join(
        [
            str(market.get("category") or "").lower(),
            str(market.get("event_ticker") or "").lower(),
            str(market.get("ticker") or "").lower(),
            str(market.get("title") or "").lower(),
        ]
    )


def _in_scope(market: dict[str, Any], scopes: list[str]) -> bool:
    if not scopes:
        return True
    b = _blob(market)
    scope_set = {s.lower().strip() for s in scopes if s.strip()}
    if "all" in scope_set:
        return True
    for s in scope_set:
        if s == "nasdaq" and ("nasdaq100" in b or "kxnasdaq100" in b):
            return True
        if s in b:
            return True
    return False


def _hours_left(close_str: str) -> float:
    if not close_str:
        return 999.0
    try:
        dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        return max((dt - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0.0)
    except (TypeError, ValueError):
        return 999.0


def _strategy_candidates(market: dict[str, Any], strategies: set[str]) -> list[WizardCandidate]:
    yes_bid = market.get("yes_bid") or 0
    yes_ask = market.get("yes_ask") or 0
    if not (yes_bid or yes_ask):
        return []

    mid = int(round((yes_bid + yes_ask) / 2.0)) if (yes_bid and yes_ask) else int(yes_bid or yes_ask)
    ticker = market.get("ticker", "")
    title = (market.get("title") or "")[:120]
    category = market.get("category") or market.get("event_ticker") or ""
    close_time = market.get("close_time") or market.get("expiration_time") or ""
    volume = int(market.get("volume") or 0)
    is_nasdaq = "NASDAQ100" in str(market.get("event_ticker") or "").upper() or "NASDAQ100" in ticker.upper()

    out: list[WizardCandidate] = []

    if "sure_thing_scrape" in strategies and 80 <= mid <= 99:
        buy_yes = yes_ask or mid
        out.append(
            WizardCandidate(
                ticker=ticker,
                title=title,
                category=category,
                side="yes",
                yes_price=int(buy_yes),
                cost_cents=int(buy_yes),
                strategy="sure_thing_scrape",
                reason=f"YES mid {mid}c in high-probability zone (80c+).",
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                volume=volume,
                close_time=close_time,
                parquet_score=0.0,
                ai_score=0.0,
                execution_score=0.0,
                final_score=0.0,
                predicted_win_rate=None,
                implied_probability=None,
                predicted_edge=None,
                net_edge_cents=None,
                sample_size=None,
                lookup_key="",
            )
        )

    if "longshot_no" in strategies and 80 <= mid <= 99:
        no_cost = max(1, (100 - yes_bid) if yes_bid else (100 - mid))
        out.append(
            WizardCandidate(
                ticker=ticker,
                title=title,
                category=category,
                side="no",
                yes_price=100 - int(no_cost),
                cost_cents=int(no_cost),
                strategy="longshot_no",
                reason=f"Fading longshot YES (mid {mid}c) by buying NO.",
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                volume=volume,
                close_time=close_time,
                parquet_score=0.0,
                ai_score=0.0,
                execution_score=0.0,
                final_score=0.0,
                predicted_win_rate=None,
                implied_probability=None,
                predicted_edge=None,
                net_edge_cents=None,
                sample_size=None,
                lookup_key="",
            )
        )

    if "nasdaq_momentum" in strategies and is_nasdaq and 10 <= mid <= 30:
        buy_yes = yes_ask or mid
        out.append(
            WizardCandidate(
                ticker=ticker,
                title=title,
                category=category,
                side="yes",
                yes_price=int(buy_yes),
                cost_cents=int(buy_yes),
                strategy="nasdaq_momentum",
                reason=f"NASDAQ momentum setup: YES entry in {mid}c band.",
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                volume=volume,
                close_time=close_time,
                parquet_score=0.0,
                ai_score=0.0,
                execution_score=0.0,
                final_score=0.0,
                predicted_win_rate=None,
                implied_probability=None,
                predicted_edge=None,
                net_edge_cents=None,
                sample_size=None,
                lookup_key="",
            )
        )

    if "nasdaq_mean_reversion" in strategies and is_nasdaq and 80 <= mid <= 95:
        no_cost = max(1, (100 - yes_bid) if yes_bid else (100 - mid))
        out.append(
            WizardCandidate(
                ticker=ticker,
                title=title,
                category=category,
                side="no",
                yes_price=100 - int(no_cost),
                cost_cents=int(no_cost),
                strategy="nasdaq_mean_reversion",
                reason=f"NASDAQ mean-reversion setup: YES extended at {mid}c.",
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                volume=volume,
                close_time=close_time,
                parquet_score=0.0,
                ai_score=0.0,
                execution_score=0.0,
                final_score=0.0,
                predicted_win_rate=None,
                implied_probability=None,
                predicted_edge=None,
                net_edge_cents=None,
                sample_size=None,
                lookup_key="",
            )
        )

    return out


def _ai_score(candidate: WizardCandidate, ai_enabled: bool) -> float:
    if not ai_enabled:
        return 50.0
    if candidate.strategy.startswith("nasdaq_"):
        return 60.0
    return 52.0


def _execution_score(candidate: WizardCandidate) -> float:
    spread = (candidate.yes_ask - candidate.yes_bid) if (candidate.yes_ask and candidate.yes_bid) else 99
    spread_score = max(0.0, 100.0 - (spread * 10.0))
    volume_score = min(100.0, float(candidate.volume) / 10.0)
    h = _hours_left(candidate.close_time)
    if h <= 1:
        time_score = 25.0
    elif h <= 12:
        time_score = 100.0
    elif h <= 24:
        time_score = 70.0
    else:
        time_score = 40.0
    return (0.5 * spread_score) + (0.3 * volume_score) + (0.2 * time_score)


def _parquet_score(candidate: WizardCandidate) -> tuple[float, dict[str, Any]]:
    _LOOKUP.load()
    if not _LOOKUP._table:
        return 40.0, {}
    try:
        h = _LOOKUP._hours_bucket(_hours_left(candidate.close_time))
        category = _LOOKUP._category_prefix({"event_ticker": candidate.category})
        side_price = candidate.cost_cents
        pb = _LOOKUP._price_bucket(side_price)
        key = (pb, candidate.side, h, category)
        row = _LOOKUP._table.get(key)
        if not row:
            return 40.0, {"lookup_key": f"{pb}|{candidate.side}|{h}|{category}"}
        edge_component = max(0.0, min(1.0, (float(row["avg_ev_cents"]) + 10.0) / 20.0)) * 100.0
        hit_component = float(row["win_rate"]) * 100.0
        sample_component = min(1.0, math.log10(max(1, int(row["n_contracts"]))) / 3.0) * 100.0
        score = (0.45 * edge_component) + (0.40 * hit_component) + (0.15 * sample_component)
        return score, {
            "predicted_win_rate": float(row["win_rate"]),
            "implied_probability": float(row["implied_probability"]),
            "predicted_edge": float(row["edge"]),
            "net_edge_cents": float(row["avg_ev_cents"]) - settings.live_lookup_fee_cents,
            "sample_size": int(row["n_contracts"]),
            "lookup_key": f"{pb}|{candidate.side}|{h}|{category}",
        }
    except Exception:
        return 40.0, {}


def _allocate_budget(candidates: list[WizardCandidate], budget_cents: int) -> list[WizardCandidate]:
    eligible = [c for c in candidates if c.final_score >= 60.0]
    if not eligible:
        return []
    weights = [(c, max(0.0, c.final_score - 55.0) ** 1.5) for c in eligible]
    total_weight = sum(w for _, w in weights)
    if total_weight <= 0:
        return []

    remaining = budget_cents
    max_per_ticker = max(1, int(budget_cents * 0.25))
    ticker_used: dict[str, int] = {}
    selected: list[WizardCandidate] = []
    for c, w in sorted(weights, key=lambda x: x[0].final_score, reverse=True):
        if remaining <= 0:
            break
        ticker_remaining = max_per_ticker - ticker_used.get(c.ticker, 0)
        if ticker_remaining <= 0:
            continue
        target_alloc = int((budget_cents * w) / total_weight)
        target_alloc = min(target_alloc, ticker_remaining, remaining)
        if target_alloc < c.cost_cents:
            continue
        count = max(1, target_alloc // max(1, c.cost_cents))
        alloc = count * c.cost_cents
        if alloc > remaining:
            count = remaining // max(1, c.cost_cents)
            alloc = count * c.cost_cents
        if alloc > ticker_remaining:
            count = ticker_remaining // max(1, c.cost_cents)
            alloc = count * c.cost_cents
        if count <= 0 or alloc <= 0:
            continue
        c.count = int(count)
        c.alloc_cents = int(alloc)
        selected.append(c)
        remaining -= alloc
        ticker_used[c.ticker] = ticker_used.get(c.ticker, 0) + alloc
    return selected


def _apply_ai_overrides(
    candidates: list[WizardCandidate], overrides: list[WizardAiOverride]
) -> None:
    if not overrides:
        return
    override_map = {
        f"{o.ticker}|{o.strategy}|{o.side.lower()}": o for o in overrides
    }
    for c in candidates:
        key = f"{c.ticker}|{c.strategy}|{c.side.lower()}"
        o = override_map.get(key)
        if not o:
            continue
        c.ai_score = round(max(0.0, min(100.0, float(o.ai_score))), 2)
        c.ai_summary = (o.summary or "")[:280]
        c.final_score = round(
            (0.65 * c.parquet_score) + (0.25 * c.ai_score) + (0.10 * c.execution_score),
            2,
        )


def _dedupe_ticker_side(candidates: list[WizardCandidate]) -> list[WizardCandidate]:
    """Keep only the strongest candidate per (ticker, side).

    Multiple strategies can nominate the same side of the same ticker. We only
    keep the highest-scoring one to avoid duplicated exposure.
    """
    best: dict[tuple[str, str], WizardCandidate] = {}
    for c in candidates:
        key = (c.ticker, c.side.lower())
        prev = best.get(key)
        if not prev or c.final_score > prev.final_score:
            best[key] = c
    return list(best.values())


@router.post("/wizard-run")
async def wizard_run(req: WizardRunRequest):
    if req.risk_budget_cents <= 0:
        raise HTTPException(status_code=400, detail="risk_budget_cents must be > 0")
    if not req.strategies:
        raise HTTPException(status_code=400, detail="Select at least one strategy")

    markets: list[dict[str, Any]] = []
    errors: list[str] = []
    now_ts = int(datetime.now(timezone.utc).timestamp())
    max_close_ts = now_ts + (max(1, req.max_hours_to_close) * 3600)
    filter_profile = "strict"
    used_min_volume = req.min_volume
    used_max_spread = req.max_spread

    try:
        async def _collect_markets(min_volume: int, max_spread: int) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            cursor: str | None = None
            for _ in range(20):
                data = await kalshi.get_markets_auth(
                    status=None,
                    limit=1000,
                    cursor=cursor,
                    min_close_ts=now_ts,
                    max_close_ts=max_close_ts,
                    mve_filter="exclude",
                )
                chunk = data.get("markets", [])
                if not chunk:
                    break
                for m in chunk:
                    status = (m.get("status") or "").lower()
                    if status not in {"active", "open"}:
                        continue
                    _normalize_market(m)
                    if not _in_scope(m, req.market_scopes):
                        continue
                    yes_bid = m.get("yes_bid") or 0
                    yes_ask = m.get("yes_ask") or 0
                    volume = int(m.get("volume") or 0)
                    spread = (yes_ask - yes_bid) if (yes_bid and yes_ask) else 99
                    if volume < min_volume or spread > max_spread:
                        continue
                    out.append(m)
                cursor = data.get("cursor")
                if not cursor or len(out) >= 4000:
                    break
            return out

        markets = await _collect_markets(req.min_volume, req.max_spread)
        if not markets and req.relax_if_empty:
            relaxed_min_volume = max(1, req.min_volume // 2)
            relaxed_max_spread = req.max_spread + 4
            markets = await _collect_markets(relaxed_min_volume, relaxed_max_spread)
            if markets:
                filter_profile = "relaxed"
                used_min_volume = relaxed_min_volume
                used_max_spread = relaxed_max_spread
                errors.append(
                    f"Applied relaxed filters: min_volume={relaxed_min_volume}, max_spread={relaxed_max_spread}"
                )

        strategy_set = {s.strip() for s in req.strategies if s.strip()}
        candidates: list[WizardCandidate] = []
        for m in markets:
            candidates.extend(_strategy_candidates(m, strategy_set))

        scored: list[WizardCandidate] = []
        for c in candidates:
            pscore, meta = _parquet_score(c)
            c.parquet_score = round(pscore, 2)
            c.ai_score = round(_ai_score(c, req.ai_enabled), 2)
            c.execution_score = round(_execution_score(c), 2)
            c.final_score = round((0.65 * c.parquet_score) + (0.25 * c.ai_score) + (0.10 * c.execution_score), 2)
            c.predicted_win_rate = meta.get("predicted_win_rate")
            c.implied_probability = meta.get("implied_probability")
            c.predicted_edge = meta.get("predicted_edge")
            c.net_edge_cents = meta.get("net_edge_cents")
            c.sample_size = meta.get("sample_size")
            c.lookup_key = meta.get("lookup_key", "")
            scored.append(c)

        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[: max(1, req.max_candidates)]
        _apply_ai_overrides(top, req.ai_overrides)
        top.sort(key=lambda x: x.final_score, reverse=True)
        deduped = _dedupe_ticker_side(top)
        deduped.sort(key=lambda x: x.final_score, reverse=True)

        if req.preview_only:
            return {
                "run_id": None,
                "dry_run": True,
                "preview_only": True,
                "filter_profile": filter_profile,
                "used_min_volume": used_min_volume,
                "used_max_spread": used_max_spread,
                "markets_scanned": len(markets),
                "raw_candidates": len(candidates),
                "ranked_candidates": len(deduped),
                "deduped_out": max(0, len(top) - len(deduped)),
                "selected_candidates": 0,
                "orders_placed": 0,
                "errors": [],
                "candidates": [
                    {
                        "ticker": c.ticker,
                        "title": c.title,
                        "category": c.category,
                        "strategy": c.strategy,
                        "side": c.side,
                        "yes_price": c.yes_price,
                        "cost_cents": c.cost_cents,
                        "count": 0,
                        "alloc_cents": 0,
                        "reason": c.reason,
                        "parquet_score": c.parquet_score,
                        "ai_score": c.ai_score,
                        "execution_score": c.execution_score,
                        "final_score": c.final_score,
                        "predicted_win_rate": c.predicted_win_rate,
                        "implied_probability": c.implied_probability,
                        "predicted_edge": c.predicted_edge,
                        "net_edge_cents": c.net_edge_cents,
                        "sample_size": c.sample_size,
                        "lookup_key": c.lookup_key,
                        "ai_summary": c.ai_summary,
                    }
                    for c in deduped
                ],
            }

        selected = _allocate_budget(deduped, req.risk_budget_cents)
        run = StrategyRun(bot_id=WIZARD_BOT_ID, started_at=datetime.now(timezone.utc), status="running")
        async with async_session() as session:
            session.add(run)
            await session.commit()
            run_id = run.id

        placed = 0
        async with async_session() as session:
            for c in selected:
                session.add(
                    Candidate(
                        bot_id=WIZARD_BOT_ID,
                        scan_id=run_id,
                        ticker=c.ticker,
                        title=c.title,
                        side=c.side,
                        action="buy",
                        price=c.yes_price,
                        cost=c.cost_cents,
                        count=c.count,
                        bias_type=c.strategy,
                        category=c.category,
                        expiry_time=c.close_time,
                        reason=c.reason,
                        ai_explanation=c.ai_summary,
                        status="auto_selected" if not req.dry_run else "dry_run",
                        yes_bid=c.yes_bid,
                        yes_ask=c.yes_ask,
                        volume=c.volume,
                        predicted_win_rate=c.predicted_win_rate,
                        implied_probability=c.implied_probability,
                        predicted_edge=c.predicted_edge,
                        net_edge_cents=c.net_edge_cents,
                        sample_size=c.sample_size,
                        lookup_key=c.lookup_key,
                    )
                )
            await session.commit()

        if not req.dry_run:
            for c in selected:
                try:
                    resp = await kalshi.place_order(
                        ticker=c.ticker,
                        action="buy",
                        side=c.side,
                        count=c.count,
                        yes_price=c.yes_price,
                    )
                    order = resp.get("order", {})
                    async with async_session() as session:
                        from app.db import Order

                        session.add(
                            Order(
                                bot_id=WIZARD_BOT_ID,
                                order_id=order.get("order_id", ""),
                                client_order_id=order.get("client_order_id", ""),
                                ticker=c.ticker,
                                side=c.side,
                                action="buy",
                                price=c.yes_price,
                                count=c.count,
                                status=order.get("status", "pending"),
                            )
                        )
                        await session.commit()
                    placed += 1
                except Exception as exc:
                    errors.append(f"{c.ticker}: {exc}")

        async with async_session() as session:
            r = await session.get(StrategyRun, run_id)
            if r:
                r.ended_at = datetime.now(timezone.utc)
                r.markets_scanned = len(markets)
                r.orders_placed = placed
                r.errors = "\n".join(errors) if errors else ""
                r.status = "completed" if not errors else "completed_with_errors"
                r.status_detail = f"{len(selected)} selected, {placed} placed"
                await session.commit()

        return {
            "run_id": run_id,
            "dry_run": req.dry_run,
            "preview_only": False,
            "filter_profile": filter_profile,
            "used_min_volume": used_min_volume,
            "used_max_spread": used_max_spread,
            "markets_scanned": len(markets),
            "raw_candidates": len(candidates),
            "ranked_candidates": len(deduped),
            "deduped_out": max(0, len(top) - len(deduped)),
            "selected_candidates": len(selected),
            "orders_placed": placed,
            "errors": errors,
            "candidates": [
                {
                    "ticker": c.ticker,
                    "title": c.title,
                    "category": c.category,
                    "strategy": c.strategy,
                    "side": c.side,
                    "yes_price": c.yes_price,
                    "cost_cents": c.cost_cents,
                    "count": c.count,
                    "alloc_cents": c.alloc_cents,
                    "reason": c.reason,
                    "parquet_score": c.parquet_score,
                    "ai_score": c.ai_score,
                    "execution_score": c.execution_score,
                    "final_score": c.final_score,
                    "predicted_win_rate": c.predicted_win_rate,
                    "implied_probability": c.implied_probability,
                    "predicted_edge": c.predicted_edge,
                    "net_edge_cents": c.net_edge_cents,
                    "sample_size": c.sample_size,
                    "lookup_key": c.lookup_key,
                    "ai_summary": c.ai_summary,
                }
                for c in selected
            ],
        }
    except Exception as exc:
        if not req.preview_only and "run_id" in locals():
            async with async_session() as session:
                r = await session.get(StrategyRun, run_id)
                if r:
                    r.ended_at = datetime.now(timezone.utc)
                    r.status = "failed"
                    r.errors = str(exc)
                    r.status_detail = f"Failed: {exc}"
                    await session.commit()
        raise HTTPException(status_code=500, detail=f"wizard run failed: {exc}") from exc


class ScanRequest(BaseModel):
    max_expiry_days: int = 7
    biases: list[str] = ["longshot_no", "favorite_yes", "asymmetry"]
    category: str = ""
    series_ticker: str = ""
    league_series: list[str] = []


@router.post("/scan")
async def trigger_scan(req: ScanRequest | None = None):
    """Scan markets and generate trade candidates (no orders placed).

    Accepts optional params: max_expiry_days, biases, category, series_ticker, league_series.
    """
    params = ScanParams(
        max_expiry_days=req.max_expiry_days if req else 7,
        biases=req.biases if req else ["longshot_no", "favorite_yes", "asymmetry"],
        category=req.category if req else "",
        series_ticker=req.series_ticker if req else "",
        league_series=req.league_series if req else [],
    )
    scan_id, slog = await scan_markets(params)
    return {
        "scan_id": scan_id,
        "log": slog.to_dict(),
    }


class PlaceRequest(BaseModel):
    candidate_ids: list[int]


@router.post("/place")
async def trigger_place(req: PlaceRequest):
    """Place orders for user-approved candidates."""
    if not req.candidate_ids:
        raise HTTPException(status_code=400, detail="No candidates selected")
    result = await place_approved(req.candidate_ids)
    return result


class UpdateCandidateRequest(BaseModel):
    status: str  # "approved" or "rejected"


@router.patch("/candidates/{candidate_id}")
async def update_candidate(candidate_id: int, req: UpdateCandidateRequest):
    """Update candidate status (approve/reject)."""
    if req.status not in ("approved", "rejected", "pending"):
        raise HTTPException(status_code=400, detail="Invalid status")
    async with async_session() as session:
        cand = await session.get(Candidate, candidate_id)
        if not cand:
            raise HTTPException(status_code=404, detail="Candidate not found")
        cand.status = req.status
        await session.commit()
        return {"id": cand.id, "status": cand.status}


@router.get("/candidates/{scan_id}")
async def get_candidates(scan_id: int):
    """Get all candidates for a scan."""
    async with async_session() as session:
        result = await session.execute(
            select(Candidate).where(
                Candidate.scan_id == scan_id,
                Candidate.bot_id == BOT_ID,
            ).order_by(Candidate.id)
        )
        rows = list(result.scalars().all())
    return {
        "candidates": [
            {
                "id": c.id,
                "scan_id": c.scan_id,
                "ticker": c.ticker,
                "title": c.title,
                "side": c.side,
                "price": c.price,
                "cost": c.cost,
                "count": c.count,
                "bias_type": c.bias_type,
                "category": c.category,
                "expiry_time": c.expiry_time,
                "reason": c.reason,
                "ai_explanation": c.ai_explanation,
                "status": c.status,
                "yes_bid": c.yes_bid,
                "yes_ask": c.yes_ask,
                "volume": c.volume,
            }
            for c in rows
        ]
    }


@router.get("/status")
async def strategy_status():
    """Return the most recent strategy run."""
    async with async_session() as session:
        result = await session.execute(
            select(StrategyRun).where(StrategyRun.bot_id == BOT_ID).order_by(StrategyRun.id.desc()).limit(1)
        )
        run = result.scalar_one_or_none()
    if not run:
        return {"last_run": None}
    return {
        "last_run": {
            "id": run.id,
            "status": run.status,
            "markets_scanned": run.markets_scanned,
            "orders_placed": run.orders_placed,
            "errors": run.errors or None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        }
    }
