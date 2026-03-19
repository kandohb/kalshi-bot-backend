"""Routes for browsing Kalshi series and markets."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter

from app.kalshi_client import kalshi

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

_series_cache: dict[str, Any] = {"data": [], "ts": 0}
CACHE_TTL = 600  # 10 minutes

LEAGUE_RULES: list[tuple[str, list[str], list[str]]] = [
    # (league_label, ticker_prefixes, title_keywords)
    ("NBA", ["KXNBA"], ["nba"]),
    ("WNBA", ["KXWNBA"], ["wnba"]),
    ("NFL", ["KXNFL", "KXAFC", "KXNFC"], ["nfl", "super bowl"]),
    ("MLB", ["KXMLB", "KXNLMVP", "KXALMVP"], ["mlb", "world series"]),
    ("NHL", ["KXNHL"], ["nhl", "stanley cup"]),
    ("MLS", ["KXMLS"], ["mls"]),
    ("Premier League", ["KXEPL", "KXEPLGAME"], ["premier league", "epl"]),
    ("La Liga", ["KXLALIGA"], ["la liga"]),
    ("Serie A", ["KXSERIEA"], ["serie a"]),
    ("Bundesliga", ["KXBUNDESLIGA", "KXBULI"], ["bundesliga"]),
    ("Ligue 1", ["KXLIGUE1"], ["ligue 1"]),
    ("Champions League", ["KXUCL", "KXCHAMPIONSLEAGUE"], ["champions league", "ucl"]),
    ("NCAA Football", ["KXNCAAF", "KXCFB"], ["college football", "cfp", "ncaaf"]),
    ("NCAA Basketball", ["KXNCAAM", "KXNCAAB", "KXMARCHM"], ["march madness", "ncaa basketball", "ncaab"]),
    ("UFC / MMA", ["KXUFC", "KXMMA"], ["ufc", "mma"]),
    ("PGA / Golf", ["KXPGA", "KXGOLF", "KXPGATOP"], ["pga", "golf", "masters", "open championship"]),
    ("Tennis", ["KXATP", "KXWTA", "KXTENNIS"], ["tennis", "wimbledon", "us open tennis", "french open"]),
    ("F1 / Motorsport", ["KXF1", "KXNASCAR", "KXINDY"], ["formula 1", "f1", "nascar", "indycar"]),
    ("Cricket", ["KXCRICKET", "KXIPL"], ["cricket", "ipl"]),
    ("Rugby", ["KXRUGBY"], ["rugby"]),
    ("Olympics", ["KXOLYMPIC"], ["olympics", "olympic"]),
    ("Esports", ["KXESPORT", "KXLOL", "KXDOTA", "KXCS", "KXVALORANT", "KXEWC", "KXMIDSEASON"], ["esports", "league of legends", "valorant"]),
    ("Boxing", ["KXBOXING"], ["boxing"]),
]


def _classify_league(ticker: str, title: str, tags: list[str] | None) -> str:
    """Assign a league/competition label to a sports series."""
    t_upper = ticker.upper()
    t_lower = title.lower()

    for label, prefixes, keywords in LEAGUE_RULES:
        for prefix in prefixes:
            if t_upper.startswith(prefix):
                return label
        for kw in keywords:
            if kw in t_lower:
                return label

    tag_set = {(t or "").lower() for t in (tags or [])}
    if "soccer" in tag_set or "football" in tag_set and "soccer" in t_lower:
        return "Soccer (Other)"
    if "football" in tag_set:
        return "Football (Other)"
    if "basketball" in tag_set:
        return "Basketball (Other)"
    if "baseball" in tag_set:
        return "Baseball (Other)"
    if "hockey" in tag_set:
        return "Hockey (Other)"

    return "Other Sports"


async def _get_all_series() -> list[dict]:
    """Fetch all series from Kalshi with caching."""
    now = time.time()
    if _series_cache["data"] and now - _series_cache["ts"] < CACHE_TTL:
        return _series_cache["data"]

    all_series: list[dict] = []
    cursor: str | None = None

    for _ in range(100):
        data = await kalshi.get_series(limit=100, cursor=cursor)
        batch = data.get("series", [])
        if not batch:
            break
        for s in batch:
            tags = s.get("tags") or []
            cat = s.get("category", "") or ""
            entry: dict[str, Any] = {
                "ticker": s.get("ticker", ""),
                "title": s.get("title", ""),
                "category": cat,
                "tags": tags,
            }
            if cat.lower() == "sports":
                entry["league"] = _classify_league(entry["ticker"], entry["title"], tags)
            all_series.append(entry)
        cursor = data.get("cursor")
        if not cursor:
            break

    all_series.sort(key=lambda s: (s["category"], s.get("league", ""), s["title"]))
    _series_cache["data"] = all_series
    _series_cache["ts"] = now
    logger.info("Fetched %d series from Kalshi API", len(all_series))
    return all_series


@router.get("/series")
async def list_series():
    """Return all Kalshi series tickers with titles, category, and league."""
    series = await _get_all_series()
    return {"series": series}


@router.get("/leagues")
async def list_leagues():
    """Return available leagues/competitions for each category that has them."""
    series = await _get_all_series()

    leagues_by_category: dict[str, dict[str, int]] = {}
    for s in series:
        league = s.get("league")
        if not league:
            continue
        cat = s["category"]
        if cat not in leagues_by_category:
            leagues_by_category[cat] = {}
        leagues_by_category[cat][league] = leagues_by_category[cat].get(league, 0) + 1

    result = {}
    for cat, leagues in sorted(leagues_by_category.items()):
        result[cat] = sorted(
            [{"name": name, "count": count} for name, count in leagues.items()],
            key=lambda x: -x["count"],
        )

    return {"leagues": result}
