from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


@dataclass
class LookupDecision:
    side: str
    yes_price: int
    cost_cents: int
    predicted_win_rate: float
    implied_probability: float
    edge: float
    avg_ev_cents: float
    net_edge_cents: float
    n_contracts: int
    hours_bucket: str
    category_prefix: str
    price_bucket: str
    reason: str


class LookupScorer:
    def __init__(
        self,
        lookup_path: str,
        min_samples: int,
        min_markets: int,
        fee_cents: float,
        min_edge_cents: float,
        min_prob_edge: float,
        allowed_hours_buckets: list[str] | None = None,
    ) -> None:
        self.lookup_path = Path(lookup_path)
        self.min_samples = min_samples
        self.min_markets = min_markets
        self.fee_cents = fee_cents
        self.min_edge_cents = min_edge_cents
        self.min_prob_edge = min_prob_edge
        self.allowed_hours_buckets = set(allowed_hours_buckets or [])
        self._table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._loaded = False

    def _hours_bucket(self, hours_left: float) -> str:
        if hours_left <= 1:
            return "0-1h"
        if hours_left <= 3:
            return "1-3h"
        if hours_left <= 6:
            return "3-6h"
        if hours_left <= 12:
            return "6-12h"
        if hours_left <= 24:
            return "12-24h"
        if hours_left <= 48:
            return "24-48h"
        return "48h+"

    def _price_bucket(self, price: int, bucket_size: int = 5) -> str:
        lo = ((max(1, price) - 1) // bucket_size) * bucket_size + 1
        hi = lo + bucket_size - 1
        return f"{lo:02d}-{hi:02d}c"

    def _category_prefix(self, market: dict[str, Any]) -> str:
        raw = (market.get("event_ticker") or "").upper()
        if not raw:
            return "UNKNOWN"
        token = []
        for ch in raw:
            if ch.isalnum():
                token.append(ch)
            else:
                break
        return "".join(token) or "UNKNOWN"

    def _hours_left(self, market: dict[str, Any]) -> float:
        close_str = market.get("close_time") or market.get("expiration_time") or ""
        if not close_str:
            return 999.0
        try:
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            return max((close_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0.0)
        except (ValueError, TypeError):
            return 999.0

    def load(self) -> None:
        if self._loaded:
            return
        if not self.lookup_path.exists():
            self._loaded = True
            return
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{self.lookup_path}'").df()
        table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = (
                str(row["price_bucket"]),
                str(row["side"]).lower(),
                str(row["hours_bucket"]),
                str(row["category_prefix"]).upper(),
            )
            table[key] = {
                "n_contracts": int(row["n_contracts"]),
                "n_markets": int(row.get("n_markets", 0)),
                "implied_probability": float(row["implied_probability"]),
                "win_rate": float(row["win_rate"]),
                "edge": float(row["edge"]),
                "avg_ev_cents": float(row["avg_ev_cents"]),
            }
        self._table = table
        self._loaded = True

    def score_market(self, market: dict[str, Any]) -> LookupDecision | None:
        self.load()
        if not self._table:
            return None
        yes_bid = market.get("yes_bid") or 0
        yes_ask = market.get("yes_ask") or 0
        if not (yes_bid or yes_ask):
            return None
        mid = (yes_bid + yes_ask) / 2.0 if (yes_bid and yes_ask) else float(yes_bid or yes_ask)
        spread = (yes_ask - yes_bid) if (yes_ask and yes_bid) else 8
        spread_cost = max(0.5, spread / 2.0)
        category = self._category_prefix(market)
        hours_left = self._hours_left(market)
        hours_bucket = self._hours_bucket(hours_left)
        if self.allowed_hours_buckets and hours_bucket not in self.allowed_hours_buckets:
            return None

        best: LookupDecision | None = None
        for side in ("yes", "no"):
            side_price = int(round(mid if side == "yes" else (100 - mid)))
            side_price = max(1, min(99, side_price))
            price_bucket = self._price_bucket(side_price)
            key = (price_bucket, side, hours_bucket, category)
            row = self._table.get(key)
            if not row:
                continue
            if row["n_contracts"] < self.min_samples:
                continue
            if row["n_markets"] < self.min_markets:
                continue
            if row["edge"] < self.min_prob_edge:
                continue
            net_edge = row["avg_ev_cents"] - spread_cost - self.fee_cents
            if net_edge < self.min_edge_cents:
                continue
            yes_price = side_price if side == "yes" else 100 - side_price
            decision = LookupDecision(
                side=side,
                yes_price=int(max(1, min(99, round(yes_price)))),
                cost_cents=int(side_price),
                predicted_win_rate=row["win_rate"],
                implied_probability=row["implied_probability"],
                edge=row["edge"],
                avg_ev_cents=row["avg_ev_cents"],
                net_edge_cents=float(net_edge),
                n_contracts=row["n_contracts"],
                hours_bucket=hours_bucket,
                category_prefix=category,
                price_bucket=price_bucket,
                reason=(
                    f"lookup edge {row['edge']:.3f}, ev {row['avg_ev_cents']:.2f}c, "
                    f"net {net_edge:.2f}c (samples={row['n_contracts']})"
                ),
            )
            if best is None or decision.net_edge_cents > best.net_edge_cents:
                best = decision
        return best

    def get_category_stats(self, category_prefix: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return raw lookup rows for a category prefix.

        Used by AI trade proposal flows that need broader context than the
        thresholded `score_market()` decision.
        """
        self.load()
        if not self._table:
            return []
        prefix = (category_prefix or "").upper().strip()
        if not prefix:
            return []

        rows: list[dict[str, Any]] = []
        for (price_bucket, side, hours_bucket, cat), row in self._table.items():
            if cat != prefix:
                continue
            rows.append(
                {
                    "price_bucket": price_bucket,
                    "side": side,
                    "hours_bucket": hours_bucket,
                    "category_prefix": cat,
                    "n_contracts": row["n_contracts"],
                    "n_markets": row["n_markets"],
                    "implied_probability": row["implied_probability"],
                    "win_rate": row["win_rate"],
                    "edge": row["edge"],
                    "avg_ev_cents": row["avg_ev_cents"],
                }
            )
        rows.sort(key=lambda x: (x["hours_bucket"], x["price_bucket"], x["side"]))
        return rows[: max(0, limit)]
