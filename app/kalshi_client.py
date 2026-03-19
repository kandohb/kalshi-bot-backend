"""Kalshi API client with RSA-PSS authentication."""

from __future__ import annotations

import base64
import asyncio
import time
import uuid
from typing import Any

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.config import settings


class KalshiClient:
    """Async client for the Kalshi trading API."""

    def __init__(self) -> None:
        self._private_key = None
        self._base = settings.kalshi_base_url
        self._prefix = settings.kalshi_api_path_prefix
        self._key_id = settings.kalshi_api_key_id

    def _load_key(self):
        if self._private_key is None:
            with open(settings.kalshi_private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        return self._private_key

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        key = self._load_key()
        path_no_qs = path.split("?")[0]
        message = f"{timestamp_ms}{method}{path_no_qs}".encode("utf-8")
        signature = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    def _url(self, endpoint: str) -> str:
        return f"{self._base}{self._prefix}{endpoint}"

    def _path(self, endpoint: str) -> str:
        return f"{self._prefix}{endpoint}"

    # ── Public endpoints (no auth) ──────────────────────────────────────

    async def _request_with_backoff(
        self,
        method: str,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        timeout: int = 30,
        retries: int = 4,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    r = await c.request(
                        method=method,
                        url=self._url(endpoint),
                        headers=headers,
                        params=params,
                        json=json_data,
                    )
                if r.status_code == 429 and attempt < retries:
                    retry_after = r.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else (0.3 * (2**attempt))
                    await asyncio.sleep(min(max(0.1, sleep_s), 5.0))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_error = e
                if attempt >= retries:
                    break
                await asyncio.sleep(0.2 * (2**attempt))
        if last_error:
            raise last_error
        raise RuntimeError("request failed without error")

    async def get_series(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request_with_backoff("GET", "/series", params=params, timeout=30)

    async def get_events(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
        series_ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return await self._request_with_backoff("GET", "/events", params=params, timeout=30)

    async def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        return await self._request_with_backoff("GET", "/markets", params=params, timeout=30)

    async def get_markets_auth(
        self,
        status: str | None = "open",
        limit: int = 200,
        cursor: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        mve_filter: str | None = None,
    ) -> dict[str, Any]:
        """Authenticated markets fetch with optional server-side filters."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts
        if mve_filter:
            params["mve_filter"] = mve_filter
        return await self._auth_get("/markets", params)

    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        return await self._request_with_backoff("GET", f"/markets/{ticker}/orderbook", timeout=10)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        return await self._request_with_backoff("GET", f"/markets/{ticker}", timeout=10)

    # ── Authenticated endpoints ─────────────────────────────────────────

    async def _auth_get(self, endpoint: str, params: dict | None = None) -> dict:
        path = self._path(endpoint)
        headers = self._auth_headers("GET", path)
        return await self._request_with_backoff(
            "GET", endpoint, headers=headers, params=params, timeout=15
        )

    async def _auth_post(self, endpoint: str, data: dict) -> dict:
        path = self._path(endpoint)
        headers = self._auth_headers("POST", path)
        headers["Content-Type"] = "application/json"
        return await self._request_with_backoff(
            "POST", endpoint, headers=headers, json_data=data, timeout=15
        )

    async def _auth_delete(self, endpoint: str) -> dict:
        path = self._path(endpoint)
        headers = self._auth_headers("DELETE", path)
        return await self._request_with_backoff("DELETE", endpoint, headers=headers, timeout=15)

    async def get_balance(self) -> dict[str, Any]:
        return await self._auth_get("/portfolio/balance")

    async def get_positions(
        self, limit: int = 200, cursor: str | None = None, settlement_status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if settlement_status:
            params["settlement_status"] = settlement_status
        return await self._auth_get("/portfolio/positions", params)

    async def get_fills(
        self, limit: int = 100, cursor: str | None = None, ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        return await self._auth_get("/portfolio/fills", params)

    async def get_orders(
        self, status: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._auth_get("/portfolio/orders", params)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self._auth_get(f"/portfolio/orders/{order_id}")

    async def place_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: int,
        order_type: str = "limit",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        data = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "yes_price": yes_price,
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        return await self._auth_post("/portfolio/orders", data)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self._auth_delete(f"/portfolio/orders/{order_id}")


kalshi = KalshiClient()
