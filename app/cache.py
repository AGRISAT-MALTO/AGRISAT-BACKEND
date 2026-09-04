"""Port 1:1 du cache parcelles de src/server.ts (lignes ~96-209) : TTL 30s avec
stale-while-revalidate (revalidation en arrière-plan) et timeout de requête DB à 20s."""

from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import select

from app.db import async_session_maker
from app.models import Parcelle

logger = logging.getLogger("agrisat.cache")

PARCELLES_CACHE_TTL_S = 30.0
PARCELLES_QUERY_TIMEOUT_S = 20.0

_cache: dict | None = None  # {"expires_at": float, "rows": list[Parcelle]}
_inflight_request: asyncio.Task | None = None


async def _fetch_parcelles_rows() -> list[Parcelle]:
    async with async_session_maker() as session:
        result = await session.execute(select(Parcelle).order_by(Parcelle.created_at.desc()))
        return list(result.scalars().all())


async def _fetch_parcelles() -> list[Parcelle]:
    global _cache, _inflight_request
    if _inflight_request is None:
        _inflight_request = asyncio.ensure_future(asyncio.wait_for(_fetch_parcelles_rows(), timeout=PARCELLES_QUERY_TIMEOUT_S))

    try:
        rows = await _inflight_request
        _cache = {"rows": rows, "expires_at": time.monotonic() + PARCELLES_CACHE_TTL_S}
        return rows
    finally:
        _inflight_request = None


def _refresh_in_background() -> None:
    if _inflight_request is not None:
        return

    async def _run() -> None:
        try:
            await _fetch_parcelles()
        except Exception as error:  # noqa: BLE001
            logger.warning("parcelles: background refresh failed, keeping stale cache: %s", error)

    asyncio.ensure_future(_run())


async def load_parcelles() -> list[Parcelle]:
    if _cache is not None:
        if _cache["expires_at"] <= time.monotonic():
            _refresh_in_background()
        return _cache["rows"]

    return await _fetch_parcelles()


def invalidate() -> None:
    global _cache
    _cache = None
