"""Port 1:1 de src/services/planet.service.ts."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from app.config import settings

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def get_planet_tile(z: int, x: int, y: int, mosaic_name: str) -> dict:
    api_key = settings.planet_api_key
    if not api_key:
        raise RuntimeError("PLANET_API_KEY n'est pas configurée dans le fichier .env")

    url = f"https://tiles.planet.com/basemaps/v1/planet-tiles/{mosaic_name}/gmap/{z}/{x}/{y}.png?api_key={quote(api_key)}"

    response = await _get_client().get(url)
    if response.status_code >= 400:
        raise RuntimeError(f"Planet tile error: {response.status_code} {response.reason_phrase}")

    return {"buffer": response.content, "contentType": response.headers.get("content-type") or "image/png"}
