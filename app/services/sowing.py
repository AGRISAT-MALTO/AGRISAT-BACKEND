"""Modèle de date de semis (orge) — inversion thermique de l'échelle Zadoks.

Principe (cohérent avec ``app/services/zadoks.py``) :
- TR (zéro de végétation orge) = 0 °C, DJ = max(0, (Tmax+Tmin)/2 - TR)
- ST(code) = seuil thermique du stade Zadoks observé (depuis le semis)
- On cumule les DJ **à rebours** depuis la date d'observation (données
  Open-Meteo archive) jusqu'à atteindre ST(code) → la date atteinte est le
  semis estimé. Intervalle d'incertitude via ST ± tolérance.

Méthodes exposées :
- ``estimate_sowing_from_stage`` : inversion à partir d'un stade observé
  (code Zadoks ou ST cible) — la plus fiable quand le stade terrain est connu.
- ``estimate_sowing_from_emergence`` : levée connue (ST = 30 °C) → semis.
- ``estimate_sowing_from_ndvi`` : série NDVI mensuelle (Sentinel-2 GEE) →
  détection du mois d'émergence → rétro-projection thermique de 30 °C.
- ``agro_window_fallback`` : fenêtre agro-climatique quand la thermique échoue.

Aucune dépendance GEE : seul Open-Meteo (archive) est interrogé.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from app.services.zadoks import (
    BARLEY_TR,
    EARLY_STAGES,
    FLOWERING_STAGES,
    STAGE_META,
    stage_for_st,
    stage_since_sowing,
)

logger = logging.getLogger("agrisat.sowing")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_S = 30.0
MAX_HISTORY_DAYS = 400  # couvre un cycle orge complet + marge
EMERGENCE_ST = 30.0  # Zadoks 09 : levée = ST 30 °C depuis le semis
HARVEST_ST = 2100.0  # Zadoks 99 : produit récolté (2000–2100 °C)
HARVEST_PROJECTION_DAYS = 120  # horizon max de projection (prévision + climatologie)

# Seuils ST par code Zadoks (séquence principale + floraison parallèle).
ST_BY_CODE: dict[str, float] = {code: float(seuil) for seuil, code, _ in EARLY_STAGES}
ST_BY_CODE.update({code: float(seuil) for seuil, code, _ in FLOWERING_STAGES})


def st_for_code(zadoks_code: str) -> float | None:
    """Seuil ST (°C) associé à un code Zadoks, ou None si inconnu."""
    if not isinstance(zadoks_code, str):
        return None
    return ST_BY_CODE.get(zadoks_code.strip())


def st_source_for_code(zadoks_code: str) -> str:
    """'user' si ST posée avec l'utilisateur, 'interpolated' sinon."""
    meta = STAGE_META.get(zadoks_code.strip()) if isinstance(zadoks_code, str) else None
    if isinstance(meta, dict) and isinstance(meta.get("st_source"), str):
        return meta["st_source"]
    return "unknown"


def degree_day(tmax: float, tmin: float, tr: float = BARLEY_TR) -> float:
    """DJ journalier, plancher à 0 (même définition que zadoks.py)."""
    try:
        return max(0.0, (float(tmax) + float(tmin)) / 2.0 - float(tr))
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def invert_sowing(
    daily: list[dict[str, Any]],
    observation_iso: str,
    target_st: float,
    tr: float = BARLEY_TR,
) -> dict[str, Any]:
    """Remonte le cumul DJ depuis l'observation jusqu'à ``target_st``.

    ``daily`` : relevés ``{date: YYYY-MM-DD, tmax, tmin}`` ou ``{date, dj}``,
    triés ou non. Retourne le semis estimé + métriques (ST atteinte,
    jours comptés, couverture). Ne fait aucun appel réseau : testable pur.
    """
    obs = _parse_iso(observation_iso)
    if obs is None:
        return {"sowingDate": None, "error": "Date d'observation invalide (YYYY-MM-DD attendue)."}
    if not isinstance(target_st, (int, float)) or target_st < 0:
        return {"sowingDate": None, "error": "ST cible invalide."}
    target = float(target_st)
    if target == 0:
        return {
            "sowingDate": obs.isoformat(),
            "stAchieved": 0.0,
            "daysCounted": 0,
            "outOfRange": False,
            "coverage": 1.0,
        }

    # Normalise : {date, dj}, jours <= observation, triés croissants.
    norm: list[tuple[date, float]] = []
    for row in daily:
        if not isinstance(row, dict):
            continue
        day = _parse_iso(row.get("date"))
        if day is None or day > obs:
            continue
        dj = row.get("dj")
        if isinstance(dj, (int, float)):
            dj_val = max(0.0, float(dj))
        elif isinstance(row.get("tmax"), (int, float)) and isinstance(row.get("tmin"), (int, float)):
            dj_val = degree_day(row["tmax"], row["tmin"], tr)
        else:
            continue
        norm.append((day, dj_val))
    norm.sort(key=lambda r: r[0])
    if not norm:
        return {"sowingDate": None, "error": "Aucune donnée thermique ≤ date d'observation."}

    # Couverture : jours manquants sur la fenêtre couverte.
    span_days = (norm[-1][0] - norm[0][0]).days + 1 if len(norm) > 1 else 1
    coverage = min(1.0, len(norm) / max(1, span_days))

    # Cumul à rebours depuis l'observation.
    cumulative = 0.0
    sowing: date | None = None
    for day, dj_val in reversed(norm):
        cumulative += dj_val
        if cumulative >= target:
            sowing = day
            break

    cumulative = round(cumulative * 10) / 10
    if sowing is None:
        return {
            "sowingDate": None,
            "stAchieved": cumulative,
            "daysCounted": len(norm),
            "outOfRange": True,
            "coverage": round(coverage * 1000) / 1000,
            "oldestAvailable": norm[0][0].isoformat(),
            "error": f"Historique insuffisant : {cumulative} °C cumulés < cible {target} °C.",
        }
    days_counted = (obs - sowing).days + 1
    return {
        "sowingDate": sowing.isoformat(),
        "stAchieved": cumulative,
        "daysCounted": days_counted,
        "outOfRange": False,
        "coverage": round(coverage * 1000) / 1000,
        "oldestAvailable": norm[0][0].isoformat(),
    }


def _confidence(
    *,
    inversion: dict[str, Any],
    st_source: str = "unknown",
    zero_dj_days: int = 0,
    days_counted: int = 0,
) -> int:
    """Score 0-95 : complétude, ambiguïté hivernale, source ST."""
    if inversion.get("sowingDate") is None:
        return 0
    score = 90.0
    coverage = inversion.get("coverage")
    if isinstance(coverage, (int, float)) and coverage < 0.95:
        score -= (0.95 - float(coverage)) * 100 * 0.8  # jusqu'à -~15
    if st_source == "interpolated":
        score -= 10
    elif st_source == "unknown":
        score -= 5
    # Longue traversée hivernale (DJ = 0) → inversion ambiguë à ± plusieurs jours.
    if zero_dj_days > 20:
        score -= 10
    elif zero_dj_days > 10:
        score -= 5
    # Fenêtre très longue → incertitude cumulée.
    if days_counted > 300:
        score -= 5
    return max(5, min(95, round(score)))


def _zero_dj_days(daily: list[dict[str, Any]], start_iso: str | None, end_iso: str) -> int:
    if not start_iso:
        return 0
    count = 0
    for row in daily:
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        if not isinstance(d, str) or d < start_iso or d > end_iso:
            continue
        dj = row.get("dj")
        if isinstance(dj, (int, float)) and float(dj) <= 0:
            count += 1
        elif isinstance(row.get("tmax"), (int, float)) and isinstance(row.get("tmin"), (int, float)):
            if degree_day(row["tmax"], row["tmin"]) <= 0:
                count += 1
    return count


async def fetch_daily_temps(lat: float, lng: float, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
    """Températures journalières Open-Meteo archive → [{date, tmax, tmin, dj}]."""
    params = {
        "latitude": str(lat),
        "longitude": str(lng),
        "start_date": start_iso,
        "end_date": end_iso,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        response = await client.get(OPEN_METEO_ARCHIVE_URL, params=params)
    if response.status_code >= 400:
        raise RuntimeError("Les données Open-Meteo sont indisponibles.")
    payload = response.json()
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict):
        raise RuntimeError("Réponse météo incomplète.")
    dates = daily.get("time")
    tmax_values = daily.get("temperature_2m_max")
    tmin_values = daily.get("temperature_2m_min")
    if not isinstance(dates, list) or not isinstance(tmax_values, list) or not isinstance(tmin_values, list):
        raise RuntimeError("Températures journalières indisponibles.")
    out: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        tmax = tmax_values[i] if i < len(tmax_values) else None
        tmin = tmin_values[i] if i < len(tmin_values) else None
        if not isinstance(d, str) or not isinstance(tmax, (int, float)) or not isinstance(tmin, (int, float)):
            continue
        out.append({"date": d, "tmax": tmax, "tmin": tmin, "dj": round(degree_day(tmax, tmin) * 10) / 10})
    return out


async def fetch_forecast_temps(lat: float, lng: float, days: int = 16) -> list[dict[str, Any]]:
    """Prévision Open-Meteo (16 j max) → [{date, tmax, tmin, dj, forecast: True}]."""
    params = {
        "latitude": str(lat),
        "longitude": str(lng),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
        "forecast_days": max(1, min(int(days), 16)),
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        response = await client.get(OPEN_METEO_FORECAST_URL, params=params)
    if response.status_code >= 400:
        raise RuntimeError("Les prévisions Open-Meteo sont indisponibles.")
    payload = response.json()
    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not isinstance(daily, dict):
        raise RuntimeError("Réponse prévision incomplète.")
    dates = daily.get("time")
    tmax_values = daily.get("temperature_2m_max")
    tmin_values = daily.get("temperature_2m_min")
    if not isinstance(dates, list) or not isinstance(tmax_values, list) or not isinstance(tmin_values, list):
        raise RuntimeError("Températures de prévision indisponibles.")
    out: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        tmax = tmax_values[i] if i < len(tmax_values) else None
        tmin = tmin_values[i] if i < len(tmin_values) else None
        if not isinstance(d, str) or not isinstance(tmax, (int, float)) or not isinstance(tmin, (int, float)):
            continue
        out.append({"date": d, "tmax": tmax, "tmin": tmin, "dj": round(degree_day(tmax, tmin) * 10) / 10, "forecast": True})
    return out


def project_harvest_date(
    daily_since_sowing: list[dict[str, Any]],
    forecast: list[dict[str, Any]] | None = None,
    *,
    harvest_st: float = HARVEST_ST,
    horizon_days: int = HARVEST_PROJECTION_DAYS,
    tr: float = BARLEY_TR,
) -> dict[str, Any]:
    """Date de récolte = jour où ST cumulée depuis le semis atteint 2100 °C.

    ``daily_since_sowing`` : relevés [{date, dj|tmax,tmin}] depuis le semis.
    Au-delà de l'archive : prévision 16 j si fournie, sinon climatologie
    (moyenne des DJ observés, plancher 1 °C/j pour ne pas projeter à l'infini
    en plein hiver). Pur, sans appel réseau.
    """
    # Normalise l'observé : {date, dj} trié croissant.
    norm: list[tuple[date, float]] = []
    for row in daily_since_sowing:
        if not isinstance(row, dict):
            continue
        day = _parse_iso(row.get("date"))
        if day is None:
            continue
        dj = row.get("dj")
        if isinstance(dj, (int, float)):
            dj_val = max(0.0, float(dj))
        elif isinstance(row.get("tmax"), (int, float)) and isinstance(row.get("tmin"), (int, float)):
            dj_val = degree_day(row["tmax"], row["tmin"], tr)
        else:
            continue
        norm.append((day, dj_val))
    norm.sort(key=lambda r: r[0])
    if not norm:
        return {"harvestDate": None, "error": "Aucune donnée thermique depuis le semis."}

    cumulative = round(sum(dj for _, dj in norm) * 10) / 10
    last_day = norm[-1][0]
    if cumulative >= harvest_st:
        # Jour exact de franchissement du seuil (cumul progressif).
        running = 0.0
        cross_day = last_day
        for day, dj_val in norm:
            running += dj_val
            if running >= harvest_st:
                cross_day = day
                break
        return {
            "harvestDate": cross_day.isoformat(),
            "stAchieved": cumulative,
            "alreadyReached": True,
            "method": "thermal-observed",
        }

    # Prévision 16 j (jours après l'archive uniquement).
    fc: list[tuple[date, float]] = []
    for row in forecast or []:
        if not isinstance(row, dict):
            continue
        day = _parse_iso(row.get("date"))
        if day is None or day <= last_day:
            continue
        dj = row.get("dj")
        if isinstance(dj, (int, float)):
            fc.append((day, max(0.0, float(dj))))
        elif isinstance(row.get("tmax"), (int, float)) and isinstance(row.get("tmin"), (int, float)):
            fc.append((day, degree_day(row["tmax"], row["tmin"], tr)))
    fc.sort(key=lambda r: r[0])

    mean_dj = sum(dj for _, dj in norm) / len(norm)
    climatology_dj = max(1.0, mean_dj)  # plancher : hiver à DJ ~0 sinon horizon infini
    running = cumulative
    cursor = last_day
    used_forecast = 0
    for i in range(horizon_days):
        cursor += timedelta(days=1)
        if i < len(fc) and fc[i][0] == cursor:
            running += fc[i][1]
            used_forecast += 1
        else:
            running += climatology_dj
        if running >= harvest_st:
            running = round(running * 10) / 10
            method = "thermal-forecast" if used_forecast and i < len(fc) else "thermal-climatology"
            return {
                "harvestDate": cursor.isoformat(),
                "stAchieved": running,
                "alreadyReached": False,
                "method": method,
                "forecastDaysUsed": used_forecast,
                "meanDj": round(mean_dj * 10) / 10,
            }
    return {
        "harvestDate": None,
        "stAchieved": round(running * 10) / 10,
        "alreadyReached": False,
        "method": "thermal-climatology",
        "error": f"ST {harvest_st} °C non atteinte dans {horizon_days} j (ST projetée {round(running * 10) / 10} °C).",
    }


def agro_window_fallback(lat: float, observation_iso: str | None = None) -> dict[str, Any]:
    """Fenêtre de semis agro-climatique (repli quand la thermique échoue).

    Hémisphère nord : orge d'hiver semée oct-nov ; hémisphère sud décalé de 6 mois.
    """
    obs = _parse_iso(observation_iso) if observation_iso else datetime.now(timezone.utc).date()
    if obs is None:
        obs = datetime.now(timezone.utc).date()
    # Année de campagne : si on est avant juillet, la campagne a commencé l'année d'avant (N).
    year = obs.year if obs.month >= 7 else obs.year - 1
    if lat < 0:
        start = date(year, 4, 1)
        end = date(year, 5, 31)
    else:
        start = date(year, 10, 1)
        end = date(year, 11, 30)
    mid = start + (end - start) / 2
    return {
        "sowingDate": mid.isoformat(),
        "sowingDateEarliest": start.isoformat(),
        "sowingDateLatest": end.isoformat(),
        "method": "agro-window",
        "confidence": 25,
        "warnings": ["Fenêtre agro-climatique par défaut (thermique indisponible)."],
    }


async def estimate_sowing_from_stage(
    lat: float,
    lng: float,
    *,
    observation_iso: str | None = None,
    zadoks_code: str | None = None,
    st_target: float | None = None,
    tr: float = BARLEY_TR,
    history_days: int = MAX_HISTORY_DAYS,
) -> dict[str, Any]:
    """Estimation du semis par inversion thermique depuis un stade observé.

    Fournir ``zadoks_code`` (ex "30", "65") OU ``st_target`` (°C). Le code prime
    sur la ST s'ils sont tous deux fournis.
    """
    obs_iso = observation_iso or datetime.now(timezone.utc).date().isoformat()
    if _parse_iso(obs_iso) is None:
        return {"sowingDate": None, "error": "Date d'observation invalide (YYYY-MM-DD attendue)."}

    target: float | None = None
    code: str | None = None
    source = "unknown"
    if isinstance(zadoks_code, str) and zadoks_code.strip():
        code = zadoks_code.strip()
        target = st_for_code(code)
        if target is None:
            return {"sowingDate": None, "error": f"Code Zadoks inconnu : {code}."}
        source = st_source_for_code(code)
    elif isinstance(st_target, (int, float)) and float(st_target) >= 0:
        target = float(st_target)
    else:
        return {"sowingDate": None, "error": "Fournir zadoksCode ou stTarget."}

    obs = _parse_iso(obs_iso)
    assert obs is not None
    end = obs.isoformat()
    # L'archive Open-Meteo a ~2 j de latence : on borne la fin à (aujourd'hui - 2 j).
    archive_end = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    truncated = False
    if obs > archive_end:
        end = archive_end.isoformat()
        truncated = True
    start = (_parse_iso(end) - timedelta(days=max(30, min(history_days, MAX_HISTORY_DAYS)))).isoformat()

    warnings: list[str] = []
    try:
        daily = await fetch_daily_temps(lat, lng, start, end)
    except Exception as exc:  # noqa: BLE001 — repli fenêtre agro
        logger.warning("sowing: Open-Meteo indisponible (%s), repli agro-window", exc)
        fallback = agro_window_fallback(lat, obs_iso)
        fallback.update({
            "observationDate": obs_iso,
            "zadoksCode": code,
            "stTarget": target,
            "warnings": [*fallback.get("warnings", []), f"Thermique indisponible : {exc}"],
        })
        return fallback
    if truncated:
        warnings.append(f"Séries thermiques arrêtées au {end} (latence archive Open-Meteo).")
    if not daily:
        fallback = agro_window_fallback(lat, obs_iso)
        fallback.update({"observationDate": obs_iso, "zadoksCode": code, "stTarget": target})
        return fallback

    inversion = invert_sowing(daily, end, target, tr)
    if inversion.get("sowingDate") is None:
        fallback = agro_window_fallback(lat, obs_iso)
        fallback.update({
            "observationDate": obs_iso,
            "zadoksCode": code,
            "stTarget": target,
            "stAchieved": inversion.get("stAchieved"),
            "warnings": [*fallback.get("warnings", []), inversion.get("error", "Inversion impossible.")],
        })
        return fallback

    sowing_iso = inversion["sowingDate"]
    zero_days = _zero_dj_days(daily, sowing_iso, end)
    confidence = _confidence(inversion=inversion, st_source=source, zero_dj_days=zero_days, days_counted=inversion.get("daysCounted", 0))

    # Intervalle : inversion à ST ± tolérance (max 15 °C ou 10 % de la cible).
    tolerance = max(15.0, target * 0.10) if target > 0 else 15.0
    earliest = invert_sowing(daily, end, target + tolerance, tr).get("sowingDate")  # ST haute → semis plus ancien
    latest = invert_sowing(daily, end, max(0.0, target - tolerance), tr).get("sowingDate")
    if zero_days > 10:
        warnings.append(f"{zero_days} j à DJ nul dans la fenêtre (pause hivernale : ± quelques jours).")
    if source == "interpolated":
        warnings.append(f"Seuil ST {target} °C interpolé pour le stade {code} (à confirmer).")
    if inversion.get("coverage", 1.0) < 0.95:
        warnings.append("Historique thermique incomplet sur la fenêtre.")

    result = {
        "sowingDate": sowing_iso,
        "sowingDateEarliest": earliest or sowing_iso,
        "sowingDateLatest": latest or sowing_iso,
        "method": "thermal-inversion",
        "observationDate": obs_iso,
        "thermalEndDate": end,
        "zadoksCode": code,
        "stTarget": target,
        "stSource": source,
        "stAchieved": inversion.get("stAchieved"),
        "daysCounted": inversion.get("daysCounted"),
        "toleranceSt": round(tolerance * 10) / 10,
        "confidence": confidence,
        "warnings": warnings,
    }
    if code is not None:
        result["stage"] = stage_for_st(target)
    return result


async def estimate_sowing_from_emergence(
    lat: float,
    lng: float,
    emergence_iso: str,
    *,
    tr: float = BARLEY_TR,
    history_days: int = 120,
) -> dict[str, Any]:
    """Semis = émergence (levée, ST 30 °C) moins ~30 °C de thermique."""
    if _parse_iso(emergence_iso) is None:
        return {"sowingDate": None, "error": "Date de levée invalide (YYYY-MM-DD attendue)."}
    result = await estimate_sowing_from_stage(
        lat, lng, observation_iso=emergence_iso, st_target=EMERGENCE_ST, tr=tr, history_days=history_days
    )
    result["method"] = "emergence-backprojection" if result.get("sowingDate") else result.get("method", "agro-window")
    result["emergenceDate"] = emergence_iso
    return result


def detect_emergence_from_ndvi(s2: list[dict[str, Any]]) -> dict[str, Any]:
    """Détecte le mois d'émergence depuis une série S2 mensuelle [{date, ndvi}].

    Port resserré de la logique ``detect_planting_date`` : plus grand saut NDVI
    mensuel (≥ 15 pts) sinon reprise depuis le minimum (≥ 5 pts). La date
    d'émergence retenue est le 10 du mois en hausse (même convention calendaire
    que l'existant, affinée ensuite par rétro-projection thermique).
    """
    valid = [p for p in s2 if isinstance(p, dict) and p.get("ndvi") is not None and isinstance(p.get("date"), str)]
    empty: dict[str, Any] = {"emergenceDate": None, "jumpMonth": None, "jump": 0.0, "confidence": 0}
    if len(valid) < 2:
        return empty
    try:
        valid = sorted(valid, key=lambda p: p["date"])
    except TypeError:
        return empty

    max_jump = 0.0
    jump_index = -1
    for i in range(1, len(valid)):
        try:
            delta = float(valid[i]["ndvi"]) - float(valid[i - 1]["ndvi"])
        except (TypeError, ValueError):
            continue
        if delta > max_jump:
            max_jump = delta
            jump_index = i

    if max_jump < 15 or jump_index < 0:
        min_idx = min(range(len(valid)), key=lambda i: float(valid[i]["ndvi"]))
        if min_idx < len(valid) - 1:
            try:
                rebound = float(valid[min_idx + 1]["ndvi"]) - float(valid[min_idx]["ndvi"])
            except (TypeError, ValueError):
                rebound = 0.0
            if rebound > 5:
                jump_index = min_idx + 1
                max_jump = rebound
            else:
                return empty
        else:
            return empty

    jump_month = str(valid[jump_index]["date"])[:7]
    try:
        year, month = int(jump_month[:4]), int(jump_month[5:7])
        emergence = date(year, month, 10).isoformat()
    except ValueError:
        return empty
    confidence = min(90, round(max_jump * 2))
    return {"emergenceDate": emergence, "jumpMonth": jump_month, "jump": round(max_jump * 10) / 10, "confidence": confidence}


async def estimate_sowing_from_ndvi(
    lat: float,
    lng: float,
    s2: list[dict[str, Any]],
    *,
    tr: float = BARLEY_TR,
) -> dict[str, Any]:
    """Chaîne complète : saut NDVI → émergence → semis par thermique (30 °C)."""
    emergence = detect_emergence_from_ndvi(s2)
    if not emergence.get("emergenceDate"):
        return {
            "sowingDate": None,
            "method": "ndvi-emergence",
            "emergenceDate": None,
            "confidence": 0,
            "warnings": ["Aucune émergence détectable dans la série NDVI."],
        }
    result = await estimate_sowing_from_emergence(lat, lng, emergence["emergenceDate"], tr=tr)
    result["method"] = "ndvi-emergence" if result.get("sowingDate") else result.get("method", "agro-window")
    result["jumpMonth"] = emergence.get("jumpMonth")
    result["ndviJump"] = emergence.get("jump")
    result["emergenceConfidence"] = emergence.get("confidence")
    if result.get("sowingDate"):
        # Confiance combinée : détection NDVI (mensuelle, grossière) × thermique.
        ndvi_conf = emergence.get("confidence", 0) or 0
        thermal_conf = result.get("confidence", 0) or 0
        result["confidence"] = max(5, min(90, round(0.4 * ndvi_conf + 0.6 * thermal_conf)))
        result.setdefault("warnings", [])
        result["warnings"].append("Émergence mensuelle (GEE) affinée par thermique journalière.")
    return result


async def track_parcel(
    lat: float,
    lng: float,
    s2: list[dict[str, Any]],
    *,
    zadoks_code: str | None = None,
    st_target: float | None = None,
    tr: float = BARLEY_TR,
) -> dict[str, Any]:
    """Branchement parcelle : semis (inversion thermique) + ST/stade courant.

    Priorité : stade observé (``zadoks_code``/``st_target``) > émergence NDVI.
    Ne fait qu'UN appel Open-Meteo : l'historique thermique sert à la fois à
    l'inversion (jours <= émergence/observation) et au cumul ST depuis le semis.
    Retourne les clés stables ``estimated_planting_date`` / ``growth_stage``
    (compatibles ``detect_planting_date``) + ``sowing`` / ``phenology`` détaillés.
    """
    warnings: list[str] = []
    sowing: dict[str, Any]
    if isinstance(zadoks_code, str) and zadoks_code.strip():
        sowing = await estimate_sowing_from_stage(lat, lng, zadoks_code=zadoks_code, tr=tr)
    elif isinstance(st_target, (int, float)):
        sowing = await estimate_sowing_from_stage(lat, lng, st_target=float(st_target), tr=tr)
    else:
        sowing = await estimate_sowing_from_ndvi(lat, lng, s2, tr=tr)
    warnings.extend(sowing.get("warnings", []) or [])

    sowing_iso = sowing.get("sowingDate")
    if not sowing_iso:
        return {
            "estimated_planting_date": None,
            "estimated_harvest_date": None,
            "days_since_planting": None,
            "growth_stage": None,
            "planting_confidence": 0,
            "sowing": sowing,
            "phenology": None,
            "warnings": warnings or [sowing.get("error", "Semis non déterminable.")],
        }
    return await _track_from_sowing(lat, lng, sowing_iso, sowing, warnings, tr=tr)


async def st_since_estimated_sowing(
    lat: float,
    lng: float,
    *,
    tr: float = BARLEY_TR,
) -> float | None:
    """ST cumulée depuis le semis estimé (fenêtre agro, SANS série GEE).

    Pré-calcul léger pour la classification : semis = milieu de la fenêtre
    agro-climatique (oct-nov N / avr-mai S), cumul DJ Open-Meteo jusqu'à
    aujourd'hui. Retourne None si la thermique est indisponible.
    """
    sowing_iso = agro_window_fallback(lat).get("sowingDate")
    if not sowing_iso:
        return None
    today = datetime.now(timezone.utc).date()
    archive_end = today - timedelta(days=2)
    try:
        daily = await fetch_daily_temps(lat, lng, sowing_iso, archive_end.isoformat())
    except Exception:  # noqa: BLE001 — pas de ST, la classification garde le repli
        return None
    return stage_since_sowing(daily, sowing_iso).get("st")


async def _track_from_sowing(
    lat: float,
    lng: float,
    sowing_iso: str,
    sowing: dict[str, Any],
    warnings: list[str] | None = None,
    *,
    tr: float = BARLEY_TR,
) -> dict[str, Any]:
    """Complète un semis connu : ST/stade courant + récolte thermique."""
    warnings = list(warnings or [])
    # Fenêtre thermique : du semis à aujourd'hui (archive, latence ~2 j).
    today = datetime.now(timezone.utc).date()
    archive_end = today - timedelta(days=2)
    try:
        daily = await fetch_daily_temps(lat, lng, sowing_iso, archive_end.isoformat())
    except Exception as exc:  # noqa: BLE001 — semis connu, stade seul indisponible
        logger.warning("sowing.track: thermique ST indisponible (%s)", exc)
        warnings.append(f"Stade courant indisponible : {exc}")
        return {
            "estimated_planting_date": sowing_iso,
            "estimated_harvest_date": None,
            "days_since_planting": (today - date.fromisoformat(sowing_iso)).days,
            "growth_stage": None,
            "planting_confidence": sowing.get("confidence", 0) or 0,
            "sowing": sowing,
            "phenology": None,
            "warnings": warnings,
        }

    phenology = stage_since_sowing(daily, sowing_iso)
    code = phenology.get("code")
    label = phenology.get("label")
    days = phenology.get("daysCounted")
    st_since_sowing = phenology.get("st")

    # Récolte thermique : ST 2100 °C (Zadoks 99) — observé, sinon prévision
    # 16 j, sinon climatologie (moyenne des DJ depuis le semis).
    harvest: dict[str, Any] = {"harvestDate": None}
    try:
        forecast = await fetch_forecast_temps(lat, lng)
    except Exception as exc:  # noqa: BLE001 — repli climatologie pure
        logger.warning("sowing.track: prévision indisponible (%s), climatologie seule", exc)
        warnings.append("Prévision météo indisponible : récolte projetée sur climatologie.")
        forecast = []
    harvest = project_harvest_date(daily, forecast, harvest_st=HARVEST_ST, tr=tr)
    if harvest.get("harvestDate") is None:
        warnings.append(harvest.get("error", "Récolte non projetable."))
    elif not harvest.get("alreadyReached"):
        warnings.append(f"Récolte projetée au {harvest['harvestDate']} (ST 2100 °C, {harvest['method']}).")

    return {
        "estimated_planting_date": sowing_iso,
        "estimated_harvest_date": harvest.get("harvestDate"),
        "days_since_planting": days,
        "growth_stage": f"Zadoks {code} — {label}" if code else None,
        "planting_confidence": sowing.get("confidence", 0) or 0,
        "sowing": sowing,
        "phenology": phenology,
        "harvest": harvest,
        "warnings": warnings,
        # Cumul thermique de référence : ST depuis le semis estimé (TR = 0 °C).
        # Les classifieurs (resolve_cereal, confirmations) doivent utiliser ce
        # cumul, pas une fenêtre calendaire fixe qui tronque le cycle.
        "stSinceSowing": st_since_sowing,
        "gddSinceSowing": st_since_sowing,
    }
