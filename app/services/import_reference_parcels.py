"""Import reference barley parcels from STAR_ORGE_TimeSeries CSV.

This script imports the STAR_ORGE_TimeSeries_2024_2026.csv file as reference
parcels in the database. These are known barley parcels with ground truth
time series data that can be used for validation and calibration.
"""

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import async_session_maker
from app.models import Parcelle


CSV_PATH = Path(__file__).parent.parent.parent.parent / "STAR_ORGE_TimeSeries_2024_2026.csv"


def parse_csv_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Parse a CSV row into a parcel data dict."""
    try:
        name = row["name"].strip()
        culture = row["culture"].strip()
        surface_m2 = float(row["surface_m2"])
        ndvi = float(row["NDVI"]) if row["NDVI"] else None
        evi = float(row["EVI"]) if row["EVI"] else None
        ndre = float(row["NDRE"]) if row["NDRE"] else None
        ndmi = float(row["NDMI"]) if row["NDMI"] else None
        date_str = row["date"]
        image_id = row["image_id"]

        # Parse bands
        bands = {}
        for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]:
            val = row.get(band)
            if val:
                bands[band] = float(val)

        return {
            "name": name,
            "culture": culture,
            "surface_m2": surface_m2,
            "surface_ha": surface_m2 / 10000,
            "ndvi": ndvi,
            "evi": evi,
            "ndre": ndre,
            "ndmi": ndmi,
            "date": date_str,
            "image_id": image_id,
            "bands": bands,
        }
    except (ValueError, KeyError) as e:
        print(f"Error parsing row {row}: {e}")
        return None


def group_by_parcel(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group time series rows by parcel name."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row["name"]
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(row)
    return grouped


def create_parcelle_from_group(name: str, time_series: list[dict[str, Any]]) -> Parcelle:
    """Create a Parcelle model from grouped time series data."""
    # Use the first entry for basic info
    first = time_series[0]
    culture = first["culture"]
    surface_ha = first["surface_ha"]

    # Build time series for S2 (NDVI, EVI, NDRE, NDMI)
    time_series_s2 = []
    for ts in time_series:
        time_series_s2.append({
            "date": ts["date"],
            "ndvi": ts["ndvi"],
            "evi": ts["evi"],
            "ndre": ts["ndre"],
            "ndmi": ts["ndmi"],
            "image_id": ts["image_id"],
            "bands": ts["bands"],
        })

    # Sort by date
    time_series_s2.sort(key=lambda x: x["date"])

    # Use latest NDVI as current value
    latest_ndvi = time_series_s2[-1]["ndvi"] if time_series_s2 else None
    ndvi_percentage = round(latest_ndvi * 100, 1) if latest_ndvi is not None else None

    # Use latest EVI, NDRE
    latest_evi = time_series_s2[-1]["evi"] if time_series_s2 else None
    latest_ndre = time_series_s2[-1]["ndre"] if time_series_s2 else None
    latest_ndmi = time_series_s2[-1]["ndmi"] if time_series_s2 else None

    # Generate a deterministic UUID based on name for reproducibility
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    parcel_uuid = uuid.uuid5(namespace, f"star_orge_{name}")

    # Create a simple polygon around a reference point (Madagascar area based on data)
    # These are approximate coordinates - in reality you'd have actual geometries
    # For now, create a small square around a reference point
    ref_lat, ref_lng = -19.848219, 47.011882  # Default center from the app
    offset = 0.0005  # ~50m
    coordinates = [[
        [ref_lng - offset, ref_lat - offset],
        [ref_lng + offset, ref_lat - offset],
        [ref_lng + offset, ref_lat + offset],
        [ref_lng - offset, ref_lat + offset],
        [ref_lng - offset, ref_lat - offset],
    ]]

    return Parcelle(
        id=parcel_uuid,
        label=f"STAR_ORGE_REF_{name}",
        coordinates=coordinates,
        center_lat=ref_lat,
        center_lng=ref_lng,
        surface_ha=surface_ha,
        culture_declared=culture,
        culture_detected=culture,
        ndvi_percentage=ndvi_percentage,
        confidence=95.0,  # High confidence for reference data
        verdict=f"{culture} confirmée (référence STAR)",
        details=f"Parcelle de référence STAR Orge - Série temporelle 2024-2026 ({len(time_series_s2)} observations)",
        saison="2024-2026",
        data_source="STAR_ORGE_TimeSeries_2024_2026.csv",
        owner_name="STAR Reference",
        notes=f"Parcelle de référence: {name}. Surface: {surface_ha:.4f} ha. {len(time_series_s2)} dates d'observation.",
        time_series_s2=time_series_s2,
        evi=round(latest_evi * 100, 1) if latest_evi is not None else None,
        ndre=round(latest_ndre * 100, 1) if latest_ndre is not None else None,
        ndwi=round(latest_ndmi * 100, 1) if latest_ndmi is not None else None,
        agro_score=85.0,
        hybrid_score=90.0,
        phenology=None,
    )


async def import_reference_parcels() -> dict[str, Any]:
    """Import reference parcels from CSV into database."""
    if not CSV_PATH.exists():
        return {"success": False, "error": f"CSV file not found at {CSV_PATH}"}

    print(f"Reading CSV from {CSV_PATH}...")

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = parse_csv_row(row)
            if parsed:
                rows.append(parsed)

    print(f"Parsed {len(rows)} rows")

    # Group by parcel name
    grouped = group_by_parcel(rows)
    print(f"Found {len(grouped)} unique parcels")

    # Create parcelle objects
    parcelles = []
    for name, time_series in grouped.items():
        parcelle = create_parcelle_from_group(name, time_series)
        parcelles.append(parcelle)

    # Save to database
    async with async_session_maker() as session:
        saved = 0
        updated = 0
        for parcelle in parcelles:
            existing = (await session.execute(
                select(Parcelle).where(Parcelle.label == parcelle.label)
            )).scalars().first()

            if existing:
                # Update existing
                existing.coordinates = parcelle.coordinates
                existing.center_lat = parcelle.center_lat
                existing.center_lng = parcelle.center_lng
                existing.surface_ha = parcelle.surface_ha
                existing.culture_declared = parcelle.culture_declared
                existing.culture_detected = parcelle.culture_detected
                existing.ndvi_percentage = parcelle.ndvi_percentage
                existing.confidence = parcelle.confidence
                existing.verdict = parcelle.verdict
                existing.details = parcelle.details
                existing.saison = parcelle.saison
                existing.data_source = parcelle.data_source
                existing.owner_name = parcelle.owner_name
                existing.notes = parcelle.notes
                existing.time_series_s2 = parcelle.time_series_s2
                existing.evi = parcelle.evi
                existing.ndre = parcelle.ndre
                existing.ndwi = parcelle.ndwi
                existing.agro_score = parcelle.agro_score
                existing.hybrid_score = parcelle.hybrid_score
                updated += 1
            else:
                session.add(parcelle)
                saved += 1

        await session.commit()

    return {
        "success": True,
        "total_parcels": len(parcelles),
        "saved": saved,
        "updated": updated,
        "parcel_names": list(grouped.keys()),
    }


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(import_reference_parcels())
    print(result)