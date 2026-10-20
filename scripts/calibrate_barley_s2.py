"""
Calibre le classifieur orge / non-orge sur Sentinel-2 a partir de releves terrain.

Entree : un ou plusieurs KML exportes de Google Earth. Chaque polygone porte sa
verite terrain par la couleur de son trait : vert = orge, rouge = autre culture
(les autres couleurs sont ignorees).

Pour chaque polygone, le script recupere l'instantane Sentinel-2 utilise par
l'analyse de parcelle (_fetch_current_snapshot), compare plusieurs jeux de
variables en validation croisee leave-one-out (chaque parcelle predite par un
modele entraine sans elle), puis ecrit le modele retenu dans
app/data/barley_s2_model.json.

    python scripts/calibrate_barley_s2.py C:/chemin/releves1.kml [releves2.kml ...]

Les exactitudes affichees sont celles de la validation croisee, pas celles sur
les donnees d'entrainement : c'est l'estimation a retenir.
"""

from __future__ import annotations

import asyncio
import colorsys
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.analyze_parcel import _fetch_current_snapshot, get_gee_access_token, map_with_concurrency  # noqa: E402
from app.services.barley_s2_classifier import MODEL_PATH, fit_logistic, predict_proba  # noqa: E402

NS = {"k": "http://www.opengis.net/kml/2.2", "gx": "http://www.google.com/kml/ext/2.2"}

CANDIDATE_FEATURE_SETS = [
    ["ndvi"],
    ["savi"],
    ["evi"],
    ["ndvi", "evi"],
    ["ndvi", "evi", "savi"],
    ["ndvi", "evi", "savi", "ndwi"],
    ["nir", "red", "green", "blue", "swir"],
]


def _label_from_kml_color(color: str) -> str | None:
    # KML : aabbggrr
    r, g, b = int(color[6:8], 16), int(color[4:6], 16), int(color[2:4], 16)
    hue = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
    if 70 <= hue <= 170:
        return "orge"
    if hue < 30 or hue > 330:
        return "non-orge"
    return None


def read_kml(path: str) -> list[dict]:
    root = ET.parse(path).getroot()
    colors: dict[str, str] = {}
    for style in root.iter("{%s}CascadingStyle" % NS["gx"]):
        c = style.find(".//k:LineStyle/k:color", NS)
        if c is not None:
            colors[style.get("{%s}id" % NS["k"])] = c.text.strip()
    for style in root.iter("{%s}Style" % NS["k"]):
        c = style.find(".//k:LineStyle/k:color", NS)
        if style.get("id") and c is not None:
            colors[style.get("id")] = c.text.strip()
    normal_style: dict[str, str] = {}
    for style_map in root.iter("{%s}StyleMap" % NS["k"]):
        for pair in style_map.findall("k:Pair", NS):
            if pair.findtext("k:key", namespaces=NS) == "normal":
                normal_style[style_map.get("id")] = pair.findtext("k:styleUrl", namespaces=NS).lstrip("#")

    parcels = []
    for placemark in root.iter("{%s}Placemark" % NS["k"]):
        name = (placemark.findtext("k:name", default="", namespaces=NS) or "").strip()
        style_id = (placemark.findtext("k:styleUrl", default="", namespaces=NS) or "").lstrip("#")
        color = colors.get(normal_style.get(style_id, style_id)) or colors.get(style_id)
        coords = placemark.find(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", NS)
        label = _label_from_kml_color(color) if color else None
        if coords is None or label is None:
            print(f"  ignore : {name!r} (pas un polygone ou couleur non reconnue)")
            continue
        points = [tuple(map(float, token.split(",")[:2])) for token in coords.text.split()]
        polygon = [{"lat": lat, "lng": lng} for lng, lat in points[:-1]]
        parcels.append({"name": name, "label": label, "polygon": polygon, "source": Path(path).name})
    return parcels


async def fetch_features(parcels: list[dict]) -> list[dict]:
    project_id = json.loads(settings.gee_service_account_key)["project_id"]
    token = await get_gee_access_token()

    async def one(parcel: dict) -> dict:
        lat = sum(p["lat"] for p in parcel["polygon"]) / len(parcel["polygon"])
        lng = sum(p["lng"] for p in parcel["polygon"]) / len(parcel["polygon"])
        snapshot = await _fetch_current_snapshot(token, lat, lng, project_id, parcel["polygon"])
        return {**parcel, "features": {k: snapshot.get(k) for k in ("ndvi", "evi", "savi", "ndwi", "nir", "red", "green", "blue", "swir")}}

    return await map_with_concurrency(parcels, 4, one)


def matrix(rows: list[dict], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    usable = [r for r in rows if all(r["features"].get(f) is not None for f in features)]
    x = np.array([[r["features"][f] for f in features] for r in usable], dtype=float)
    y = np.array([1.0 if r["label"] == "orge" else 0.0 for r in usable])
    return x, y


def leave_one_out_accuracy(x: np.ndarray, y: np.ndarray, features: list[str]) -> float:
    correct = 0
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        model = fit_logistic(x[mask], y[mask], features)
        p = predict_proba(model, dict(zip(features, x[i])))
        correct += int((p >= 0.5) == (y[i] == 1))
    return correct / len(y)


async def main(paths: list[str]) -> None:
    parcels = [p for path in paths for p in read_kml(path)]
    n_barley = sum(p["label"] == "orge" for p in parcels)
    print(f"{len(parcels)} parcelles terrain : {n_barley} orge, {len(parcels) - n_barley} non-orge")
    if n_barley < 5 or len(parcels) - n_barley < 5:
        sys.exit("Il faut au moins 5 parcelles de chaque classe pour calibrer.")

    rows = await fetch_features(parcels)
    majority = max(n_barley, len(parcels) - n_barley) / len(parcels)
    print(f"\nReference (toujours predire la classe majoritaire) : {majority:.0%}")
    print("Validation croisee leave-one-out :")
    scored = []
    for rank, features in enumerate(CANDIDATE_FEATURE_SETS):
        x, y = matrix(rows, features)
        if len(y) < len(rows) * 0.8:
            continue
        acc = leave_one_out_accuracy(x, y, features)
        # A exactitude egale : le moins de variables, puis l'ordre de la liste (NDVI d'abord).
        scored.append((acc, -len(features), -rank, features))
        print(f"  {'+'.join(features):<32} {acc:.0%}  (n={len(y)})")

    acc, _, _, features = max(scored)
    x, y = matrix(rows, features)
    model = fit_logistic(x, y, features)
    model["calibration"] = {
        "date": date.today().isoformat(),
        "sources": sorted({r["source"] for r in rows}),
        "n_parcels": int(len(y)),
        "n_barley": int(y.sum()),
        "loo_accuracy": round(acc, 3),
        "majority_baseline": round(majority, 3),
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModele retenu : {'+'.join(features)} (validation croisee {acc:.0%}) -> {MODEL_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1:]))
