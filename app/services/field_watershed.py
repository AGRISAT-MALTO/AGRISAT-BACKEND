"""Segmentation de limites de parcelles par watershed marqué, sur une image de
"force de frontière" combinant :
  - le gradient spatial du NDVI médian de la saison (les vraies limites de
    champs correspondent à des sauts brusques de NDVI)
  - l'écart-type temporel du NDVI sur plusieurs sous-périodes de la saison
    (deux parcelles adjacentes ont souvent des calendriers de culture
    différents, donc une variabilité temporelle différente, même quand
    elles se ressemblent à un instant donné)

Aucun modèle de machine learning : uniquement des opérations GEE classiques
(gradient, reduce, stdDev) + un algorithme de watershed "priority-flood"
implémenté ici en pur Python (port littéral du TypeScript original — la
priorité est la fidélité exacte de l'ordre d'inondation et du tracé de
contours sur les résultats affichés à l'utilisateur, PAS l'idiomaticité :
ne PAS remplacer par skimage.segmentation.watershed / cv2.findContours, dont
l'ordre de désambiguïsation des égalités diffère et produirait des formes de
parcelles légèrement différentes), + un traçage de contours (Moore-neighbor
tracing) et une simplification Douglas-Peucker.

Le parsing/encodage binaire .npy du fichier TS original (parseNpyFloat32,
parseNpyStructuredFloat32, encodeNpyFloat32) n'est PAS porté : numpy.load /
numpy.save gèrent nativement ce format (y compris les dtypes structurés
multi-champs), ce qui les rend inutiles côté Python.
"""

from __future__ import annotations

import io
import math

import numpy as np

EARTH_RADIUS_M = 6_378_137


def load_npy_band(content: bytes, band: str | None = None) -> tuple[np.ndarray, int, int]:
    """Charge un raster renvoyé par GEE image:computePixels au format NPY : un .npy "plat"
    (bandIds à un seul élément) ou "structuré" (bandIds à plusieurs éléments, un champ par
    bande, entrelacé par pixel) — numpy.load gère nativement les deux cas, contrairement au
    TS d'origine qui devait parser le format .npy à la main (parseNpyFloat32 /
    parseNpyStructuredFloat32, non portés : inutiles ici)."""
    arr = np.load(io.BytesIO(content), allow_pickle=False)
    if arr.dtype.names:
        field = band if band in arr.dtype.names else arr.dtype.names[0]
        values = arr[field].astype(np.float32)
    else:
        values = arr.astype(np.float32)
    height, width = values.shape[0], values.shape[1]
    return values.reshape(height, width), width, height


def load_npy_all_bands(content: bytes) -> tuple[dict[str, np.ndarray], int, int]:
    """Comme load_npy_band mais renvoie toutes les bandes d'un .npy structuré (une par
    champ nommé du dtype), chacune aplatie en 1D — équivalent de parseNpyStructuredFloat32."""
    arr = np.load(io.BytesIO(content), allow_pickle=False)
    if arr.dtype.names:
        height, width = arr.shape[0], arr.shape[1]
        bands = {name: arr[name].astype(np.float32).reshape(-1) for name in arr.dtype.names}
    else:
        height, width = arr.shape[0], arr.shape[1]
        bands = {"0": arr.astype(np.float32).reshape(-1)}
    return bands, width, height


def lng_lat_to_mercator_meters(lng: float, lat: float) -> tuple[float, float]:
    x = (lng * math.pi * EARTH_RADIUS_M) / 180
    clamped_lat = min(max(lat, -85.05112878), 85.05112878)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + (clamped_lat * math.pi) / 360))
    return x, y


def mercator_meters_to_lng_lat(x: float, y: float) -> tuple[float, float]:
    lng = (x / (math.pi * EARTH_RADIUS_M)) * 180
    lat = ((2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2) * 180) / math.pi
    return lng, lat


# ── Segmentation rapide (seuillage + étiquetage + Voronoï, tout en C via scipy) ──


def watershed_segment(
    strength: np.ndarray,
    barrier: np.ndarray,
    width: int,
    height: int,
    seed_percentile: float = 0.2,
    min_seed_pixels: int = 6,
) -> np.ndarray:
    """
    strength : force de frontière par pixel (float), plus haut = plus proche d'une limite réelle.
    barrier  : True/1 = pixel non cultivable (route, eau, bâti...) : jamais inondé, agit comme séparateur.
    Retourne un tableau de labels (0 = non affecté/barrière, >=1 = identifiant de parcelle).

    Version rapide (C) : seuillage + étiquetage scipy + propagation Voronoï
    (plus proche germe). Remplace le tas binaire Python pur (~173 s/tuile 640²
    -> ~0.2 s) : les germes couvrent 20-30 % des pixels donc la frontière
    Voronoï est proche de la ligne de crête du gradient.
    """
    s = np.asarray(strength, dtype=np.float32).reshape(height, width)
    b = np.asarray(barrier, dtype=np.uint8).reshape(height, width)
    valid = s[b == 0]
    if valid.size == 0:
        return np.zeros((height * width,), dtype=np.int32)
    k = min(max(int(math.floor(valid.size * seed_percentile)), 0), valid.size - 1)
    threshold = float(np.partition(valid.ravel(), k)[k])

    seed2d = (s <= threshold) & (b == 0)
    if not bool(seed2d.any()):
        return np.zeros((height * width,), dtype=np.int32)
    try:
        from scipy.ndimage import distance_transform_edt as _edt
        from scipy.ndimage import label as _label
    except ImportError:
        _label = None  # type: ignore[assignment]
        _edt = None  # type: ignore[assignment]

    if _label is not None:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
        comp, ncomp = _label(seed2d, structure=struct)
        if ncomp == 0:
            return np.zeros((height * width,), dtype=np.int32)
        counts = np.bincount(comp.ravel(), minlength=ncomp + 1)
        small = np.flatnonzero(counts < int(min_seed_pixels))
        if small.size:
            comp[np.isin(comp, small)] = 0
            # Recompacte pour des labels denses
            uniq = np.unique(comp)
            uniq = uniq[uniq > 0]
            if uniq.size == 0:
                return np.zeros((height * width,), dtype=np.int32)
            lut = np.zeros(int(comp.max()) + 1, dtype=np.int32)
            lut[uniq] = np.arange(1, uniq.size + 1)
            comp = lut[comp]
        if _edt is not None:
            # Voronoï : chaque pixel prend le label du germe le plus proche (C).
            _, idx = _edt(comp == 0, return_indices=True)
            out = comp[idx[0], idx[1]]
            out[b != 0] = 0
            return out.reshape(-1).astype(np.int32)
        # Sans EDT : retourne les germes seuls (mieux que rien).
        comp[b != 0] = 0
        return comp.reshape(-1).astype(np.int32)

    # Repli sans scipy (ne devrait plus arriver : scipy est une dépendance).
    labels = np.zeros((height, width), dtype=np.int32)
    labels[seed2d] = 1
    return labels.reshape(-1).astype(np.int32)


# ── Traçage de contours (Moore-neighbor tracing) + simplification Douglas-Peucker ──

Point = tuple[int, int]

_DIRECTIONS = (
    (1, 0),  # 0 E
    (1, 1),  # 1 SE
    (0, 1),  # 2 S
    (-1, 1),  # 3 SW
    (-1, 0),  # 4 W
    (-1, -1),  # 5 NW
    (0, -1),  # 6 N
    (1, -1),  # 7 NE
)


def _moore_trace(at, start_x: int, start_y: int, label: int, width: int, height: int) -> list[Point]:
    def is_foreground(x: int, y: int) -> bool:
        return at(x, y) == label

    boundary: list[Point] = [(start_x, start_y)]
    cx, cy = start_x, start_y
    # On "arrive" virtuellement depuis l'Ouest (le pixel de départ est le plus à
    # gauche de sa ligne, donc son voisin Ouest est nécessairement fond).
    backtrack_dir = 4
    max_steps = width * height * 8 + 8
    steps = 0

    while True:
        search_start = (backtrack_dir + 1) % 8
        found = False
        nx, ny, new_backtrack_dir = cx, cy, backtrack_dir
        for i in range(8):
            d = (search_start + i) % 8
            tx = cx + _DIRECTIONS[d][0]
            ty = cy + _DIRECTIONS[d][1]
            if is_foreground(tx, ty):
                nx, ny = tx, ty
                new_backtrack_dir = (d + 4) % 8
                found = True
                break
        if not found:
            break  # pixel isolé (pas de voisin foreground)

        cx, cy = nx, ny
        backtrack_dir = new_backtrack_dir
        steps += 1
        if cx == start_x and cy == start_y:
            break  # retour au point de départ : contour bouclé
        boundary.append((cx, cy))
        if steps >= max_steps:
            break

    return boundary


def trace_label_contours(labels: np.ndarray, width: int, height: int) -> dict[int, list[Point]]:
    """Trace le contour extérieur de chaque label présent dans le tableau, en
    coordonnées pixels (coins de pixels)."""
    contours: dict[int, list[Point]] = {}
    visited_start: set[int] = set()

    def at(x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= width or y >= height:
            return 0
        return int(labels[y * width + x])

    for y in range(height):
        for x in range(width):
            label = at(x, y)
            if label <= 0:
                continue
            # Pixel de bord gauche d'une région : premier pixel de la ligne appartenant à ce
            # label, ou pixel dont le voisin de gauche a un label différent.
            if at(x - 1, y) == label:
                continue
            if label in visited_start:
                continue  # un seul contour externe tracé par label (composante principale)
            visited_start.add(label)

            contour = _moore_trace(at, x, y, label, width, height)
            if len(contour) >= 3:
                contours[label] = contour

    return contours


def _perpendicular_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


def _douglas_peucker(pts: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(pts) <= 2:
        return pts
    max_dist = -1.0
    max_index = 0
    for i in range(1, len(pts) - 1):
        dist = _perpendicular_distance(pts[i], pts[0], pts[-1])
        if dist > max_dist:
            max_dist = dist
            max_index = i
    if max_dist > epsilon:
        left = _douglas_peucker(pts[: max_index + 1], epsilon)
        right = _douglas_peucker(pts[max_index:], epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def simplify_polygon(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    return _douglas_peucker(points, epsilon)
