import {
  buildTrueColorVisualizeExpression,
  computePixelsPng,
  geeCall,
  geeConstant,
  type GeeValue,
  type PixelGrid,
} from "./analyze-parcel.js";

// ── Serveur de tuiles XYZ Sentinel-2 (fond de carte, mode "analyse" uniquement) ──
// Sert exactement l'image Sentinel-2 choisie par selectBestImageWindow (barley-detect-simple.ts)
// pour une zone analysée : le fond affiché correspond toujours à l'image réellement analysée,
// jamais un composite générique. Un mode "auto" (fond permanent, indépendant de toute analyse)
// a existé mais a été retiré : il nécessitait de mosaïquer plusieurs images par tuile pour éviter
// des trous (une tuile à faible zoom dépasse la taille d'une scène Sentinel-2), ce qui s'est
// révélé fragile (algorithmes GEE mal devinés, temps de calcul par tuile ~1-2s incompatible avec
// un pan/zoom libre). Réutilise computePixelsPng (image:computePixels, fileFormat "PNG") ajouté
// dans analyze-parcel.ts pour la miniature CNN.

const WEB_MERCATOR_EXTENT_METERS = 20_037_508.342789244;
const TILE_SIZE_PX = 256;

export interface TileCoordinates {
  z: number;
  x: number;
  y: number;
}

/** Convertit une tuile slippy-map (z/x/y) en grille EPSG:3857 pour image:computePixels. */
export function tileToMercatorGrid({ z, x, y }: TileCoordinates): PixelGrid {
  const tileSizeMeters = (2 * WEB_MERCATOR_EXTENT_METERS) / 2 ** z;
  const originXMeters = -WEB_MERCATOR_EXTENT_METERS + x * tileSizeMeters;
  const originYMeters = WEB_MERCATOR_EXTENT_METERS - y * tileSizeMeters;
  return {
    widthPx: TILE_SIZE_PX,
    heightPx: TILE_SIZE_PX,
    originXMeters,
    originYMeters,
    scaleMeters: tileSizeMeters / TILE_SIZE_PX,
  };
}

function buildTileRegion(grid: PixelGrid): GeeValue {
  const west = grid.originXMeters;
  const north = grid.originYMeters;
  const east = grid.originXMeters + grid.widthPx * grid.scaleMeters;
  const south = grid.originYMeters - grid.heightPx * grid.scaleMeters;
  return geeCall("GeometryConstructors.Rectangle", {
    coordinates: geeConstant([west, south, east, north]),
    crs: geeCall("Projection", { crs: geeConstant("EPSG:3857") }),
    geodesic: geeConstant(false),
  });
}

// Cache mémoire simple (clé -> octets PNG) pour éviter de rappeler GEE en boucle pendant un pan/zoom
// sur la zone déjà analysée.
const TILE_CACHE_MAX_ENTRIES = 500;
const TILE_CACHE_TTL_MS = 30 * 60_000;
const tileCache = new Map<string, { expiresAt: number; bytes: Uint8Array }>();

function cacheGet(key: string): Uint8Array | null {
  const entry = tileCache.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= Date.now()) {
    tileCache.delete(key);
    return null;
  }
  return entry.bytes;
}

function cacheSet(key: string, bytes: Uint8Array): void {
  if (tileCache.size >= TILE_CACHE_MAX_ENTRIES) {
    const oldestKey = tileCache.keys().next().value;
    if (oldestKey !== undefined) tileCache.delete(oldestKey);
  }
  tileCache.set(key, { expiresAt: Date.now() + TILE_CACHE_TTL_MS, bytes });
}

/**
 * Récupère la tuile PNG vrai-couleur Sentinel-2 pour z/x/y, en sélectionnant l'image
 * COPERNICUS/S2_SR_HARMONIZED dont system:time_start correspond exactement à imageTimestampMs
 * (la même image que celle choisie pour l'analyse). Retourne null si aucune image ne couvre
 * cette tuile à cet instant précis (ex. tuile en dehors de l'empreinte de la scène S2).
 */
export async function fetchSentinel2TilePng(
  accessToken: string,
  projectId: string,
  tile: TileCoordinates,
  imageTimestampMs: number,
): Promise<Uint8Array | null> {
  const cacheKey = `${tile.z}-${tile.x}-${tile.y}-${imageTimestampMs}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  const grid = tileToMercatorGrid(tile);
  const values: Record<string, GeeValue> = {};
  const ref = (name: string): GeeValue => ({ valueReference: name });

  values.region = buildTileRegion(grid);
  values.intersects = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: ref("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", {
      start: geeConstant(new Date(imageTimestampMs).toISOString()),
      end: geeConstant(new Date(imageTimestampMs + 1_000).toISOString()),
    }),
    rightField: geeConstant("system:time_start"),
  });
  values.raw = geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") });
  values.byRegion = geeCall("Collection.filter", { collection: ref("raw"), filter: ref("intersects") });
  values.byDate = geeCall("Collection.filter", { collection: ref("byRegion"), filter: ref("dateRange") });
  values.image = geeCall("Collection.first", { collection: ref("byDate") });
  values.visualized = buildTrueColorVisualizeExpression(ref("image"));

  try {
    const bytes = await computePixelsPng(accessToken, projectId, { result: "visualized", values }, grid);
    cacheSet(cacheKey, bytes);
    return bytes;
  } catch (error) {
    console.warn(`[sentinel-tiles] Tuile ${tile.z}/${tile.x}/${tile.y} indisponible :`, error);
    return null;
  }
}
