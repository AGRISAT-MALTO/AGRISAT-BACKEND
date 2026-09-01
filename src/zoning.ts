import {
  geeCall,
  geeConstant,
  addSpectralIndex,
  getGeeAccessToken,
  getGeeProjectId,
  fetchWithRetry,
  normalizePolygon,
  polygonCoordinates,
  GEE_COMPUTE_TIMEOUT_MS,
  type GeeValue,
  type LatLng,
  type PixelGrid,
} from "./analyze-parcel.js";
import { lngLatToMercatorMeters, mercatorMetersToLngLat, parseNpyStructuredFloat32 } from "./field-watershed.js";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "content-type",
};

// ── Définition des 3 zones VRA (mêmes seuils NDVI que la référence : 0.75 / 0.35) ──
// Couleurs = RGB littéral des tokens --vegetation-high/medium/none (index.css), pour rester
// visuellement cohérent avec le reste de l'app sans inventer une nouvelle palette.
interface ZoneDefinition {
  key: "high" | "mid" | "low";
  label: string;
  ndviMin: number;
  ndviMax: number;
  color: string;
}

const ZONE_DEFINITIONS: ZoneDefinition[] = [
  { key: "high", label: "Zone de végétation forte", ndviMin: 0.75, ndviMax: 1.01, color: "#2eb860" },
  { key: "mid", label: "Zone de végétation intermédiaire", ndviMin: 0.35, ndviMax: 0.75, color: "#86ac39" },
  { key: "low", label: "Zone de végétation faible", ndviMin: -1, ndviMax: 0.35, color: "#bf4040" },
];

const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;
const ZONING_THUMBNAIL_SIZE = "320x200";

/**
 * Vignette satellite compacte de la parcelle (contour tracé), pour l'aperçu de l'onglet
 * "Proche en zone" et pour l'export PDF — même service (Google Static Maps) que
 * captureParcelImage dans analyze-parcel.ts, mais sans center/zoom explicites : Static Maps
 * calcule automatiquement le cadrage à partir du path fourni. Échec non bloquant (thumbnail
 * "nice to have", pas le cœur du zoning) : renvoie null plutôt que de faire échouer la requête.
 */
async function captureZoningThumbnail(polygon: LatLng[]): Promise<string | null> {
  if (!GOOGLE_MAPS_API_KEY || GOOGLE_MAPS_API_KEY.startsWith("VOTRE_")) return null;
  try {
    const path = polygon.map((point) => `${point.lat},${point.lng}`).join("|");
    const params = new URLSearchParams({
      size: ZONING_THUMBNAIL_SIZE,
      maptype: "satellite",
      path: `color:0xfbbf24ff|weight:2|fillcolor:0x00000000|${path}`,
      key: GOOGLE_MAPS_API_KEY,
    });
    const resp = await fetchWithRetry(`https://maps.googleapis.com/maps/api/staticmap?${params.toString()}`, {}, 15_000);
    if (!resp.ok) return null;
    const arrayBuffer = await resp.arrayBuffer();
    return Buffer.from(arrayBuffer).toString("base64");
  } catch {
    return null;
  }
}

const ZONING_TARGET_SCALE_M = 10; // résolution native Sentinel-2
const ZONING_MAX_DIMENSION_PX = 400; // borne la taille du raster (perf + limites computePixels)
const ZONING_SEASON_DAYS = 180; // même fenêtre que fetchCurrentSnapshot (cohérence avec le NDVI déjà affiché)

export async function computeZoning(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { polygon } = await req.json();
    const parcelPolygon = normalizePolygon(polygon);
    if (!parcelPolygon) {
      return new Response(
        JSON.stringify({ error: "Le contour réel de la parcelle est requis (au moins 3 points valides)." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
    if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) {
      return new Response(
        JSON.stringify({ error: "GEE_SERVICE_ACCOUNT_KEY n’est pas configurée : le zoning nécessite un accès Earth Engine réel." }),
        { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    let accessToken: string;
    try {
      accessToken = await getGeeAccessToken();
    } catch (error) {
      return new Response(
        JSON.stringify({ error: `GEE indisponible : ${error instanceof Error ? error.message : "authentification impossible"}` }),
        { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }
    const projectId = getGeeProjectId();

    const now = new Date();
    const endDate = now.toISOString().slice(0, 10);
    const startDate = new Date(now.getTime() - ZONING_SEASON_DAYS * 86_400_000).toISOString().slice(0, 10);

    const expression = buildZoningNdviExpression(parcelPolygon, startDate, endDate);
    const grid = computeZoningGrid(parcelPolygon);
    const [{ data, width, height }, thumbnail] = await Promise.all([
      fetchNdviRasterResilient(accessToken, projectId, expression, grid),
      captureZoningThumbnail(parcelPolygon),
    ]);

    const result = classifyZoningRaster(parcelPolygon, grid, data, width, height);

    return new Response(JSON.stringify({
      thumbnail,
      bounds: result.bounds,
      widthPx: width,
      heightPx: height,
      classes: Buffer.from(result.classes).toString("base64"),
      zones: result.zones,
      totalAreaHa: result.totalAreaHa,
      imageStartDate: startDate,
      imageEndDate: endDate,
    }), { headers: { ...corsHeaders, "Content-Type": "application/json" } });
  } catch (error) {
    console.error("zoning error:", error);
    return new Response(JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
}

// ── Expression GEE : composite NDVI médian (180 j, nuages < 35%), même schéma que
// fetchCurrentSnapshot/buildSNICVectorsExpression mais sans reduceRegion final : le résultat
// reste une image (bande "NDVI"), consommée ici par image:computePixels (raster brut). ──
function buildZoningNdviExpression(polygon: LatLng[], startDate: string, endDate: string): { result: string; values: Record<string, GeeValue> } {
  const values: Record<string, GeeValue> = {};
  const ref = (name: string): GeeValue => ({ valueReference: name });

  values.region = geeCall("GeometryConstructors.Polygon", { coordinates: geeConstant(polygonCoordinates(polygon)) });
  values.intersectsRegion = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: ref("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", { start: geeConstant(startDate), end: geeConstant(endDate) }),
    rightField: geeConstant("system:time_start"),
  });
  values.lowCloudCover = geeCall("Filter.lessThan", {
    leftField: geeConstant("CLOUDY_PIXEL_PERCENTAGE"),
    rightValue: geeConstant(35),
  });
  values.collectionByRegion = geeCall("Collection.filter", {
    collection: geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") }),
    filter: ref("intersectsRegion"),
  });
  values.collectionByDate = geeCall("Collection.filter", { collection: ref("collectionByRegion"), filter: ref("dateRange") });
  values.collection = geeCall("Collection.filter", { collection: ref("collectionByDate"), filter: ref("lowCloudCover") });
  values.composite = geeCall("reduce.median", { collection: ref("collection") });
  values.withNdvi = addSpectralIndex(ref("composite"), "NDVI", ["B8", "B4"]);
  values.ndviBand = geeCall("Image.select", { input: ref("withNdvi"), bandSelectors: geeConstant(["NDVI"]) });

  return { result: "ndviBand", values };
}

// ── Grille de calcul (bbox de la parcelle, ~10 m/pixel, plafonnée à 400 px de côté) ──
function computeZoningGrid(polygon: LatLng[]): PixelGrid {
  const corners = polygon.map((p) => lngLatToMercatorMeters(p.lng, p.lat));
  const minX = Math.min(...corners.map((c) => c.x));
  const maxX = Math.max(...corners.map((c) => c.x));
  const minY = Math.min(...corners.map((c) => c.y));
  const maxY = Math.max(...corners.map((c) => c.y));

  const pad = ZONING_TARGET_SCALE_M;
  const paddedMinX = minX - pad;
  const paddedMaxX = maxX + pad;
  const paddedMinY = minY - pad;
  const paddedMaxY = maxY + pad;

  const rawWidthPx = (paddedMaxX - paddedMinX) / ZONING_TARGET_SCALE_M;
  const rawHeightPx = (paddedMaxY - paddedMinY) / ZONING_TARGET_SCALE_M;
  const scaleFactor = Math.max(1, Math.max(rawWidthPx, rawHeightPx) / ZONING_MAX_DIMENSION_PX);
  const scaleMeters = ZONING_TARGET_SCALE_M * scaleFactor;

  return {
    widthPx: Math.max(2, Math.round((paddedMaxX - paddedMinX) / scaleMeters)),
    heightPx: Math.max(2, Math.round((paddedMaxY - paddedMinY) / scaleMeters)),
    scaleMeters,
    originXMeters: paddedMinX,
    originYMeters: paddedMaxY,
  };
}

// ── Fetch du raster NDVI brut (NPY float32) — variante numérique de computePixelsPng ──
async function fetchNdviRaster(
  accessToken: string,
  projectId: string,
  expression: { result: string; values: Record<string, GeeValue> },
  grid: PixelGrid,
): Promise<{ data: Float32Array; width: number; height: number }> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/image:computePixels`;
  const body = {
    expression,
    fileFormat: "NPY",
    bandIds: ["NDVI"],
    grid: {
      dimensions: { width: grid.widthPx, height: grid.heightPx },
      affineTransform: {
        scaleX: grid.scaleMeters,
        shearX: 0,
        translateX: grid.originXMeters,
        shearY: 0,
        scaleY: -grid.scaleMeters,
        translateY: grid.originYMeters,
      },
      crsCode: "EPSG:3857",
    },
  };
  const response = await fetchWithRetry(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, GEE_COMPUTE_TIMEOUT_MS);
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`GEE computePixels (NDVI) : erreur ${response.status} ${text.slice(0, 300)}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  // image:computePixels renvoie un .npy structuré (un champ nommé par bande) dès que
  // bandIds est fourni, même pour une seule bande — pas le format plat "float32 simple".
  const { bands, shape } = parseNpyStructuredFloat32(arrayBuffer);
  const [height, width] = shape;
  return { data: bands.NDVI, width, height };
}

/**
 * fetchNdviRaster échoue parfois avec une erreur réseau générique ("fetch failed") sous charge
 * (plusieurs zonings déclenchés coup sur coup, ex. navigation rapide entre parcelles) — malgré
 * les 3 tentatives déjà internes à fetchWithRetry. Une tentative supplémentaire après un délai
 * plus long absorbe ces creux transitoires sans changer le comportement partagé de
 * fetchWithRetry (utilisé par d'autres endpoints).
 */
async function fetchNdviRasterResilient(
  accessToken: string,
  projectId: string,
  expression: { result: string; values: Record<string, GeeValue> },
  grid: PixelGrid,
): Promise<{ data: Float32Array; width: number; height: number }> {
  try {
    return await fetchNdviRaster(accessToken, projectId, expression, grid);
  } catch (error) {
    console.warn("[zoning] fetchNdviRaster a échoué, nouvelle tentative dans 1.5s :", error);
    await new Promise((resolve) => setTimeout(resolve, 1_500));
    return fetchNdviRaster(accessToken, projectId, expression, grid);
  }
}

// ── Classification pixel-par-pixel : masque le raster (rectangle) au vrai contour de la
// parcelle (ray-casting), puis répartit chaque pixel intérieur dans une des 3 zones NDVI. ──
function pointInPolygon(x: number, y: number, ring: Array<{ x: number; y: number }>): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i].x, yi = ring[i].y;
    const xj = ring[j].x, yj = ring[j].y;
    const intersects = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function classifyZoningRaster(
  polygon: LatLng[],
  grid: PixelGrid,
  data: Float32Array,
  width: number,
  height: number,
) {
  const ring = polygon.map((p) => lngLatToMercatorMeters(p.lng, p.lat));
  const classes = new Uint8Array(width * height);
  const sums = [0, 0, 0];
  const counts = [0, 0, 0];

  for (let row = 0; row < height; row++) {
    const y = grid.originYMeters - (row + 0.5) * grid.scaleMeters;
    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      const ndvi = data[index];
      if (!Number.isFinite(ndvi)) {
        classes[index] = 255;
        continue;
      }
      const x = grid.originXMeters + (col + 0.5) * grid.scaleMeters;
      if (!pointInPolygon(x, y, ring)) {
        classes[index] = 255;
        continue;
      }
      const zoneIndex = ZONE_DEFINITIONS.findIndex((zone) => ndvi >= zone.ndviMin && ndvi < zone.ndviMax);
      const resolvedIndex = zoneIndex === -1 ? 2 : zoneIndex;
      classes[index] = resolvedIndex;
      sums[resolvedIndex] += ndvi;
      counts[resolvedIndex] += 1;
    }
  }

  const pixelAreaHa = (grid.scaleMeters * grid.scaleMeters) / 10_000;
  const zones = ZONE_DEFINITIONS.map((zone, i) => ({
    key: zone.key,
    label: zone.label,
    ndviMin: zone.ndviMin,
    ndviMax: Math.min(zone.ndviMax, 1),
    color: zone.color,
    avgNdvi: counts[i] > 0 ? Math.round((sums[i] / counts[i]) * 100) / 100 : null,
    areaHa: Math.round(counts[i] * pixelAreaHa * 100) / 100,
  }));
  const totalAreaHa = Math.round(zones.reduce((sum, z) => sum + z.areaHa, 0) * 100) / 100;

  const topLeft = mercatorMetersToLngLat(grid.originXMeters, grid.originYMeters);
  const bottomRight = mercatorMetersToLngLat(
    grid.originXMeters + width * grid.scaleMeters,
    grid.originYMeters - height * grid.scaleMeters,
  );

  return {
    classes,
    zones,
    totalAreaHa,
    bounds: { south: bottomRight.lat, west: topLeft.lng, north: topLeft.lat, east: bottomRight.lng },
  };
}
