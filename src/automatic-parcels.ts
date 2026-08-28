import { analyzeParcel } from "./analyze-parcel.js";
import {
  addBareSoilIndex,
  addSpectralIndex,
  approximatePolygonAreaM2,
  callGeeComputeRaw,
  extractLatLngFromGeometry,
  fetchWithRetry,
  geeCall,
  geeConstant,
  geeImageConstant,
  GEE_COMPUTE_TIMEOUT_MS,
  getGeeAccessToken,
  getGeeProjectId,
  isGeeFeatureCollection,
  type GeeValue,
} from "./analyze-parcel.js";
import {
  encodeNpyFloat32,
  lngLatToMercatorMeters,
  mercatorMetersToLngLat,
  parseNpyFloat32,
  parseNpyStructuredFloat32,
  simplifyPolygon,
  traceLabelContours,
  watershedSegment,
} from "./field-watershed.js";
import { Prisma } from "@prisma/client";
import { prisma } from "./db.js";

export interface BarleyDetectionConfig {
  baseTemperature: number;
  threshold: number;
  periodDays: number;
}

export interface AutoDetectionRequest extends BarleyDetectionConfig {
  lat: number;
  lng: number;
  radiusKm: number;
}

interface CandidateParcel {
  id: string;
  coordinates: Array<{ lat: number; lng: number }>;
  center: { lat: number; lng: number };
  tags: Record<string, string>;
  persist?: boolean;
}

export interface AutoDetectedParcel extends CandidateParcel {
  analysis: Record<string, unknown> | null;
  analysis_error: string | null;
}

export interface AutoDetectionResponse {
  center: { lat: number; lng: number };
  radius_km: number;
  base_temperature: number;
  threshold: number;
  period_days: number;
  candidates_found: number;
  analyzed_count: number;
  parcels: AutoDetectedParcel[];
  notice: string | null;
}

const OVERPASS_API_URLS = [
  process.env.OVERPASS_API_URL ?? "https://overpass-api.de/api/interpreter",
  "https://overpass.openstreetmap.fr/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];
const OVERPASS_USER_AGENT = process.env.OVERPASS_USER_AGENT ?? "fieldscan-ai/1.0 (+https://localhost)";
const OVERPASS_REQUEST_TIMEOUT_MS = 10_000; // par miroir ; 3 miroirs dans OVERPASS_API_URLS = 30s max au lieu de 75s
const DATABASE_DISCOVERY_TIMEOUT_MS = 8_000; // couvre un cold-start Neon normal sans bloquer toute la chaîne
const MAX_CANDIDATES = 48;
const MAX_SATELLITE_CELLS = 36;
const ANALYSIS_CONCURRENCY = 4;
const ANALYSIS_TIME_BUDGET_MS = 60_000;

// Empêche un appel qui ne répond jamais (DB endormie, réseau qui pend) de bloquer toute
// la chaîne de découverte — sans ça, une seule étape lente empêche les suivantes (dont
// le modèle de segmentation) de jamais s'exécuter.
function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} : dépassement de ${timeoutMs}ms`)), timeoutMs);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error) => { clearTimeout(timer); reject(error); },
    );
  });
}

// ── Segmentation par modèle de segmentation de parcelles (U-Net, agri_field_segmentation) ──
// Contrairement au watershed/SNIC ci-dessous (heuristiques classiques), ce modèle a été
// entraîné spécifiquement pour délimiter des parcelles agricoles (dataset AI4Boundaries),
// à partir de 30 canaux : B2/B3/B4/B8/NDVI sur 6 dates mensuelles (mars→août). Voir
// agri_field_segmentation/data/dataset.py::load_ai4boundaries_image pour le format exact.
const FIELD_SEGMENTATION_MODEL_URL = process.env.FIELD_SEGMENTATION_MODEL_URL;
const FIELD_MODEL_TILE_PX = 256;
const FIELD_MODEL_SCALE_M = 10; // doit correspondre à la résolution d'entraînement (10m/px)
const FIELD_MODEL_REGION_RADIUS_KM = 2; // couvre la tuile carrée (2,56km de côté, demi-diagonale ~1,81km) avec marge
const MAX_FIELD_MODEL_RADIUS_KM = 1.2; // tuile fixe : ne couvre pas un rayon de recherche plus large
const FIELD_MODEL_MONTHS = ["03", "04", "05", "06", "07", "08"]; // fenêtre d'entraînement AI4Boundaries
const FIELD_MODEL_CLOUD_LIMIT = 35;
const FIELD_MODEL_GEE_CONCURRENCY = 3; // 30 appels image:computePixels par requête (5 bandes × 6 mois)
const FIELD_MODEL_MIN_SEGMENT_AREA_M2 = 1_000;   // 0.1 ha, cohérent avec le watershed
const FIELD_MODEL_MAX_SEGMENT_AREA_M2 = 800_000; // 80 ha
const FIELD_MODEL_REQUEST_TIMEOUT_MS = 30_000;

/** Dernière fenêtre mars-août complète : année en cours si on est en septembre ou après, sinon l'année précédente. */
function fieldModelSeasonYear(): number {
  const now = new Date();
  return now.getUTCMonth() + 1 >= 9 ? now.getUTCFullYear() : now.getUTCFullYear() - 1;
}

/** Composite médian S2 d'un mois donné, bandes B2/B3/B4/B8 + NDVI, trous nuageux comblés à 0. */
function buildMonthlyBandsExpression(lat: number, lng: number, year: number, month: string) {
  const startDate = `${year}-${month}-01`;
  const endDate = new Date(Date.UTC(year, Number(month), 1)).toISOString().slice(0, 10);
  const values: Record<string, GeeValue> = {};
  const reference = (name: string): GeeValue => ({ valueReference: name });

  values.region = geeCall("GeometryConstructors.Polygon", {
    coordinates: geeConstant(circleRegionCoordinates(lat, lng, FIELD_MODEL_REGION_RADIUS_KM)),
  });
  values.intersectsRegion = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: reference("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", { start: geeConstant(startDate), end: geeConstant(endDate) }),
    rightField: geeConstant("system:time_start"),
  });
  values.lowCloud = geeCall("Filter.lessThan", {
    leftField: geeConstant("CLOUDY_PIXEL_PERCENTAGE"),
    rightValue: geeConstant(FIELD_MODEL_CLOUD_LIMIT),
  });
  values.byRegion = geeCall("Collection.filter", {
    collection: geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") }),
    filter: reference("intersectsRegion"),
  });
  values.byDate = geeCall("Collection.filter", { collection: reference("byRegion"), filter: reference("dateRange") });
  values.collection = geeCall("Collection.filter", { collection: reference("byDate"), filter: reference("lowCloud") });
  values.composite = geeCall("reduce.median", { collection: reference("collection") });
  values.spectral = geeCall("Image.select", {
    input: reference("composite"),
    bandSelectors: geeConstant(["B2", "B3", "B4", "B8"]),
  });
  values.withNdvi = addSpectralIndex(reference("spectral"), "NDVI", ["B8", "B4"]);
  // Comble les trous nuageux/absents avec 0, comme nan_to_num(nan=0.0) côté modèle
  // (load_ai4boundaries_image) — fait ici côté GEE plutôt que côté client.
  // Note : pas de cast Image.toFloat ici — l'export NPY multi-bandes structuré renvoie
  // du float64 (<f8) quel que soit le type source, et GEE a systématiquement rejeté
  // Image.toFloat sur une image multi-bandes lors des tests (erreur "Parameter 'value'
  // is required"). parseNpyStructuredFloat32 gère nativement f4 et f8, donc inutile.
  values.finalImage = geeCall("Image.unmask", { input: reference("withNdvi"), value: geeImageConstant(0) });

  return { result: "finalImage", values };
}

interface FieldModelGrid {
  widthPx: number;
  heightPx: number;
  originXMeters: number;
  originYMeters: number;
  scaleMeters: number;
}

async function fetchFieldModelInputArray(
  accessToken: string,
  projectId: string,
  lat: number,
  lng: number,
): Promise<{ array: Float32Array; grid: FieldModelGrid }> {
  const year = fieldModelSeasonYear();
  const center = lngLatToMercatorMeters(lng, lat);
  const halfSizeMeters = (FIELD_MODEL_TILE_PX * FIELD_MODEL_SCALE_M) / 2;
  const grid: FieldModelGrid = {
    widthPx: FIELD_MODEL_TILE_PX,
    heightPx: FIELD_MODEL_TILE_PX,
    originXMeters: center.x - halfSizeMeters,
    originYMeters: center.y + halfSizeMeters, // coin haut-gauche ; Y décroît vers le bas
    scaleMeters: FIELD_MODEL_SCALE_M,
  };

  // Ordre = VARIABLES dans dataset.py : B2,B3,B4,B8,NDVI.
  const bandNames = ["B2", "B3", "B4", "B8", "NDVI"];

  // Un seul appel GEE par mois (NPY multi-bandes structuré) plutôt qu'un par bande :
  // 6 appels au lieu de 30, même résultat, latence cumulée ~5x moindre.
  const monthTasks = FIELD_MODEL_MONTHS.map((month, monthIndex) => ({ month, monthIndex }));
  const monthlyResults = await mapWithConcurrency(monthTasks, FIELD_MODEL_GEE_CONCURRENCY, async ({ month, monthIndex }) => {
    const expression = buildMonthlyBandsExpression(lat, lng, year, month);
    const raster = await callGeeComputePixelsMultiBand(accessToken, projectId, expression, bandNames, grid);
    return { monthIndex, raster };
  });

  const tileSize = FIELD_MODEL_TILE_PX * FIELD_MODEL_TILE_PX;
  const array = new Float32Array(bandNames.length * FIELD_MODEL_MONTHS.length * tileSize);
  for (const { monthIndex, raster } of monthlyResults) {
    bandNames.forEach((bandName, bandIndex) => {
      // Ordre canal = [B2×6,B3×6,B4×6,B8×6,NDVI×6], identique à VARIABLES dans dataset.py.
      const channelIndex = bandIndex * FIELD_MODEL_MONTHS.length + monthIndex;
      array.set(raster.bands[bandName], channelIndex * tileSize);
    });
  }

  return { array, grid };
}

interface FieldModelPolygon {
  points: Array<{ x: number; y: number }>;
  score?: number;
}

interface FieldModelSegmentResponse {
  polygons: FieldModelPolygon[];
  image_width: number;
  image_height: number;
}

/** Contrat attendu : POST multipart "file" -> tableau .npy (30,256,256) brut, réponse
 * { polygons: [{points:[{x,y},...], score}], image_width, image_height } en coordonnées pixels
 * (origine haut-gauche) — voir agri_field_segmentation/api.py::segment_endpoint. */
async function callFieldBoundaryModel(rawArray: Float32Array): Promise<FieldModelSegmentResponse> {
  const npyBytes = encodeNpyFloat32(rawArray, [5 * FIELD_MODEL_MONTHS.length, FIELD_MODEL_TILE_PX, FIELD_MODEL_TILE_PX]);
  const formData = new FormData();
  formData.append("file", new Blob([new Uint8Array(npyBytes)]), "field-input.npy");

  const response = await fetchWithRetry(`${FIELD_SEGMENTATION_MODEL_URL}/segment`, {
    method: "POST",
    body: formData,
  }, FIELD_MODEL_REQUEST_TIMEOUT_MS);

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Modèle de segmentation de parcelles : erreur ${response.status} ${text.slice(0, 300)}`);
  }
  const data: unknown = await response.json();
  if (!data || typeof data !== "object" || !Array.isArray((data as Record<string, unknown>).polygons)) {
    throw new Error("Réponse du modèle de segmentation de parcelles invalide (champ 'polygons' manquant).");
  }
  return data as FieldModelSegmentResponse;
}

async function discoverAgriculturalParcelsFromFieldModel(
  lat: number,
  lng: number,
  radiusKm: number,
): Promise<CandidateParcel[]> {
  if (!FIELD_SEGMENTATION_MODEL_URL) return [];
  if (radiusKm > MAX_FIELD_MODEL_RADIUS_KM) return [];

  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) return [];

  const accessToken = await getGeeAccessToken();
  const projectId = getGeeProjectId();
  const { array, grid } = await fetchFieldModelInputArray(accessToken, projectId, lat, lng);
  const result = await callFieldBoundaryModel(array);

  const candidates: CandidateParcel[] = [];
  let index = 0;
  for (const polygon of result.polygons) {
    if (!Array.isArray(polygon.points) || polygon.points.length < 3) continue;

    const coordinates = polygon.points.map((point) => {
      const xMeters = grid.originXMeters + point.x * grid.scaleMeters;
      const yMeters = grid.originYMeters - point.y * grid.scaleMeters;
      const { lng: pointLng, lat: pointLat } = mercatorMetersToLngLat(xMeters, yMeters);
      return { lat: pointLat, lng: pointLng };
    });

    const areaM2 = approximatePolygonAreaM2(coordinates);
    if (areaM2 < FIELD_MODEL_MIN_SEGMENT_AREA_M2 || areaM2 > FIELD_MODEL_MAX_SEGMENT_AREA_M2) continue;
    if (!isPlausibleFieldShape(coordinates, areaM2)) continue;

    index++;
    candidates.push({
      id: `field-model-${lat.toFixed(6)}-${lng.toFixed(6)}-${index}`,
      coordinates,
      center: polygonCenter(coordinates),
      tags: { source: "field-boundary-model" },
    });
  }

  return candidates;
}

// ── Segmentation par watershed marqué + variance NDVI multi-temporelle (sans ML) ──
// Approche la plus fiable des trois : suit les vrais gradients NDVI (comme un
// vrai contour de champ) au lieu de regrouper des pixels similaires (SNIC),
// et exploite la différence de calendrier cultural entre parcelles voisines
// (variance temporelle) pour distinguer deux champs qui se ressemblent à un
// instant T. Voir src/field-watershed.ts pour l'algorithme (priority-flood).
const WATERSHED_RASTER_SCALE_M = 10; // résolution Sentinel-2
const MAX_WATERSHED_RADIUS_KM = 1.5; // borne la taille du raster (perf + limites computePixels)
const WATERSHED_SEASON_DAYS = 120;
const WATERSHED_TEMPORAL_PERIODS = 4;
const WATERSHED_MIN_SEGMENT_AREA_M2 = 1_000;   // 0.1 ha
const WATERSHED_MAX_SEGMENT_AREA_M2 = 800_000; // 80 ha
const WATERSHED_SIMPLIFY_EPS_PX = 1.2;

// ── Watershed marqué + variance NDVI multi-temporelle ──

async function callGeeComputePixels(
  accessToken: string,
  projectId: string,
  expression: { result: string; values: Record<string, GeeValue> },
  bandId: string,
  grid: { widthPx: number; heightPx: number; originXMeters: number; originYMeters: number; scaleMeters: number },
): Promise<{ data: Float32Array; width: number; height: number }> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/image:computePixels`;
  const body = {
    expression,
    fileFormat: "NPY",
    bandIds: [bandId],
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
    throw new Error(`GEE computePixels (${bandId}) : erreur ${response.status} ${text.slice(0, 300)}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const { data, shape } = parseNpyFloat32(arrayBuffer);
  const [height, width] = shape;
  return { data, width, height };
}

/**
 * Variante multi-bandes de callGeeComputePixels : récupère plusieurs bandes en un seul
 * appel GEE (NPY structuré) au lieu d'un appel par bande — réduit le nombre de requêtes
 * (et donc la latence cumulée) d'un facteur égal au nombre de bandes demandées.
 */
async function callGeeComputePixelsMultiBand(
  accessToken: string,
  projectId: string,
  expression: { result: string; values: Record<string, GeeValue> },
  bandIds: string[],
  grid: { widthPx: number; heightPx: number; originXMeters: number; originYMeters: number; scaleMeters: number },
): Promise<{ bands: Record<string, Float32Array>; width: number; height: number }> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/image:computePixels`;
  const body = {
    expression,
    fileFormat: "NPY",
    bandIds,
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
    throw new Error(`GEE computePixels (${bandIds.join(",")}) : erreur ${response.status} ${text.slice(0, 300)}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const { bands, shape } = parseNpyStructuredFloat32(arrayBuffer);
  const [height, width] = shape;
  return { bands, width, height };
}

/**
 * Construit l'image 2 bandes ("grad", "tstd") utilisée comme carte de "force
 * de frontière" pour le watershed :
 *  - grad : magnitude du gradient spatial du NDVI médian saisonnier
 *  - tstd : écart-type du NDVI entre WATERSHED_TEMPORAL_PERIODS sous-périodes
 * Les deux bandes valent WATERSHED_BARRIER_VALUE hors du masque "surface
 * agricole plausible" (végétatif, non-eau, non-bâti, non-sol-nu), ce qui les
 * transforme en barrières infranchissables pour le watershed.
 */
const WATERSHED_BARRIER_VALUE = 1_000_000;

function buildBoundaryStrengthExpression(lat: number, lng: number, radiusKm: number) {
  const endDate = new Date();
  const seasonStart = new Date(endDate.getTime() - WATERSHED_SEASON_DAYS * 86_400_000);
  const values: Record<string, GeeValue> = {};
  const reference = (name: string): GeeValue => ({ valueReference: name });

  values.region = geeCall("GeometryConstructors.Polygon", {
    coordinates: geeConstant(circleRegionCoordinates(lat, lng, radiusKm)),
  });
  values.intersectsRegion = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: reference("region") }),
  });
  values.collectionByRegion = geeCall("Collection.filter", {
    collection: geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") }),
    filter: reference("intersectsRegion"),
  });

  // Composite NDVI de toute la saison, pour le gradient spatial + le masque cropland.
  const buildPeriodNdvi = (varPrefix: string, startDate: string, endDate: string, cloudLimit: number) => {
    values[`${varPrefix}DateRange`] = geeCall("Filter.dateRangeContains", {
      leftValue: geeCall("DateRange", { start: geeConstant(startDate), end: geeConstant(endDate) }),
      rightField: geeConstant("system:time_start"),
    });
    values[`${varPrefix}ByDate`] = geeCall("Collection.filter", {
      collection: reference("collectionByRegion"),
      filter: reference(`${varPrefix}DateRange`),
    });
    values[`${varPrefix}LowCloud`] = geeCall("Filter.lessThan", {
      leftField: geeConstant("CLOUDY_PIXEL_PERCENTAGE"),
      rightValue: geeConstant(cloudLimit),
    });
    values[`${varPrefix}Collection`] = geeCall("Collection.filter", {
      collection: reference(`${varPrefix}ByDate`),
      filter: reference(`${varPrefix}LowCloud`),
    });
    values[`${varPrefix}Composite`] = geeCall("reduce.median", { collection: reference(`${varPrefix}Collection`) });
    values[`${varPrefix}Spectral`] = geeCall("Image.select", {
      input: reference(`${varPrefix}Composite`),
      bandSelectors: geeConstant(["B2", "B3", "B4", "B8", "B11", "B12"]),
    });
    values[`${varPrefix}WithNdvi`] = addSpectralIndex(reference(`${varPrefix}Spectral`), "NDVI", ["B8", "B4"]);
    values[`${varPrefix}NdviBand`] = geeCall("Image.select", {
      input: reference(`${varPrefix}WithNdvi`),
      bandSelectors: geeConstant(["NDVI"]),
    });
    return `${varPrefix}NdviBand`;
  };

  const seasonNdviRef = buildPeriodNdvi("season", seasonStart.toISOString().slice(0, 10), endDate.toISOString().slice(0, 10), 35);

  // Bandes de masquage cropland, calculées sur le composite saisonnier complet (NDWI/NDBI/BSI).
  values.seasonWithNdwi = addSpectralIndex(reference("seasonWithNdvi"), "NDWI", ["B3", "B8"]);
  values.seasonWithNdbi = addSpectralIndex(reference("seasonWithNdwi"), "NDBI", ["B11", "B8"]);
  values.seasonWithBsi = addBareSoilIndex(reference("seasonWithNdbi"));
  const seasonBand = (name: string) => geeCall("Image.select", {
    input: reference("seasonWithBsi"),
    bandSelectors: geeConstant([name]),
  });
  values.ndviMask = geeCall("Image.gt", { image1: seasonBand("NDVI"), image2: geeImageConstant(0.15) });
  values.waterMask = geeCall("Image.lt", { image1: seasonBand("NDWI"), image2: geeImageConstant(0.1) });
  values.urbanMask = geeCall("Image.lt", { image1: seasonBand("NDBI"), image2: geeImageConstant(0.05) });
  values.bareSoilMask = geeCall("Image.lt", { image1: seasonBand("BSI"), image2: geeImageConstant(0.25) });
  values.fieldMask = geeCall("Image.and", {
    image1: geeCall("Image.and", {
      image1: geeCall("Image.and", { image1: reference("ndviMask"), image2: reference("waterMask") }),
      image2: reference("urbanMask"),
    }),
    image2: reference("bareSoilMask"),
  });

  // Gradient spatial du NDVI saisonnier -> magnitude.
  values.ndviGradient = geeCall("Image.gradient", { input: reference(seasonNdviRef) });
  values.gradX = geeCall("Image.select", { input: reference("ndviGradient"), bandSelectors: geeConstant(["x"]) });
  values.gradY = geeCall("Image.select", { input: reference("ndviGradient"), bandSelectors: geeConstant(["y"]) });
  values.gradMagnitude = geeCall("Image.hypot", { image1: reference("gradX"), image2: reference("gradY") });

  // NDVI sur WATERSHED_TEMPORAL_PERIODS sous-périodes -> écart-type temporel.
  const periodMs = (WATERSHED_SEASON_DAYS * 86_400_000) / WATERSHED_TEMPORAL_PERIODS;
  let stackRef: string | null = null;
  for (let i = 0; i < WATERSHED_TEMPORAL_PERIODS; i++) {
    const periodStart = new Date(seasonStart.getTime() + i * periodMs);
    const periodEnd = new Date(seasonStart.getTime() + (i + 1) * periodMs);
    const ndviRef = buildPeriodNdvi(`p${i}`, periodStart.toISOString().slice(0, 10), periodEnd.toISOString().slice(0, 10), 50);
    const renamedRef = `p${i}NdviRenamed`;
    values[renamedRef] = geeCall("Image.rename", { input: reference(ndviRef), names: geeConstant([`ndvi_${i}`]) });
    stackRef = stackRef === null
      ? renamedRef
      : (() => {
        const combinedRef = `tempStack${i}`;
        values[combinedRef] = geeCall("Image.addBands", { dstImg: reference(stackRef as string), srcImg: reference(renamedRef) });
        return combinedRef;
      })();
  }
  values.temporalStdDev = geeCall("Image.reduce", {
    image: reference(stackRef as string),
    reducer: geeCall("Reducer.stdDev", {}),
  });

  // Application du masque cropland comme barrière (valeur très haute hors zone agricole).
  const applyBarrier = (imageRef: string, outName: string) => {
    const maskedRef = `${outName}Masked`;
    const unmaskedRef = `${outName}Barrier`;
    values[maskedRef] = geeCall("Image.updateMask", { image: reference(imageRef), mask: reference("fieldMask") });
    values[unmaskedRef] = geeCall("Image.unmask", { input: reference(maskedRef), value: geeImageConstant(WATERSHED_BARRIER_VALUE) });
    return unmaskedRef;
  };
  const gradBarrierRef = applyBarrier("gradMagnitude", "grad");
  const tstdBarrierRef = applyBarrier("temporalStdDev", "tstd");

  values.gradFloat = geeCall("Image.rename", { input: geeCall("Image.toFloat", { input: reference(gradBarrierRef) }), names: geeConstant(["grad"]) });
  values.tstdFloat = geeCall("Image.rename", { input: geeCall("Image.toFloat", { input: reference(tstdBarrierRef) }), names: geeConstant(["tstd"]) });
  values.finalImage = geeCall("Image.clip", {
    input: geeCall("Image.addBands", { dstImg: reference("gradFloat"), srcImg: reference("tstdFloat") }),
    geometry: reference("region"),
  });

  return { result: "finalImage", values };
}

async function discoverAgriculturalParcelsFromWatershed(
  lat: number,
  lng: number,
  radiusKm: number,
): Promise<CandidateParcel[]> {
  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) return [];
  if (radiusKm > MAX_WATERSHED_RADIUS_KM) return [];

  const accessToken = await getGeeAccessToken();
  const projectId = getGeeProjectId();
  const expression = buildBoundaryStrengthExpression(lat, lng, radiusKm);

  const center = lngLatToMercatorMeters(lng, lat);
  const radiusMeters = radiusKm * 1000 * 1.05; // légère marge
  const widthPx = Math.min(400, Math.ceil((2 * radiusMeters) / WATERSHED_RASTER_SCALE_M));
  const heightPx = widthPx;
  const grid = {
    widthPx,
    heightPx,
    originXMeters: center.x - radiusMeters,
    originYMeters: center.y + radiusMeters, // origine = coin haut-gauche (Y décroît vers le bas)
    scaleMeters: WATERSHED_RASTER_SCALE_M,
  };

  const [gradRaster, tstdRaster] = await Promise.all([
    callGeeComputePixels(accessToken, projectId, expression, "grad", grid),
    callGeeComputePixels(accessToken, projectId, expression, "tstd", grid),
  ]);
  const { width, height } = gradRaster;
  const size = width * height;

  // Normalisation min-max (hors barrière) + fusion des deux signaux de frontière.
  const barrier = new Uint8Array(size);
  let gradMin = Infinity, gradMax = -Infinity, tstdMin = Infinity, tstdMax = -Infinity;
  for (let i = 0; i < size; i++) {
    const isBarrier = gradRaster.data[i] >= WATERSHED_BARRIER_VALUE || tstdRaster.data[i] >= WATERSHED_BARRIER_VALUE;
    barrier[i] = isBarrier ? 1 : 0;
    if (!isBarrier) {
      if (gradRaster.data[i] < gradMin) gradMin = gradRaster.data[i];
      if (gradRaster.data[i] > gradMax) gradMax = gradRaster.data[i];
      if (tstdRaster.data[i] < tstdMin) tstdMin = tstdRaster.data[i];
      if (tstdRaster.data[i] > tstdMax) tstdMax = tstdRaster.data[i];
    }
  }
  if (!Number.isFinite(gradMin) || !Number.isFinite(tstdMin)) return []; // rien d'exploitable (tout barrière)

  const strength = new Float32Array(size);
  const gradRange = Math.max(gradMax - gradMin, 1e-9);
  const tstdRange = Math.max(tstdMax - tstdMin, 1e-9);
  for (let i = 0; i < size; i++) {
    if (barrier[i]) {
      strength[i] = WATERSHED_BARRIER_VALUE;
      continue;
    }
    const normGrad = (gradRaster.data[i] - gradMin) / gradRange;
    const normTstd = (tstdRaster.data[i] - tstdMin) / tstdRange;
    strength[i] = 0.6 * normGrad + 0.4 * normTstd;
  }

  const labels = watershedSegment(strength, barrier, width, height);
  const contours = traceLabelContours(labels, width, height);

  const candidates: CandidateParcel[] = [];
  let index = 0;
  for (const pixelContour of contours.values()) {
    const simplified = simplifyPolygon(pixelContour, WATERSHED_SIMPLIFY_EPS_PX);
    if (simplified.length < 3) continue;

    const coordinates = simplified.map((point: { x: number; y: number }) => {
      const xMeters = grid.originXMeters + point.x * grid.scaleMeters;
      const yMeters = grid.originYMeters - point.y * grid.scaleMeters;
      const { lng: pointLng, lat: pointLat } = mercatorMetersToLngLat(xMeters, yMeters);
      return { lat: pointLat, lng: pointLng };
    });

    const areaM2 = approximatePolygonAreaM2(coordinates);
    if (areaM2 < WATERSHED_MIN_SEGMENT_AREA_M2 || areaM2 > WATERSHED_MAX_SEGMENT_AREA_M2) continue;
    if (!isPlausibleFieldShape(coordinates, areaM2)) continue;

    index++;
    candidates.push({
      id: `gee-watershed-${lat.toFixed(6)}-${lng.toFixed(6)}-${index}`,
      coordinates,
      center: polygonCenter(coordinates),
      tags: { source: "gee-watershed-segmentation" },
    });
  }

  return candidates;
}

// ── Segmentation automatique des limites de parcelles (SNIC via Google Earth Engine) ──
// Aucun modèle ML : SNIC (Simple Non-Iterative Clustering) est un algorithme de
// segmentation classique par superpixels, appliqué ici à une image Sentinel-2
// (NDVI/NDWI/NDBI/BSI) sur une région circulaire autour du point demandé.
// C'est le même algorithme que celui déjà utilisé par analyze-parcel.ts pour
// isoler des sous-parcelles d'orge à l'intérieur d'un contour connu — on l'
// applique ici sans contour préalable, sur un disque de rayon radiusKm.
const REGION_SNIC_PARAMETERS = {
  size: 24,          // taille cible des superpixels (px, plus grand ici que la détection d'orge intra-parcelle)
  compactness: 0.6,  // plus bas = suit mieux les contours réels des champs, plus haut = formes plus régulières
  connectivity: 8,
  scale: 10,         // résolution Sentinel-2 (m/pixel)
} as const;
const MIN_PARCEL_SEGMENT_AREA_M2 = 1_000;   // 0.1 ha
const MAX_PARCEL_SEGMENT_AREA_M2 = 800_000; // 80 ha
const MAX_SEGMENTATION_RADIUS_KM = 5; // au-delà, le coût de calcul GEE devient trop élevé (timeout probable)
const CIRCLE_REGION_VERTICES = 48;

type OverpassElement = {
  type?: string;
  id?: number;
  geometry?: Array<{ lat?: number; lon?: number }>;
  tags?: Record<string, string>;
};

export async function detectAutomaticParcels({ lat, lng, radiusKm, baseTemperature, threshold, periodDays }: AutoDetectionRequest): Promise<AutoDetectionResponse> {
  try {
    const candidates = await discoverAgriculturalParcels(lat, lng, radiusKm);
    const config = { baseTemperature, threshold, periodDays };
    const analyzedCandidates = candidates.slice(0, MAX_CANDIDATES);
    const fieldModelFallback = analyzedCandidates.some((candidate) => candidate.tags.source === "field-boundary-model");
    const watershedFallback = !fieldModelFallback && analyzedCandidates.some((candidate) => candidate.tags.source === "gee-watershed-segmentation");
    const geeSegmentationFallback = !fieldModelFallback && !watershedFallback && analyzedCandidates.some((candidate) => candidate.tags.source === "gee-snic-segmentation");
    const satelliteWindowFallback = !fieldModelFallback && !watershedFallback && !geeSegmentationFallback && analyzedCandidates.some((candidate) => candidate.tags.source?.startsWith("satellite-search"));
    // Chaque analyse (analyzeCandidate -> analyzeParcel, SNIC + CNN + GEE) coûte ~30-40s :
    // avec beaucoup de candidats (ex: Overpass en retourne parfois des dizaines), les
    // analyser tous séquentiellement/à faible concurrence peut prendre 10+ minutes.
    // On borne donc la phase d'analyse par un budget de temps global plutôt que de
    // deviner un nombre de candidats : les candidats non traités à temps repartent avec
    // le même statut "non analysé" que ceux au-delà de MAX_CANDIDATES (voir plus bas).
    const { results: analyzedResults } = await mapWithConcurrencyDeadline(
      analyzedCandidates,
      ANALYSIS_CONCURRENCY,
      ANALYSIS_TIME_BUDGET_MS,
      (candidate) => analyzeCandidate(candidate, config),
    );
    const analyzedParcels = analyzedCandidates.map((candidate, index) => analyzedResults.get(index) ?? {
      ...candidate,
      analysis: null,
      analysis_error: "Analyse IA non exécutée : délai de recherche automatique dépassé (trop de parcelles candidates).",
    });
    const analyzedIds = new Set(analyzedCandidates.map((candidate) => candidate.id));
    const parcels = [
      ...analyzedParcels,
      ...candidates
        .filter((candidate) => !analyzedIds.has(candidate.id))
        .map((candidate) => ({
          ...candidate,
          analysis: null,
          analysis_error: "Analyse IA non exécutée pour cette parcelle.",
        })),
    ];

    return {
      center: { lat, lng },
      radius_km: radiusKm,
      base_temperature: baseTemperature,
      threshold,
      period_days: periodDays,
      candidates_found: candidates.length,
      analyzed_count: parcels.filter((parcel) => parcel.analysis !== null).length,
      parcels,
      notice: fieldModelFallback
        ? "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par le modèle de segmentation U-Net (IA, entraîné sur AI4Boundaries)."
        : watershedFallback
          ? "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par watershed + analyse NDVI multi-temporelle (Google Earth Engine, sans modèle IA)."
          : geeSegmentationFallback
            ? "Aucun contour vectoriel référencé : les limites de parcelles ont été détectées automatiquement par segmentation d’image satellite (SNIC / Google Earth Engine, sans modèle IA)."
            : satelliteWindowFallback
              ? "Aucun contour vectoriel n’a été trouvé : plusieurs cellules satellite autour du point sont analysées sans créer de fausse parcelle en base."
              : candidates.length === 0
                ? "Aucun contour agricole fiable n’est référencé dans ce rayon."
                : null,
    };
  } catch (error) {
    // On erreur (DB, Overpass, etc.) renvoyer un fallback lisible pour le frontend
    console.warn("detectAutomaticParcels: discovery/analysis failure:", error);
    const message = error instanceof Error ? error.message : "Service de découverte indisponible.";
    return {
      center: { lat, lng },
      radius_km: radiusKm,
      base_temperature: baseTemperature,
      threshold,
      period_days: periodDays,
      candidates_found: 0,
      analyzed_count: 0,
      parcels: [],
      notice: message.includes("Database") || message.toLowerCase().includes("parcelles") || message.toLowerCase().includes("database")
        ? "La base des parcelles agricoles est indisponible. Essayez un rayon plus large ou vérifiez la configuration de la base de données."
        : message,
    };
  }
}

async function discoverAgriculturalParcels(lat: number, lng: number, radiusKm: number): Promise<CandidateParcel[]> {
  // Overpass et la base sont indépendants l'un de l'autre : les lancer en parallèle
  // (plutôt qu'en séquence) coupe le pire cas cumulé en deux, sans changer l'ordre de
  // priorité — Overpass reste préféré à la base si les deux ont trouvé quelque chose.
  const [candidates, databaseCandidates] = await Promise.all([
    discoverAgriculturalParcelsFromOverpass(lat, lng, radiusKm),
    discoverAgriculturalParcelsFromDatabase(lat, lng, radiusKm),
  ]);
  if (candidates.length > 0) return ensureCoordinateCoverage(candidates, lat, lng, radiusKm);
  if (databaseCandidates.length > 0) return ensureCoordinateCoverage(databaseCandidates, lat, lng, radiusKm);

  try {
    const fieldModelCandidates = await discoverAgriculturalParcelsFromFieldModel(lat, lng, radiusKm);
    if (fieldModelCandidates.length > 0) return ensureCoordinateCoverage(fieldModelCandidates, lat, lng, radiusKm);
  } catch (error) {
    console.warn("discoverAgriculturalParcelsFromFieldModel failed, trying next fallback:", error);
  }

  try {
    const watershedCandidates = await discoverAgriculturalParcelsFromWatershed(lat, lng, radiusKm);
    if (watershedCandidates.length > 0) return ensureCoordinateCoverage(watershedCandidates, lat, lng, radiusKm);
  } catch (error) {
    console.warn("discoverAgriculturalParcelsFromWatershed failed, trying next fallback:", error);
  }

  try {
    const geeSegmentedCandidates = await discoverAgriculturalParcelsFromGeeSegmentation(lat, lng, radiusKm);
    if (geeSegmentedCandidates.length > 0) return ensureCoordinateCoverage(geeSegmentedCandidates, lat, lng, radiusKm);
  } catch (error) {
    console.warn("discoverAgriculturalParcelsFromGeeSegmentation failed, trying next fallback:", error);
  }

  return createSatelliteSearchCellCandidates(lat, lng, radiusKm);
}

// Distance en dessous de laquelle deux candidats sont considérés comme la même parcelle
// (ex: doublons en base) plutôt que deux champs voisins distincts.
const DUPLICATE_CANDIDATE_DISTANCE_KM = 0.005; // 5 m

/** Ne garde qu'un seul candidat par groupe de centres quasi-identiques — évite de
 * gaspiller le budget d'analyse sur des doublons (ex: parcelles enregistrées plusieurs
 * fois en base). */
function deduplicateByCenter<T extends { center: { lat: number; lng: number } }>(candidates: T[]): T[] {
  const kept: T[] = [];
  for (const candidate of candidates) {
    const isDuplicate = kept.some((existing) => distanceBetweenPoints(existing.center, candidate.center) < DUPLICATE_CANDIDATE_DISTANCE_KM);
    if (!isDuplicate) kept.push(candidate);
  }
  return kept;
}

function ensureCoordinateCoverage(candidates: CandidateParcel[], lat: number, lng: number, radiusKm: number): CandidateParcel[] {
  const constrainedCandidates = deduplicateByCenter(candidates.flatMap((candidate) => {
    const coordinates = clipPolygonToRadius(candidate.coordinates, { lat, lng }, radiusKm);
    if (coordinates.length < 3) return [];
    return [{ ...candidate, coordinates, center: polygonCenter(coordinates) }];
  }));
  const satelliteCells = createSatelliteSearchCellCandidates(lat, lng, radiusKm)
    .filter((cell) => !constrainedCandidates.some((candidate) => pointInPolygon(cell.center, candidate.coordinates)));
  return [...constrainedCandidates, ...satelliteCells].slice(0, MAX_CANDIDATES);
}

function createSatelliteSearchCellCandidates(lat: number, lng: number, radiusKm: number): CandidateParcel[] {
  const center = { lat, lng };
  const cellSizeKm = Math.max(0.1, radiusKm / Math.sqrt(MAX_SATELLITE_CELLS / Math.PI));
  const halfCellKm = cellSizeKm / 2;
  const cellCount = Math.ceil(radiusKm / cellSizeKm);
  const cells: CandidateParcel[] = [];

  for (let row = -cellCount; row <= cellCount; row++) {
    for (let column = -cellCount; column <= cellCount; column++) {
      const cellCenter = offsetPoint(center, column * cellSizeKm, row * cellSizeKm);
      if (distanceBetweenPoints(center, cellCenter) > radiusKm + halfCellKm) continue;
      const cell = squareAround(cellCenter, halfCellKm);
      const coordinates = clipPolygonToRadius(cell, center, radiusKm);
      if (coordinates.length < 3) continue;
      cells.push({
        id: `satellite-search-cell-${lat.toFixed(6)}-${lng.toFixed(6)}-${row}-${column}`,
        coordinates,
        center: polygonCenter(coordinates),
        tags: { source: "satellite-search-cell" },
        persist: false,
      });
    }
  }

  return cells
    .sort((left, right) => distanceBetweenPoints(center, left.center) - distanceBetweenPoints(center, right.center))
    .slice(0, MAX_SATELLITE_CELLS);
}

// ── Segmentation GEE/SNIC (sans modèle ML) ──

/** Approxime un disque de rayon radiusKm autour de (lat,lng) par un polygone à N côtés. */
function circleRegionCoordinates(lat: number, lng: number, radiusKm: number, vertices = CIRCLE_REGION_VERTICES): number[][][] {
  const latRad = (lat * Math.PI) / 180;
  const metersPerDegLat = 111_320;
  const metersPerDegLng = 111_320 * Math.cos(latRad);
  const radiusMeters = radiusKm * 1000;
  const ring: number[][] = [];
  for (let i = 0; i <= vertices; i++) {
    const angle = (2 * Math.PI * i) / vertices;
    const dLat = (radiusMeters * Math.sin(angle)) / metersPerDegLat;
    const dLng = (radiusMeters * Math.cos(angle)) / metersPerDegLng;
    ring.push([lng + dLng, lat + dLat]);
  }
  return [ring];
}

/**
 * Construit l'expression GEE : composite Sentinel-2 (90 derniers jours, faible
 * nuage) -> indices spectraux (NDVI/NDWI/NDBI/BSI) -> masque végétation ->
 * segmentation SNIC -> vectorisation en polygones lat/lng, sur un disque de
 * rayon radiusKm autour de (lat,lng). Même logique que buildSNICVectorsExpression
 * dans analyze-parcel.ts, mais la région est un disque libre plutôt qu'un
 * contour de parcelle déjà connu.
 */
function buildRegionSnicExpression(lat: number, lng: number, radiusKm: number) {
  const endDate = new Date().toISOString().slice(0, 10);
  const startDate = new Date(Date.now() - 90 * 86_400_000).toISOString().slice(0, 10);
  const values: Record<string, GeeValue> = {};
  const reference = (name: string): GeeValue => ({ valueReference: name });

  values.region = geeCall("GeometryConstructors.Polygon", {
    coordinates: geeConstant(circleRegionCoordinates(lat, lng, radiusKm)),
  });
  values.intersectsRegion = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: reference("region") }),
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
    filter: reference("intersectsRegion"),
  });
  values.collectionByDate = geeCall("Collection.filter", {
    collection: reference("collectionByRegion"),
    filter: reference("dateRange"),
  });
  values.collection = geeCall("Collection.filter", {
    collection: reference("collectionByDate"),
    filter: reference("lowCloudCover"),
  });
  values.composite = geeCall("reduce.median", { collection: reference("collection") });
  values.spectralImage = geeCall("Image.select", {
    input: reference("composite"),
    bandSelectors: geeConstant(["B2", "B3", "B4", "B8", "B11", "B12"]),
  });
  values.withNdvi = addSpectralIndex(reference("spectralImage"), "NDVI", ["B8", "B4"]);
  values.withNdwi = addSpectralIndex(reference("withNdvi"), "NDWI", ["B3", "B8"]);
  values.withNdbi = addSpectralIndex(reference("withNdwi"), "NDBI", ["B11", "B8"]);
  values.segmentationImage = addBareSoilIndex(reference("withNdbi"));

  const imageBand = (name: string) => geeCall("Image.select", {
    input: reference("segmentationImage"),
    bandSelectors: geeConstant([name]),
  });
  // Masque "surface agricole plausible" : végétatif, non-eau, non-bâti, non-sol nu.
  values.ndviMask = geeCall("Image.gt", { image1: imageBand("NDVI"), image2: geeImageConstant(0.15) });
  values.waterMask = geeCall("Image.lt", { image1: imageBand("NDWI"), image2: geeImageConstant(0.1) });
  values.urbanMask = geeCall("Image.lt", { image1: imageBand("NDBI"), image2: geeImageConstant(0.05) });
  values.bareSoilMask = geeCall("Image.lt", { image1: imageBand("BSI"), image2: geeImageConstant(0.25) });
  values.fieldMask = geeCall("Image.and", {
    image1: geeCall("Image.and", {
      image1: geeCall("Image.and", { image1: reference("ndviMask"), image2: reference("waterMask") }),
      image2: reference("urbanMask"),
    }),
    image2: reference("bareSoilMask"),
  });
  values.maskedImage = geeCall("Image.updateMask", {
    image: reference("segmentationImage"),
    mask: reference("fieldMask"),
  });
  values.fieldImage = geeCall("Image.clip", {
    input: reference("maskedImage"),
    geometry: reference("region"),
  });
  values.snic = geeCall("Image.Segmentation.SNIC", {
    image: reference("fieldImage"),
    size: geeConstant(REGION_SNIC_PARAMETERS.size),
    compactness: geeConstant(REGION_SNIC_PARAMETERS.compactness),
    connectivity: geeConstant(REGION_SNIC_PARAMETERS.connectivity),
    neighborhoodSize: geeConstant(REGION_SNIC_PARAMETERS.size * 4),
  });
  values.snicClusters = geeCall("Image.select", {
    input: reference("snic"),
    bandSelectors: geeConstant(["clusters"]),
  });
  values.vectorsImage = geeCall("Image.addBands", {
    dstImg: reference("snicClusters"),
    srcImg: reference("fieldImage"),
  });
  values.vectors = geeCall("Image.reduceToVectors", {
    image: reference("vectorsImage"),
    reducer: geeCall("Reducer.mean", {}),
    geometry: reference("region"),
    scale: geeConstant(REGION_SNIC_PARAMETERS.scale),
    geometryType: geeConstant("polygon"),
    eightConnected: geeConstant(true),
    labelProperty: geeConstant("segment_id"),
    bestEffort: geeConstant(true),
    maxPixels: geeConstant(30_000_000),
    tileScale: geeConstant(4),
  });

  return { expression: { result: "vectors", values } };
}

async function discoverAgriculturalParcelsFromGeeSegmentation(
  lat: number,
  lng: number,
  radiusKm: number,
): Promise<CandidateParcel[]> {
  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) return [];
  if (radiusKm > MAX_SEGMENTATION_RADIUS_KM) return [];

  const accessToken = await getGeeAccessToken();
  const projectId = getGeeProjectId();
  const raw = await callGeeComputeRaw(accessToken, projectId, buildRegionSnicExpression(lat, lng, radiusKm));
  const result = raw.result;
  if (!isGeeFeatureCollection(result)) return [];

  return result.features.flatMap((feature, index): CandidateParcel[] => {
    if (!feature || typeof feature !== "object") return [];
    const { geometry, properties } = feature as { geometry?: unknown; properties?: unknown };
    const coordinates = extractLatLngFromGeometry(geometry);
    if (coordinates.length < 3) return [];

    const areaM2 = approximatePolygonAreaM2(coordinates);
    if (areaM2 < MIN_PARCEL_SEGMENT_AREA_M2 || areaM2 > MAX_PARCEL_SEGMENT_AREA_M2) return [];
    if (!isPlausibleFieldShape(coordinates, areaM2)) return [];

    const props = properties && typeof properties === "object" ? properties as Record<string, unknown> : {};
    const ndvi = typeof props.NDVI === "number" ? props.NDVI : undefined;

    return [{
      id: `gee-snic-${lat.toFixed(6)}-${lng.toFixed(6)}-${index}`,
      coordinates,
      center: polygonCenter(coordinates),
      tags: {
        source: "gee-snic-segmentation",
        ...(ndvi !== undefined ? { ndvi_hint: String(Math.round(ndvi * 1000) / 1000) } : {}),
      },
    }];
  });
}

async function discoverAgriculturalParcelsFromOverpass(lat: number, lng: number, radiusKm: number): Promise<CandidateParcel[]> {
  const query = `[out:json][timeout:20];(way["landuse"~"farmland|farm|orchard|vineyard|meadow"](around:${radiusKm * 1000},${lat},${lng});relation["landuse"~"farmland|farm|orchard|vineyard|meadow"](around:${radiusKm * 1000},${lat},${lng});way["crop"](around:${radiusKm * 1000},${lat},${lng});relation["crop"](around:${radiusKm * 1000},${lat},${lng}););out tags geom;`;
  for (const endpoint of [...new Set(OVERPASS_API_URLS)]) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          "User-Agent": OVERPASS_USER_AGENT,
        },
        body: new URLSearchParams({ data: query }),
        signal: AbortSignal.timeout(OVERPASS_REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        console.warn(`Overpass endpoint ${endpoint} returned ${response.status}: ${text}`);
        continue;
      }
      const payload: unknown = await response.json();
      const candidateElements = payload && typeof payload === "object" && "elements" in payload ? payload.elements : null;
      if (!Array.isArray(candidateElements)) continue;
      const parsed = candidateElements.flatMap((element): CandidateParcel[] => {
        if (!element || typeof element !== "object") return [];
        const item = element as OverpassElement;
        if (item.type !== "way" || typeof item.id !== "number" || !Array.isArray(item.geometry)) return [];
        const coordinates = item.geometry.flatMap((point) => {
          if (typeof point.lat !== "number" || typeof point.lon !== "number") return [];
          return [{ lat: point.lat, lng: point.lon }];
        });
        if (coordinates.length < 3) return [];
        const first = coordinates[0];
        const last = coordinates.at(-1);
        const closedCoordinates = last && first.lat === last.lat && first.lng === last.lng ? coordinates.slice(0, -1) : coordinates;
        if (closedCoordinates.length < 3) return [];
        const center = {
          lat: closedCoordinates.reduce((sum, point) => sum + point.lat, 0) / closedCoordinates.length,
          lng: closedCoordinates.reduce((sum, point) => sum + point.lng, 0) / closedCoordinates.length,
        };
        return [{ id: `osm-way-${item.id}`, coordinates: closedCoordinates, center, tags: { ...(item.tags ?? {}), source: "osm" } }];
      });
      const candidates = parsed.filter((candidate) => isParcelWithinRadius(candidate, { lat, lng }, radiusKm));
      if (candidates.length > 0) return candidates;
    } catch {
      continue;
    }
  }
  return [];
}

async function discoverAgriculturalParcelsFromDatabase(lat: number, lng: number, radiusKm: number): Promise<CandidateParcel[]> {
  const latDelta = radiusKm / 110.574;
  const longitudeScale = Math.max(Math.abs(Math.cos((lat * Math.PI) / 180)), 0.1);
  const lngDelta = radiusKm / (111.32 * longitudeScale);
  const minLat = lat - latDelta;
  const maxLat = lat + latDelta;
  const minLng = lng - lngDelta;
  const maxLng = lng + lngDelta;

  let rows;
  try {
    rows = await withTimeout(
      prisma.parcelle.findMany({
        where: {
          center_lat: { gte: minLat, lte: maxLat },
          center_lng: { gte: minLng, lte: maxLng },
        },
      }),
      DATABASE_DISCOVERY_TIMEOUT_MS,
      "discoverAgriculturalParcelsFromDatabase",
    );
  } catch (error) {
    console.error("Database access error for parcel discovery:", error);
    return [];
  }

  return rows.flatMap((row) => {
    const coordinates = Array.isArray(row.coordinates)
      ? row.coordinates.flatMap((point) => {
          if (!point || typeof point !== "object") return [];
          const values = point as Record<string, unknown>;
          const latValue = values.lat;
          const lngValue = values.lng;
          if (typeof latValue !== "number" || typeof lngValue !== "number") return [];
          return [{ lat: latValue, lng: lngValue }];
        })
      : [];
    if (coordinates.length < 3) return [];
    const center = {
      lat: coordinates.reduce((sum, point) => sum + point.lat, 0) / coordinates.length,
      lng: coordinates.reduce((sum, point) => sum + point.lng, 0) / coordinates.length,
    };
    const candidate = { id: row.id, coordinates, center, tags: { source: "database" } };
    return isParcelWithinRadius(candidate, { lat, lng }, radiusKm) ? [candidate] : [];
  });
}

export interface GrowingDegreeDaySummary {
  cumulative: number;
  detected: boolean;
  startDate: string;
  endDate: string;
  dailyValues: Array<{ date: string; tmax: number; tmin: number; dj: number }>;
}

function pointInPolygon(point: { lat: number; lng: number }, polygon: Array<{ lat: number; lng: number }>): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].lng;
    const yi = polygon[i].lat;
    const xj = polygon[j].lng;
    const yj = polygon[j].lat;
    const intersect = (yi > point.lat) !== (yj > point.lat)
      && point.lng < ((xj - xi) * (point.lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function distanceToSegment(point: { x: number; y: number }, start: { x: number; y: number }, end: { x: number; y: number }): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(point.x - start.x, point.y - start.y);
  const ratio = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  return Math.hypot(point.x - (start.x + ratio * dx), point.y - (start.y + ratio * dy));
}

function polygonCenter(points: Array<{ lat: number; lng: number }>): { lat: number; lng: number } {
  return {
    lat: points.reduce((sum, point) => sum + point.lat, 0) / points.length,
    lng: points.reduce((sum, point) => sum + point.lng, 0) / points.length,
  };
}

function polygonPerimeterM(coords: Array<{ lat: number; lng: number }>): number {
  const centroid = polygonCenter(coords);
  const latFactor = 111_320;
  const lngFactor = 111_320 * Math.cos((centroid.lat * Math.PI) / 180);
  const points = coords.map((point) => ({
    x: (point.lng - centroid.lng) * lngFactor,
    y: (point.lat - centroid.lat) * latFactor,
  }));
  let perimeter = 0;
  for (let i = 0; i < points.length; i++) {
    const next = (i + 1) % points.length;
    perimeter += Math.hypot(points[next].x - points[i].x, points[next].y - points[i].y);
  }
  return perimeter;
}

// Indice de compacité de Polsby-Popper (4π·aire/périmètre²) : 1 = cercle parfait, proche de 0 =
// forme très découpée/allongée. Watershed et SNIC peuvent sous-segmenter (plusieurs champs, routes
// et rivières fusionnés en un seul "bassin") quand le gradient NDVI interne est trop faible pour
// bien séparer les parcelles réelles ; ce bassin fusionné a un contour très irrégulier (suit les
// routes/rivières) et donc une compacité bien plus basse qu'un vrai champ, même de forme imparfaite.
// Seuil de départ empirique (pas une vérité agronomique figée), à affiner avec des cas réels.
const MIN_FIELD_COMPACTNESS = 0.12;

function isPlausibleFieldShape(coords: Array<{ lat: number; lng: number }>, areaM2: number): boolean {
  const perimeterM = polygonPerimeterM(coords);
  if (perimeterM <= 0) return false;
  const compactness = (4 * Math.PI * areaM2) / (perimeterM * perimeterM);
  return compactness >= MIN_FIELD_COMPACTNESS;
}

function offsetPoint(center: { lat: number; lng: number }, eastKm: number, northKm: number): { lat: number; lng: number } {
  const longitudeScale = Math.max(Math.abs(Math.cos((center.lat * Math.PI) / 180)), 0.1);
  return {
    lat: center.lat + northKm / 110.574,
    lng: center.lng + eastKm / (111.32 * longitudeScale),
  };
}

function distanceBetweenPoints(left: { lat: number; lng: number }, right: { lat: number; lng: number }): number {
  const longitudeScale = Math.max(Math.abs(Math.cos((left.lat * Math.PI) / 180)), 0.1);
  const eastKm = (right.lng - left.lng) * 111.32 * longitudeScale;
  const northKm = (right.lat - left.lat) * 110.574;
  return Math.hypot(eastKm, northKm);
}

function squareAround(center: { lat: number; lng: number }, halfSideKm: number): Array<{ lat: number; lng: number }> {
  return [
    offsetPoint(center, -halfSideKm, -halfSideKm),
    offsetPoint(center, halfSideKm, -halfSideKm),
    offsetPoint(center, halfSideKm, halfSideKm),
    offsetPoint(center, -halfSideKm, halfSideKm),
  ];
}

function radiusPolygon(center: { lat: number; lng: number }, radiusKm: number): Array<{ lat: number; lng: number }> {
  const radiusM = radiusKm * 1_000;
  const longitudeScale = Math.max(Math.abs(Math.cos((center.lat * Math.PI) / 180)), 0.1);
  return Array.from({ length: 48 }, (_, index) => {
    const angle = (index / 48) * Math.PI * 2;
    return {
      lat: center.lat + (Math.sin(angle) * radiusM) / 110_574,
      lng: center.lng + (Math.cos(angle) * radiusM) / (111_320 * longitudeScale),
    };
  });
}

type LocalPoint = { x: number; y: number };

function clipPolygonToRadius(
  polygon: Array<{ lat: number; lng: number }>,
  center: { lat: number; lng: number },
  radiusKm: number,
): Array<{ lat: number; lng: number }> {
  const longitudeScale = Math.max(Math.abs(Math.cos((center.lat * Math.PI) / 180)), 0.1);
  const toLocal = (point: { lat: number; lng: number }): LocalPoint => ({
    x: (point.lng - center.lng) * 111_320 * longitudeScale,
    y: (point.lat - center.lat) * 110_574,
  });
  const fromLocal = (point: LocalPoint) => ({
    lat: center.lat + point.y / 110_574,
    lng: center.lng + point.x / (111_320 * longitudeScale),
  });
  const points = polygon.slice();
  const first = points[0];
  const last = points.at(-1);
  if (first && last && first.lat === last.lat && first.lng === last.lng) points.pop();
  let clipped = points.map(toLocal);
  const boundary = radiusPolygon(center, radiusKm).map(toLocal);

  for (let index = 0; index < boundary.length && clipped.length > 0; index++) {
    const start = boundary[index];
    const end = boundary[(index + 1) % boundary.length];
    const input = clipped;
    clipped = [];
    for (let pointIndex = 0; pointIndex < input.length; pointIndex++) {
      const previous = input[(pointIndex + input.length - 1) % input.length];
      const current = input[pointIndex];
      const previousInside = crossProduct(start, end, previous) >= 0;
      const currentInside = crossProduct(start, end, current) >= 0;
      if (currentInside !== previousInside) clipped.push(lineIntersection(previous, current, start, end));
      if (currentInside) clipped.push(current);
    }
  }

  return clipped.map(fromLocal);
}

function crossProduct(start: LocalPoint, end: LocalPoint, point: LocalPoint): number {
  return (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x);
}

function lineIntersection(start: LocalPoint, end: LocalPoint, boundaryStart: LocalPoint, boundaryEnd: LocalPoint): LocalPoint {
  const direction = { x: end.x - start.x, y: end.y - start.y };
  const boundaryDirection = { x: boundaryEnd.x - boundaryStart.x, y: boundaryEnd.y - boundaryStart.y };
  const denominator = direction.x * boundaryDirection.y - direction.y * boundaryDirection.x;
  if (denominator === 0) return end;
  const offset = { x: boundaryStart.x - start.x, y: boundaryStart.y - start.y };
  const ratio = (offset.x * boundaryDirection.y - offset.y * boundaryDirection.x) / denominator;
  return { x: start.x + ratio * direction.x, y: start.y + ratio * direction.y };
}

function isParcelWithinRadius(candidate: CandidateParcel, center: { lat: number; lng: number }, radiusKm: number): boolean {
  if (pointInPolygon(center, candidate.coordinates)) return true;
  const longitudeScale = Math.cos((center.lat * Math.PI) / 180);
  const toKilometers = (point: { lat: number; lng: number }) => ({
    x: (point.lng - center.lng) * 111.32 * longitudeScale,
    y: (point.lat - center.lat) * 110.574,
  });
  const origin = { x: 0, y: 0 };
  const points = candidate.coordinates.map(toKilometers);
  return points.some((point) => Math.hypot(point.x, point.y) <= radiusKm)
    || points.some((point, index) => distanceToSegment(origin, point, points[(index + 1) % points.length]) <= radiusKm);
}

async function analyzeCandidate(candidate: CandidateParcel, config: BarleyDetectionConfig): Promise<AutoDetectedParcel> {
  let response: Response;
  try {
    response = await analyzeParcel(new Request("http://backend/auto-analyze-parcel", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        lat: candidate.center.lat,
        lng: candidate.center.lng,
        zoom: 15,
        polygon: candidate.coordinates,
      }),
    }));
  } catch (error) {
    return {
      ...candidate,
      analysis: null,
      analysis_error: error instanceof Error ? error.message : "Analyse satellite indisponible.",
    };
  }

  const body = await response.text();
  let data: unknown = null;
  try {
    data = JSON.parse(body);
  } catch {
    data = null;
  }
  if (!response.ok || !data || typeof data !== "object") {
    return { ...candidate, analysis: null, analysis_error: "Analyse satellite indisponible." };
  }

  // Le cumul de degrés-jours ne fait que confirmer une détection déjà positive.
  // Une panne/limite de débit ponctuelle sur Open-Meteo ne doit pas faire disparaître
  // un résultat satellite/HF valide (la parcelle reste "probable" au lieu d'être perdue).
  let gdd: GrowingDegreeDaySummary | null = null;
  let gddError: string | null = null;
  try {
    gdd = await fetchGrowingDegreeDays(candidate.center.lat, candidate.center.lng, config);
  } catch (error) {
    gddError = error instanceof Error ? error.message : "Données degrés-jours indisponibles.";
  }

  const modelAnalysis = data as Record<string, unknown>;
  const modelDetectsBarley = modelAnalysis.is_barley === true;
  const barleyPresence = modelDetectsBarley
    ? gdd?.detected === true ? "confirmed" : "probable"
    : "none";
  const analysis = {
    ...modelAnalysis,
    barley_presence: barleyPresence,
    gdd_cumulative: gdd?.cumulative,
    gdd_threshold: config.threshold,
    gdd_base_temperature: config.baseTemperature,
    gdd_start_date: gdd?.startDate,
    gdd_end_date: gdd?.endDate,
    gdd_detected: gdd?.detected,
    gdd_daily_values: gdd?.dailyValues,
    gdd_error: gddError,
  };

  if (candidate.persist !== false) {
    try {
      await saveAutomaticAnalysis(candidate, analysis);
    } catch (error) {
      console.warn(`Automatic analysis ${candidate.id} was not persisted:`, error);
    }
  }

  return { ...candidate, analysis, analysis_error: null };
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

async function saveAutomaticAnalysis(candidate: CandidateParcel, analysis: Record<string, unknown>): Promise<void> {
  const gddSummary = asNumber(analysis.gdd_cumulative) != null && asNumber(analysis.gdd_threshold) != null
    ? `Degrés-jours : ${analysis.gdd_cumulative}/${analysis.gdd_threshold} °C`
    : "";
  const lastDay = Array.isArray(analysis.gdd_daily_values) ? analysis.gdd_daily_values.at(-1) : null;
  const temperatureSummary = lastDay && typeof lastDay === "object"
    ? `Dernier relevé : Tmax ${String((lastDay as Record<string, unknown>).tmax ?? "—")}°C · Tmin ${String((lastDay as Record<string, unknown>).tmin ?? "—")}°C`
    : "";
  const details = [asString(analysis.details), gddSummary, temperatureSummary].filter(Boolean).join(" · ") || null;
  const recommendations = [asString(analysis.recommendations), asString(analysis.details), gddSummary, temperatureSummary].filter(Boolean).join(" · ") || null;
  const data: Prisma.ParcelleUncheckedCreateInput = {
    label: candidate.id,
    coordinates: candidate.coordinates,
    center_lat: candidate.center.lat,
    center_lng: candidate.center.lng,
    surface_ha: null,
    culture_declared: asString(analysis.culture_declared),
    culture_detected: asString(analysis.culture_detected),
    ndvi_percentage: asNumber(analysis.percentage),
    confidence: asNumber(analysis.confidence),
    verdict: asString(analysis.verdict),
    details,
    saison: asString(analysis.saison),
    soil_type: asString(analysis.soil_type),
    risk_factors: asStringArray(analysis.risk_factors),
    recommendations,
    data_source: asString(analysis.data_source),
    owner_name: asString(candidate.tags.owner),
    notes: null,
    time_series_s1: Array.isArray(analysis.time_series_s1) ? analysis.time_series_s1 : [],
    time_series_s2: Array.isArray(analysis.time_series_s2) ? analysis.time_series_s2 : [],
    estimated_planting_date: asString(analysis.estimated_planting_date),
    estimated_harvest_date: asString(analysis.estimated_harvest_date),
    days_since_planting: Number.isInteger(analysis.days_since_planting) ? analysis.days_since_planting as number : null,
    growth_stage: asString(analysis.growth_stage),
    planting_confidence: asNumber(analysis.planting_confidence),
    evi: asNumber(analysis.evi),
    savi: asNumber(analysis.savi),
    ndwi: asNumber(analysis.ndwi),
    agro_score: asNumber(analysis.agro_score),
    hybrid_score: asNumber(analysis.hybrid_score),
    cnn_prob_barley: asNumber(analysis.cnn_prob_barley),
    cnn_prob_non_barley: asNumber(analysis.cnn_prob_non_barley),
  };
  const existing = await prisma.parcelle.findFirst({ where: { label: candidate.id }, select: { id: true } });
  if (existing) {
    await prisma.parcelle.update({ where: { id: existing.id }, data });
  } else {
    await prisma.parcelle.create({ data });
  }
}

function formatDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export async function fetchGrowingDegreeDays(lat: number, lng: number, config: BarleyDetectionConfig): Promise<GrowingDegreeDaySummary> {
  const endDate = new Date();
  endDate.setUTCDate(endDate.getUTCDate() - 2);
  const startDate = new Date(endDate);
  startDate.setUTCDate(startDate.getUTCDate() - config.periodDays + 1);
  const url = new URL("https://archive-api.open-meteo.com/v1/archive");
  url.search = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lng),
    start_date: formatDate(startDate),
    end_date: formatDate(endDate),
    daily: "temperature_2m_max,temperature_2m_min",
    timezone: "auto",
  }).toString();

  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error("Les données Open-Meteo sont indisponibles.");
  const payload: unknown = await response.json();
  const daily = payload && typeof payload === "object" && "daily" in payload ? payload.daily : null;
  if (!daily || typeof daily !== "object") throw new Error("Réponse météo incomplète.");
  const values = daily as { time?: unknown; temperature_2m_max?: unknown; temperature_2m_min?: unknown };
  if (!Array.isArray(values.time) || !Array.isArray(values.temperature_2m_max) || !Array.isArray(values.temperature_2m_min)) {
    throw new Error("Températures journalières indisponibles.");
  }
  const dates = values.time as unknown[];
  const tmaxValues = values.temperature_2m_max as unknown[];
  const tminValues = values.temperature_2m_min as unknown[];

  let cumulative = 0;
  let validDays = 0;
  const dailyValues: Array<{ date: string; tmax: number; tmin: number; dj: number }> = [];
  dates.forEach((date, index) => {
    const tmax = tmaxValues[index];
    const tmin = tminValues[index];
    if (typeof date !== "string" || typeof tmax !== "number" || typeof tmin !== "number") return;
    const dj = Math.round(((tmax + tmin) / 2 - config.baseTemperature) * 10) / 10;
    dailyValues.push({ date, tmax, tmin, dj });
    cumulative += dj;
    validDays += 1;
  });
  if (validDays === 0) throw new Error("Aucune température valide n’a été reçue.");

  return {
    dailyValues,
    cumulative: Math.round(cumulative * 10) / 10,
    detected: cumulative >= config.threshold,
    startDate: String(dates[0] ?? ""),
    endDate: String(dates.at(-1) ?? ""),
  };
}

async function mapWithConcurrency<T, R>(items: T[], concurrency: number, mapper: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await mapper(items[index]);
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
}

/**
 * Variante de mapWithConcurrency bornée par un budget de temps global plutôt que par le
 * nombre d'éléments : une fois le délai dépassé, aucun nouvel élément n'est démarré (ceux
 * déjà en cours vont à leur terme). Les éléments non traités sont simplement absents de
 * la Map résultat — à l'appelant de leur donner un statut "non traité".
 */
async function mapWithConcurrencyDeadline<T, R>(
  items: T[],
  concurrency: number,
  deadlineMs: number,
  mapper: (item: T) => Promise<R>,
): Promise<{ results: Map<number, R>; timedOut: boolean }> {
  const deadlineAt = Date.now() + deadlineMs;
  const results = new Map<number, R>();
  let cursor = 0;
  let timedOut = false;

  async function worker() {
    while (cursor < items.length) {
      if (Date.now() >= deadlineAt) {
        timedOut = true;
        return;
      }
      const index = cursor++;
      results.set(index, await mapper(items[index]));
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return { results, timedOut };
}
