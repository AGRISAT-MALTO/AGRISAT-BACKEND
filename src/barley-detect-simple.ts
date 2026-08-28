import {
  addSpectralIndex,
  approximatePolygonAreaM2,
  callGeeComputeRaw,
  callHFModel,
  captureSentinel2ParcelImage,
  extractLatLngFromGeometry,
  geeCall,
  geeConstant,
  geeImageConstant,
  getGeeAccessToken,
  isGeeFeatureCollection,
  SNIC_PARAMETERS,
  type GeeValue,
  type LatLng,
} from "./analyze-parcel.js";
import {
  fetchGrowingDegreeDays,
  type BarleyDetectionConfig,
} from "./automatic-parcels.js";
import { Prisma } from "@prisma/client";
import { prisma } from "./db.js";

// ── Pipeline "Version A" : Sentinel-2 L2A + NDVI/NDRE + segmentation SNIC + CNN externe existant ──
// Voir /home/tiavina/.claude/plans/recursive-sleeping-goblet.md et
// /home/tiavina/.claude/plans/warm-rolling-riddle.md pour le contexte complet.
// Flux principal exposé côté UI : GPS+rayon -> GEE -> Sentinel-2 L2A -> sélection de la
// meilleure image + masque nuage/ombre (SCL) -> bandes/NDVI/NDRE -> segmentation SNIC des
// zones végétatives plausibles -> classification CNN (image Sentinel-2, pas Google Static
// Maps) -> GeoJSON des parcelles ORGE. La "Version B" (automatic-parcels.ts, cascade
// Overpass/DB/watershed/SNIC/Google Static Maps) reste inchangée pour comparaison.

const MAX_IMAGE_AGE_DAYS_PRIMARY = 5;
const MAX_IMAGE_AGE_DAYS_FALLBACK = 10;

// Tuiles carrées couvrant la zone demandée : image:computePixels/reduceToVectors de GEE plafonne
// à 5000 entités vectorisées par appel. 2.5 km de rayon est resté largement sous cette limite lors
// des tests empiriques contre l'API réelle (voir le plan) ; on garde une marge de sécurité.
const TILE_RADIUS_M = 2_500;
const MAX_TILES = 40;
const TILE_CONCURRENCY = 3;
const CLASSIFY_CONCURRENCY = 3;

const DEFAULT_MIN_AREA_HA = 0.05; // spec §9
const MAX_CANDIDATE_AREA_M2 = 800_000; // 80 ha, borne réutilisée du reste du code (automatic-parcels.ts)
const MAX_CANDIDATES_TO_CLASSIFY = 20;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.7; // spec §7

// Confirmation phénologique (degrés-jours) : mêmes valeurs par défaut que le frontend
// (frontend/src/lib/barley-detection.ts). Best-effort — ne fait que confirmer une détection
// CNN déjà positive (barleyPresence "confirmed" vs "probable"), jamais la rejeter : une panne
// Open-Meteo ne doit pas faire perdre une détection satellite valide.
const DEFAULT_GDD_CONFIG: BarleyDetectionConfig = { baseTemperature: 0, threshold: 2200, periodDays: 365 };

// Seuils de plausibilité spectrale (pré-filtre avant CNN, spec §8). Valeurs de départ à affiner
// avec des données de validation — pas une vérité agronomique figée.
const NDVI_MIN = 0.3;
const NDWI_MAX = 0.1;
const NDRE_MIN = 0.1;

export interface SimpleAnalysisInput {
  lat: number;
  lng: number;
  radiusM: number;
  confidenceThreshold?: number;
  minAreaHa?: number;
  gddConfig?: BarleyDetectionConfig;
}

export interface SimpleFieldFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: {
    class: "ORGE";
    confidence: number;
    areaHa: number;
    meanNDVI: number | null;
    meanNDRE: number | null;
    imageDate: string | null;
    imageAgeDays: number | null;
    cloudPercentage: number | null;
    /** "confirmed" si le cumul de degrés-jours a atteint le seuil (spec agronomique), "probable" sinon (CNN seul). */
    barleyPresence: "confirmed" | "probable";
  };
}

export interface SimpleAnalysisResult {
  type: "FeatureCollection";
  features: SimpleFieldFeature[];
  center: { lat: number; lng: number };
  radiusM: number;
  imageDate: string | null;
  imageAgeDays: number | null;
  cloudPercentage: number | null;
  /** Timestamp exact (ms) de l'image Sentinel-2 analysée — sert de clé à /api/sentinel-tiles pour afficher le même fond de carte. */
  imageTimestampMs: number | null;
  confidenceThreshold: number;
  minAreaHa: number;
  candidatesFound: number;
  candidatesClassified: number;
  gddCumulative: number | null;
  gddThreshold: number;
  warnings: string[];
}

interface SelectedImageWindow {
  startDate: string;
  endDate: string;
  imageDate: string | null;
  imageAgeDays: number | null;
  cloudPercentage: number | null;
  imageTimestampMs: number | null;
}

export async function analyzeFieldsSimple(input: SimpleAnalysisInput): Promise<SimpleAnalysisResult> {
  const { lat, lng, radiusM } = input;
  const confidenceThreshold = input.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD;
  const minAreaHa = input.minAreaHa ?? DEFAULT_MIN_AREA_HA;
  const gddConfig = input.gddConfig ?? DEFAULT_GDD_CONFIG;
  const warnings: string[] = [];

  // Démarré tôt et en parallèle du reste (GEE) : indépendant de la fenêtre d'image, ne sert
  // qu'à confirmer une détection CNN déjà positive, jamais à la rejeter (cf. automatic-parcels.ts).
  const gddPromise = fetchGrowingDegreeDays(lat, lng, gddConfig).catch((error) => {
    warnings.push(`Données degrés-jours indisponibles : ${getErrorMessage(error)}`);
    return null;
  });

  const empty = (): SimpleAnalysisResult => ({
    type: "FeatureCollection", features: [],
    center: { lat, lng }, radiusM,
    imageDate: null, imageAgeDays: null, cloudPercentage: null, imageTimestampMs: null,
    confidenceThreshold, minAreaHa,
    candidatesFound: 0, candidatesClassified: 0,
    gddCumulative: null, gddThreshold: gddConfig.threshold,
    warnings,
  });

  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) {
    warnings.push("GEE_SERVICE_ACCOUNT_KEY n'est pas configurée.");
    return empty();
  }

  let projectId = "earthengine-legacy";
  let accessToken: string;
  try {
    const serviceAccount = JSON.parse(serviceAccountJson) as { project_id?: unknown };
    if (typeof serviceAccount.project_id === "string" && serviceAccount.project_id.length > 0) {
      projectId = serviceAccount.project_id;
    }
    accessToken = await getGeeAccessToken();
  } catch (error) {
    warnings.push(`GEE indisponible : ${getErrorMessage(error)}`);
    return empty();
  }

  const window = await selectBestImageWindow(accessToken, projectId, lat, lng, radiusM);
  if (!window) {
    warnings.push(`Aucune image Sentinel-2 exploitable dans les ${MAX_IMAGE_AGE_DAYS_FALLBACK} derniers jours.`);
    return empty();
  }

  const tileCenters = buildTileCenters(lat, lng, radiusM, TILE_RADIUS_M).slice(0, MAX_TILES);
  const candidateLists = await mapWithConcurrency(tileCenters, TILE_CONCURRENCY, (tile) =>
    fetchCandidatePolygons(accessToken, projectId, tile.lat, tile.lng, TILE_RADIUS_M, window, minAreaHa, warnings));

  const allCandidates = candidateLists.flat();
  const candidates = dedupeAndRankCandidates(allCandidates).slice(0, MAX_CANDIDATES_TO_CLASSIFY);
  const gdd = await gddPromise;

  const classifiedFeatures = await mapWithConcurrency(candidates, CLASSIFY_CONCURRENCY, async (candidate): Promise<SimpleFieldFeature | null> => {
    try {
      const center = polygonCentroid(candidate.coordinates);
      const thumbnail = await captureSentinel2ParcelImage(accessToken, projectId, center.lat, center.lng, 17, window.startDate, window.endDate);
      const classification = await callHFModel(thumbnail);
      const confidenceFraction = classification.confidence / 100;
      if (!classification.is_barley || confidenceFraction < confidenceThreshold) return null;
      return {
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [candidate.coordinates.map((p) => [p.lng, p.lat])] },
        properties: {
          class: "ORGE",
          confidence: Math.round(confidenceFraction * 1000) / 1000,
          areaHa: Math.round((candidate.areaM2 / 10_000) * 100) / 100,
          meanNDVI: candidate.ndvi,
          meanNDRE: candidate.ndre,
          imageDate: window.imageDate,
          imageAgeDays: window.imageAgeDays,
          cloudPercentage: window.cloudPercentage,
          barleyPresence: gdd?.detected === true ? "confirmed" : "probable",
        },
      };
    } catch (error) {
      warnings.push(`Classification d'un candidat échouée : ${getErrorMessage(error)}`);
      return null;
    }
  });
  const features = classifiedFeatures.filter((feature): feature is SimpleFieldFeature => feature !== null);

  return {
    type: "FeatureCollection", features,
    center: { lat, lng }, radiusM,
    imageDate: window.imageDate, imageAgeDays: window.imageAgeDays, cloudPercentage: window.cloudPercentage,
    imageTimestampMs: window.imageTimestampMs,
    gddCumulative: gdd?.cumulative ?? null, gddThreshold: gddConfig.threshold,
    confidenceThreshold, minAreaHa,
    candidatesFound: allCandidates.length, candidatesClassified: candidates.length,
    warnings,
  };
}

// ── Persistance dans la table `parcelles` (registre/dashboard) ──
// La pipeline simple ne sauvegardait jusqu'ici que le blob complet du run (FieldScanRun,
// pour /api/analyze/latest), jamais de ligne individuelle par parcelle détectée — contrairement
// à la Version B (`saveAutomaticAnalysis`, automatic-parcels.ts). Sans ça, l'orge détectée par
// Sentinel-2 s'affiche sur la carte au retour de la requête mais disparaît au rechargement et
// n'apparaît jamais dans le registre/tableau de bord. Même pattern upsert-par-label que la
// Version B : label dérivé du centroïde du polygone, stable d'une exécution à l'autre pour un
// même champ réel.
export async function saveSimpleFieldParcelles(features: SimpleFieldFeature[], warnings: string[]): Promise<void> {
  for (const feature of features) {
    try {
      await saveSimpleFieldParcelle(feature);
    } catch (error) {
      console.warn("saveSimpleFieldParcelles: échec de la persistance d'un candidat :", error);
      warnings.push(`Une parcelle détectée n'a pas pu être enregistrée dans le registre : ${getErrorMessage(error)}`);
    }
  }
}

async function saveSimpleFieldParcelle(feature: SimpleFieldFeature): Promise<void> {
  const ring = feature.geometry.coordinates[0];
  const coordinates: LatLng[] = ring.map(([lng, lat]) => ({ lat, lng }));
  const center = polygonCentroid(coordinates);
  const label = `simple-v1-${center.lat.toFixed(5)}-${center.lng.toFixed(5)}`;
  const { confidence, areaHa, meanNDVI, meanNDRE, imageDate, imageAgeDays, cloudPercentage, barleyPresence } = feature.properties;
  const confidencePercent = Math.round(confidence * 1000) / 10;
  const presenceLabel = barleyPresence === "confirmed" ? "confirmée par degrés-jours" : "probable (CNN seul)";

  const data: Prisma.ParcelleUncheckedCreateInput = {
    label,
    coordinates,
    center_lat: center.lat,
    center_lng: center.lng,
    surface_ha: areaHa,
    culture_declared: null,
    culture_detected: "Orge",
    ndvi_percentage: meanNDVI != null ? Math.round(meanNDVI * 1000) / 10 : null,
    ndre: meanNDRE,
    confidence: confidencePercent,
    verdict: `Orge détectée (Sentinel-2, SNIC) — confiance ${confidencePercent}% · présence ${presenceLabel}`,
    details: `Image Sentinel-2 du ${imageDate ?? "—"} (${imageAgeDays ?? "?"} j) · nuages ${cloudPercentage ?? "?"}%`,
    saison: null,
    soil_type: null,
    risk_factors: [],
    recommendations: null,
    data_source: "Sentinel-2 simple (v1, segmentation SNIC)",
    owner_name: null,
    notes: null,
    time_series_s1: [],
    time_series_s2: [],
    estimated_planting_date: null,
    estimated_harvest_date: null,
    days_since_planting: null,
    growth_stage: null,
    planting_confidence: null,
    evi: null,
    savi: null,
    ndwi: null,
    agro_score: null,
    hybrid_score: null,
    cnn_prob_barley: null,
    cnn_prob_non_barley: null,
  };

  const existing = await prisma.parcelle.findFirst({ where: { label }, select: { id: true } });
  if (existing) {
    await prisma.parcelle.update({ where: { id: existing.id }, data });
  } else {
    await prisma.parcelle.create({ data });
  }
}

// ── Sélection de la meilleure image S2 L2A (spec §1) ──
// Fenêtre primaire de 5 jours triée par couverture nuageuse croissante ; repli à 10 jours
// seulement si la fenêtre primaire ne contient aucune image. Trier par cloud% (pas par date)
// à l'intérieur de chaque fenêtre reproduit exactement les exemples de la spec.

async function selectBestImageWindow(
  accessToken: string, projectId: string, lat: number, lng: number, radiusM: number,
): Promise<SelectedImageWindow | null> {
  const now = new Date();
  const end = now.toISOString().slice(0, 10);
  const primaryStart = new Date(now.getTime() - MAX_IMAGE_AGE_DAYS_PRIMARY * 86400000).toISOString().slice(0, 10);
  const fallbackStart = new Date(now.getTime() - MAX_IMAGE_AGE_DAYS_FALLBACK * 86400000).toISOString().slice(0, 10);

  const primary = await fetchImageMeta(accessToken, projectId, lat, lng, radiusM, primaryStart, end);
  const chosen = primary && primary.size > 0
    ? primary
    : await fetchImageMeta(accessToken, projectId, lat, lng, radiusM, fallbackStart, end);
  if (!chosen || chosen.size === 0 || chosen.imageDateMillis == null) return null;

  const imageDate = new Date(chosen.imageDateMillis);
  const imageAgeDays = Math.max(0, Math.round((now.getTime() - chosen.imageDateMillis) / 86400000));
  return {
    startDate: chosen === primary ? primaryStart : fallbackStart,
    endDate: end,
    imageDate: imageDate.toISOString().slice(0, 10),
    imageAgeDays,
    cloudPercentage: chosen.cloudPercentage,
    imageTimestampMs: chosen.imageDateMillis,
  };
}

interface ImageMeta { size: number; imageDateMillis: number | null; cloudPercentage: number | null }

async function fetchImageMeta(
  accessToken: string, projectId: string, lat: number, lng: number, radiusM: number, start: string, end: string,
): Promise<ImageMeta | null> {
  const values: Record<string, GeeValue> = {};
  const ref = (name: string): GeeValue => ({ valueReference: name });

  values.point = geeCall("GeometryConstructors.Point", { coordinates: geeConstant([lng, lat]) });
  values.region = geeCall("Geometry.buffer", { geometry: ref("point"), distance: geeConstant(radiusM) });
  values.intersects = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: ref("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", { start: geeConstant(start), end: geeConstant(end) }),
    rightField: geeConstant("system:time_start"),
  });
  values.raw = geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") });
  values.byRegion = geeCall("Collection.filter", { collection: ref("raw"), filter: ref("intersects") });
  values.byDate = geeCall("Collection.filter", { collection: ref("byRegion"), filter: ref("dateRange") });
  values.sorted = geeCall("Collection.limit", {
    collection: ref("byDate"), limit: geeConstant(1),
    key: geeConstant("CLOUDY_PIXEL_PERCENTAGE"), ascending: geeConstant(true),
  });
  values.size = geeCall("Collection.size", { collection: ref("sorted") });
  values.first = geeCall("Collection.first", { collection: ref("sorted") });
  values.date = geeCall("Element.get", { object: ref("first"), property: geeConstant("system:time_start") });
  values.cloud = geeCall("Element.get", { object: ref("first"), property: geeConstant("CLOUDY_PIXEL_PERCENTAGE") });
  values.dict = geeCall("Dictionary.set", {
    dictionary: geeCall("Dictionary.set", {
      dictionary: geeCall("Dictionary.set", { dictionary: geeCall("Dictionary", {}), key: geeConstant("size"), value: ref("size") }),
      key: geeConstant("date"), value: ref("date"),
    }),
    key: geeConstant("cloud"), value: ref("cloud"),
  });

  try {
    const raw = await callGeeComputeRaw(accessToken, projectId, { expression: { result: "dict", values } });
    const result = (raw as { result?: unknown }).result;
    if (!result || typeof result !== "object") return null;
    const values_ = result as { size?: unknown; date?: unknown; cloud?: unknown };
    const size = typeof values_.size === "number" ? values_.size : 0;
    return {
      size,
      imageDateMillis: typeof values_.date === "number" ? values_.date : null,
      cloudPercentage: typeof values_.cloud === "number" ? Math.round(values_.cloud * 10) / 10 : null,
    };
  } catch {
    return null;
  }
}

// ── Masque nuage/ombre (SCL) + bandes + NDVI/NDRE + masque de plausibilité + segmentation SNIC ──
// (spec §2, §3, §4, §5, §9)

interface PolygonCandidate {
  coordinates: LatLng[];
  areaM2: number;
  ndvi: number | null;
  ndre: number | null;
}

async function fetchCandidatePolygons(
  accessToken: string, projectId: string, lat: number, lng: number, radiusM: number,
  window: SelectedImageWindow, minAreaHa: number, warnings: string[],
): Promise<PolygonCandidate[]> {
  const values: Record<string, GeeValue> = {};
  const ref = (name: string): GeeValue => ({ valueReference: name });

  values.point = geeCall("GeometryConstructors.Point", { coordinates: geeConstant([lng, lat]) });
  values.region = geeCall("Geometry.buffer", { geometry: ref("point"), distance: geeConstant(radiusM) });
  values.intersects = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: ref("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", { start: geeConstant(window.startDate), end: geeConstant(window.endDate) }),
    rightField: geeConstant("system:time_start"),
  });
  values.raw = geeCall("ImageCollection.load", { id: geeConstant("COPERNICUS/S2_SR_HARMONIZED") });
  values.byRegion = geeCall("Collection.filter", { collection: ref("raw"), filter: ref("intersects") });
  values.byDate = geeCall("Collection.filter", { collection: ref("byRegion"), filter: ref("dateRange") });
  values.sorted = geeCall("Collection.limit", {
    collection: ref("byDate"), limit: geeConstant(1),
    key: geeConstant("CLOUDY_PIXEL_PERCENTAGE"), ascending: geeConstant(true),
  });
  values.image = geeCall("Collection.first", { collection: ref("sorted") });

  // Masque nuage/ombre pixel via SCL : exclut ombre (3), nuage moyen/haut (8/9), cirrus fin (10).
  // Appliqué AVANT tout calcul de bande/indice (spec §3).
  values.scl = geeCall("Image.select", { input: ref("image"), bandSelectors: geeConstant(["SCL"]) });
  values.isShadow = geeCall("Image.eq", { image1: ref("scl"), image2: geeImageConstant(3) });
  values.isCloudMed = geeCall("Image.eq", { image1: ref("scl"), image2: geeImageConstant(8) });
  values.isCloudHigh = geeCall("Image.eq", { image1: ref("scl"), image2: geeImageConstant(9) });
  values.isCirrus = geeCall("Image.eq", { image1: ref("scl"), image2: geeImageConstant(10) });
  values.isBad1 = geeCall("Image.or", { image1: ref("isShadow"), image2: ref("isCloudMed") });
  values.isBad2 = geeCall("Image.or", { image1: ref("isCloudHigh"), image2: ref("isCirrus") });
  values.isBad = geeCall("Image.or", { image1: ref("isBad1"), image2: ref("isBad2") });
  values.cloudMask = geeCall("Image.not", { value: ref("isBad") });

  values.bands = geeCall("Image.select", {
    input: ref("image"),
    bandSelectors: geeConstant(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]),
  });
  values.maskedBands = geeCall("Image.updateMask", { image: ref("bands"), mask: ref("cloudMask") });
  values.withNdvi = addSpectralIndex(ref("maskedBands"), "NDVI", ["B8", "B4"]);
  values.withNdre = addSpectralIndex(ref("withNdvi"), "NDRE", ["B8A", "B5"]);
  values.withNdwi = addSpectralIndex(ref("withNdre"), "NDWI", ["B3", "B8"]);

  // Masque de plausibilité spectrale (spec §8/§9) : NDVI/NDRE dans une plage plausible, pas d'eau.
  values.ndviBand = geeCall("Image.select", { input: ref("withNdwi"), bandSelectors: geeConstant(["NDVI"]) });
  values.ndreBand = geeCall("Image.select", { input: ref("withNdwi"), bandSelectors: geeConstant(["NDRE"]) });
  values.ndwiBand = geeCall("Image.select", { input: ref("withNdwi"), bandSelectors: geeConstant(["NDWI"]) });
  values.ndviOk = geeCall("Image.gt", { image1: ref("ndviBand"), image2: geeImageConstant(NDVI_MIN) });
  values.ndreOk = geeCall("Image.gt", { image1: ref("ndreBand"), image2: geeImageConstant(NDRE_MIN) });
  values.ndwiOk = geeCall("Image.lt", { image1: ref("ndwiBand"), image2: geeImageConstant(NDWI_MAX) });
  values.plausible1 = geeCall("Image.and", { image1: ref("ndviOk"), image2: ref("ndreOk") });
  values.plausibleMask = geeCall("Image.and", { image1: ref("plausible1"), image2: ref("ndwiOk") });

  // Segmentation SNIC (spec §9, « modèle de segmentation ») sur les zones plausibles : même
  // algorithme et mêmes paramètres que buildSNICVectorsExpression (analyze-parcel.ts) /
  // buildRegionSnicExpression (automatic-parcels.ts), appliqué ici à l'image mono-date
  // masquée nuage/ombre (SCL) + NDRE de cette pipeline, plutôt qu'à un composite médian.
  values.maskedForSegmentation = geeCall("Image.updateMask", { image: ref("withNdwi"), mask: ref("plausibleMask") });
  values.vegetationImage = geeCall("Image.clip", { input: ref("maskedForSegmentation"), geometry: ref("region") });
  values.snic = geeCall("Image.Segmentation.SNIC", {
    image: ref("vegetationImage"),
    size: geeConstant(SNIC_PARAMETERS.size),
    compactness: geeConstant(SNIC_PARAMETERS.compactness),
    connectivity: geeConstant(SNIC_PARAMETERS.connectivity),
    neighborhoodSize: geeConstant(SNIC_PARAMETERS.neighborhoodSize),
  });
  values.snicClusters = geeCall("Image.select", { input: ref("snic"), bandSelectors: geeConstant(["clusters"]) });
  values.vectorsImage = geeCall("Image.addBands", { dstImg: ref("snicClusters"), srcImg: ref("vegetationImage") });
  values.vectors = geeCall("Image.reduceToVectors", {
    image: ref("vectorsImage"),
    reducer: geeCall("Reducer.mean", {}),
    geometry: ref("region"),
    scale: geeConstant(SNIC_PARAMETERS.scale),
    geometryType: geeConstant("polygon"),
    eightConnected: geeConstant(true),
    labelProperty: geeConstant("segment_id"),
    bestEffort: geeConstant(true),
    maxPixels: geeConstant(20_000_000),
    tileScale: geeConstant(4),
  });

  try {
    const raw = await callGeeComputeRaw(accessToken, projectId, { expression: { result: "vectors", values } });
    const result = (raw as { result?: unknown }).result;
    if (!isGeeFeatureCollection(result)) return [];
    return result.features.flatMap((feature) => toPolygonCandidate(feature, minAreaHa));
  } catch (error) {
    warnings.push(`Tuile Sentinel-2 (${lat.toFixed(4)},${lng.toFixed(4)}) indisponible : ${getErrorMessage(error)}`);
    return [];
  }
}

function toPolygonCandidate(feature: unknown, minAreaHa: number): PolygonCandidate[] {
  if (!feature || typeof feature !== "object") return [];
  const value = feature as { geometry?: unknown; properties?: unknown };
  const coordinates = extractLatLngFromGeometry(value.geometry);
  if (coordinates.length < 3) return [];

  const areaM2 = approximatePolygonAreaM2(coordinates);
  if (areaM2 < minAreaHa * 10_000 || areaM2 > MAX_CANDIDATE_AREA_M2) return [];

  const properties = value.properties && typeof value.properties === "object"
    ? value.properties as Record<string, unknown>
    : {};
  const ndvi = typeof properties.NDVI === "number" ? properties.NDVI : null;
  const ndre = typeof properties.NDRE === "number" ? properties.NDRE : null;

  return [{ coordinates, areaM2, ndvi, ndre }];
}

function dedupeAndRankCandidates(candidates: PolygonCandidate[]): PolygonCandidate[] {
  // Les tuiles voisines se chevauchent légèrement (marge anti-lacune) : un même champ proche d'une
  // frontière de tuile peut apparaître deux fois avec un centroïde très proche.
  const seen: LatLng[] = [];
  const deduped: PolygonCandidate[] = [];
  for (const candidate of candidates) {
    const center = polygonCentroid(candidate.coordinates);
    const isDuplicate = seen.some((point) => haversineMeters(point, center) < 15);
    if (isDuplicate) continue;
    seen.push(center);
    deduped.push(candidate);
  }
  return deduped.sort((left, right) => {
    const ndviDiff = (right.ndvi ?? -1) - (left.ndvi ?? -1);
    return ndviDiff !== 0 ? ndviDiff : right.areaM2 - left.areaM2;
  });
}

// ── Tuilage de la zone (GPS + rayon) en cercles couvrant l'AOI ──

function buildTileCenters(lat: number, lng: number, radiusM: number, tileRadiusM: number): LatLng[] {
  if (radiusM <= tileRadiusM) return [{ lat, lng }];

  const metersPerDegLat = 111_320;
  const metersPerDegLng = 111_320 * Math.cos((lat * Math.PI) / 180);
  const step = tileRadiusM * 1.6; // léger recouvrement pour éviter les lacunes entre tuiles circulaires
  const steps = Math.ceil(radiusM / step);
  const centers: LatLng[] = [];

  for (let row = -steps; row <= steps; row++) {
    for (let col = -steps; col <= steps; col++) {
      const offsetM = { x: col * step, y: row * step };
      const distanceFromCenter = Math.hypot(offsetM.x, offsetM.y);
      if (distanceFromCenter > radiusM + tileRadiusM) continue;
      centers.push({
        lat: lat + offsetM.y / metersPerDegLat,
        lng: lng + offsetM.x / metersPerDegLng,
      });
    }
  }

  return centers.sort((left, right) => haversineMeters({ lat, lng }, left) - haversineMeters({ lat, lng }, right));
}

function haversineMeters(a: LatLng, b: LatLng): number {
  const metersPerDegLat = 111_320;
  const metersPerDegLng = 111_320 * Math.cos((a.lat * Math.PI) / 180);
  const dx = (b.lng - a.lng) * metersPerDegLng;
  const dy = (b.lat - a.lat) * metersPerDegLat;
  return Math.hypot(dx, dy);
}

function polygonCentroid(coords: LatLng[]): LatLng {
  const total = coords.reduce((sum, point) => ({ lat: sum.lat + point.lat, lng: sum.lng + point.lng }), { lat: 0, lng: 0 });
  return { lat: total.lat / coords.length, lng: total.lng / coords.length };
}

async function mapWithConcurrency<T, R>(items: T[], concurrency: number, task: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await task(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Erreur inconnue";
}
