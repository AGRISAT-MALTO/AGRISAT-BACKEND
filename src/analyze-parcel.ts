import { setDefaultResultOrder } from "node:dns";
import { lngLatToMercatorMeters } from "./field-watershed.js";

setDefaultResultOrder("ipv4first");

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "content-type",
};

const HF_MODEL_URL = process.env.HF_MODEL_URL ?? "https://andritinatonny-agrisat.hf.space";
const GOOGLE_MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY;

// ── GEE Auth ──

export async function analyzeParcel(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { lat, lng, zoom, polygon } = await req.json();

    if (typeof lat !== "number" || typeof lng !== "number") {
      return new Response(
        JSON.stringify({ error: "lat et lng sont requis (nombres)" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const parcelPolygon = normalizePolygon(polygon);
    if (!parcelPolygon) {
      return new Response(
        JSON.stringify({ error: "Le contour réel de la parcelle est requis (au moins 3 points valides)." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const effectiveZoom = zoom || 16;
    const warnings: string[] = [];
    let projectId = "earthengine-legacy";
    let accessToken: string | null = null;
    const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;

    if (!serviceAccountJson || serviceAccountJson.startsWith("VOTRE_")) {
      warnings.push("GEE_SERVICE_ACCOUNT_KEY n’est pas configurée.");
    } else {
      try {
        const serviceAccount = JSON.parse(serviceAccountJson) as { project_id?: unknown };
        if (typeof serviceAccount.project_id === "string" && serviceAccount.project_id.length > 0) {
          projectId = serviceAccount.project_id;
        }
        accessToken = await getGeeAccessToken();
      } catch (error) {
        warnings.push(`GEE indisponible : ${error instanceof Error ? error.message : "authentification impossible"}`);
      }
    }

    const [snapshotResult, timeSeriesResult, rainResult] = await Promise.allSettled([
      accessToken ? fetchCurrentSnapshot(accessToken, lat, lng, projectId, parcelPolygon) : Promise.resolve(null),
      accessToken ? fetchTimeSeries(accessToken, lat, lng, projectId, parcelPolygon) : Promise.resolve(null),
      fetchPrecipitationTimeSeries(lat, lng, getMonthlyRanges(6)),
    ]);
    const satData = snapshotResult.status === "fulfilled" && snapshotResult.value
      ? snapshotResult.value
      : createEmptySatelliteData();
    const timeSeries = timeSeriesResult.status === "fulfilled" && timeSeriesResult.value
      ? timeSeriesResult.value
      : { s2: [], s1: [] };
    const timeSeriesRain = rainResult.status === "fulfilled" ? rainResult.value : [];
    if (snapshotResult.status === "rejected") warnings.push(`Sentinel-2 indisponible : ${getErrorMessage(snapshotResult.reason)}`);
    if (timeSeriesResult.status === "rejected") warnings.push(`Séries temporelles indisponibles : ${getErrorMessage(timeSeriesResult.reason)}`);
    if (rainResult.status === "rejected") warnings.push(`Précipitations indisponibles : ${getErrorMessage(rainResult.reason)}`);

    const detectedSegments = accessToken
      ? await detectBarleySegments(accessToken, projectId, lat, lng, parcelPolygon, effectiveZoom, warnings)
      : [];
    if (!accessToken) warnings.push("SNIC non exécuté : authentification GEE indisponible.");

    let satelliteImage: string | null = null;
    try {
      satelliteImage = await captureParcelImage(lat, lng, effectiveZoom, parcelPolygon);
    } catch (error) {
      warnings.push(`Google Static Maps indisponible : ${getErrorMessage(error)}`);
    }

    const planting = detectPlantingDate(timeSeries.s2, timeSeries.s1);
    const agro = computeAgroScore(satData, planting, timeSeries);
    const hfResult = satelliteImage ? await callHFModelSafely(satelliteImage, warnings) : null;
    const hybrid = hfResult
      ? computeHybridScore(hfResult.confidence, hfResult.is_barley, agro.score)
      : createAgroOnlyHybridScore(agro.score);
    const season = detectSeason(lat);
    const dataSource = ["Contour parcellaire réel", ...satData.dataSource, ...(hfResult ? [`HF OrgeDetector (${HF_MODEL_URL})`] : [])].join(" + ");
    const spectralParts: string[] = [];
    if (satData.ndvi != null) spectralParts.push(`NDVI=${satData.ndvi}%`);
    if (satData.evi != null) spectralParts.push(`EVI=${satData.evi}%`);
    if (satData.savi != null) spectralParts.push(`SAVI=${satData.savi}%`);
    if (satData.ndwi != null) spectralParts.push(`NDWI=${satData.ndwi}`);
    const spectralAnalysis = spectralParts.length ? spectralParts.join(", ") : null;
    let radarAnalysis: string | null = null;
    if (satData.vv != null && satData.vh != null) {
      radarAnalysis = `VV=${satData.vv}dB, VH=${satData.vh}dB`;
      if (satData.vhVvRatio != null) radarAnalysis += `, VH/VV=${satData.vhVvRatio}`;
    }
    const riskFactors: string[] = [];
    if (satData.ndvi != null && satData.ndvi < 15) riskFactors.push("Végétation très faible");
    if (satData.ndvi != null && satData.ndvi > 85) riskFactors.push("Végétation trop dense pour de l'orge");
    if (satData.vhVvRatio != null && satData.vhVvRatio > 0.45) riskFactors.push("Structure radar type forêt");
    if (!hybrid.final_is_barley && hybrid.final_confidence > 70) riskFactors.push("Score hybride confirme non-orge");
    const detailParts: string[] = [];
    detailParts.push(hfResult
      ? `CNN: ${hfResult.is_barley ? "orge" : "non-orge"} (${hfResult.confidence.toFixed(1)}%)`
      : "CNN: indisponible");
    detailParts.push(`Agro: ${agro.score}%`);
    detailParts.push(`Hybride: ${hybrid.hybrid_score}%`);

    return new Response(JSON.stringify({
      percentage: satData.ndvi, is_barley: hfResult?.is_barley === true,
      detected_segments: detectedSegments,
      hf_model_url: HF_MODEL_URL, hf_available: hfResult !== null,
      verdict: hybrid.final_verdict, details: detailParts.join(" | "),
      culture_detected: hfResult ? (hfResult.is_barley ? "Orge" : "Non-orge") : null,
      confidence: hfResult?.confidence ?? null, cnn_prob_barley: hfResult?.prob_barley ?? null,
      cnn_prob_non_barley: hfResult?.prob_non_barley ?? null, cnn_available: hfResult !== null,
      evi: satData.evi, savi: satData.savi, ndwi: satData.ndwi,
      agro_score: agro.score, agro_breakdown: agro.breakdown, agro_details: agro.details,
      hybrid_score: hybrid.hybrid_score, saison: season,
      soil_type: satData.swir != null && satData.nir != null && satData.nir !== 0
        ? (satData.swir / satData.nir > 1.5 ? "Sol argileux sec" : satData.swir / satData.nir > 1.0 ? "Sol limoneux" : "Sol humide")
        : null,
      risk_factors: riskFactors,
      boundary_source: "Contour fourni par l’utilisateur, le cadastre ou une source agricole fiable",
      recommendations: hfResult
        ? hybrid.final_is_barley
          ? `Orge ${hybrid.final_confidence > 70 ? "confirmée" : "probable"}. Score hybride ${hybrid.hybrid_score}% (CNN ${Math.round(hfResult.confidence)}% + Agro ${agro.score}%).`
          : `Non-orge détecté. Score hybride ${hybrid.hybrid_score}%. Vérification terrain recommandée.`
        : `${hybrid.final_is_barley ? "Orge probable" : "Non-orge probable"} (modèle CNN indisponible, estimation basée uniquement sur les règles agronomiques, score ${agro.score}%). Vérification terrain recommandée.`,
      anomaly_level: hybrid.final_is_barley ? "AUCUNE" : "FORTE", data_source: dataSource,
      warnings,
      radar_analysis: radarAnalysis, spectral_analysis: spectralAnalysis,
      time_series_s2: timeSeries.s2, time_series_s1: timeSeries.s1, time_series_rain: timeSeriesRain,
      estimated_planting_date: planting.estimated_planting_date,
      estimated_harvest_date: planting.estimated_harvest_date,
      days_since_planting: planting.days_since_planting, growth_stage: planting.growth_stage,
      planting_confidence: planting.planting_confidence,
    }), { headers: { ...corsHeaders, "Content-Type": "application/json" } });
  } catch (e) {
    console.error("analyze-parcel error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
}

export type LatLng = { lat: number; lng: number };

type DetectedSegment = {
  coordinates: LatLng[];
  confidence: number;
  ndvi: number | null;
  area_ha: number;
};

type SegmentCandidate = {
  coordinates: LatLng[];
  areaM2: number;
  ndvi: number | null;
};

export type GeeValue = {
  constantValue?: unknown;
  valueReference?: string;
  functionInvocationValue?: {
    functionName: string;
    arguments: Record<string, GeeValue>;
  };
};

export const SNIC_PARAMETERS = {
  size: 15,
  compactness: 0.75,
  connectivity: 8,
  neighborhoodSize: 60,
  scale: 10,
} as const;
const MIN_SEGMENT_AREA_M2 = 2_500;
const MAX_SEGMENT_AREA_M2 = 500_000;
const MAX_SEGMENTS_TO_CLASSIFY = 12;
const SEGMENT_CLASSIFY_CONCURRENCY = 3;
const MIN_BARLEY_CONFIDENCE = 70;
const TIME_SERIES_CONCURRENCY = 1;
export const GEE_COMPUTE_TIMEOUT_MS = 60_000;
const EXTERNAL_REQUEST_TIMEOUT_MS = 30_000;
const RETRY_DELAYS_MS = [250, 1_000] as const;

export async function fetchWithRetry(input: string | URL, init: RequestInit, timeoutMs: number): Promise<Response> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    try {
      const response = await fetch(input, { ...init, signal: AbortSignal.timeout(timeoutMs) });
      if (response.status < 500 && response.status !== 429) return response;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    const delay = RETRY_DELAYS_MS[attempt];
    if (delay != null) await new Promise((resolve) => setTimeout(resolve, delay));
  }
  throw lastError instanceof Error ? lastError : new Error("Échec réseau après plusieurs tentatives.");
}

async function detectBarleySegments(
  accessToken: string,
  projectId: string,
  lat: number,
  lng: number,
  polygon: unknown,
  fallbackZoom: number,
  warnings: string[],
): Promise<DetectedSegment[]> {
  try {
    console.log("[SNIC] Segmentation démarrée");
    const raw = await callGeeComputeRaw(accessToken, projectId, buildSNICVectorsExpression(lat, lng, polygon));
    const result = raw.result;
    if (!isGeeFeatureCollection(result)) {
      console.warn("[SNIC] Aucune image Sentinel-2 exploitable ou aucun objet vectorisé.");
      return [];
    }

    console.log("[SNIC] Image Sentinel-2 récupérée");
    console.log("[SNIC] Prétraitement terminé");
    console.log("[SNIC] Segmentation terminée");
    console.log(`[SNIC] Nombre d'objets détectés : ${result.features.length}`);

    const candidates = result.features
      .flatMap(toSegmentCandidate)
      .sort((left, right) => {
        const vegetationDifference = (right.ndvi ?? -1) - (left.ndvi ?? -1);
        return vegetationDifference !== 0 ? vegetationDifference : right.areaM2 - left.areaM2;
      })
      .slice(0, MAX_SEGMENTS_TO_CLASSIFY);

    console.log(`[CLASSIFICATION] Objets envoyés au modèle : ${candidates.length}`);

    // Chaque candidat est classifié indépendamment (Google Static Maps + modèle HF) :
    // en série, jusqu'à 12 candidats pouvaient prendre 30-40s. mapWithConcurrency (déjà
    // utilisé pour les mêmes appels dans barley-detect-simple.ts) ramène ça à ~10-15s.
    const classifiedSegments = await mapWithConcurrency(candidates, SEGMENT_CLASSIFY_CONCURRENCY, async (candidate): Promise<DetectedSegment | null> => {
      try {
        const center = polygonCentroid(candidate.coordinates);
        const thumbnail = await captureParcelImage(
          center.lat,
          center.lng,
          segmentZoomForArea(candidate.areaM2, center.lat, fallbackZoom),
          candidate.coordinates,
        );
        const classification = await callHFModel(thumbnail);
        if (classification.is_barley && classification.confidence >= MIN_BARLEY_CONFIDENCE) {
          return {
            coordinates: candidate.coordinates,
            confidence: Math.round(classification.confidence),
            ndvi: candidate.ndvi,
            area_ha: Math.round((candidate.areaM2 / 10_000) * 100) / 100,
          };
        }
        return null;
      } catch (error) {
        console.warn("[CLASSIFICATION] Échec de la classification d'un objet :", error);
        return null;
      }
    });
    const barleySegments = classifiedSegments.filter((segment): segment is DetectedSegment => segment !== null);

    console.log(`[CLASSIFICATION] Orge détectée : ${barleySegments.length}`);
    return barleySegments;
  } catch (error) {
    const message = getErrorMessage(error);
    warnings.push(`SNIC indisponible : ${message}`);
    console.warn("[SNIC] Échec de la segmentation :", error);
    return [];
  }
}

export function isGeeFeatureCollection(value: unknown): value is { type: "FeatureCollection"; features: unknown[] } {
  return !!value
    && typeof value === "object"
    && (value as { type?: unknown }).type === "FeatureCollection"
    && Array.isArray((value as { features?: unknown }).features);
}

function toSegmentCandidate(feature: unknown): SegmentCandidate[] {
  if (!feature || typeof feature !== "object") return [];
  const value = feature as { geometry?: unknown; properties?: unknown };
  const coordinates = extractLatLngFromGeometry(value.geometry);
  if (coordinates.length < 3) return [];

  const areaM2 = approximatePolygonAreaM2(coordinates);
  if (areaM2 < MIN_SEGMENT_AREA_M2 || areaM2 > MAX_SEGMENT_AREA_M2) return [];

  const properties = value.properties && typeof value.properties === "object"
    ? value.properties as Record<string, unknown>
    : {};
  const ndvi = featureNumber(properties, ["NDVI", "NDVI_mean", "ndvi", "ndvi_mean"]);
  const ndwi = featureNumber(properties, ["NDWI", "NDWI_mean", "ndwi", "ndwi_mean"]);
  const ndbi = featureNumber(properties, ["NDBI", "NDBI_mean", "ndbi", "ndbi_mean"]);

  if ((ndvi != null && ndvi < 0.2) || (ndwi != null && ndwi > 0.1) || (ndbi != null && ndbi > 0.05)) {
    return [];
  }

  return [{ coordinates, areaM2, ndvi }];
}

function featureNumber(properties: Record<string, unknown>, names: string[]): number | null {
  for (const name of names) {
    const value = properties[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function buildSNICVectorsExpression(_lat: number, _lng: number, polygon: unknown) {
  const endDate = new Date().toISOString().slice(0, 10);
  const startDate = new Date(Date.now() - 90 * 86400000).toISOString().slice(0, 10);
  const values: Record<string, GeeValue> = {};
  const reference = (name: string): GeeValue => ({ valueReference: name });

  values.region = geeCall("GeometryConstructors.Polygon", {
    coordinates: geeConstant(polygonCoordinates(polygon)),
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
  values.ndviMask = geeCall("Image.gt", { image1: imageBand("NDVI"), image2: geeImageConstant(0.2) });
  values.waterMask = geeCall("Image.lt", { image1: imageBand("NDWI"), image2: geeImageConstant(0.1) });
  values.urbanMask = geeCall("Image.lt", { image1: imageBand("NDBI"), image2: geeImageConstant(0.05) });
  values.bareSoilMask = geeCall("Image.lt", { image1: imageBand("BSI"), image2: geeImageConstant(0.2) });
  values.vegetationMask = geeCall("Image.and", {
    image1: geeCall("Image.and", {
      image1: geeCall("Image.and", {
        image1: reference("ndviMask"),
        image2: reference("waterMask"),
      }),
      image2: reference("urbanMask"),
    }),
    image2: reference("bareSoilMask"),
  });
  values.maskedImage = geeCall("Image.updateMask", {
    image: reference("segmentationImage"),
    mask: reference("vegetationMask"),
  });
  values.vegetationImage = geeCall("Image.clip", {
    input: reference("maskedImage"),
    geometry: reference("region"),
  });
  values.snic = geeCall("Image.Segmentation.SNIC", {
    image: reference("vegetationImage"),
    size: geeConstant(SNIC_PARAMETERS.size),
    compactness: geeConstant(SNIC_PARAMETERS.compactness),
    connectivity: geeConstant(SNIC_PARAMETERS.connectivity),
    neighborhoodSize: geeConstant(SNIC_PARAMETERS.neighborhoodSize),
  });
  values.snicClusters = geeCall("Image.select", {
    input: reference("snic"),
    bandSelectors: geeConstant(["clusters"]),
  });
  values.vectorsImage = geeCall("Image.addBands", {
    dstImg: reference("snicClusters"),
    srcImg: reference("vegetationImage"),
  });
  values.vectors = geeCall("Image.reduceToVectors", {
    image: reference("vectorsImage"),
    reducer: geeCall("Reducer.mean", {}),
    geometry: reference("region"),
    scale: geeConstant(SNIC_PARAMETERS.scale),
    geometryType: geeConstant("polygon"),
    eightConnected: geeConstant(true),
    labelProperty: geeConstant("segment_id"),
    bestEffort: geeConstant(true),
    maxPixels: geeConstant(10_000_000),
    tileScale: geeConstant(2),
  });

  return { expression: { result: "vectors", values } };
}

export function addSpectralIndex(image: GeeValue, name: string, bands: [string, string]): GeeValue {
  const index = geeCall("Image.rename", {
    input: geeCall("Image.normalizedDifference", {
      input: image,
      bandNames: geeConstant(bands),
    }),
    names: geeConstant([name]),
  });
  return geeCall("Image.addBands", { dstImg: image, srcImg: index });
}

export function addBareSoilIndex(image: GeeValue): GeeValue {
  const band = (name: string) => geeCall("Image.select", {
    input: image,
    bandSelectors: geeConstant([name]),
  });
  const swirPlusRed = geeCall("Image.add", { image1: band("B11"), image2: band("B4") });
  const nirPlusBlue = geeCall("Image.add", { image1: band("B8"), image2: band("B2") });
  const bsi = geeCall("Image.divide", {
    image1: geeCall("Image.subtract", { image1: swirPlusRed, image2: nirPlusBlue }),
    image2: geeCall("Image.add", { image1: swirPlusRed, image2: nirPlusBlue }),
  });
  return geeCall("Image.addBands", {
    dstImg: image,
    srcImg: geeCall("Image.rename", { input: bsi, names: geeConstant(["BSI"]) }),
  });
}

export function geeCall(functionName: string, arguments_: Record<string, GeeValue>): GeeValue {
  return { functionInvocationValue: { functionName, arguments: arguments_ } };
}

export function geeConstant(value: unknown): GeeValue {
  return { constantValue: value };
}

export function geeImageConstant(value: number): GeeValue {
  return geeCall("Image.constant", { value: geeConstant(value) });
}

export function normalizePolygon(value: unknown): LatLng[] | null {
  if (!Array.isArray(value)) return null;
  const points = value.flatMap((point): LatLng[] => {
    if (!point || typeof point !== "object") return [];
    const coordinates = point as { lat?: unknown; lng?: unknown };
    return typeof coordinates.lat === "number"
      && Number.isFinite(coordinates.lat)
      && coordinates.lat >= -90
      && coordinates.lat <= 90
      && typeof coordinates.lng === "number"
      && Number.isFinite(coordinates.lng)
      && coordinates.lng >= -180
      && coordinates.lng <= 180
      ? [{ lat: coordinates.lat, lng: coordinates.lng }]
      : [];
  });
  if (points.length < 3) return null;
  const first = points[0];
  const last = points.at(-1);
  return last && first.lat === last.lat && first.lng === last.lng ? points.slice(0, -1) : points;
}

export function polygonCoordinates(polygon: unknown): number[][][] {
  const points = normalizePolygon(polygon);
  if (!points) throw new Error("Le contour de la parcelle est invalide.");
  const ring = points.map((point) => [point.lng, point.lat]);
  ring.push([...ring[0]]);
  return [ring];
}

export function extractLatLngFromGeometry(geometry: unknown): LatLng[] {
  if (!geometry || typeof geometry !== "object") return [];
  const value = geometry as { type?: unknown; coordinates?: unknown };
  const ring = value.type === "Polygon" && Array.isArray(value.coordinates)
    ? value.coordinates[0]
    : value.type === "MultiPolygon" && Array.isArray(value.coordinates)
      ? value.coordinates[0]?.[0]
      : null;
  if (!Array.isArray(ring)) return [];

  return ring.flatMap((coordinate): LatLng[] => Array.isArray(coordinate)
    && typeof coordinate[0] === "number"
    && typeof coordinate[1] === "number"
    ? [{ lat: coordinate[1], lng: coordinate[0] }]
    : []);
}

export function approximatePolygonAreaM2(coords: LatLng[]): number {
  if (coords.length < 3) return 0;
  const centroid = polygonCentroid(coords);
  const latFactor = 111_320;
  const lngFactor = 111_320 * Math.cos((centroid.lat * Math.PI) / 180);
  const points = coords.map((point) => ({
    x: (point.lng - centroid.lng) * lngFactor,
    y: (point.lat - centroid.lat) * latFactor,
  }));
  let area = 0;
  for (let index = 0; index < points.length; index++) {
    const next = (index + 1) % points.length;
    area += points[index].x * points[next].y - points[next].x * points[index].y;
  }
  return Math.abs(area) / 2;
}

function polygonCentroid(coords: LatLng[]): LatLng {
  const total = coords.reduce((sum, point) => ({ lat: sum.lat + point.lat, lng: sum.lng + point.lng }), { lat: 0, lng: 0 });
  return { lat: total.lat / coords.length, lng: total.lng / coords.length };
}

function segmentZoomForArea(areaM2: number, latitude: number, fallbackZoom: number): number {
  const targetWidthM = Math.max(180, Math.sqrt(areaM2) * 2.2);
  const zoom = Math.floor(Math.log2((156_543.03392 * Math.cos(latitude * Math.PI / 180) * 640) / targetWidthM));
  return Math.max(fallbackZoom, Math.min(20, zoom));
}

export function getGeeProjectId(): string {
  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson) return "earthengine-legacy";
  try {
    const serviceAccount = JSON.parse(serviceAccountJson) as { project_id?: unknown };
    return typeof serviceAccount.project_id === "string" && serviceAccount.project_id.length > 0
      ? serviceAccount.project_id
      : "earthengine-legacy";
  } catch {
    return "earthengine-legacy";
  }
}

// Le token GEE (JWT signé RSA + échange OAuth) est valable 1h côté Google, mais était
// jusqu'ici re-signé/re-échangé à CHAQUE appel — y compris pour chaque tuile /api/sentinel-tiles
// affichée, ce qui ajoutait un aller-retour OAuth complet (signature RSA + requête réseau) avant
// même de commencer le vrai calcul GEE. Mis en cache en mémoire, avec une marge de sécurité avant
// l'expiration réelle, et une dédup des appels concurrents (plusieurs tuiles demandées en même
// temps ne doivent déclencher qu'un seul échange de token).
const ACCESS_TOKEN_SAFETY_MARGIN_MS = 5 * 60_000;
let cachedAccessToken: { token: string; expiresAt: number } | null = null;
let accessTokenRequest: Promise<string> | null = null;

export async function getGeeAccessToken(): Promise<string> {
  if (cachedAccessToken && cachedAccessToken.expiresAt > Date.now()) {
    return cachedAccessToken.token;
  }
  if (!accessTokenRequest) {
    accessTokenRequest = fetchGeeAccessToken().finally(() => {
      accessTokenRequest = null;
    });
  }
  return accessTokenRequest;
}

async function fetchGeeAccessToken(): Promise<string> {
  const serviceAccountJson = process.env.GEE_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountJson) throw new Error("GEE_SERVICE_ACCOUNT_KEY is not configured");

  const sa = JSON.parse(serviceAccountJson);
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  const claimSet = {
    iss: sa.client_email,
    scope: "https://www.googleapis.com/auth/earthengine.readonly",
    aud: "https://oauth2.googleapis.com/token",
    iat: now,
    exp: now + 3600,
  };

  const enc = (obj: unknown) =>
    btoa(JSON.stringify(obj)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const unsignedToken = `${enc(header)}.${enc(claimSet)}`;

  const pemBody = sa.private_key
    .replace(/-----BEGIN PRIVATE KEY-----/, "")
    .replace(/-----END PRIVATE KEY-----/, "")
    .replace(/\n/g, "");
  const binaryDer = Uint8Array.from(atob(pemBody), (c) => c.charCodeAt(0));

  const cryptoKey = await crypto.subtle.importKey(
    "pkcs8", binaryDer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"]
  );

  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5", cryptoKey, new TextEncoder().encode(unsignedToken)
  );

  const sig = btoa(String.fromCharCode(...new Uint8Array(signature)))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

  const jwt = `${unsignedToken}.${sig}`;

  const tokenResp = await fetchWithRetry("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion=${jwt}`,
  }, EXTERNAL_REQUEST_TIMEOUT_MS);

  if (!tokenResp.ok) {
    const err = await tokenResp.text();
    console.error("GEE token error:", err);
    throw new Error(`Failed to get GEE access token: ${tokenResp.status}`);
  }

  const token = (await tokenResp.json()).access_token as string;
  cachedAccessToken = { token, expiresAt: Date.now() + 3_600_000 - ACCESS_TOKEN_SAFETY_MARGIN_MS };
  return token;
}

// ── GEE Expression Builders ──

function buildS2Expression(lat: number, lng: number, startDate: string, endDate: string, polygon: unknown) {
  return {
    expression: {
      result: "0",
      values: {
        "1": {
          functionInvocationValue: {
            functionName: "GeometryConstructors.Polygon",
            arguments: { coordinates: { constantValue: polygonCoordinates(polygon) } },
          },
        },
        "0": {
          functionInvocationValue: {
            functionName: "Image.reduceRegion",
            arguments: {
              geometry: { valueReference: "1" },
              image: {
                functionInvocationValue: {
                  functionName: "reduce.median",
                  arguments: {
                    collection: {
                      functionInvocationValue: {
                        functionName: "Collection.filter",
                        arguments: {
                          collection: {
                            functionInvocationValue: {
                              functionName: "Collection.filter",
                              arguments: {
                                collection: {
                                  functionInvocationValue: {
                                    functionName: "Collection.filter",
                                    arguments: {
                                      collection: {
                                        functionInvocationValue: {
                                          functionName: "ImageCollection.load",
                                          arguments: { id: { constantValue: "COPERNICUS/S2_SR_HARMONIZED" } },
                                        },
                                      },
                                      filter: {
                                        functionInvocationValue: {
                                          functionName: "Filter.intersects",
                                          arguments: {
                                            leftField: { constantValue: ".all" },
                                            rightValue: {
                                              functionInvocationValue: {
                                                functionName: "Feature",
                                                arguments: { geometry: { valueReference: "1" } },
                                              },
                                            },
                                          },
                                        },
                                      },
                                    },
                                  },
                                },
                                filter: {
                                  functionInvocationValue: {
                                    functionName: "Filter.dateRangeContains",
                                    arguments: {
                                      leftValue: {
                                        functionInvocationValue: {
                                          functionName: "DateRange",
                                          arguments: {
                                            start: { constantValue: startDate },
                                            end: { constantValue: endDate },
                                          },
                                        },
                                      },
                                      rightField: { constantValue: "system:time_start" },
                                    },
                                  },
                                },
                              },
                            },
                          },
                          filter: {
                            functionInvocationValue: {
                              functionName: "Filter.lessThan",
                              arguments: {
                                leftField: { constantValue: "CLOUDY_PIXEL_PERCENTAGE" },
                                rightValue: { constantValue: 30 },
                              },
                            },
                          },
                        },
                      },
                    },
                  },
                },
              },
              reducer: {
                functionInvocationValue: { functionName: "Reducer.mean", arguments: {} },
              },
              scale: { constantValue: 10 },
            },
          },
        },
      },
    },
  };
}

function buildS1Expression(lat: number, lng: number, startDate: string, endDate: string, polygon: unknown) {
  return {
    expression: {
      result: "0",
      values: {
        "1": {
          functionInvocationValue: {
            functionName: "GeometryConstructors.Polygon",
            arguments: { coordinates: { constantValue: polygonCoordinates(polygon) } },
          },
        },
        "0": {
          functionInvocationValue: {
            functionName: "Image.reduceRegion",
            arguments: {
              geometry: { valueReference: "1" },
              image: {
                functionInvocationValue: {
                  functionName: "reduce.median",
                  arguments: {
                    collection: {
                      functionInvocationValue: {
                        functionName: "Collection.filter",
                        arguments: {
                          collection: {
                            functionInvocationValue: {
                              functionName: "Collection.filter",
                              arguments: {
                                collection: {
                                  functionInvocationValue: {
                                    functionName: "Collection.filter",
                                    arguments: {
                                      collection: {
                                        functionInvocationValue: {
                                          functionName: "ImageCollection.load",
                                          arguments: { id: { constantValue: "COPERNICUS/S1_GRD" } },
                                        },
                                      },
                                      filter: {
                                        functionInvocationValue: {
                                          functionName: "Filter.intersects",
                                          arguments: {
                                            leftField: { constantValue: ".all" },
                                            rightValue: {
                                              functionInvocationValue: {
                                                functionName: "Feature",
                                                arguments: { geometry: { valueReference: "1" } },
                                              },
                                            },
                                          },
                                        },
                                      },
                                    },
                                  },
                                },
                                filter: {
                                  functionInvocationValue: {
                                    functionName: "Filter.dateRangeContains",
                                    arguments: {
                                      leftValue: {
                                        functionInvocationValue: {
                                          functionName: "DateRange",
                                          arguments: {
                                            start: { constantValue: startDate },
                                            end: { constantValue: endDate },
                                          },
                                        },
                                      },
                                      rightField: { constantValue: "system:time_start" },
                                    },
                                  },
                                },
                              },
                            },
                          },
                          filter: {
                            functionInvocationValue: {
                              functionName: "Filter.equals",
                              arguments: {
                                leftField: { constantValue: "instrumentMode" },
                                rightValue: { constantValue: "IW" },
                              },
                            },
                          },
                        },
                      },
                    },
                  },
                },
              },
              reducer: {
                functionInvocationValue: { functionName: "Reducer.mean", arguments: {} },
              },
              scale: { constantValue: 10 },
            },
          },
        },
      },
    },
  };
}

// ── GEE compute helper ──

async function callGeeCompute(accessToken: string, projectId: string, expression: unknown): Promise<Record<string, number | null> | null> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/value:compute`;
  try {
    const resp = await fetchWithRetry(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: JSON.stringify(expression),
    }, GEE_COMPUTE_TIMEOUT_MS);
    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      console.error("GEE API error:", resp.status, errText.slice(0, 300));
      return null;
    }
    const data = await resp.json();
    return data.result || null;
  } catch (e) {
    console.error("GEE fetch error:", e);
    return null;
  }
}

// Variant that returns the raw compute response (useful for vector outputs)
export async function callGeeComputeRaw(accessToken: string, projectId: string, expression: unknown): Promise<Record<string, unknown>> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/value:compute`;
  const resp = await fetchWithRetry(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
    body: JSON.stringify(expression),
  }, GEE_COMPUTE_TIMEOUT_MS);
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "");
    console.error("GEE API error (raw):", resp.status, errText.slice(0, 300));
    throw new Error(`Erreur GEE lors de la segmentation (${resp.status}).`);
  }
  const data: unknown = await resp.json();
  if (!data || typeof data !== "object") throw new Error("Réponse GEE de segmentation invalide.");
  return data as Record<string, unknown>;
}

// ── Satellite data types ──

interface SpectralIndices {
  ndvi: number | null;
  ndwi: number | null;
  evi: number | null;
  savi: number | null;
}

interface SatelliteData {
  ndvi: number | null;
  ndwi: number | null;
  evi: number | null;
  savi: number | null;
  blue: number | null;
  nir: number | null;
  red: number | null;
  green: number | null;
  swir: number | null;
  vv: number | null;
  vh: number | null;
  vhVvRatio: number | null;
  dataSource: string[];
}

function createEmptySatelliteData(): SatelliteData {
  return {
    ndvi: null, ndwi: null, evi: null, savi: null,
    blue: null, nir: null, red: null, green: null, swir: null,
    vv: null, vh: null, vhVvRatio: null, dataSource: [],
  };
}

function getErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) return "erreur réseau inconnue";
  const cause = error.cause;
  if (cause && typeof cause === "object") {
    const details = cause as { code?: unknown; message?: unknown };
    if (typeof details.code === "string") return `${error.message} (${details.code})`;
    if (typeof details.message === "string") return `${error.message} (${details.message})`;
  }
  return error.message;
}

function parseS2Bands(result: Record<string, number | null>) {
  const nir = result.B8_mean ?? result.B8_median ?? result.B8 ?? null;
  const red = result.B4_mean ?? result.B4_median ?? result.B4 ?? null;
  const green = result.B3_mean ?? result.B3_median ?? result.B3 ?? null;
  const blue = result.B2_mean ?? result.B2_median ?? result.B2 ?? null;
  const swir = result.B11_mean ?? result.B11_median ?? result.B11 ?? null;
  return { nir, red, green, blue, swir };
}

function parseS1Bands(result: Record<string, number | null>) {
  const vv = result.VV_mean ?? result.VV_median ?? result.VV ?? null;
  const vh = result.VH_mean ?? result.VH_median ?? result.VH ?? null;
  return { vv, vh };
}

/** Compute all spectral indices from raw bands */
function computeSpectralIndices(nir: number | null, red: number | null, blue: number | null, green: number | null, swir: number | null): SpectralIndices {
  let ndvi: number | null = null;
  let ndwi: number | null = null;
  let evi: number | null = null;
  let savi: number | null = null;

  // NDVI = (NIR - RED) / (NIR + RED)
  if (nir != null && red != null && (nir + red) !== 0) {
    ndvi = Math.round(((nir - red) / (nir + red)) * 1000) / 10; // as percentage
  }

  // NDWI = (GREEN - NIR) / (GREEN + NIR)
  if (green != null && nir != null && (green + nir) !== 0) {
    ndwi = Math.round(((green - nir) / (green + nir)) * 1000) / 1000;
  }

  // EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
  if (nir != null && red != null && blue != null) {
    const denom = nir + 6 * red - 7.5 * blue + 10000; // +10000 for S2 SR scale
    if (denom !== 0) {
      const rawEvi = 2.5 * (nir - red) / denom;
      evi = Math.round(rawEvi * 1000) / 10; // as percentage
    }
  }

  // SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L), L=0.5
  if (nir != null && red != null && (nir + red + 0.5) !== 0) {
    const L = 0.5;
    const rawSavi = ((nir - red) / (nir + red + L)) * (1 + L);
    savi = Math.round(rawSavi * 1000) / 10; // as percentage-like
  }

  return { ndvi, ndwi, evi, savi };
}

async function fetchCurrentSnapshot(accessToken: string, lat: number, lng: number, projectId: string, polygon: unknown): Promise<SatelliteData> {
  const now = new Date();
  const end = now.toISOString().slice(0, 10);
  const start = new Date(now.getTime() - 180 * 86400000).toISOString().slice(0, 10);

  const [s2Result, s1Result] = await Promise.all([
    callGeeCompute(accessToken, projectId, buildS2Expression(lat, lng, start, end, polygon)),
    callGeeCompute(accessToken, projectId, buildS1Expression(lat, lng, start, end, polygon)),
  ]);

  const result: SatelliteData = {
    ndvi: null, ndwi: null, evi: null, savi: null,
    blue: null, nir: null, red: null, green: null, swir: null,
    vv: null, vh: null, vhVvRatio: null, dataSource: [],
  };

  if (s2Result) {
    const { nir, red, green, blue, swir } = parseS2Bands(s2Result);
    result.nir = nir;
    result.red = red;
    result.green = green;
    result.blue = blue;
    result.swir = swir;

    const indices = computeSpectralIndices(nir, red, blue, green, swir);
    result.ndvi = indices.ndvi != null ? Math.round(indices.ndvi) : null;
    result.ndwi = indices.ndwi;
    result.evi = indices.evi != null ? Math.round(indices.evi * 10) / 10 : null;
    result.savi = indices.savi != null ? Math.round(indices.savi * 10) / 10 : null;

    result.dataSource.push("Sentinel-2");
  }

  if (s1Result) {
    const { vv, vh } = parseS1Bands(s1Result);
    if (vv != null) result.vv = Math.round(vv * 100) / 100;
    if (vh != null) result.vh = Math.round(vh * 100) / 100;
    if (vv != null && vh != null && vv !== 0) {
      result.vhVvRatio = Math.round((vh / vv) * 1000) / 1000;
    }
    result.dataSource.push("Sentinel-1");
  }

  return result;
}

// ── Time Series ──

interface TimeSeriesPointS2 { date: string; ndvi: number | null; cloud_cover: number | null; }
interface TimeSeriesPointS1 { date: string; vv: number | null; vh: number | null; }
interface TimeSeriesPointRain { date: string; precipitation_mm: number | null; }

function getMonthlyRanges(numMonths: number): Array<{ label: string; start: string; end: string }> {
  const now = new Date();
  const ranges: Array<{ label: string; start: string; end: string }> = [];
  for (let i = numMonths - 1; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    const start = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`;
    const endD = new Date(d.getFullYear(), d.getMonth() + 2, 0);
    const end = `${endD.getFullYear()}-${String(endD.getMonth() + 1).padStart(2, "0")}-${String(endD.getDate()).padStart(2, "0")}`;
    ranges.push({ label: start.slice(0, 7), start, end });
  }
  return ranges;
}

async function fetchTimeSeries(
  accessToken: string, lat: number, lng: number, projectId: string, polygon: unknown
): Promise<{ s2: TimeSeriesPointS2[]; s1: TimeSeriesPointS1[] }> {
  const months = getMonthlyRanges(6);

  const s2 = await mapWithConcurrency(months, TIME_SERIES_CONCURRENCY, async (m) => {
    const result = await callGeeCompute(accessToken, projectId, buildS2Expression(lat, lng, m.start, m.end, polygon));
    let ndvi: number | null = null;
    if (result) {
      const { nir, red } = parseS2Bands(result);
      if (nir != null && red != null && (nir + red) !== 0) {
        ndvi = Math.round(((nir - red) / (nir + red)) * 1000) / 10;
      }
    }
    return { date: m.label, ndvi, cloud_cover: null } as TimeSeriesPointS2;
  });

  const s1 = await mapWithConcurrency(months, TIME_SERIES_CONCURRENCY, async (m) => {
    const result = await callGeeCompute(accessToken, projectId, buildS1Expression(lat, lng, m.start, m.end, polygon));
    let vv: number | null = null;
    let vh: number | null = null;
    if (result) {
      const parsed = parseS1Bands(result);
      if (parsed.vv != null) vv = Math.round(parsed.vv * 100) / 100;
      if (parsed.vh != null) vh = Math.round(parsed.vh * 100) / 100;
    }
    return { date: m.label, vv, vh } as TimeSeriesPointS1;
  });

  return { s2, s1 };
}

/**
 * Précipitations réelles (Open-Meteo, historique gratuit sans clé) agrégées par mois pour
 * s'aligner sur les mêmes libellés que `time_series_s2`/`time_series_s1` (ex. "2026-03").
 */
async function fetchPrecipitationTimeSeries(
  lat: number, lng: number, months: Array<{ label: string; start: string; end: string }>
): Promise<TimeSeriesPointRain[]> {
  if (months.length === 0) return [];

  const archiveCutoff = new Date();
  archiveCutoff.setUTCDate(archiveCutoff.getUTCDate() - 2);
  const cutoffStr = archiveCutoff.toISOString().slice(0, 10);
  const startDate = months[0].start;
  const lastMonthEnd = months[months.length - 1].end;
  const endDate = lastMonthEnd > cutoffStr ? cutoffStr : lastMonthEnd;

  const url = new URL("https://archive-api.open-meteo.com/v1/archive");
  url.search = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lng),
    start_date: startDate,
    end_date: endDate,
    daily: "precipitation_sum",
    timezone: "auto",
  }).toString();

  const response = await fetchWithRetry(url, {}, EXTERNAL_REQUEST_TIMEOUT_MS);
  if (!response.ok) throw new Error("Les données de précipitations Open-Meteo sont indisponibles.");
  const payload: unknown = await response.json();
  const daily = payload && typeof payload === "object" && "daily" in payload ? (payload as { daily: unknown }).daily : null;
  if (!daily || typeof daily !== "object") throw new Error("Réponse météo (pluie) incomplète.");
  const values = daily as { time?: unknown; precipitation_sum?: unknown };
  if (!Array.isArray(values.time) || !Array.isArray(values.precipitation_sum)) {
    throw new Error("Précipitations journalières indisponibles.");
  }
  const dates = values.time as unknown[];
  const precipValues = values.precipitation_sum as unknown[];

  const monthlyTotals = new Map<string, number>();
  dates.forEach((date, index) => {
    const value = precipValues[index];
    if (typeof date !== "string" || typeof value !== "number") return;
    const monthLabel = date.slice(0, 7);
    monthlyTotals.set(monthLabel, (monthlyTotals.get(monthLabel) ?? 0) + value);
  });

  return months.map((m) => ({
    date: m.label,
    precipitation_mm: monthlyTotals.has(m.label) ? Math.round((monthlyTotals.get(m.label) as number) * 10) / 10 : null,
  }));
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

// ── Planting Date Detection ──

interface PlantingDetection {
  estimated_planting_date: string | null;
  estimated_harvest_date: string | null;
  days_since_planting: number | null;
  growth_stage: string | null;
  planting_confidence: number;
}

function detectPlantingDate(s2: TimeSeriesPointS2[], s1: TimeSeriesPointS1[]): PlantingDetection {
  const validS2 = s2.filter((p) => p.ndvi != null);
  if (validS2.length < 2) {
    return { estimated_planting_date: null, estimated_harvest_date: null, days_since_planting: null, growth_stage: null, planting_confidence: 0 };
  }

  let maxJump = 0;
  let jumpIndex = -1;
  for (let i = 1; i < validS2.length; i++) {
    const delta = validS2[i].ndvi! - validS2[i - 1].ndvi!;
    if (delta > maxJump) { maxJump = delta; jumpIndex = i; }
  }

  if (maxJump < 15 || jumpIndex < 0) {
    const minIdx = validS2.reduce((mi, p, i) => (p.ndvi! < validS2[mi].ndvi! ? i : mi), 0);
    if (minIdx < validS2.length - 1 && validS2[minIdx + 1].ndvi! - validS2[minIdx].ndvi! > 5) {
      jumpIndex = minIdx + 1;
      maxJump = validS2[jumpIndex].ndvi! - validS2[minIdx].ndvi!;
    } else {
      return { estimated_planting_date: null, estimated_harvest_date: null, days_since_planting: null, growth_stage: null, planting_confidence: 0 };
    }
  }

  const jumpMonth = validS2[jumpIndex].date;
  const [year, month] = jumpMonth.split("-").map(Number);
  const plantingDate = new Date(year, month - 1, 10);

  let confidence = Math.min(90, Math.round(maxJump * 2));
  if (s1.length > 0 && jumpIndex < s1.length) {
    const s1Jump = s1[jumpIndex];
    const s1Prev = jumpIndex > 0 ? s1[jumpIndex - 1] : null;
    if (s1Jump.vv != null && s1Prev?.vv != null && s1Jump.vv - s1Prev.vv > 0.5) {
      confidence = Math.min(95, confidence + 10);
    }
  }

  const now = new Date();
  const daysSincePlanting = Math.round((now.getTime() - plantingDate.getTime()) / 86400000);

  let growthStage: string;
  if (daysSincePlanting < 0) growthStage = "Pré-semis";
  else if (daysSincePlanting <= 7) growthStage = "Semis";
  else if (daysSincePlanting <= 20) growthStage = "Levée";
  else if (daysSincePlanting <= 45) growthStage = "Tallage";
  else if (daysSincePlanting <= 65) growthStage = "Montaison";
  else if (daysSincePlanting <= 80) growthStage = "Épiaison";
  else if (daysSincePlanting <= 110) growthStage = "Maturation";
  else growthStage = "Récolte";

  const harvestDate = new Date(plantingDate.getTime() + 100 * 86400000);

  return {
    estimated_planting_date: plantingDate.toISOString().slice(0, 10),
    estimated_harvest_date: harvestDate.toISOString().slice(0, 10),
    days_since_planting: Math.max(0, daysSincePlanting),
    growth_stage: growthStage,
    planting_confidence: confidence,
  };
}

// ── Agronomic Rules Scoring ──

interface AgroScore {
  score: number; // 0-100
  breakdown: {
    ndvi_score: number;
    cycle_score: number;
    radar_score: number;
    ndvi_trend_score: number;
  };
  details: string;
}

function computeAgroScore(
  satData: SatelliteData,
  planting: PlantingDetection,
  timeSeries: { s2: TimeSeriesPointS2[]; s1: TimeSeriesPointS1[] }
): AgroScore {
  let ndviScore = 0;
  let cycleScore = 0;
  let radarScore = 0;
  let ndviTrendScore = 0;
  const details: string[] = [];

  // 1. NDVI range check for barley (typical: 30-80% at peak, 20-60% average)
  if (satData.ndvi != null) {
    const ndvi = satData.ndvi;
    if (ndvi >= 25 && ndvi <= 75) {
      ndviScore = 80 + (1 - Math.abs(ndvi - 50) / 25) * 20; // peak around 50%
      details.push(`NDVI ${ndvi}% dans la plage orge`);
    } else if (ndvi >= 15 && ndvi <= 85) {
      ndviScore = 40;
      details.push(`NDVI ${ndvi}% limite pour l'orge`);
    } else {
      ndviScore = 10;
      details.push(`NDVI ${ndvi}% hors plage orge typique`);
    }
  }

  // 2. Growth cycle coherence (barley = 90-120 days)
  if (planting.days_since_planting != null && planting.growth_stage) {
    const days = planting.days_since_planting;
    if (days >= 0 && days <= 130) {
      cycleScore = 80;
      // Check NDVI vs expected stage
      if (satData.ndvi != null) {
        const ndvi = satData.ndvi;
        if (planting.growth_stage === "Levée" && ndvi >= 15 && ndvi <= 40) cycleScore = 95;
        else if (planting.growth_stage === "Tallage" && ndvi >= 30 && ndvi <= 65) cycleScore = 95;
        else if (planting.growth_stage === "Montaison" && ndvi >= 40 && ndvi <= 80) cycleScore = 95;
        else if (planting.growth_stage === "Épiaison" && ndvi >= 50 && ndvi <= 80) cycleScore = 95;
        else if (planting.growth_stage === "Maturation" && ndvi >= 30 && ndvi <= 60) cycleScore = 90;
        else if (planting.growth_stage === "Récolte" && ndvi >= 10 && ndvi <= 40) cycleScore = 90;
      }
      details.push(`Cycle ${days}j cohérent (stade ${planting.growth_stage})`);
    } else if (days > 130) {
      cycleScore = 30;
      details.push(`Cycle ${days}j > 130j, dépasse le cycle orge`);
    }
  } else {
    cycleScore = 50; // neutral if no planting detected
  }

  // 3. Radar signature check (barley typically: VV -8 to -14 dB, VH -14 to -22 dB)
  if (satData.vv != null && satData.vh != null) {
    const vv = satData.vv;
    const vh = satData.vh;
    if (vv >= -14 && vv <= -8 && vh >= -22 && vh <= -14) {
      radarScore = 90;
      details.push(`Radar VV=${vv}dB VH=${vh}dB typique cultures`);
    } else if (vv >= -16 && vv <= -6 && vh >= -24 && vh <= -12) {
      radarScore = 60;
      details.push(`Radar acceptable mais pas typique orge`);
    } else {
      radarScore = 20;
      details.push(`Signature radar atypique pour céréales`);
    }

    // VH/VV ratio: barley typically 0.3-0.6
    if (satData.vhVvRatio != null) {
      const ratio = satData.vhVvRatio;
      if (ratio >= 0.3 && ratio <= 0.6) radarScore = Math.min(100, radarScore + 10);
    }
  } else {
    radarScore = 50; // neutral if no radar
  }

  // 4. NDVI temporal trend (barley: rise then fall pattern)
  const validNdvi = timeSeries.s2.filter(p => p.ndvi != null).map(p => p.ndvi!);
  if (validNdvi.length >= 3) {
    // Check for rise-fall pattern
    const maxNdvi = Math.max(...validNdvi);
    const minNdvi = Math.min(...validNdvi);
    const range = maxNdvi - minNdvi;

    if (range >= 15) {
      // Good dynamic range - typical of crops
      ndviTrendScore = 70 + Math.min(30, range);
      details.push(`Dynamique NDVI ${range.toFixed(0)}% (bon signal cultural)`);
    } else if (range >= 5) {
      ndviTrendScore = 50;
      details.push(`Dynamique NDVI faible (${range.toFixed(0)}%)`);
    } else {
      ndviTrendScore = 20;
      details.push(`NDVI stable - pas typique d'une culture`);
    }
  } else {
    ndviTrendScore = 50;
  }

  // Weighted average
  const score = Math.round(
    ndviScore * 0.3 + cycleScore * 0.25 + radarScore * 0.25 + ndviTrendScore * 0.2
  );

  return {
    score: Math.max(0, Math.min(100, score)),
    breakdown: {
      ndvi_score: Math.round(ndviScore),
      cycle_score: Math.round(cycleScore),
      radar_score: Math.round(radarScore),
      ndvi_trend_score: Math.round(ndviTrendScore),
    },
    details: details.join(" | "),
  };
}

// ── Hybrid Score Fusion ──

function computeHybridScore(cnnConfidence: number, cnnIsBarley: boolean, agroScore: number): {
  hybrid_score: number;
  final_is_barley: boolean;
  final_verdict: string;
  final_confidence: number;
} {
  // CNN score: positive if barley, inverted if not
  const cnnScore = cnnIsBarley ? cnnConfidence : (100 - cnnConfidence);

  // Hybrid = 0.6 * CNN + 0.4 * Agro
  const hybridScore = Math.round(cnnScore * 0.6 + agroScore * 0.4);

  // Disagreement penalty — seuil et poids assouplis pour ne pénaliser que les désaccords
  // vraiment francs (au lieu de faire basculer en incertain dès qu'un écart modéré existe).
  const disagreement = Math.abs(cnnScore - agroScore);
  const penalty = disagreement > 55 ? disagreement * 0.08 : 0;
  const finalConfidence = Math.max(0, Math.min(100, Math.round(hybridScore - penalty)));

  const finalIsBarley = hybridScore > 50;

  let verdict: string;
  if (finalIsBarley) {
    if (finalConfidence > 80) verdict = "✅ ORGE CONFIRMÉE — CNN + Règles agro concordent";
    else if (finalConfidence > 45) verdict = "🟡 ORGE PROBABLE — Confiance modérée";
    else verdict = "⚠️ ORGE INCERTAIN — Validation terrain recommandée";
  } else {
    if (finalConfidence > 80) verdict = "❌ NON-ORGE CONFIRMÉ — CNN + Agro concordent";
    else if (finalConfidence > 45) verdict = "🟡 PROBABLEMENT NON-ORGE";
    else verdict = "⚠️ INDÉTERMINÉ — Données insuffisantes";
  }

  if (disagreement > 55) {
    verdict += ` (⚠️ désaccord CNN/Agro: ${disagreement.toFixed(0)}%)`;
  }

  return {
    hybrid_score: hybridScore,
    final_is_barley: finalIsBarley,
    final_verdict: verdict,
    final_confidence: finalConfidence,
  };
}

// ── HuggingFace Model Integration ──

export async function captureParcelImage(lat: number, lng: number, zoom: number, polygon?: unknown): Promise<string> {
  if (!GOOGLE_MAPS_API_KEY || GOOGLE_MAPS_API_KEY.startsWith("VOTRE_")) {
    throw new Error("GOOGLE_MAPS_API_KEY n’est pas configurée.");
  }

  const params = new URLSearchParams({
    center: `${lat},${lng}`,
    zoom: String(zoom),
    size: "640x640",
    maptype: "satellite",
    key: GOOGLE_MAPS_API_KEY,
  });
  const parcelPolygon = normalizePolygon(polygon);
  if (parcelPolygon) {
    const path = parcelPolygon.map((point) => `${point.lat},${point.lng}`).join("|");
    params.set("path", `color:0xfbbf24ff|weight:2|${path}`);
  }

  const resp = await fetchWithRetry(`https://maps.googleapis.com/maps/api/staticmap?${params.toString()}`, {}, EXTERNAL_REQUEST_TIMEOUT_MS);
  if (!resp.ok) {
    throw new Error(`Google Maps Static API error: ${resp.status}`);
  }

  const arrayBuffer = await resp.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// ── Sentinel-2 comme source d'imagerie (remplace Google Static Maps pour la Version A) ──

export interface PixelGrid {
  widthPx: number;
  heightPx: number;
  originXMeters: number;
  originYMeters: number;
  scaleMeters: number;
}

/** Point de vérité unique pour la colorimétrie vrai-couleur Sentinel-2 (tuiles carto + miniatures CNN). */
export function buildTrueColorVisualizeExpression(imageRef: GeeValue): GeeValue {
  return geeCall("Image.visualize", {
    image: imageRef,
    bands: geeConstant(["B4", "B3", "B2"]),
    min: geeConstant(0),
    max: geeConstant(3000),
  });
}

/** Variante de callGeeComputeRaw pour une image visualisée (3 bandes uint8) : renvoie directement les octets PNG. */
export async function computePixelsPng(
  accessToken: string,
  projectId: string,
  expression: { result: string; values: Record<string, GeeValue> },
  grid: PixelGrid,
): Promise<Uint8Array> {
  const url = `https://earthengine.googleapis.com/v1/projects/${projectId}/image:computePixels`;
  const body = {
    expression,
    fileFormat: "PNG",
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
    throw new Error(`GEE computePixels (PNG) : erreur ${response.status} ${text.slice(0, 300)}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Mètres/pixel à un zoom donné (même formule que zoomToFitRadius/segmentZoomForArea). */
function metersPerPixelAtZoom(lat: number, zoom: number): number {
  return (156_543.03392 * Math.cos((lat * Math.PI) / 180)) / (2 ** zoom);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/**
 * Remplace captureParcelImage (Google Static Maps) pour la Version A (barley-detect-simple.ts) :
 * miniature vrai-couleur Sentinel-2 centrée sur lat/lng, même contrat de sortie (PNG base64,
 * 640x640) pour que callHFModel n'ait rien à changer.
 *
 * Sélectionne l'image la moins nuageuse dans [startDate, endDate] qui couvre RÉELLEMENT ce point
 * — pas forcément l'image "globale" choisie par selectBestImageWindow pour toute la zone
 * recherchée. Nécessaire dès que la zone dépasse une seule scène Sentinel-2 (~110 km) : un
 * candidat peut venir d'une tuile de segmentation dont l'image la plus propre diffère de l'image
 * globale, et forcer cette dernière produisait une collection vide -> Image.visualize(null) ->
 * erreur 400 (constaté en conditions réelles avec un rayon de recherche de 3 km).
 */
export async function captureSentinel2ParcelImage(
  accessToken: string,
  projectId: string,
  lat: number,
  lng: number,
  zoom: number,
  startDate: string,
  endDate: string,
): Promise<string> {
  const values: Record<string, GeeValue> = {};
  const ref = (name: string): GeeValue => ({ valueReference: name });

  values.point = geeCall("GeometryConstructors.Point", { coordinates: geeConstant([lng, lat]) });
  values.region = geeCall("Geometry.buffer", { geometry: ref("point"), distance: geeConstant(5_000) });
  values.intersects = geeCall("Filter.intersects", {
    leftField: geeConstant(".all"),
    rightValue: geeCall("Feature", { geometry: ref("region") }),
  });
  values.dateRange = geeCall("Filter.dateRangeContains", {
    leftValue: geeCall("DateRange", { start: geeConstant(startDate), end: geeConstant(endDate) }),
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
  values.visualized = buildTrueColorVisualizeExpression(ref("image"));

  const sizePx = 640;
  const scaleMeters = metersPerPixelAtZoom(lat, zoom);
  const halfSizeMeters = (sizePx / 2) * scaleMeters;
  const center = lngLatToMercatorMeters(lng, lat);
  const grid: PixelGrid = {
    widthPx: sizePx,
    heightPx: sizePx,
    originXMeters: center.x - halfSizeMeters,
    originYMeters: center.y + halfSizeMeters,
    scaleMeters,
  };

  const pngBytes = await computePixelsPng(accessToken, projectId, { result: "visualized", values }, grid);
  return bytesToBase64(pngBytes);
}

export interface HFModelResult {
  is_barley: boolean;
  confidence: number;
  prob_barley: number;
  prob_non_barley: number;
}

async function callHFModelSafely(imageBase64: string, warnings: string[]): Promise<HFModelResult | null> {
  try {
    return await callHFModel(imageBase64);
  } catch (error) {
    const message = getErrorMessage(error);
    warnings.push(`Modèle Hugging Face indisponible : ${message}`);
    console.warn("[CLASSIFICATION] Modèle indisponible :", error);
    return null;
  }
}

// Repli quand le CNN Hugging Face est indisponible : on ne peut pas se permettre de
// déclarer "non-orge" par défaut (ça masquerait de vraies parcelles d'orge en cas de
// panne du Space externe) — on retombe sur le score agronomique seul, avec un verdict
// qui indique explicitement que la classification CNN n'a pas pu être faite.
function createAgroOnlyHybridScore(agroScore: number): ReturnType<typeof computeHybridScore> {
  const finalIsBarley = agroScore > 50;
  const verdict = finalIsBarley
    ? `⚠️ ORGE PROBABLE (CNN indisponible, estimation agro seule) — Score agro ${agroScore}%`
    : `⚠️ NON-ORGE PROBABLE (CNN indisponible, estimation agro seule) — Score agro ${agroScore}%`;
  return {
    hybrid_score: agroScore,
    final_is_barley: finalIsBarley,
    final_verdict: verdict,
    final_confidence: agroScore,
  };
}

export async function callHFModel(satelliteImageBase64: string): Promise<HFModelResult> {
  console.log("Calling HF model /predict at:", HF_MODEL_URL);

  const binaryStr = atob(satelliteImageBase64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }

  const formData = new FormData();
  const blob = new Blob([bytes], { type: "image/png" });
  formData.append("file", blob, "parcel.png");

  const resp = await fetchWithRetry(`${HF_MODEL_URL}/predict`, {
    method: "POST",
    body: formData,
  }, EXTERNAL_REQUEST_TIMEOUT_MS);

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "");
    console.error("HF /predict error:", resp.status, errText.slice(0, 500));
    throw new Error(`HF model error: ${resp.status}`);
  }

  const data: unknown = await resp.json();
  console.log("HF /predict response:", JSON.stringify(data));
  if (!data || typeof data !== "object") throw new Error("Réponse du modèle HF invalide.");

  const values = data as Record<string, unknown>;
  if (
    typeof values.is_barley !== "boolean"
    || typeof values.confidence !== "number"
    || !Number.isFinite(values.confidence)
    || typeof values.prob_barley !== "number"
    || !Number.isFinite(values.prob_barley)
    || typeof values.prob_non_barley !== "number"
    || !Number.isFinite(values.prob_non_barley)
  ) {
    throw new Error("La réponse du modèle HF ne contient pas les valeurs attendues.");
  }

  return {
    is_barley: values.is_barley,
    confidence: values.confidence,
    prob_barley: values.prob_barley,
    prob_non_barley: values.prob_non_barley,
  };
}

// ── Season detection ──

function detectSeason(lat: number): string {
  const calendarMonth = new Date().getMonth() + 1; // 1-12, cycle calé sur l'hémisphère nord (semis oct-nov)
  // Hémisphère sud : cycle décalé de 6 mois (semis avril-mai).
  const month = lat < 0 ? (((calendarMonth - 1 + 6) % 12) + 1) : calendarMonth;

  if (month === 10 || month === 11) return "Semis / Levée";
  if (month === 12 || month === 1 || month === 2) return "Tallage";
  if (month === 3 || month === 4) return "Montaison";
  if (month === 5) return "Épiaison";
  if (month === 6) return "Maturation";
  if (month === 7 || month === 8) return "Récolte / Post-récolte";
  return "Jachère"; // septembre
}