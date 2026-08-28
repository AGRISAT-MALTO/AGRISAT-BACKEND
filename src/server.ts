import "dotenv/config";
import "dotenv/config";
import dns from "node:dns";
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Prisma } from "@prisma/client";
import { z } from "zod";
import { analyzeParcel, getGeeAccessToken, getGeeProjectId } from "./analyze-parcel.js";
import { detectAutomaticParcels } from "./automatic-parcels.js";
import { analyzeFieldsSimple, saveSimpleFieldParcelles } from "./barley-detect-simple.js";
import { fetchSentinel2TilePng } from "./sentinel-tiles.js";
import { callFieldSegmentationModel } from "./field-segmentation.js";
import { prisma } from "./db.js";

// Sur certains réseaux (box 4G/domestique), les adresses IPv6 sont annoncées mais peu fiables,
// ce qui fait échouer/traîner les connexions Prisma vers Neon avant leur timeout. On force IPv4.
dns.setDefaultResultOrder("ipv4first");

// PNG transparent 1x1, servi quand une tuile Sentinel-2 n'est pas disponible (tuile hors
// empreinte de la scène, image manquante) pour éviter une icône "image cassée" côté carte.
const TRANSPARENT_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

const app = Fastify({ logger: true });

const jsonValue = (value: unknown) => value as Prisma.InputJsonValue;
type ParcelleRow = Awaited<ReturnType<typeof prisma.parcelle.findMany>>[number];

const PARCELLES_CACHE_TTL_MS = 30_000;
const PARCELLES_QUERY_TIMEOUT_MS = 20_000;

let parcellesCache: { expiresAt: number; rows: ParcelleRow[] } | null = null;
let parcellesRequest: Promise<ParcelleRow[]> | null = null;

async function loadParcelles(): Promise<ParcelleRow[]> {
  if (parcellesCache) {
    // Cache expiré mais présent : on sert la version connue tout de suite et on
    // rafraîchit en arrière-plan, pour ne jamais faire attendre une requête
    // utilisateur sur la latence Neon (souvent plusieurs secondes sur ce réseau).
    if (parcellesCache.expiresAt <= Date.now()) refreshParcellesInBackground();
    return parcellesCache.rows;
  }
  return fetchParcelles();
}

function refreshParcellesInBackground(): void {
  if (parcellesRequest) return;
  fetchParcelles().catch((error) => {
    app.log.warn({ err: error }, "parcelles: background refresh failed, keeping stale cache");
  });
}

async function fetchParcelles(): Promise<ParcelleRow[]> {
  if (!parcellesRequest) {
    parcellesRequest = withTimeout(
      prisma.parcelle.findMany({ orderBy: { created_at: "desc" } }),
      PARCELLES_QUERY_TIMEOUT_MS,
    );
  }
  try {
    const rows = await parcellesRequest;
    parcellesCache = { rows, expiresAt: Date.now() + PARCELLES_CACHE_TTL_MS };
    return rows;
  } finally {
    parcellesRequest = null;
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Requête base de données dépassant ${timeoutMs} ms.`));
    }, timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

const pointSchema = z.object({ lat: z.number(), lng: z.number() });
const analyzeSimpleSchema = z.object({
  latitude: z.number().finite().min(-90).max(90),
  longitude: z.number().finite().min(-180).max(180),
  radius: z.number().finite().min(50).max(20_000),
  confidenceThreshold: z.number().finite().min(0.5).max(0.95).optional(),
  minAreaHa: z.number().finite().min(0.01).max(5).optional(),
  baseTemperature: z.number().finite().min(-20).max(30).optional(),
  threshold: z.number().finite().min(2200).max(10000).optional(),
  periodDays: z.number().int().min(1).max(730).optional(),
});
const automaticDetectionSchema = z.object({
  lat: z.number().finite().min(-90).max(90),
  lng: z.number().finite().min(-180).max(180),
  radiusKm: z.number().finite().min(0.05).max(20),
  baseTemperature: z.number().finite().min(-20).max(30),
  threshold: z.number().finite().min(2200).max(10000),
  periodDays: z.number().int().min(1).max(730),
});
const fieldSegmentationSchema = z.object({
  fileBase64: z.string().min(1),
  filename: z.string().min(1),
  threshold: z.number().finite().min(0).max(1).optional(),
});

const parcelleSchema = z.object({
  label: z.string().max(200),
  coordinates: z.array(pointSchema).min(3),
  center_lat: z.number(),
  center_lng: z.number(),
  surface_ha: z.number().nullable(),
  culture_declared: z.string(),
  culture_detected: z.string().nullable(),
  ndvi_percentage: z.number().nullable(),
  ndre: z.number().nullable(),
  spectral_bands: z.record(z.string(), z.number().nullable()).nullable(),
  confidence: z.number().nullable(),
  verdict: z.string().nullable(),
  details: z.string().nullable(),
  saison: z.string().nullable(),
  soil_type: z.string().nullable(),
  risk_factors: z.array(z.string()),
  recommendations: z.string().nullable(),
  data_source: z.string().nullable(),
  owner_name: z.string(),
  notes: z.string(),
  time_series_s2: z.array(z.unknown()),
  time_series_s1: z.array(z.unknown()),
  estimated_planting_date: z.string().nullable(),
  estimated_harvest_date: z.string().nullable(),
  days_since_planting: z.number().nullable(),
  growth_stage: z.string().nullable(),
  planting_confidence: z.number().nullable(),
  evi: z.number().nullable(),
  savi: z.number().nullable(),
  ndwi: z.number().nullable(),
  agro_score: z.number().nullable(),
  hybrid_score: z.number().nullable(),
  cnn_prob_barley: z.number().nullable(),
  cnn_prob_non_barley: z.number().nullable(),
});

app.register(cors, { origin: process.env.FRONTEND_ORIGIN?.split(",") ?? true });

app.setErrorHandler((error, _request, reply) => {
  app.log.error(error);
  const statusCode = typeof error === "object" && error !== null && "statusCode" in error && typeof error.statusCode === "number" && error.statusCode < 500
    ? error.statusCode
    : 500;
  const message = error instanceof Error ? error.message : "Erreur interne du serveur";
  return reply.code(statusCode).send({ error: statusCode === 500 ? "Erreur interne du serveur" : message });
});

app.get("/health", async () => ({ status: "ok" }));

// Fallback endpoint that always returns a detection result (uses internal fallback on errors)
app.post("/api/detect-parcels-fallback", async (request, reply) => {
  const parsed = automaticDetectionSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: "Coordonnées ou rayon invalides.", details: parsed.error.flatten() });
  const result = await detectAutomaticParcels(parsed.data);
  parcellesCache = null;
  return reply.send(result);
});

app.get("/api/parcelles", async (_request, reply) => {
  try {
    return reply.send(await loadParcelles());
  } catch (error) {
    app.log.error({ err: error }, "parcelles: read failed");
    return reply.code(503).send({ error: "Les parcelles sont temporairement indisponibles." });
  }
});

app.post("/api/parcelles", async (request, reply) => {
  const parsed = parcelleSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: "Données de parcelle invalides", details: parsed.error.flatten() });

  const { coordinates, risk_factors, time_series_s1, time_series_s2, spectral_bands, ...scalarData } = parsed.data;
  const parcelle = await prisma.parcelle.create({
    data: {
      ...scalarData,
      coordinates: jsonValue(coordinates),
      risk_factors: jsonValue(risk_factors),
      time_series_s1: jsonValue(time_series_s1),
      time_series_s2: jsonValue(time_series_s2),
      spectral_bands: spectral_bands === null ? Prisma.JsonNull : jsonValue(spectral_bands),
    },
  });
  parcellesCache = null;
  return reply.code(201).send(parcelle);
});

app.delete("/api/parcelles/:id", async (request, reply) => {
  const id = z.string().uuid().safeParse((request.params as { id: string }).id);
  if (!id.success) return reply.code(400).send({ error: "Identifiant invalide" });
  const result = await prisma.parcelle.deleteMany({ where: { id: id.data } });
  if (result.count === 0) return reply.code(404).send({ error: "Parcelle introuvable" });
  parcellesCache = null;
  return reply.code(204).send();
});

app.post("/api/analyze-parcel", async (request, reply) => {
  const response = await analyzeParcel(new Request("http://backend/analyze-parcel", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request.body),
  }));
  return reply.code(response.status).headers(Object.fromEntries(response.headers.entries())).send(await response.text());
});

app.post("/api/detect-parcels", async (request, reply) => {
  const parsed = automaticDetectionSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: "Coordonnées ou rayon invalides.", details: parsed.error.flatten() });
  try {
    const result = await detectAutomaticParcels(parsed.data);
    parcellesCache = null;
    return reply.send(result);
  } catch (error) {
    app.log.warn({ err: error }, "detect-parcels: discovery failed, returning fallback notice");
    const msg = error instanceof Error ? error.message : "Le service de recherche des parcelles est indisponible.";
    // Return a graceful fallback so the frontend can display a notice instead of 502
    const fallback = {
      center: { lat: parsed.data.lat, lng: parsed.data.lng },
      radius_km: parsed.data.radiusKm,
      base_temperature: parsed.data.baseTemperature,
      threshold: parsed.data.threshold,
      period_days: parsed.data.periodDays,
      candidates_found: 0,
      analyzed_count: 0,
      parcels: [],
      notice: msg.includes("database") || msg.includes("Database") || msg.toLowerCase().includes("parcelles") ? "La base des parcelles agricoles est indisponible. Essayez un rayon plus large ou vérifiez la configuration de la base de données." : msg,
    };
    return reply.send(fallback);
  }
});

const FIELD_SCAN_VERSION = "v1-simple";
const FIELD_SCAN_PROXIMITY_DEGREES = 0.01; // ~1 km, tolérance pour retrouver "la même zone"

app.post("/api/analyze", async (request, reply) => {
  const parsed = analyzeSimpleSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: "Coordonnées ou rayon invalides.", details: parsed.error.flatten() });

  const { latitude, longitude, radius, confidenceThreshold, minAreaHa, baseTemperature, threshold, periodDays } = parsed.data;
  const gddConfig = baseTemperature != null && threshold != null && periodDays != null
    ? { baseTemperature, threshold, periodDays }
    : undefined;
  const result = await analyzeFieldsSimple({ lat: latitude, lng: longitude, radiusM: radius, confidenceThreshold, minAreaHa, gddConfig });

  // Enregistre chaque parcelle ORGE détectée dans le registre (table `parcelles`), sinon elle
  // ne survit qu'en mémoire côté frontend et disparaît au rechargement / n'apparaît jamais dans
  // le tableau de bord. Best-effort : un échec ne doit pas faire perdre le résultat déjà calculé.
  if (result.features.length > 0) {
    await saveSimpleFieldParcelles(result.features, result.warnings);
    parcellesCache = null;
  }

  // La persistance ne doit jamais faire perdre une analyse déjà calculée (appels GEE + CNN coûteux) :
  // un incident DB transitoire devient un avertissement, pas un 500 qui jette tout le résultat.
  try {
    const run = await prisma.fieldScanRun.create({
      data: {
        version: FIELD_SCAN_VERSION,
        center_lat: latitude,
        center_lng: longitude,
        radius_m: radius,
        image_date: result.imageDate,
        image_age_days: result.imageAgeDays,
        cloud_percentage: result.cloudPercentage,
        confidence_threshold: result.confidenceThreshold,
        min_area_ha: result.minAreaHa,
        result_geojson: jsonValue(result),
        candidates_found: result.candidatesFound,
        candidates_kept: result.candidatesClassified,
        warnings: jsonValue(result.warnings),
      },
    });
    return reply.send({ analysisId: run.id, analysisDate: run.created_at, ...result });
  } catch (error) {
    app.log.warn({ err: error }, "analyze: persistence failed, returning result without analysisId");
    result.warnings.push("Résultat non sauvegardé : la base de données est temporairement indisponible.");
    return reply.send({ analysisId: null, analysisDate: new Date().toISOString(), ...result });
  }
});

app.get("/api/analyze/latest", async (request, reply) => {
  const query = z.object({
    lat: z.coerce.number().finite().min(-90).max(90),
    lng: z.coerce.number().finite().min(-180).max(180),
  }).safeParse(request.query);
  if (!query.success) return reply.code(400).send({ error: "Coordonnées invalides.", details: query.error.flatten() });

  const { lat, lng } = query.data;
  const run = await prisma.fieldScanRun.findFirst({
    where: {
      version: FIELD_SCAN_VERSION,
      center_lat: { gte: lat - FIELD_SCAN_PROXIMITY_DEGREES, lte: lat + FIELD_SCAN_PROXIMITY_DEGREES },
      center_lng: { gte: lng - FIELD_SCAN_PROXIMITY_DEGREES, lte: lng + FIELD_SCAN_PROXIMITY_DEGREES },
    },
    orderBy: { created_at: "desc" },
  });
  if (!run) return reply.code(404).send({ error: "Aucune analyse enregistrée pour cette zone." });

  return reply.send({ analysisId: run.id, analysisDate: run.created_at, ...(run.result_geojson as Record<string, unknown>) });
});

const sentinelTileParamsSchema = z.object({
  z: z.coerce.number().int().min(10).max(19),
  x: z.coerce.number().int().min(0),
  y: z.coerce.number().int().min(0),
});
const sentinelTileQuerySchema = z.object({
  imageTimestampMs: z.coerce.number().finite().positive(),
});

app.get("/api/sentinel-tiles/:z/:x/:y", async (request, reply) => {
  const params = sentinelTileParamsSchema.safeParse(request.params);
  const query = sentinelTileQuerySchema.safeParse(request.query);
  if (!params.success || !query.success) {
    return reply.code(400).send({ error: "Coordonnées de tuile ou timestamp d'image invalides." });
  }

  try {
    const accessToken = await getGeeAccessToken();
    const projectId = getGeeProjectId();
    const pngBytes = await fetchSentinel2TilePng(accessToken, projectId, params.data, query.data.imageTimestampMs);
    reply.header("Cache-Control", "public, max-age=3600");
    if (!pngBytes) return reply.type("image/png").send(TRANSPARENT_PNG);
    return reply.type("image/png").send(Buffer.from(pngBytes));
  } catch (error) {
    app.log.warn({ err: error }, "sentinel-tiles: tuile indisponible");
    reply.header("Cache-Control", "public, max-age=60");
    return reply.type("image/png").send(TRANSPARENT_PNG);
  }
});

// Fichier .nc/.npy en base64 : un patch 256×256×30 canaux float32 pèse ~10.5 Mo encodé,
// bien au-delà du bodyLimit JSON par défaut de Fastify (1 Mo) — on l'augmente pour cette route.
app.post("/api/field-segmentation", { bodyLimit: 20 * 1024 * 1024 }, async (request, reply) => {
  const parsed = fieldSegmentationSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: "Fichier ou paramètres invalides.", details: parsed.error.flatten() });

  const { fileBase64, filename, threshold } = parsed.data;
  let fileBytes: Buffer;
  try {
    fileBytes = Buffer.from(fileBase64, "base64");
  } catch {
    return reply.code(400).send({ error: "fileBase64 invalide (attendu : encodage base64)." });
  }

  try {
    const result = await callFieldSegmentationModel(fileBytes, filename, threshold ?? 0.5);
    return reply.send(result);
  } catch (error) {
    app.log.warn({ err: error }, "field-segmentation: modèle indisponible");
    const message = error instanceof Error ? error.message : "Modèle de segmentation indisponible.";
    return reply.code(502).send({ error: message });
  }
});

const port = Number(process.env.PORT ?? 3001);
const host = process.env.HOST ?? "0.0.0.0";

app.listen({ port, host }).catch(async (error) => {
  app.log.error(error);
  await prisma.$disconnect();
  process.exit(1);
});

// Préchauffe la connexion Neon et le cache au démarrage : sans ça, c'est la première
// requête utilisateur qui paie le coût de connexion initial (plusieurs secondes ici).
loadParcelles().catch((error) => {
  app.log.warn({ err: error }, "parcelles: préchauffage initial échoué, réessai à la première requête");
});
