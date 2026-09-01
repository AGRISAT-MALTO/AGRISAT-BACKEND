import "dotenv/config";
import dns from "node:dns";
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Prisma } from "@prisma/client";
import { z } from "zod";

import {
  analyzeParcel,
  getGeeAccessToken,
  getGeeProjectId,
} from "./analyze-parcel.js";

import { detectAutomaticParcels } from "./automatic-parcels.js";

import {
  analyzeFieldsSimple,
  saveSimpleFieldParcelles,
} from "./barley-detect-simple.js";

import { fetchSentinel2TilePng } from "./sentinel-tiles.js";

import { computeZoning } from "./zoning.js";

import { callFieldSegmentationModel } from "./field-segmentation.js";

import planetRoutes from "./routes/planet.routes.js";

import { prisma } from "./db.js";


// ============================================================
// DNS
// ============================================================

dns.setDefaultResultOrder("ipv4first");


// ============================================================
// SERVER
// ============================================================

const app = Fastify({
  logger: true,
});


// ============================================================
// PLANET CONFIGURATION
// ============================================================

const PLANET_API_KEY = process.env.PLANET_API_KEY;

if (!PLANET_API_KEY) {
  app.log.warn(
    "PLANET_API_KEY n'est pas configurée dans le fichier .env"
  );
}


// ============================================================
// CORS
// ============================================================

app.register(cors, {
  origin: process.env.FRONTEND_ORIGIN?.split(",") ?? true,
});

app.register(planetRoutes);


// ============================================================
// CONSTANTS
// ============================================================

const TRANSPARENT_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);


// ============================================================
// HELPERS
// ============================================================

const jsonValue = (value: unknown) =>
  value as Prisma.InputJsonValue;

type ParcelleRow =
  Awaited<
    ReturnType<typeof prisma.parcelle.findMany>
  >[number];


// ============================================================
// PARCELLES CACHE
// ============================================================

const PARCELLES_CACHE_TTL_MS = 30_000;
const PARCELLES_QUERY_TIMEOUT_MS = 20_000;

let parcellesCache:
  | {
      expiresAt: number;
      rows: ParcelleRow[];
    }
  | null = null;

let parcellesRequest:
  | Promise<ParcelleRow[]>
  | null = null;


// ============================================================
// LOAD PARCELLES
// ============================================================

async function loadParcelles(): Promise<ParcelleRow[]> {
  if (parcellesCache) {
    if (parcellesCache.expiresAt <= Date.now()) {
      refreshParcellesInBackground();
    }

    return parcellesCache.rows;
  }

  return fetchParcelles();
}


// ============================================================
// BACKGROUND REFRESH
// ============================================================

function refreshParcellesInBackground(): void {
  if (parcellesRequest) return;

  fetchParcelles().catch((error) => {
    app.log.warn(
      {
        err: error,
      },
      "parcelles: background refresh failed, keeping stale cache"
    );
  });
}


// ============================================================
// FETCH PARCELLES
// ============================================================

async function fetchParcelles(): Promise<ParcelleRow[]> {
  if (!parcellesRequest) {
    parcellesRequest = withTimeout(
      prisma.parcelle.findMany({
        orderBy: {
          created_at: "desc",
        },
      }),
      PARCELLES_QUERY_TIMEOUT_MS
    );
  }

  try {
    const rows = await parcellesRequest;

    parcellesCache = {
      rows,
      expiresAt:
        Date.now() + PARCELLES_CACHE_TTL_MS,
    };

    return rows;
  } finally {
    parcellesRequest = null;
  }
}


// ============================================================
// TIMEOUT
// ============================================================

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(
        new Error(
          `Requête base de données dépassant ${timeoutMs} ms.`
        )
      );
    }, timeoutMs);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}


// ============================================================
// SCHEMAS
// ============================================================

const pointSchema = z.object({
  lat: z.number(),
  lng: z.number(),
});


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

  spectral_bands:
    z.record(z.string(), z.number().nullable()).nullable(),

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
  time_series_rain: z.array(z.unknown()),

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


// ============================================================
// ERROR HANDLER
// ============================================================

app.setErrorHandler(
  (error, _request, reply) => {
    app.log.error(error);

    const statusCode =
      typeof error === "object" &&
      error !== null &&
      "statusCode" in error &&
      typeof error.statusCode === "number" &&
      error.statusCode < 500
        ? error.statusCode
        : 500;

    const message =
      error instanceof Error
        ? error.message
        : "Erreur interne du serveur";

    return reply
      .code(statusCode)
      .send({
        error:
          statusCode === 500
            ? "Erreur interne du serveur"
            : message,
      });
  }
);


// ============================================================
// HEALTH
// ============================================================

app.get("/health", async () => ({
  status: "ok",
}));


// ============================================================
// ============================================================
// PLANET
// ============================================================
// ============================================================


// ------------------------------------------------------------
// PLANET STATUS
// ------------------------------------------------------------

app.get("/api/planet/status", async (_request, reply) => {
  if (!PLANET_API_KEY) {
    return reply.code(500).send({
      success: false,
      configured: false,
      message:
        "PLANET_API_KEY est absente du fichier .env",
    });
  }

  return reply.send({
    success: true,
    configured: true,
    provider: "Planet",
    message: "Planet API Key configurée.",
  });
});


// ------------------------------------------------------------
// PLANET SERIES / MOSAICS
// ------------------------------------------------------------

app.get(
  "/api/planet/series",
  async (_request, reply) => {
    if (!PLANET_API_KEY) {
      return reply.code(500).send({
        success: false,
        error: "PLANET_API_KEY non configurée.",
      });
    }

    try {
      const response = await fetch(
        "https://api.planet.com/basemaps/v1/series/",
        {
          headers: {
            Authorization:
              `api-key ${PLANET_API_KEY}`,

            Accept: "application/json",
          },
        }
      );

      const text = await response.text();

      if (!response.ok) {
        app.log.error(
          {
            status: response.status,
            body: text,
          },
          "Planet series request failed"
        );

        return reply
          .code(response.status)
          .send({
            success: false,
            error: "Planet API error",
            details: text,
          });
      }

      return reply
        .type("application/json")
        .send(JSON.parse(text));
    } catch (error) {
      app.log.error(
        {
          err: error,
        },
        "Planet series request failed"
      );

      return reply.code(502).send({
        success: false,
        error:
          "Impossible de contacter Planet API.",
      });
    }
  }
);


// ============================================================
// AUTOMATIC PARCEL DETECTION
// ============================================================

app.post(
  "/api/detect-parcels-fallback",
  async (request, reply) => {
    const parsed =
      automaticDetectionSchema.safeParse(
        request.body
      );

    if (!parsed.success) {
      return reply.code(400).send({
        error:
          "Coordonnées ou rayon invalides.",
        details:
          parsed.error.flatten(),
      });
    }

    const result =
      await detectAutomaticParcels(
        parsed.data
      );

    parcellesCache = null;

    return reply.send(result);
  }
);


// ============================================================
// GET PARCELLES
// ============================================================

app.get(
  "/api/parcelles",
  async (_request, reply) => {
    try {
      return reply.send(
        await loadParcelles()
      );
    } catch (error) {
      app.log.error(
        {
          err: error,
        },
        "parcelles: read failed"
      );

      return reply.code(503).send({
        error:
          "Les parcelles sont temporairement indisponibles.",
      });
    }
  }
);


// ============================================================
// CREATE PARCELLE
// ============================================================

app.post(
  "/api/parcelles",
  async (request, reply) => {
    const parsed =
      parcelleSchema.safeParse(
        request.body
      );

    if (!parsed.success) {
      return reply.code(400).send({
        error:
          "Données de parcelle invalides",
        details:
          parsed.error.flatten(),
      });
    }


    const {
      coordinates,
      risk_factors,
      time_series_s1,
      time_series_s2,
      time_series_rain,
      spectral_bands,
      ...scalarData
    } = parsed.data;


    const parcelle =
      await prisma.parcelle.create({
        data: {
          ...scalarData,

          coordinates:
            jsonValue(coordinates),

          risk_factors:
            jsonValue(risk_factors),

          time_series_s1:
            jsonValue(time_series_s1),

          time_series_s2:
            jsonValue(time_series_s2),

          time_series_rain:
            jsonValue(time_series_rain),

          spectral_bands:
            spectral_bands === null
              ? Prisma.JsonNull
              : jsonValue(
                  spectral_bands
                ),
        },
      });


    parcellesCache = null;

    return reply
      .code(201)
      .send(parcelle);
  }
);


// ============================================================
// DELETE PARCELLE
// ============================================================

app.delete(
  "/api/parcelles/:id",
  async (request, reply) => {
    const id =
      z.string()
        .uuid()
        .safeParse(
          (request.params as { id: string })
            .id
        );


    if (!id.success) {
      return reply.code(400).send({
        error:
          "Identifiant invalide",
      });
    }


    const result =
      await prisma.parcelle.deleteMany({
        where: {
          id: id.data,
        },
      });


    if (result.count === 0) {
      return reply.code(404).send({
        error:
          "Parcelle introuvable",
      });
    }


    parcellesCache = null;

    return reply
      .code(204)
      .send();
  }
);


// ============================================================
// ANALYZE PARCEL
// ============================================================

app.post(
  "/api/analyze-parcel",
  async (request, reply) => {
    const response =
      await analyzeParcel(
        new Request(
          "http://backend/analyze-parcel",
          {
            method: "POST",

            headers: {
              "content-type":
                "application/json",
            },

            body: JSON.stringify(
              request.body
            ),
          }
        )
      );


    return reply
      .code(response.status)
      .headers(
        Object.fromEntries(
          response.headers.entries()
        )
      )
      .send(
        await response.text()
      );
  }
);


// ============================================================
// ZONING (VRA)
// ============================================================

app.post(
  "/api/zoning-parcel",
  async (request, reply) => {
    const response = await computeZoning(
      new Request("http://backend/zoning-parcel", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request.body),
      })
    );

    return reply
      .code(response.status)
      .headers(Object.fromEntries(response.headers.entries()))
      .send(await response.text());
  }
);


// ============================================================
// DETECT PARCELS
// ============================================================

app.post(
  "/api/detect-parcels",
  async (request, reply) => {
    const parsed =
      automaticDetectionSchema.safeParse(
        request.body
      );


    if (!parsed.success) {
      return reply.code(400).send({
        error:
          "Coordonnées ou rayon invalides.",
        details:
          parsed.error.flatten(),
      });
    }


    try {
      const result =
        await detectAutomaticParcels(
          parsed.data
        );


      parcellesCache = null;

      return reply.send(result);

    } catch (error) {
      app.log.warn(
        {
          err: error,
        },
        "detect-parcels: discovery failed, returning fallback notice"
      );


      const msg =
        error instanceof Error
          ? error.message
          : "Le service de recherche des parcelles est indisponible.";


      const fallback = {
        center: {
          lat: parsed.data.lat,
          lng: parsed.data.lng,
        },

        radius_km:
          parsed.data.radiusKm,

        base_temperature:
          parsed.data.baseTemperature,

        threshold:
          parsed.data.threshold,

        period_days:
          parsed.data.periodDays,

        candidates_found: 0,
        analyzed_count: 0,

        parcels: [],

        notice:
          msg.includes("database") ||
          msg.includes("Database") ||
          msg
            .toLowerCase()
            .includes("parcelles")
            ? "La base des parcelles agricoles est indisponible. Essayez un rayon plus large ou vérifiez la configuration de la base de données."
            : msg,
      };


      return reply.send(
        fallback
      );
    }
  }
);


// ============================================================
// SIMPLE FIELD ANALYSIS
// ============================================================

const FIELD_SCAN_VERSION =
  "v1-simple";

const FIELD_SCAN_PROXIMITY_DEGREES =
  0.01;


app.post(
  "/api/analyze",
  async (request, reply) => {
    const parsed =
      analyzeSimpleSchema.safeParse(
        request.body
      );


    if (!parsed.success) {
      return reply.code(400).send({
        error:
          "Coordonnées ou rayon invalides.",
        details:
          parsed.error.flatten(),
      });
    }


    const {
      latitude,
      longitude,
      radius,
      confidenceThreshold,
      minAreaHa,
      baseTemperature,
      threshold,
      periodDays,
    } = parsed.data;


    const gddConfig =
      baseTemperature != null &&
      threshold != null &&
      periodDays != null
        ? {
            baseTemperature,
            threshold,
            periodDays,
          }
        : undefined;


    const result =
      await analyzeFieldsSimple({
        lat: latitude,
        lng: longitude,
        radiusM: radius,
        confidenceThreshold,
        minAreaHa,
        gddConfig,
      });


    if (
      result.features.length > 0
    ) {
      await saveSimpleFieldParcelles(
        result.features,
        result.warnings
      );

      parcellesCache = null;
    }


    try {
      const run =
        await prisma.fieldScanRun.create({
          data: {
            version:
              FIELD_SCAN_VERSION,

            center_lat:
              latitude,

            center_lng:
              longitude,

            radius_m:
              radius,

            image_date:
              result.imageDate,

            image_age_days:
              result.imageAgeDays,

            cloud_percentage:
              result.cloudPercentage,

            confidence_threshold:
              result.confidenceThreshold,

            min_area_ha:
              result.minAreaHa,

            result_geojson:
              jsonValue(result),

            candidates_found:
              result.candidatesFound,

            candidates_kept:
              result.candidatesClassified,

            warnings:
              jsonValue(
                result.warnings
              ),
          },
        });


      return reply.send({
        analysisId:
          run.id,

        analysisDate:
          run.created_at,

        ...result,
      });

    } catch (error) {
      app.log.warn(
        {
          err: error,
        },
        "analyze: persistence failed, returning result without analysisId"
      );


      result.warnings.push(
        "Résultat non sauvegardé : la base de données est temporairement indisponible."
      );


      return reply.send({
        analysisId: null,

        analysisDate:
          new Date().toISOString(),

        ...result,
      });
    }
  }
);


// ============================================================
// LAST ANALYSIS
// ============================================================

app.get(
  "/api/analyze/latest",
  async (request, reply) => {
    const query =
      z.object({
        lat: z.coerce
          .number()
          .finite()
          .min(-90)
          .max(90),

        lng: z.coerce
          .number()
          .finite()
          .min(-180)
          .max(180),
      })
      .safeParse(
        request.query
      );


    if (!query.success) {
      return reply.code(400).send({
        error:
          "Coordonnées invalides.",
        details:
          query.error.flatten(),
      });
    }


    const {
      lat,
      lng,
    } = query.data;


    const run =
      await prisma.fieldScanRun.findFirst({
        where: {
          version:
            FIELD_SCAN_VERSION,

          center_lat: {
            gte:
              lat -
              FIELD_SCAN_PROXIMITY_DEGREES,

            lte:
              lat +
              FIELD_SCAN_PROXIMITY_DEGREES,
          },

          center_lng: {
            gte:
              lng -
              FIELD_SCAN_PROXIMITY_DEGREES,

            lte:
              lng +
              FIELD_SCAN_PROXIMITY_DEGREES,
          },
        },

        orderBy: {
          created_at: "desc",
        },
      });


    if (!run) {
      return reply.code(404).send({
        error:
          "Aucune analyse enregistrée pour cette zone.",
      });
    }


    return reply.send({
      analysisId:
        run.id,

      analysisDate:
        run.created_at,

      ...(run.result_geojson as Record<string, unknown>),
    });
  }
);


// ============================================================
// SENTINEL-2 TILE
// ============================================================

const sentinelTileParamsSchema =
  z.object({
    z: z.coerce
      .number()
      .int()
      .min(10)
      .max(19),

    x: z.coerce
      .number()
      .int()
      .min(0),

    y: z.coerce
      .number()
      .int()
      .min(0),
  });


const sentinelTileQuerySchema =
  z.object({
    imageTimestampMs:
      z.coerce
        .number()
        .finite()
        .positive(),
  });


app.get(
  "/api/sentinel-tiles/:z/:x/:y",
  async (request, reply) => {
    const params =
      sentinelTileParamsSchema.safeParse(
        request.params
      );

    const query =
      sentinelTileQuerySchema.safeParse(
        request.query
      );


    if (
      !params.success ||
      !query.success
    ) {
      return reply.code(400).send({
        error:
          "Coordonnées de tuile ou timestamp d'image invalides.",
      });
    }


    try {
      const accessToken =
        await getGeeAccessToken();

      const projectId =
        getGeeProjectId();

      const pngBytes =
        await fetchSentinel2TilePng(
          accessToken,
          projectId,
          params.data,
          query.data
            .imageTimestampMs
        );


      reply.header(
        "Cache-Control",
        "public, max-age=3600"
      );


      if (!pngBytes) {
        return reply
          .type("image/png")
          .send(
            TRANSPARENT_PNG
          );
      }


      return reply
        .type("image/png")
        .send(
          Buffer.from(
            pngBytes
          )
        );

    } catch (error) {
      app.log.warn(
        {
          err: error,
        },
        "sentinel-tiles: tuile indisponible"
      );


      reply.header(
        "Cache-Control",
        "public, max-age=60"
      );


      return reply
        .type("image/png")
        .send(
          TRANSPARENT_PNG
        );
    }
  }
);


// ============================================================
// FIELD SEGMENTATION
// ============================================================

const FIELD_SEGMENTATION_BODY_LIMIT =
  20 * 1024 * 1024;


app.post(
  "/api/field-segmentation",
  {
    bodyLimit:
      FIELD_SEGMENTATION_BODY_LIMIT,
  },

  async (request, reply) => {
    const parsed =
      fieldSegmentationSchema.safeParse(
        request.body
      );


    if (!parsed.success) {
      return reply.code(400).send({
        error:
          "Fichier ou paramètres invalides.",
        details:
          parsed.error.flatten(),
      });
    }


    const {
      fileBase64,
      filename,
      threshold,
    } = parsed.data;


    let fileBytes: Buffer;


    try {
      fileBytes =
        Buffer.from(
          fileBase64,
          "base64"
        );

    } catch {
      return reply.code(400).send({
        error:
          "fileBase64 invalide (attendu : encodage base64).",
      });
    }


    try {
      const result =
        await callFieldSegmentationModel(
          fileBytes,
          filename,
          threshold ?? 0.5
        );


      return reply.send(
        result
      );

    } catch (error) {
      app.log.warn(
        {
          err: error,
        },
        "field-segmentation: modèle indisponible"
      );


      const message =
        error instanceof Error
          ? error.message
          : "Modèle de segmentation indisponible.";


      return reply.code(502).send({
        error: message,
      });
    }
  }
);


// ============================================================
// SERVER START
// ============================================================

const port =
  Number(
    process.env.PORT ?? 3001
  );

const host =
  process.env.HOST ??
  "0.0.0.0";


app.listen({
  port,
  host,
})
  .catch(
    async (error) => {
      app.log.error(error);

      await prisma.$disconnect();

      process.exit(1);
    }
  );


// ============================================================
// DATABASE WARMUP
// ============================================================

loadParcelles()
  .catch((error) => {
    app.log.warn(
      {
        err: error,
      },
      "parcelles: préchauffage initial échoué, réessai à la première requête"
    );
  });