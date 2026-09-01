import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { getPlanetTile } from "../services/planet.service.js";

const planetTileParamsSchema = z.object({
  z: z.coerce.number().int().min(0).max(22),
  x: z.coerce.number().int().min(0),
  y: z.coerce.number().int().min(0),
});

const planetTileQuerySchema = z.object({
  mosaic: z.string().min(1).max(300).optional(),
});

export default async function planetRoutes(
  fastify: FastifyInstance
) {
  fastify.get(
    "/api/planet/tiles/:z/:x/:y.png",
    async (request, reply) => {
      const params = planetTileParamsSchema.safeParse(request.params);
      const query = planetTileQuerySchema.safeParse(request.query);

      if (!params.success || !query.success) {
        return reply.code(400).send({
          success: false,
          error: "Paramètres de tuile Planet invalides.",
        });
      }

      const { z, x, y } = params.data;

      const maxTile = Math.pow(2, z);

      if (x >= maxTile || y >= maxTile) {
        return reply.code(400).send({
          success: false,
          error: "Coordonnées XYZ invalides.",
        });
      }

      const mosaicName =
        query.data.mosaic ||
        "planet_medres_normalized_analytic_2024-07_mosaic";

      try {
        const tile = await getPlanetTile(z, x, y, mosaicName);

        return reply
          .header("Content-Type", tile.contentType)
          .header("Cache-Control", "public, max-age=3600")
          .send(tile.buffer);
      } catch (error) {
        fastify.log.error(error);

        return reply
          .code(502)
          .send({
            success: false,
            message: "Impossible de récupérer la tuile Planet",
          });
      }
    }
  );
}
