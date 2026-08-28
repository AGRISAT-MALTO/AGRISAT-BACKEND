import { PrismaClient } from "@prisma/client";

// Instance unique partagée par tout le backend : chaque `new PrismaClient()` ouvre son
// propre pool de connexions vers Neon, et en avoir plusieurs (server/automatic-parcels/
// barley-detect-simple) épuisait le pool disponible plus vite qu'un seul client partagé.
export const prisma = new PrismaClient();
