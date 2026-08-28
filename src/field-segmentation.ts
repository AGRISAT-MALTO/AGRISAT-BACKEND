import { fetchWithRetry } from "./analyze-parcel.js";

const FIELD_SEGMENTATION_MODEL_URL = process.env.FIELD_SEGMENTATION_MODEL_URL ?? "http://localhost:8000";
const EXTERNAL_REQUEST_TIMEOUT_MS = 30_000;

export interface FieldSegmentationResult {
  threshold: number;
  fieldPercentage: number;
  maskShape: number[];
  maskPngBase64: string;
  probabilityPngBase64: string;
}


export async function callFieldSegmentationModel(
  fileBytes: Buffer,
  filename: string,
  threshold = 0.5,
): Promise<FieldSegmentationResult> {
  const formData = new FormData();

  formData.append("file", new Blob([new Uint8Array(fileBytes)]), filename);

  const url = `${FIELD_SEGMENTATION_MODEL_URL}/predict?threshold=${threshold}`;
  console.log("Calling field segmentation model /predict at:", url);

  const resp = await fetchWithRetry(url, { method: "POST", body: formData }, EXTERNAL_REQUEST_TIMEOUT_MS);

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "");
    console.error("Field segmentation /predict error:", resp.status, errText.slice(0, 500));
    throw new Error(`Modèle de segmentation indisponible : HTTP ${resp.status}`);
  }

  const data: unknown = await resp.json();
  if (!data || typeof data !== "object") throw new Error("Réponse du modèle de segmentation invalide.");

  const values = data as Record<string, unknown>;
  if (
    typeof values.field_percentage !== "number"
    || !Number.isFinite(values.field_percentage)
    || !Array.isArray(values.mask_shape)
    || typeof values.mask_png_base64 !== "string"
    || typeof values.probability_png_base64 !== "string"
  ) {
    throw new Error("La réponse du modèle de segmentation ne contient pas les valeurs attendues.");
  }

  return {
    threshold: typeof values.threshold === "number" ? values.threshold : threshold,
    fieldPercentage: values.field_percentage,
    maskShape: values.mask_shape as number[],
    maskPngBase64: values.mask_png_base64,
    probabilityPngBase64: values.probability_png_base64,
  };
}
