/**
 * Segmentation de limites de parcelles par watershed marqué, sur une image de
 * "force de frontière" combinant :
 *   - le gradient spatial du NDVI médian de la saison (les vraies limites de
 *     champs correspondent à des sauts brusques de NDVI)
 *   - l'écart-type temporel du NDVI sur plusieurs sous-périodes de la saison
 *     (deux parcelles adjacentes ont souvent des calendriers de culture
 *     différents, donc une variabilité temporelle différente, même quand
 *     elles se ressemblent à un instant donné)
 *
 * Aucun modèle de machine learning : uniquement des opérations GEE classiques
 * (gradient, reduce, stdDev) + un algorithme de watershed "priority-flood"
 * implémenté ici en pur TypeScript, + un traçage de contours (Moore-neighbor
 * tracing) et une simplification Douglas-Peucker.
 */

// ── Web Mercator (mètres) : conversion lat/lng <-> mètres EPSG:3857 ──
// (utilisé comme grille pour image:computePixels, car les distances y sont en
// mètres, contrairement à EPSG:4326 en degrés)
const EARTH_RADIUS_M = 6_378_137;

export function lngLatToMercatorMeters(lng: number, lat: number): { x: number; y: number } {
  const x = (lng * Math.PI * EARTH_RADIUS_M) / 180;
  const clampedLat = Math.min(Math.max(lat, -85.05112878), 85.05112878);
  const y = EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + (clampedLat * Math.PI) / 360));
  return { x, y };
}

export function mercatorMetersToLngLat(x: number, y: number): { lng: number; lat: number } {
  const lng = (x / (Math.PI * EARTH_RADIUS_M)) * 180;
  const lat = ((2 * Math.atan(Math.exp(y / EARTH_RADIUS_M)) - Math.PI / 2) * 180) / Math.PI;
  return { lng, lat };
}

// ── Parsing minimal du format .npy (tableau 2D float32, C-order) ──

function readNpyHeader(buffer: ArrayBuffer): { headerText: string; dataStart: number } {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(1, 6));
  if (magic !== "NUMPY") throw new Error("Format .npy invalide (magic manquant).");

  const majorVersion = bytes[6];
  const headerLenBytes = majorVersion >= 2 ? 4 : 2;
  const headerLenOffset = 8;
  const headerLen = majorVersion >= 2
    ? new DataView(buffer, headerLenOffset, 4).getUint32(0, true)
    : new DataView(buffer, headerLenOffset, 2).getUint16(0, true);
  const headerStart = headerLenOffset + headerLenBytes;
  const headerText = String.fromCharCode(...bytes.slice(headerStart, headerStart + headerLen));
  return { headerText, dataStart: headerStart + headerLen };
}

export function parseNpyFloat32(buffer: ArrayBuffer): { data: Float32Array; shape: number[] } {
  const { headerText, dataStart } = readNpyHeader(buffer);

  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  if (!shapeMatch || !descrMatch) throw new Error("En-tête .npy illisible.");
  const shape = shapeMatch[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const descr = descrMatch[1];
  if (!/f4$/.test(descr)) throw new Error(`Type .npy non supporté : ${descr} (float32 attendu).`);
  const littleEndian = !descr.startsWith(">");

  const count = shape.reduce((a, b) => a * b, 1);
  const view = new DataView(buffer, dataStart, count * 4);
  const data = new Float32Array(count);
  for (let i = 0; i < count; i++) data[i] = view.getFloat32(i * 4, littleEndian);
  return { data, shape };
}

/**
 * Parse un .npy "structuré" (dtype = liste de champs), tel que renvoyé par
 * image:computePixels quand plusieurs bandIds sont demandés en un seul appel :
 * chaque pixel est un enregistrement contenant une valeur par bande, dans l'ordre
 * du dtype — pas un tableau plan par bande (BSQ), mais entrelacé par pixel (BIP).
 * Accepte f4 (float32) et f8 (float64) par champ — GEE renvoie f8 pour un export
 * multi-bandes même quand les bandes sources sont castées en float32 côté serveur.
 * Retourne un Float32Array par bande, déjà dé-entrelacé (une bande = un tableau plat).
 */
export function parseNpyStructuredFloat32(buffer: ArrayBuffer): { bands: Record<string, Float32Array>; shape: number[] } {
  const { headerText, dataStart } = readNpyHeader(buffer);

  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error("En-tête .npy illisible (shape manquant).");
  const shape = shapeMatch[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);

  const fieldMatches = [...headerText.matchAll(/\('([^']+)',\s*'([^']+)'\)/g)];
  if (fieldMatches.length === 0) throw new Error("En-tête .npy illisible (dtype structuré manquant).");

  let offset = 0;
  const fields = fieldMatches.map(([, name, descr]) => {
    const byteSize = descr.endsWith("f4") ? 4 : descr.endsWith("f8") ? 8 : null;
    if (byteSize === null) throw new Error(`Type .npy non supporté pour '${name}' : ${descr} (float32/float64 attendu).`);
    const field = { name, byteSize, littleEndian: !descr.startsWith(">"), offset };
    offset += byteSize;
    return field;
  });
  const recordSize = offset;

  const count = shape.reduce((a, b) => a * b, 1);
  const view = new DataView(buffer, dataStart, count * recordSize);

  const bands: Record<string, Float32Array> = {};
  for (const field of fields) bands[field.name] = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const recordOffset = i * recordSize;
    for (const field of fields) {
      const fieldOffset = recordOffset + field.offset;
      bands[field.name][i] = field.byteSize === 4
        ? view.getFloat32(fieldOffset, field.littleEndian)
        : view.getFloat64(fieldOffset, field.littleEndian);
    }
  }

  return { bands, shape };
}

/** Encode un tableau float32 (C-order) en bytes .npy — symétrique de parseNpyFloat32. */
export function encodeNpyFloat32(data: Float32Array, shape: number[]): Uint8Array {
  const magic = "\x93NUMPY";
  const version = new Uint8Array([1, 0]); // v1.0
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;

  // L'en-tête total (magic + version + longueur + dict) doit être un multiple de 64 octets,
  // complété par des espaces puis un '\n' final — convention du format .npy.
  const unpaddedLen = magic.length + version.length + 2 + dict.length + 1;
  const padLen = (64 - (unpaddedLen % 64)) % 64;
  const header = dict + " ".repeat(padLen) + "\n";
  const headerLen = header.length;

  const buffer = new ArrayBuffer(magic.length + version.length + 2 + headerLen + data.length * 4);
  const bytes = new Uint8Array(buffer);
  let offset = 0;
  for (let i = 0; i < magic.length; i++) bytes[offset++] = magic.charCodeAt(i);
  bytes[offset++] = version[0];
  bytes[offset++] = version[1];
  new DataView(buffer, offset, 2).setUint16(0, headerLen, true);
  offset += 2;
  for (let i = 0; i < header.length; i++) bytes[offset++] = header.charCodeAt(i);

  const dataView = new DataView(buffer, offset, data.length * 4);
  for (let i = 0; i < data.length; i++) dataView.setFloat32(i * 4, data[i], true);

  return bytes;
}

// ── Watershed marqué (priority-flood / immersion à la Vincent-Soille) ──

export interface WatershedOptions {
  /** Percentile (0-1) sous lequel un pixel est candidat "germe" (fond de bassin). */
  seedPercentile?: number;
  /** Taille minimale (en pixels) d'une composante connexe pour être un germe valide. */
  minSeedPixels?: number;
}

class MinHeap<T> {
  private items: Array<{ priority: number; value: T }> = [];
  push(priority: number, value: T) {
    this.items.push({ priority, value });
    let i = this.items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.items[parent].priority <= this.items[i].priority) break;
      [this.items[parent], this.items[i]] = [this.items[i], this.items[parent]];
      i = parent;
    }
  }
  pop(): T | undefined {
    if (this.items.length === 0) return undefined;
    const top = this.items[0];
    const last = this.items.pop()!;
    if (this.items.length > 0) {
      this.items[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let smallest = i;
        if (left < this.items.length && this.items[left].priority < this.items[smallest].priority) smallest = left;
        if (right < this.items.length && this.items[right].priority < this.items[smallest].priority) smallest = right;
        if (smallest === i) break;
        [this.items[smallest], this.items[i]] = [this.items[i], this.items[smallest]];
        i = smallest;
      }
    }
    return top.value;
  }
  get size() {
    return this.items.length;
  }
}

/**
 * @param strength Force de frontière par pixel (plus haut = plus proche d'une limite réelle).
 * @param barrier  true = pixel non cultivable (route, eau, bâti...) : jamais inondé, agit comme séparateur naturel.
 * @returns Un tableau de labels (0 = non affecté/barrière, >=1 = identifiant de parcelle).
 */
export function watershedSegment(
  strength: Float32Array,
  barrier: Uint8Array,
  width: number,
  height: number,
  options: WatershedOptions = {},
): Int32Array {
  const seedPercentile = options.seedPercentile ?? 0.2;
  const minSeedPixels = options.minSeedPixels ?? 6;

  const validValues: number[] = [];
  for (let i = 0; i < strength.length; i++) if (!barrier[i]) validValues.push(strength[i]);
  if (validValues.length === 0) return new Int32Array(strength.length);
  validValues.sort((a, b) => a - b);
  const threshold = validValues[Math.floor(validValues.length * seedPercentile)];

  const labels = new Int32Array(strength.length); // 0 = non labellisé
  const isSeedCandidate = (i: number) => !barrier[i] && strength[i] <= threshold;

  // Étiquetage des composantes connexes (4-connexité) parmi les candidats germes.
  let nextLabel = 0;
  const stack: number[] = [];
  for (let start = 0; start < strength.length; start++) {
    if (labels[start] !== 0 || !isSeedCandidate(start)) continue;
    nextLabel++;
    const componentPixels: number[] = [];
    stack.push(start);
    labels[start] = nextLabel;
    while (stack.length > 0) {
      const idx = stack.pop()!;
      componentPixels.push(idx);
      const x = idx % width;
      const y = (idx / width) | 0;
      const neighbors = [
        x > 0 ? idx - 1 : -1,
        x < width - 1 ? idx + 1 : -1,
        y > 0 ? idx - width : -1,
        y < height - 1 ? idx + width : -1,
      ];
      for (const n of neighbors) {
        if (n >= 0 && labels[n] === 0 && isSeedCandidate(n)) {
          labels[n] = nextLabel;
          stack.push(n);
        }
      }
    }
    if (componentPixels.length < minSeedPixels) {
      for (const idx of componentPixels) labels[idx] = 0;
      nextLabel--;
    }
  }

  // Inondation par priorité (priority-flood) depuis chaque germe.
  const heap = new MinHeap<number>();
  for (let idx = 0; idx < labels.length; idx++) {
    if (labels[idx] <= 0) continue;
    const x = idx % width;
    const y = (idx / width) | 0;
    const neighbors = [
      x > 0 ? idx - 1 : -1,
      x < width - 1 ? idx + 1 : -1,
      y > 0 ? idx - width : -1,
      y < height - 1 ? idx + width : -1,
    ];
    for (const n of neighbors) {
      if (n >= 0 && labels[n] === 0 && !barrier[n]) heap.push(strength[n], n);
    }
  }

  const visited = new Uint8Array(strength.length);
  while (heap.size > 0) {
    const idx = heap.pop()!;
    if (labels[idx] !== 0 || barrier[idx] || visited[idx]) continue;
    visited[idx] = 1;

    const x = idx % width;
    const y = (idx / width) | 0;
    const neighbors = [
      x > 0 ? idx - 1 : -1,
      x < width - 1 ? idx + 1 : -1,
      y > 0 ? idx - width : -1,
      y < height - 1 ? idx + width : -1,
    ];
    let assignedLabel = 0;
    for (const n of neighbors) {
      if (n >= 0 && labels[n] > 0) {
        assignedLabel = labels[n];
        break;
      }
    }
    if (assignedLabel === 0) continue;
    labels[idx] = assignedLabel;

    for (const n of neighbors) {
      if (n >= 0 && labels[n] === 0 && !barrier[n] && !visited[n]) heap.push(strength[n], n);
    }
  }

  return labels;
}

// ── Traçage de contours (Moore-neighbor tracing) + simplification Douglas-Peucker ──

/** Trace le contour extérieur de chaque label présent dans le tableau, en coordonnées pixels (coins de pixels). */
export function traceLabelContours(labels: Int32Array, width: number, height: number): Map<number, Array<{ x: number; y: number }>> {
  const contours = new Map<number, Array<{ x: number; y: number }>>();
  const visitedStart = new Set<string>();

  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= width || y >= height ? 0 : labels[y * width + x]);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const label = at(x, y);
      if (label <= 0) continue;
      // Pixel de bord gauche d'une région : premier pixel de la ligne appartenant à ce label,
      // ou pixel dont le voisin de gauche a un label différent.
      if (at(x - 1, y) === label) continue;
      const key = `${label}`;
      if (visitedStart.has(key)) continue; // un seul contour externe tracé par label (composante principale)
      visitedStart.add(key);

      const contour = mooreTrace(at, x, y, label, width, height);
      if (contour.length >= 3) contours.set(label, contour);
    }
  }
  return contours;
}

function mooreTrace(
  at: (x: number, y: number) => number,
  startX: number,
  startY: number,
  label: number,
  width: number,
  height: number,
): Array<{ x: number; y: number }> {
  // Directions en 8-connexité, sens horaire, en partant de l'Est (index 0).
  const directions = [
    [1, 0],   // 0 E
    [1, 1],   // 1 SE
    [0, 1],   // 2 S
    [-1, 1],  // 3 SW
    [-1, 0],  // 4 W
    [-1, -1], // 5 NW
    [0, -1],  // 6 N
    [1, -1],  // 7 NE
  ];
  const isForeground = (x: number, y: number) => at(x, y) === label;

  const boundary: Array<{ x: number; y: number }> = [{ x: startX, y: startY }];
  let cx = startX;
  let cy = startY;
  // On "arrive" virtuellement depuis l'Ouest (le pixel de départ est le plus à
  // gauche de sa ligne, donc son voisin Ouest est nécessairement fond).
  let backtrackDir = 4;
  const maxSteps = width * height * 8 + 8;
  let steps = 0;

  for (;;) {
    const searchStart = (backtrackDir + 1) % 8;
    let found = false;
    let nx = cx, ny = cy, newBacktrackDir = backtrackDir;
    for (let i = 0; i < 8; i++) {
      const d = (searchStart + i) % 8;
      const tx = cx + directions[d][0];
      const ty = cy + directions[d][1];
      if (isForeground(tx, ty)) {
        nx = tx;
        ny = ty;
        newBacktrackDir = (d + 4) % 8;
        found = true;
        break;
      }
    }
    if (!found) break; // pixel isolé (pas de voisin foreground)

    cx = nx;
    cy = ny;
    backtrackDir = newBacktrackDir;
    steps++;
    if (cx === startX && cy === startY) break; // retour au point de départ : contour bouclé
    boundary.push({ x: cx, y: cy });
    if (steps >= maxSteps) break;
  }

  return boundary;
}

export function simplifyPolygon(points: Array<{ x: number; y: number }>, epsilon: number): Array<{ x: number; y: number }> {
  if (points.length <= 2) return points;

  const perpendicularDistance = (p: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq === 0) return Math.hypot(p.x - a.x, p.y - a.y);
    const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq;
    const projX = a.x + t * dx;
    const projY = a.y + t * dy;
    return Math.hypot(p.x - projX, p.y - projY);
  };

  const douglasPeucker = (pts: Array<{ x: number; y: number }>): Array<{ x: number; y: number }> => {
    if (pts.length <= 2) return pts;
    let maxDist = -1;
    let maxIndex = 0;
    for (let i = 1; i < pts.length - 1; i++) {
      const dist = perpendicularDistance(pts[i], pts[0], pts[pts.length - 1]);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }
    if (maxDist > epsilon) {
      const left = douglasPeucker(pts.slice(0, maxIndex + 1));
      const right = douglasPeucker(pts.slice(maxIndex));
      return [...left.slice(0, -1), ...right];
    }
    return [pts[0], pts[pts.length - 1]];
  };

  return douglasPeucker(points);
}