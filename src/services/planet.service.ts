export async function getPlanetTile(
  z: number,
  x: number,
  y: number,
  mosaicName: string
) {
  const apiKey = process.env.PLANET_API_KEY;

  if (!apiKey) {
    throw new Error("PLANET_API_KEY n'est pas configurée dans le fichier .env");
  }

  const url =
    `https://tiles.planet.com/basemaps/v1/planet-tiles/` +
    `${mosaicName}/gmap/${z}/${x}/${y}.png` +
    `?api_key=${encodeURIComponent(apiKey)}`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Planet tile error: ${response.status} ${response.statusText}`
    );
  }

  const buffer = Buffer.from(await response.arrayBuffer());

  return {
    buffer,
    contentType: response.headers.get("content-type") || "image/png",
  };
}