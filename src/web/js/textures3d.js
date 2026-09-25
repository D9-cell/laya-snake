import * as THREE from "three";

// Smooth tileable value noise in [0, 1] on a square of `period` cells.
function valueNoise(period, seed) {
  const grid = new Float32Array(period * period);
  let s = seed * 9301 + 49297;
  for (let i = 0; i < grid.length; i++) { s = (s * 16807) % 2147483647; grid[i] = s / 2147483647; }
  const at = (x, y) => grid[((y % period + period) % period) * period + ((x % period + period) % period)];
  return (u, v) => {
    const x = u * period, y = v * period, x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx;
    const b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx;
    return a + (b - a) * sy;
  };
}

// Tileable fractal noise built from a few octaves of value noise.
function fbm(seed, base = 4, octaves = 4) {
  const layers = Array.from({ length: octaves }, (_, i) => valueNoise(base << i, seed + i * 17));
  return (u, v) => {
    let sum = 0, amp = 0.5, norm = 0;
    for (const n of layers) { sum += n(u, v) * amp; norm += amp; amp *= 0.5; }
    return sum / norm;
  };
}

// Converts a tileable height field into a tangent-space normal map canvas.
function normalCanvas(height, size, strength) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(size, size), d = img.data;
  const h = (x, y) => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const nx = (h(x - 1, y) - h(x + 1, y)) * strength, ny = (h(x, y + 1) - h(x, y - 1)) * strength;
    const l = Math.hypot(nx, ny, 1), i = (y * size + x) * 4;
    d[i] = (nx / l * 0.5 + 0.5) * 255; d[i + 1] = (ny / l * 0.5 + 0.5) * 255; d[i + 2] = (1 / l * 0.5 + 0.5) * 255; d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

// Paints a canvas pixel by pixel from a function returning [r, g, b] in 0..255.
function paint(size, fn) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(size, size), d = img.data;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const c = fn(x / size, y / size, x, y), i = (y * size + x) * 4;
    d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

// Wraps a canvas as a repeating texture.
function tex(cv, srgb) {
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.anisotropy = 8;
  return t;
}

const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
const mixRGB = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

let skinCache = null;
// Snake skin: u runs around the body (belly at 0 and 1, spine at 0.5), v runs along it.
// Dorsal scales are staggered rows; the belly gets wide ventral plates.
export function snakeSkinTextures() {
  if (skinCache) return skinCache;
  const S = 512, COLS = 34, ROWS = 34;
  const blotch = fbm(3, 3, 4), grain = fbm(11, 16, 3);
  const height = new Float32Array(S * S), rough = new Float32Array(S * S);
  const scaleAt = (u, v) => {
    const t = Math.abs(u - 0.5) * 2;
    if (t > 0.8) {
      const fy = (v * 16) % 1;
      return { h: 1 - Math.pow(fy, 2.2) * 0.9, edge: fy > 0.88 ? 1 : 0, belly: 1, tint: 0.5 };
    }
    const row = Math.floor(v * ROWS), fy = v * ROWS - row;
    const fx = (u * COLS + (row % 2) * 0.5) % 1;
    const dx = (fx - 0.5) * 2, dy = fy;
    const dome = Math.max(0, 1 - dx * dx * 0.9 - Math.pow(dy, 1.6) * 0.75);
    const col = Math.floor(u * COLS + (row % 2) * 0.5) % COLS;
    let hsh = Math.imul(col ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(row + 0x632be5ab, 0xc2b2ae35);
    hsh = Math.imul(hsh ^ (hsh >>> 15), 0x2c1b3c6d);
    return { h: dome, edge: dome < 0.18 ? 1 : 0, belly: 0, tint: ((hsh ^ (hsh >>> 13)) >>> 0) / 4294967296 };
  };
  const dark = [44, 66, 26], mid = [96, 132, 52], side = [128, 158, 62], belly = [196, 196, 112];
  const color = paint(S, (u, v, x, y) => {
    const t = Math.abs(u - 0.5) * 2, sc = scaleAt(u, v);
    height[y * S + x] = sc.h;
    rough[y * S + x] = sc.edge ? 0.78 : 0.5 + grain(u, v) * 0.12;
    let c = t < 0.35 ? mixRGB(dark, mid, smooth(0.0, 0.35, t)) : t < 0.72 ? mixRGB(mid, side, smooth(0.35, 0.72, t)) : mixRGB(side, belly, smooth(0.72, 0.86, t));
    const spot = blotch(t * 0.5, v);
    if (t < 0.6 && spot > 0.54) c = mixRGB(c, [26, 38, 16], smooth(0.54, 0.64, spot) * (1 - t / 0.6) * 0.9);
    if (t > 0.45 && t < 0.72 && spot < 0.3) c = mixRGB(c, [40, 58, 24], 0.35);
    const shade = (0.86 + 0.14 * sc.h + (grain(u, v) - 0.5) * 0.08) * (0.95 + sc.tint * 0.1);
    return c.map((k) => Math.min(255, k * shade));
  });
  const roughCv = paint(S, (u, v, x, y) => { const r = rough[y * S + x] * 255; return [r, r, r]; });
  skinCache = { map: tex(color, true), normalMap: tex(normalCanvas(height, S, 2.2), false), roughnessMap: tex(roughCv, false) };
  return skinCache;
}

let stoneCache = null;
// Weathered paving stone: colour mottling, pitted normal and roughness.
export function stoneTextures() {
  if (stoneCache) return stoneCache;
  const S = 256, n = fbm(5, 3, 6), pits = fbm(9, 24, 2), grain = fbm(13, 64, 2);
  const height = new Float32Array(S * S);
  const map = paint(S, (u, v, x, y) => {
    const a = n(u, v), p = smooth(0.7, 0.8, pits(u, v)), gr = grain(u, v);
    height[y * S + x] = a * 0.5 + gr * 0.35 - p * 0.3;
    const l = 116 + (a - 0.5) * 34 + (gr - 0.5) * 14 - p * 12;
    return [l * 1.02, l * 0.98, l * 0.9];
  });
  const rough = paint(S, (u, v) => { const r = 205 + (grain(u, v) - 0.5) * 50; return [r, r, r]; });
  stoneCache = { map: tex(map, true), normalMap: tex(normalCanvas(height, S, 3), false), roughnessMap: tex(rough, false) };
  return stoneCache;
}

let grassCache = null;
// Short garden grass and soil, tiled across the yard.
export function grassTextures() {
  if (grassCache) return grassCache;
  const S = 512, n = fbm(21, 4, 5), fine = fbm(23, 64, 2), soil = fbm(29, 3, 3);
  const height = new Float32Array(S * S);
  const map = paint(S, (u, v, x, y) => {
    const f = fine(u, v), s = soil(u, v);
    height[y * S + x] = f;
    const grass = mixRGB([52, 84, 30], [96, 128, 52], n(u, v) * 0.7 + f * 0.3);
    return mixRGB(grass, [74, 70, 46], smooth(0.66, 0.8, s) * 0.55);
  });
  grassCache = { map: tex(map, true), normalMap: tex(normalCanvas(height, S, 4), false) };
  return grassCache;
}

let woodCache = null;
// Weathered wood grain for the fence and lantern posts.
export function woodTextures() {
  if (woodCache) return woodCache;
  const S = 256, n = fbm(31, 4, 4), fine = fbm(37, 32, 2);
  const height = new Float32Array(S * S);
  const map = paint(S, (u, v, x, y) => {
    const ring = Math.sin((u * 18 + n(u, v * 0.25) * 6) * Math.PI) * 0.5 + 0.5;
    height[y * S + x] = ring * 0.6 + fine(u, v) * 0.4;
    return mixRGB([70, 50, 32], [124, 94, 62], ring * 0.6 + fine(u, v) * 0.4);
  });
  woodCache = { map: tex(map, true), normalMap: tex(normalCanvas(height, S, 2), false) };
  return woodCache;
}

let furCache = null;
// Soft fur: fine directional strands used as a normal map on the rabbits.
export function furTextures() {
  if (furCache) return furCache;
  const S = 256, strands = fbm(41, 64, 2), clumps = fbm(43, 8, 3);
  const height = new Float32Array(S * S);
  paint(S, (u, v, x, y) => { height[y * S + x] = strands(u * 0.25, v) * 0.7 + clumps(u, v) * 0.3; return [0, 0, 0]; });
  furCache = { normalMap: tex(normalCanvas(height, S, 1.6), false) };
  return furCache;
}
