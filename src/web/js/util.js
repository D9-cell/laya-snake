// Deterministic pseudo-random number in [0, 1) for a cell and a salt.
export function hash(x, y, k = 0) {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(k | 0, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// Blends two colours given as hex or rgb() strings.
export function mix(a, b, t) {
  const rgb = (v) => v[0] === "#" ? [16, 8, 0].map((s) => (parseInt(v.slice(1), 16) >> s) & 255) : v.match(/\d+/g).map(Number);
  const pa = rgb(a), pb = rgb(b);
  const c = (i) => Math.round(pa[i] + (pb[i] - pa[i]) * t);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}
