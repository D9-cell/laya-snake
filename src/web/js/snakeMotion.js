export const GLIDE_MS = 160;

// The previous snake and when the latest move started, for visual interpolation only.
export const motion = { prev: null, at: 0 };
// Recent food pickups, for the 2D sparkle.
export const effects = [];

// Remembers the previous snake so frames can glide between cells, and marks eaten food.
export function trackMotion(old, s) {
  const now = performance.now();
  while (effects.length && now - effects[0].at > 1000) effects.shift();
  if (!old || !old.snake.length || !s.snake.length || old.round !== s.round) {
    motion.prev = null;
    return;
  }
  const [ox, oy] = old.snake[0], [nx, ny] = s.snake[0];
  if (ox === nx && oy === ny) return;
  if (Math.abs(ox - nx) + Math.abs(oy - ny) === 1) { motion.prev = old.snake; motion.at = now; }
  else motion.prev = null;
  if (s.score > old.score) effects.push({ x: nx + 0.5, y: ny + 0.5, at: now });
}

// Allocation-free polyline through the snake in cell units, head first, with arc lengths.
export class Centerline {
  constructor() {
    this.n = 0; this.total = 0; this.cap = 0;
    this.x = this.y = this.d = null;
    this.px = new Float32Array(64); this.py = new Float32Array(64);
  }

  // Grows the buffers to hold at least `n` samples.
  ensure(n) {
    if (n <= this.cap) return;
    this.cap = Math.max(n, this.cap * 2, 256);
    this.x = new Float32Array(this.cap); this.y = new Float32Array(this.cap); this.d = new Float32Array(this.cap);
  }

  // Appends one sample.
  push(x, y) { this.x[this.n] = x; this.y[this.n] = y; this.n++; }

  // Writes the point `dist` cells from the head into `out` (clamped to the body).
  pointAt(dist, out) {
    const n = this.n;
    if (n === 1 || dist <= 0) { out.x = this.x[0]; out.y = this.y[0]; return out; }
    if (dist >= this.total) { out.x = this.x[n - 1]; out.y = this.y[n - 1]; return out; }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.d[m] <= dist) lo = m; else hi = m; }
    const span = this.d[hi] - this.d[lo], t = span > 1e-9 ? (dist - this.d[lo]) / span : 0;
    out.x = this.x[lo] + (this.x[hi] - this.x[lo]) * t;
    out.y = this.y[lo] + (this.y[hi] - this.y[lo]) * t;
    return out;
  }
}

// Fills `cl` for this frame: cell centres, head and tail gliding between the last two states, with
// every corner replaced by a quadratic curve between the midpoints of its two edges. The curve
// stays inside the corner cell, so it never cuts across a cell the snake is not in.
export function buildCenterline(state, now, cl) {
  const cells = state.snake, n = cells.length;
  cl.n = 0; cl.total = 0;
  if (!n) return cl;
  if (cl.px.length < n + 1) { cl.px = new Float32Array((n + 1) * 2); cl.py = new Float32Array((n + 1) * 2); }
  const px = cl.px, py = cl.py;
  for (let i = 0; i < n; i++) { px[i] = cells[i][0] + 0.5; py[i] = cells[i][1] + 0.5; }
  let m = n;
  const e = motion.prev ? Math.min(1, Math.max(0, (now - motion.at) / GLIDE_MS)) : 1;
  const ease = e * (2 - e);
  if (n > 1 && ease < 1) {
    px[0] = px[1] + (px[0] - px[1]) * ease; py[0] = py[1] + (py[0] - py[1]) * ease;
    const oldTail = motion.prev[motion.prev.length - 1], newTail = cells[n - 1];
    if (oldTail[0] !== newTail[0] || oldTail[1] !== newTail[1]) {
      const ox = oldTail[0] + 0.5, oy = oldTail[1] + 0.5;
      px[m] = ox + (px[n - 1] - ox) * ease; py[m] = oy + (py[n - 1] - oy) * ease; m++;
    }
  }
  cl.ensure(m * 8 + 16);
  if (m === 1) { cl.push(px[0], py[0]); cl.d[0] = 0; return cl; }
  const fx = (px[0] + px[1]) / 2, fy = (py[0] + py[1]) / 2;
  for (let k = 0; k < 4; k++) cl.push(px[0] + (fx - px[0]) * k / 4, py[0] + (fy - py[0]) * k / 4);
  for (let i = 1; i < m - 1; i++) {
    const sx = (px[i - 1] + px[i]) / 2, sy = (py[i - 1] + py[i]) / 2;
    const ex = (px[i] + px[i + 1]) / 2, ey = (py[i] + py[i + 1]) / 2;
    for (let k = 0; k < 8; k++) {
      const t = k / 8, u = 1 - t;
      cl.push(u * u * sx + 2 * u * t * px[i] + t * t * ex, u * u * sy + 2 * u * t * py[i] + t * t * ey);
    }
  }
  const lx = px[m - 1], ly = py[m - 1], mx = (px[m - 2] + lx) / 2, my = (py[m - 2] + ly) / 2;
  for (let k = 0; k <= 4; k++) cl.push(mx + (lx - mx) * k / 4, my + (ly - my) * k / 4);
  let len = 0;
  cl.d[0] = 0;
  for (let i = 1; i < cl.n; i++) { len += Math.hypot(cl.x[i] - cl.x[i - 1], cl.y[i] - cl.y[i - 1]); cl.d[i] = len; }
  cl.total = len;
  return cl;
}
