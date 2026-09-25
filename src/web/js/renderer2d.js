import { hash, mix } from "./util.js";
import { effects, Centerline, buildCenterline } from "./snakeMotion.js";

let stage = null, g = null, S = null, prefs = null;
const MARGIN = 1.3;
let VW = 800, VH = 500;
const C = { ready: false, tilt: 0, F: 1, ox: 0, oy: 0, fx: 0, fy: 0, sin: 0, cos: 1, D: 40 };

// Fits the scene canvas to its box at the device pixel ratio.
function sizeStage() {
  const w = stage.parentElement.clientWidth;
  const h = Math.round(Math.max(340, Math.min(w * 0.64, window.innerHeight - 140)));
  const dpr = window.devicePixelRatio || 1;
  if (stage.width !== Math.round(w * dpr) || stage.height !== Math.round(h * dpr)) {
    stage.width = Math.round(w * dpr); stage.height = Math.round(h * dpr);
  }
  stage.style.height = h + "px";
  VW = w; VH = h;
}

// Projects a world point (cells, z up) to screen pixels; `s` is pixels per cell at that depth.
function P(x, y, z) {
  const X = x - C.fx, Y = y - C.fy;
  const sy = Y * C.cos - z * C.sin;
  const k = C.F / (C.D - Y * C.sin - z * C.cos);
  return { x: C.ox + X * k, y: C.oy + sy * k, s: k };
}

// Fills a polygon given in world coordinates.
function poly(pts, fill) {
  g.beginPath();
  pts.forEach(([x, y, z], i) => { const p = P(x, y, z); i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y); });
  g.closePath(); g.fillStyle = fill; g.fill();
}

// Adds a flat ellipse lying on the ground (or at height z) to the current path.
function groundBlob(x, y, r, z = 0) {
  const p = P(x, y, z), rx = r * p.s, ry = Math.max(0.5, r * p.s * (C.cos + 0.08));
  g.moveTo(p.x + rx, p.y); g.ellipse(p.x, p.y, rx, ry, 0, 0, Math.PI * 2);
}

// Draws an axis-aligned block with its top, front and camera-facing side.
function block(x0, y0, x1, y1, z0, z1, top, front, side) {
  const sx = (x0 + x1) / 2 > C.fx ? x0 : x1;
  poly([[sx, y0, z0], [sx, y1, z0], [sx, y1, z1], [sx, y0, z1]], side);
  poly([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], front);
  poly([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], top);
}

// Moves the camera toward the fitted overview, or toward the snake's head when following.
function updateCamera(dt, head) {
  const W = S.width, H = S.height, M = MARGIN;
  const a = C.ready ? 1 - Math.exp(-dt / 180) : 1;
  const tiltT = 0;
  C.tilt += (tiltT - C.tilt) * a;
  if (Math.abs(C.tilt - tiltT) < 1e-3) C.tilt = tiltT;
  C.sin = Math.sin(C.tilt); C.cos = Math.cos(C.tilt);
  C.D = Math.max(W, H) * 1.7 + 8;

  const keep = { F: C.F, ox: C.ox, oy: C.oy, fx: C.fx, fy: C.fy };
  Object.assign(C, { F: 1, ox: 0, oy: 0, fx: W / 2, fy: H / 2 });
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y, z] of [[-M, -M, 0], [W + M, -M, 0], [-M, H + M, 0], [W + M, H + M, 0], [-M, -M, 1.2], [W + M, -M, 1.2]]) {
    const p = P(x, y, z);
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  Object.assign(C, keep);
  const fit = Math.min(VW / (x1 - x0), VH / (y1 - y0));
  const T = { F: fit, fx: W / 2, fy: H / 2, ox: VW / 2 - fit * (x0 + x1) / 2, oy: VH / 2 - fit * (y0 + y1) / 2 };
  if (prefs.follow && head) {
    Object.assign(T, { F: fit * Math.max(1.4, Math.min(2.6, Math.max(W, H) / 12)), fx: head.x, fy: head.y, ox: VW / 2, oy: VH / 2 });
  }
  for (const k of ["F", "fx", "fy", "ox", "oy"]) {
    C[k] += (T[k] - C[k]) * a;
    if (Math.abs(T[k] - C[k]) < 1e-3) C[k] = T[k];
  }
  C.ready = true;
}

/* scenery that only depends on the board size */
let decor = { key: "", bushes: [], blades: [], flowers: [], lanterns: [] };
// Scatters bushes, grass and flowers around the yard for the current board size.
function buildDecor(W, H) {
  const key = W + "x" + H;
  if (decor.key === key) return decor;
  const M = MARGIN, bushes = [], blades = [], flowers = [];
  const ring = (x, y) => x < -0.35 || y < -0.35 || x > W + 0.35 || y > H + 0.35;
  let i = 0;
  for (let t = 0; t < 2 * (W + H) + 8; t += 0.8, i++) {
    const r1 = hash(i, 7), r2 = hash(i, 11), r3 = hash(i, 13);
    if (r1 < 0.3) continue;
    const off = 0.55 + r2 * (M - 0.35);
    let x, y;
    if (t < W + 2) { x = t - 1; y = -off; }
    else if (t < W + H + 4) { x = W + off; y = t - (W + 2) - 1; }
    else if (t < 2 * W + H + 6) { x = W + 1 - (t - (W + H + 4)); y = H + off; }
    else { x = -off; y = H + 1 - (t - (2 * W + H + 6)); }
    if (!ring(x, y)) continue;
    const r = 0.28 + r3 * 0.34;
    const parts = [];
    for (let k = 0; k < 4; k++) parts.push([(hash(i, 20 + k) - 0.5) * r * 1.3, (hash(i, 30 + k) - 0.5) * r * 0.9, r * (0.55 + hash(i, 40 + k) * 0.45)]);
    const hue = hash(i, 50);
    bushes.push({ x, y, r, parts, dark: hue < 0.5 ? "#1e3a17" : "#23421b", mid: hue < 0.5 ? "#2f5a22" : "#3b6526", light: hue < 0.5 ? "#4f8233" : "#5f8f35",
      bloom: hash(i, 60) < 0.35 ? (hash(i, 61) < 0.5 ? "#f3a6c8" : "#fff3d6") : null });
  }
  for (let k = 0; k < (W + H) * 14; k++) {
    const x = -M + hash(k, 71) * (W + 2 * M), y = -M + hash(k, 73) * (H + 2 * M);
    if (!ring(x, y)) continue;
    blades.push([x, y, (hash(k, 75) - 0.5) * 0.12, 0.08 + hash(k, 77) * 0.14, hash(k, 79) < 0.5]);
  }
  for (let k = 0; k < (W + H) * 1.2; k++) {
    const x = -M + hash(k, 81) * (W + 2 * M), y = -M + hash(k, 83) * (H + 2 * M);
    if (ring(x, y)) flowers.push([x, y, ["#f7b6d2", "#fff5dc", "#ffd66b", "#c9b6ff"][Math.floor(hash(k, 85) * 4)]]);
  }
  const lanterns = [[0, 0], [W, 0], [0, H], [W, H]];
  decor = { key, bushes, blades, flowers, lanterns };
  return decor;
}

/* ground layer, cached while the camera and board stand still */
const groundCv = document.createElement("canvas");
const gctx = groundCv.getContext("2d");
let groundKey = "";
// Paints the yard, the paving stones, rock shadows and grass into the cached ground layer.
function paintGround(W, H, dpr) {
  const key = [VW, VH, dpr, C.F.toFixed(3), C.ox.toFixed(2), C.oy.toFixed(2), C.fx.toFixed(3), C.fy.toFixed(3), C.tilt.toFixed(4), prefs.grid, W, H, S.obstacles.join(";")].join("|");
  if (key === groundKey) return;
  groundKey = key;
  if (groundCv.width !== stage.width || groundCv.height !== stage.height) { groundCv.width = stage.width; groundCv.height = stage.height; }
  gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  withCtx(gctx, () => {
    const M = MARGIN, d = buildDecor(W, H);
    const bg = g.createLinearGradient(0, 0, 0, VH);
    bg.addColorStop(0, "#15260f"); bg.addColorStop(1, "#0e1b0b");
    g.fillStyle = bg; g.fillRect(0, 0, VW, VH);
    poly([[-M - 3, -M - 3, 0], [W + M + 3, -M - 3, 0], [W + M + 3, H + M + 3, 0], [-M - 3, H + M + 3, 0]], "#2a4a1d");
    poly([[-M, -M, 0], [W + M, -M, 0], [W + M, H + M, 0], [-M, H + M, 0]], "#3a6128");
    g.lineCap = "round";
    for (const [x, y, lean, h, light] of d.blades) {
      const a = P(x, y, 0), b = P(x + lean, y, h);
      g.strokeStyle = light ? "rgba(150,200,90,.55)" : "rgba(20,45,12,.55)";
      g.lineWidth = Math.max(0.6, a.s * 0.035);
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
    }
    for (const [x, y, c] of d.flowers) { g.beginPath(); groundBlob(x, y, 0.05, 0.02); g.fillStyle = c; g.fill(); }

    poly([[-0.08, -0.08, 0], [W + 0.08, -0.08, 0], [W + 0.08, H + 0.08, 0], [-0.08, H + 0.08, 0]], prefs.grid ? "#2b3322" : "#4a4a3f");
    const L = [];
    for (let j = 0; j <= H; j++) { const row = []; for (let i = 0; i <= W; i++) row.push(P(i, j, 0)); L.push(row); }
    const inset = prefs.grid ? 0.06 : 0.018;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const q = [L[j][i], L[j][i + 1], L[j + 1][i + 1], L[j + 1][i]];
      const cx = (q[0].x + q[1].x + q[2].x + q[3].x) / 4, cy = (q[0].y + q[1].y + q[2].y + q[3].y) / 4;
      const r = hash(i, j, 1), r2 = hash(i, j, 2);
      const l = 36 + r * 14, sat = 6 + r2 * 10, hue = 30 + hash(i, j, 3) * 40;
      g.beginPath();
      q.forEach((p, k) => { const x = p.x + (cx - p.x) * inset * 2, y = p.y + (cy - p.y) * inset * 2; k ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.closePath();
      g.fillStyle = `hsl(${hue},${sat}%,${l}%)`; g.fill();
      g.strokeStyle = `hsla(${hue},${sat}%,${l + 14}%,.35)`; g.lineWidth = Math.max(0.5, q[0].s * 0.03);
      g.beginPath(); g.moveTo(q[3].x + (cx - q[3].x) * inset * 2, q[3].y + (cy - q[3].y) * inset * 2);
      g.lineTo(q[0].x + (cx - q[0].x) * inset * 2, q[0].y + (cy - q[0].y) * inset * 2);
      g.lineTo(q[1].x + (cx - q[1].x) * inset * 2, q[1].y + (cy - q[1].y) * inset * 2); g.stroke();
      if (hash(i, j, 4) < 0.12) { g.beginPath(); groundBlob(i + 0.2 + hash(i, j, 5) * 0.6, j + 0.2 + hash(i, j, 6) * 0.6, 0.06 + hash(i, j, 7) * 0.08); g.fillStyle = "rgba(0,0,0,.12)"; g.fill(); }
    }
    g.fillStyle = "rgba(92,140,52,.85)";
    g.beginPath();
    for (let j = 0; j <= H; j++) for (let i = 0; i <= W; i++) if (hash(i, j, 8) < 0.28) groundBlob(i, j, 0.07 + hash(i, j, 9) * 0.09, 0.005);
    g.fill();
    g.fillStyle = "rgba(0,0,0,.32)";
    g.beginPath();
    for (const [x, y] of S.obstacles) {
      const p = [[x + 0.1, y + 0.2], [x + 1.25, y + 0.2], [x + 1.25, y + 1.3], [x + 0.1, y + 1.3]].map(([a, b]) => P(a, b, 0));
      p.forEach((q, k) => k ? g.lineTo(q.x, q.y) : g.moveTo(q.x, q.y)); g.closePath();
    }
    g.fill();
  });
}

// Runs a drawing callback with the shared helpers pointed at another context.
function withCtx(ctx, fn) { const prev = g; g = ctx; try { fn(); } finally { g = prev; } }

const SKIN = Array.from({ length: 33 }, (_, i) => {
  const base = mix("#86c94a", "#3d7426", i / 32);
  return { base, dark: mix(base, "#0b1a06", 0.55), light: mix(base, "#f1ffd0", 0.5), spot: mix(base, "#14290b", 0.62) };
});
const FUR = [["#f5f0e8", "#d9cfc0"], ["#ead6b8", "#c8ab86"], ["#c9a07a", "#9c7550"], ["#9a8f86", "#6f655c"]];

// Draws a sleeping rabbit (a piece of food) standing on the given cell.
function drawRabbit(fx, fy, now) {
  const seed = hash(fx, fy, 17), p = P(fx + 0.5, fy + 0.62, 0), u = p.s * 1.35;
  const flip = seed < 0.5 ? -1 : 1, [fur, shade] = FUR[Math.floor(hash(fx, fy, 19) * FUR.length)];
  const br = 1 + 0.04 * Math.sin(now / 540 + seed * 30);
  g.save(); g.translate(p.x, p.y); g.scale(flip, 1);
  g.fillStyle = "rgba(0,0,0,.3)"; g.beginPath(); g.ellipse(0.02 * u, 0, 0.38 * u, 0.12 * u, 0, 0, Math.PI * 2); g.fill();
  const body = g.createRadialGradient(-0.1 * u, -0.34 * u, 0.03 * u, 0, -0.18 * u, 0.42 * u);
  body.addColorStop(0, "#ffffff"); body.addColorStop(0.35, fur); body.addColorStop(1, shade);
  g.fillStyle = body; g.beginPath(); g.ellipse(-0.05 * u, -0.19 * u * br, 0.34 * u, 0.22 * u * br, 0, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#fffaf2"; g.beginPath(); g.arc(-0.37 * u, -0.2 * u, 0.075 * u, 0, Math.PI * 2); g.fill();
  for (const [dx, dy] of [[0.02, -0.36], [0.07, -0.31]]) {
    g.save(); g.translate(dx * u, dy * u * br); g.rotate(-0.28);
    g.fillStyle = shade; g.beginPath(); g.ellipse(0, 0, 0.21 * u, 0.06 * u, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#efb3b3"; g.beginPath(); g.ellipse(0.01 * u, 0.005 * u, 0.15 * u, 0.028 * u, 0, 0, Math.PI * 2); g.fill();
    g.restore();
  }
  const head = g.createRadialGradient(0.2 * u, -0.33 * u, 0.02 * u, 0.24 * u, -0.26 * u, 0.2 * u);
  head.addColorStop(0, "#ffffff"); head.addColorStop(0.4, fur); head.addColorStop(1, shade);
  g.fillStyle = head; g.beginPath(); g.ellipse(0.24 * u, -0.25 * u, 0.16 * u, 0.14 * u, 0, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "#3a2a22"; g.lineWidth = Math.max(0.7, 0.02 * u); g.lineCap = "round";
  g.beginPath(); g.arc(0.27 * u, -0.28 * u, 0.035 * u, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke();
  g.fillStyle = "#e58f9b"; g.beginPath(); g.arc(0.39 * u, -0.24 * u, 0.022 * u, 0, Math.PI * 2); g.fill();
  g.restore();
  g.textAlign = "center"; g.textBaseline = "middle";
  for (let k = 0; k < 3; k++) {
    const ph = ((now / 1000 + seed * 5 + k * 0.8) % 2.4) / 2.4;
    g.globalAlpha = Math.sin(ph * Math.PI) * 0.9;
    g.fillStyle = "#ffffff";
    g.font = `700 ${Math.max(6, (0.16 + 0.12 * ph) * u)}px ui-sans-serif, system-ui, sans-serif`;
    g.fillText("z", p.x + flip * (0.2 + ph * 0.3) * u, p.y - (0.5 + ph * 0.55) * u);
  }
  g.globalAlpha = 1;
}

// Draws one stone block of an obstacle with moss on top.
function drawRock(x, y) {
  const h = 0.72 + hash(x, y, 21) * 0.26, i = 0.03, t = hash(x, y, 23);
  const l = 34 + t * 9, hue = 30 + hash(x, y, 37) * 25;
  const pt = P(x + 0.5, y + 0.5, h), pb = P(x + 0.5, y + 1, 0);
  const fg = g.createLinearGradient(pt.x, pt.y, pb.x, pb.y);
  fg.addColorStop(0, `hsl(${hue},8%,${l - 2}%)`); fg.addColorStop(1, `hsl(${hue},9%,${l - 14}%)`);
  const tg = g.createLinearGradient(pt.x - pt.s * 0.5, pt.y - pt.s * 0.4, pt.x + pt.s * 0.5, pt.y + pt.s * 0.3);
  tg.addColorStop(0, `hsl(${hue},7%,${l + 16}%)`); tg.addColorStop(1, `hsl(${hue},8%,${l + 4}%)`);
  block(x + i, y + i, x + 1 - i, y + 1 - i, 0, h, tg, fg, `hsl(${hue},10%,${l - 18}%)`);
  g.fillStyle = "rgba(0,0,0,.18)";
  for (let k = 0; k < 5; k++) {
    const sp = P(x + 0.12 + hash(x, y, 40 + k) * 0.76, y + 1 - i, h * (0.1 + hash(x, y, 50 + k) * 0.8));
    g.beginPath(); g.arc(sp.x, sp.y, Math.max(0.6, sp.s * 0.03), 0, Math.PI * 2); g.fill();
  }
  const e0 = P(x + i, y + 1 - i, h), e1 = P(x + 1 - i, y + 1 - i, h);
  g.strokeStyle = `hsla(${hue},10%,${l + 24}%,.55)`; g.lineWidth = Math.max(0.6, e0.s * 0.025);
  g.beginPath(); g.moveTo(e0.x, e0.y); g.lineTo(e1.x, e1.y); g.stroke();
  const p0 = P(x + 0.2, y + 1 - i, h * 0.7), p1 = P(x + 0.45, y + 1 - i, h * 0.45), p2 = P(x + 0.4, y + 1 - i, h * 0.15);
  g.strokeStyle = "rgba(0,0,0,.28)"; g.lineWidth = Math.max(0.6, p0.s * 0.02);
  g.beginPath(); g.moveTo(p0.x, p0.y); g.lineTo(p1.x, p1.y); g.lineTo(p2.x, p2.y); g.stroke();
  g.beginPath();
  groundBlob(x + 0.3 + hash(x, y, 25) * 0.4, y + 0.3 + hash(x, y, 27) * 0.4, 0.16 + hash(x, y, 29) * 0.12, h + 0.001);
  if (hash(x, y, 31) < 0.6) groundBlob(x + 0.2 + hash(x, y, 33) * 0.6, y + 0.25 + hash(x, y, 35) * 0.5, 0.1, h + 0.001);
  g.fillStyle = "rgba(88,132,46,.92)"; g.fill();
  g.beginPath(); groundBlob(x + 0.5, y + 1.02, 0.12 + t * 0.1, 0.01); g.fillStyle = "rgba(70,120,40,.9)"; g.fill();
}

// Draws a wooden fence post (taller with a lantern on the corners).
function drawPost(x, y, lantern, lit) {
  const w = lantern ? 0.13 : 0.08, h = lantern ? 1.25 : 0.82;
  block(x - w, y - w, x + w, y + w, 0, h, "#8d6a45", "#5f432a", "#4a331f");
  if (!lantern) return;
  block(x - 0.2, y - 0.2, x + 0.2, y + 0.2, h, h + 0.05, "#3b2a1c", "#2d2016", "#241a12");
  block(x - 0.15, y - 0.15, x + 0.15, y + 0.15, h + 0.05, h + 0.42, lit ? "#ffe3a0" : "#d8cfb8", lit ? "#ffc864" : "#bdb39b", lit ? "#f0a848" : "#a39a84");
  const f = P(x, y + 0.151, h + 0.24);
  g.strokeStyle = "#2d2016"; g.lineWidth = Math.max(0.8, f.s * 0.03);
  const a = P(x, y + 0.151, h + 0.05), b = P(x, y + 0.151, h + 0.42);
  g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  block(x - 0.22, y - 0.22, x + 0.22, y + 0.22, h + 0.42, h + 0.5, "#4a3423", "#33241a", "#2a1e15");
}

// Draws a pair of fence rails between two posts.
function drawRails(ax, ay, bx, by) {
  for (const z of [0.36, 0.66]) {
    const a = P(ax, ay, z), b = P(bx, by, z);
    g.lineCap = "butt";
    g.strokeStyle = "#4c3521"; g.lineWidth = Math.max(1, a.s * 0.1);
    g.beginPath(); g.moveTo(a.x, a.y + a.s * 0.02); g.lineTo(b.x, b.y + b.s * 0.02); g.stroke();
    g.strokeStyle = "#7d5b3a"; g.lineWidth = Math.max(0.8, a.s * 0.06);
    g.beginPath(); g.moveTo(a.x, a.y - a.s * 0.015); g.lineTo(b.x, b.y - b.s * 0.015); g.stroke();
  }
}

// Draws one leafy bush, optionally flowering.
function drawBush(b) {
  for (const [dx, dy, r] of b.parts) {
    const p = P(b.x + dx, b.y + dy, r * 0.75), R = r * p.s;
    g.fillStyle = b.dark; g.beginPath(); g.arc(p.x, p.y, R, 0, Math.PI * 2); g.fill();
    g.fillStyle = b.mid; g.beginPath(); g.arc(p.x - R * 0.12, p.y - R * 0.16, R * 0.8, 0, Math.PI * 2); g.fill();
    g.fillStyle = b.light; g.beginPath(); g.arc(p.x - R * 0.3, p.y - R * 0.36, R * 0.38, 0, Math.PI * 2); g.fill();
    if (b.bloom) {
      g.fillStyle = b.bloom;
      for (let k = 0; k < 3; k++) { g.beginPath(); g.arc(p.x + (k - 1) * R * 0.45, p.y - R * (0.25 + 0.2 * (k % 2)), Math.max(1, R * 0.12), 0, Math.PI * 2); g.fill(); }
    }
  }
}

// Collects the snake's body samples and head as depth-sorted drawables.
function snakeItems(cl, items, now) {
  const total = Math.max(0.001, cl.total);
  const R = 0.36;
  const rad = (d) => { const t = d / total; return R * (t < 0.5 ? 1 : 1 - (t - 0.5) / 0.5 * 0.65); };
  let lastD = -1;
  for (let i = cl.n - 1; i >= 0; i--) {
    const q = { x: cl.x[i], y: cl.y[i], d: cl.d[i] };
    if (lastD >= 0 && lastD - q.d < 0.1 && i > 0) continue;
    lastD = q.d;
    const r = rad(q.d), sk = SKIN[Math.round(Math.min(1, q.d / Math.max(total, 4)) * 32)];
    const spot = Math.floor(q.d / 0.45) % 2 === 1 && q.d > 0.5;
    items.push({ k: q.y + r + (1 - q.d / total) * 1e-3, f: () => {
      const p = P(q.x, q.y, r), pr = r * p.s;
      g.fillStyle = sk.dark; g.beginPath(); g.arc(p.x, p.y, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = sk.base; g.beginPath(); g.arc(p.x - pr * 0.05, p.y - pr * 0.1, pr * 0.86, 0, Math.PI * 2); g.fill();
      if (spot) { g.fillStyle = sk.spot; g.beginPath(); g.ellipse(p.x, p.y - pr * 0.18, pr * 0.42, pr * 0.3, 0, 0, Math.PI * 2); g.fill(); }
      g.fillStyle = sk.light; g.beginPath(); g.ellipse(p.x - pr * 0.3, p.y - pr * 0.42, pr * 0.28, pr * 0.16, -0.4, 0, Math.PI * 2); g.fill();
    } });
  }
  const h = { x: cl.x[0], y: cl.y[0] }, k4 = Math.min(cl.n - 1, 4), nx = { x: cl.x[k4], y: cl.y[k4] };
  let dx = h.x - nx.x, dy = h.y - nx.y;
  if (cl.n < 2 || Math.hypot(dx, dy) < 1e-6) [dx, dy] = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[S.direction] || [1, 0];
  const len = Math.hypot(dx, dy); dx /= len; dy /= len;
  items.push({ k: h.y + 0.4 + 2e-3, f: () => drawHead(h.x, h.y, dx, dy, now) });
}

// Draws the snake's head: skull, snout, eyes, nostrils and a flicking tongue.
function drawHead(x, y, dx, dy, now) {
  const px = -dy, py = dx, sk = SKIN[0];
  const flick = (now % 2600) < 360 && !S.game_over;
  if (flick) {
    const out = 0.62 + 0.2 * Math.sin((now % 2600) / 360 * Math.PI);
    const a = P(x + dx * 0.4, y + dy * 0.4, 0.2), b = P(x + dx * out, y + dy * out, 0.2);
    const f1 = P(x + dx * (out + 0.12) + px * 0.07, y + dy * (out + 0.12) + py * 0.07, 0.2);
    const f2 = P(x + dx * (out + 0.12) - px * 0.07, y + dy * (out + 0.12) - py * 0.07, 0.2);
    g.strokeStyle = "#d42a4c"; g.lineWidth = Math.max(1, a.s * 0.035); g.lineCap = "round";
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.lineTo(f1.x, f1.y); g.moveTo(b.x, b.y); g.lineTo(f2.x, f2.y); g.stroke();
  }
  const skull = P(x - dx * 0.02, y - dy * 0.02, 0.38), snout = P(x + dx * 0.24, y + dy * 0.24, 0.34);
  const back = skull.y < snout.y ? [skull, 0.44] : [snout, 0.34], front = skull.y < snout.y ? [snout, 0.34] : [skull, 0.44];
  for (const [p, r] of [back, front]) {
    const pr = r * p.s;
    g.fillStyle = sk.dark; g.beginPath(); g.arc(p.x, p.y, pr, 0, Math.PI * 2); g.fill();
    const gr = g.createRadialGradient(p.x - pr * 0.3, p.y - pr * 0.4, pr * 0.1, p.x, p.y, pr);
    gr.addColorStop(0, sk.light); gr.addColorStop(0.5, sk.base); gr.addColorStop(1, sk.dark);
    g.fillStyle = gr; g.beginPath(); g.arc(p.x - pr * 0.04, p.y - pr * 0.08, pr * 0.9, 0, Math.PI * 2); g.fill();
  }
  for (const s of [1, -1]) {
    const n = P(x + dx * 0.5 + px * 0.09 * s, y + dy * 0.5 + py * 0.09 * s, 0.46);
    g.fillStyle = "#16300c"; g.beginPath(); g.arc(n.x, n.y, Math.max(0.8, n.s * 0.025), 0, Math.PI * 2); g.fill();
  }
  const eyes = [1, -1].map((s) => P(x + dx * 0.12 + px * 0.25 * s, y + dy * 0.12 + py * 0.25 * s, 0.6)).sort((a, b) => a.y - b.y);
  for (const e of eyes) {
    const er = Math.max(1.8, e.s * 0.115);
    g.fillStyle = "#1a1a10"; g.beginPath(); g.arc(e.x, e.y, er * 1.15, 0, Math.PI * 2); g.fill();
    g.fillStyle = "#f2d45c"; g.beginPath(); g.arc(e.x, e.y, er, 0, Math.PI * 2); g.fill();
    if (S.game_over) {
      g.strokeStyle = "#111"; g.lineWidth = Math.max(1, er * 0.35);
      g.beginPath(); g.moveTo(e.x - er * 0.7, e.y - er * 0.7); g.lineTo(e.x + er * 0.7, e.y + er * 0.7);
      g.moveTo(e.x + er * 0.7, e.y - er * 0.7); g.lineTo(e.x - er * 0.7, e.y + er * 0.7); g.stroke();
    } else {
      const sx = P(x + dx * 0.2, y + dy * 0.2, 0.6), ang = Math.atan2(sx.y - e.y, sx.x - e.x);
      g.fillStyle = "#0d0d0a"; g.beginPath(); g.ellipse(e.x + Math.cos(ang) * er * 0.25, e.y + Math.sin(ang) * er * 0.25, er * 0.28, er * 0.8, ang, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(255,255,255,.9)"; g.beginPath(); g.arc(e.x - er * 0.35, e.y - er * 0.4, er * 0.28, 0, Math.PI * 2); g.fill();
    }
  }
}

// Tints the finished frame for the time of day and lights the lanterns.
function lighting(now) {
  const tod = prefs.tod, d = decor;
  g.globalCompositeOperation = "multiply";
  const lg = g.createLinearGradient(0, 0, VW, VH);
  if (tod === "golden") { lg.addColorStop(0, "#ffe7b8"); lg.addColorStop(1, "#d8935a"); }
  else if (tod === "night") { lg.addColorStop(0, "#6a7db0"); lg.addColorStop(1, "#2f3a66"); }
  else { lg.addColorStop(0, "#ffffff"); lg.addColorStop(1, "#e9eee0"); }
  g.fillStyle = lg; g.fillRect(0, 0, VW, VH);
  g.globalCompositeOperation = "screen";
  if (tod !== "night") {
    const sun = g.createRadialGradient(VW * 0.08, -VH * 0.15, 0, VW * 0.08, -VH * 0.15, VW * 0.95);
    sun.addColorStop(0, tod === "golden" ? "rgba(255,186,100,.42)" : "rgba(255,250,225,.22)"); sun.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = sun; g.fillRect(0, 0, VW, VH);
  }
  if (tod !== "day") {
    g.globalCompositeOperation = "lighter";
    const flicker = 1 + 0.05 * Math.sin(now / 90) + 0.03 * Math.sin(now / 37);
    for (const [x, y] of d.lanterns) {
      const p = P(x, y, 1.45), R = p.s * (tod === "night" ? 5.5 : 3.2) * flicker;
      const gl = g.createRadialGradient(p.x, p.y, 0, p.x, p.y, R);
      gl.addColorStop(0, tod === "night" ? "rgba(255,176,80,.55)" : "rgba(255,170,80,.32)"); gl.addColorStop(1, "rgba(255,150,60,0)");
      g.fillStyle = gl; g.fillRect(p.x - R, p.y - R, 2 * R, 2 * R);
    }
  }
  g.globalCompositeOperation = "source-over";
  const v = g.createRadialGradient(VW / 2, VH / 2, Math.min(VW, VH) * 0.35, VW / 2, VH / 2, Math.hypot(VW, VH) * 0.62);
  v.addColorStop(0, "rgba(0,0,0,0)"); v.addColorStop(1, "rgba(0,0,0,.5)");
  g.fillStyle = v; g.fillRect(0, 0, VW, VH);
}

// Draws the sparkle burst where food was just eaten.
function drawEffects(now) {
  for (let i = effects.length - 1; i >= 0; i--) {
    const e = effects[i], t = (now - e.at) / 750;
    if (t >= 1) { effects.splice(i, 1); continue; }
    const c = P(e.x, e.y, 0.4 + t * 0.6), a = 1 - t;
    g.fillStyle = `rgba(255,220,120,${a})`;
    for (let k = 0; k < 10; k++) {
      const ang = k / 10 * Math.PI * 2, r = c.s * (0.2 + t * 0.7);
      g.beginPath(); g.arc(c.x + Math.cos(ang) * r, c.y + Math.sin(ang) * r * 0.7, Math.max(1, c.s * 0.05 * a), 0, Math.PI * 2); g.fill();
    }
    g.font = `700 ${Math.max(10, c.s * 0.4)}px ui-sans-serif, system-ui, sans-serif`;
    g.textAlign = "center"; g.fillStyle = `rgba(255,236,160,${a})`;
    g.fillText("+1", c.x, c.y - c.s * (0.5 + t * 0.5));
  }
}

let lastFrame = 0;
const CL = new Centerline();
// Renders one frame of the yard: ground, then every object back to front, then lighting.
function frame(now) {
  if (!S) return;
  const dt = lastFrame ? Math.min(100, now - lastFrame) : 16;
  lastFrame = now;
  const W = S.width, H = S.height, dpr = window.devicePixelRatio || 1;
  const path = S.snake.length ? buildCenterline(S, now, CL) : null;
  updateCamera(dt, path && { x: path.x[0], y: path.y[0] });
  const d = buildDecor(W, H);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  paintGround(W, H, dpr);
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(groundCv, 0, 0);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (path) {
    const total = Math.max(0.001, path.total);
    g.fillStyle = "rgba(0,0,0,.26)"; g.beginPath();
    for (let i = 0; i < path.n; i += 3) {
      const t = path.d[i] / total, r = 0.36 * (t < 0.5 ? 1 : 1 - (t - 0.5) / 0.5 * 0.65);
      groundBlob(path.x[i] + 0.1, path.y[i] + 0.14, r);
    }
    g.fill();
  }

  const lit = prefs.tod !== "day";
  const items = [];
  for (const b of d.bushes) items.push({ k: b.y + b.r, f: () => drawBush(b) });
  for (let i = 0; i <= W; i++) {
    for (const y of [0, H]) {
      const corner = i === 0 || i === W;
      items.push({ k: y + 0.1, f: () => drawPost(i, y, corner, lit) });
      if (i < W) items.push({ k: y + 0.02, f: () => drawRails(i, y, i + 1, y) });
    }
  }
  for (let j = 0; j < H; j++) {
    for (const x of [0, W]) {
      if (j > 0) items.push({ k: j + 0.1, f: () => drawPost(x, j, false, lit) });
      items.push({ k: j + 1.02, f: () => drawRails(x, j, x, j + 1) });
    }
  }
  for (const [x, y] of S.obstacles) items.push({ k: y + 0.96 - Math.abs(x + 0.5 - C.fx) * 1e-4, f: () => drawRock(x, y) });
  for (const [x, y] of S.foods) items.push({ k: y + 0.8, f: () => drawRabbit(x, y, now) });
  if (path) snakeItems(path, items, now);
  items.sort((a, b) => a.k - b.k);
  for (const it of items) it.f();

  lighting(now);
  drawEffects(now);
  if (S.game_over) { g.fillStyle = "rgba(140,10,10,.18)"; g.fillRect(0, 0, VW, VH); }
}

// The flat top-down board, drawn with the 2D canvas API.
export class CanvasRenderer {
  constructor(canvas, viewPrefs) {
    stage = canvas; g = canvas.getContext("2d"); prefs = viewPrefs;
  }

  // Takes a new server snapshot.
  update(state) {
    const first = !S || S.width !== state.width || S.height !== state.height;
    S = state;
    if (first) sizeStage();
  }

  // Draws one frame.
  render(now) { frame(now); }

  // Refits the canvas to its box.
  resize() { sizeStage(); groundKey = ""; }

  // Called when another renderer takes over; the camera snaps into place next time.
  deactivate() { C.ready = false; lastFrame = 0; }

  destroy() {}
}
