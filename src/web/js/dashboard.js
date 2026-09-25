const $ = (id) => document.getElementById(id);
const DIRS = ["up", "down", "left", "right"];
const ARROW = { up: "▲", down: "▼", left: "◀", right: "▶" };
const TRAIL = { up: "↑", down: "↓", left: "←", right: "→" };
const SIZES = ["16x10", "24x14", "32x20", "40x24", "48x28", "56x34", "64x40"];
let S = null;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pad3 = (n) => String(n).padStart(3, "0");
const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(ms < 10 ? 1 : 0) + " ms";
const fmtDur = (s) => { s = Math.floor(s); const m = Math.floor(s / 60); return m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`; };
const clock = (s) => { s = Math.floor(s); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60; return (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); };

/* ---------- probability rows ---------- */
const probRows = {};
DIRS.forEach((d) => {
  const row = document.createElement("div");
  row.className = "prob";
  row.innerHTML = `<span class="arrow">${ARROW[d]}</span><span class="name">${d}</span><span class="bar"><i></i></span><span class="val num">0.00</span>`;
  $("probs").appendChild(row);
  probRows[d] = row;
});
const dpad = {};
document.querySelectorAll(".dpad span").forEach((el) => dpad[el.dataset.d] = el);

/* ---------- charts ---------- */
function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
const PAD = { l: 44, r: 10, t: 8, b: 22 };

function axes(ctx, w, h, yMax, yFmt, xLabels) {
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--border"); ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = yMax * i / 4, y = PAD.t + (h - PAD.t - PAD.b) * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(PAD.l, Math.round(y) + .5); ctx.lineTo(w - PAD.r, Math.round(y) + .5); ctx.stroke();
    ctx.fillText(yFmt(v), PAD.l - 6, y);
  }
  ctx.textBaseline = "top";
  xLabels.forEach(([x, t]) => {
    const tw = ctx.measureText(t).width;
    ctx.textAlign = "left";
    ctx.fillText(t, Math.max(PAD.l - tw / 2, Math.min(x - tw / 2, w - PAD.r - tw)), h - PAD.b + 6);
  });
}

const charts = {};
function lineChart(id, pts, { yMax, yFmt, tip, ref }) {
  const cv = $(id), { ctx, w, h } = setupCanvas(cv);
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
  const n = pts.length;
  const X = (i) => PAD.l + (n <= 1 ? pw / 2 : pw * i / (n - 1));
  const Y = (v) => PAD.t + ph * (1 - Math.min(v, yMax) / yMax);
  const xl = n ? [[X(0), "#" + pts[0].x], [X(n - 1), "#" + pts[n - 1].x]] : [];
  axes(ctx, w, h, yMax, yFmt, n > 1 ? xl : xl.slice(0, 1));
  if (!n) {
    ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("waiting for the first decision", PAD.l + pw / 2, PAD.t + ph / 2);
  } else {
    const series = css("--series");
    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(i), Y(p.y)) : ctx.moveTo(X(i), Y(p.y)));
    ctx.lineTo(X(n - 1), PAD.t + ph); ctx.lineTo(X(0), PAD.t + ph); ctx.closePath();
    ctx.fillStyle = css("--series-soft"); ctx.fill();
    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(i), Y(p.y)) : ctx.moveTo(X(i), Y(p.y)));
    ctx.strokeStyle = series; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
    if (ref != null) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--text-2"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(PAD.l, Y(ref.v)); ctx.lineTo(w - PAD.r, Y(ref.v)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      const lw = ctx.measureText(ref.label).width;
      ctx.fillStyle = css("--surface"); ctx.fillRect(PAD.l + 4, PAD.t + 2, lw + 10, 16);
      ctx.fillStyle = css("--text-2"); ctx.fillText(ref.label, PAD.l + 9, PAD.t + 4);
    }
    const hi = charts[id] && charts[id].hover;
    if (hi != null && hi < n) {
      ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(hi), PAD.t); ctx.lineTo(X(hi), PAD.t + ph); ctx.stroke();
      ctx.fillStyle = series; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(hi), Y(pts[hi].y), 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
  }
  charts[id] = Object.assign(charts[id] || {}, { kind: "line", n, X, pts, tip, w });
}

function barChart(id, bars, { yMax, yFmt, tip, empty }) {
  const cv = $(id), { ctx, w, h } = setupCanvas(cv);
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b, n = bars.length;
  const slot = n ? pw / n : pw, bw = Math.max(2, Math.min(28, slot - 2));
  const X = (i) => PAD.l + slot * i + slot / 2;
  const every = Math.max(1, Math.ceil(n / 10));
  axes(ctx, w, h, yMax, yFmt, bars.map((b, i) => [X(i), b.label]).filter((_, i) => i % every === 0 || i === n - 1));
  if (!n) {
    ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(empty, PAD.l + pw / 2, PAD.t + ph / 2);
  }
  const hi = charts[id] && charts[id].hover;
  bars.forEach((b, i) => {
    const bh = Math.max(b.y > 0 ? 2 : 0, ph * Math.min(b.y, yMax) / yMax), x = X(i) - bw / 2, y = PAD.t + ph - bh;
    ctx.globalAlpha = hi == null || hi === i ? 1 : 0.55;
    ctx.fillStyle = css("--series");
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, y, bw, bh, [Math.min(4, bw / 2), Math.min(4, bw / 2), 0, 0]) : ctx.rect(x, y, bw, bh);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  charts[id] = Object.assign(charts[id] || {}, { kind: "bar", n, X, slot, bars, tip, w });
}

function hookHover(id) {
  const cv = $(id), tipEl = cv.parentElement.querySelector(".tip");
  const move = (ev) => {
    const c = charts[id]; if (!c || !c.n) return;
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left;
    let idx;
    if (c.kind === "line") {
      idx = 0; let best = Infinity;
      for (let i = 0; i < c.n; i++) { const d = Math.abs(c.X(i) - mx); if (d < best) { best = d; idx = i; } }
    } else {
      idx = Math.floor((mx - PAD.l) / c.slot);
      if (idx < 0 || idx >= c.n) { leave(); return; }
    }
    c.hover = idx; drawCharts();
    tipEl.innerHTML = c.tip(idx);
    tipEl.style.display = "block";
    const tx = Math.min(Math.max(c.X(idx) + 12, 0), c.w - tipEl.offsetWidth - 4);
    tipEl.style.left = (c.X(idx) + 12 + tipEl.offsetWidth > c.w ? c.X(idx) - tipEl.offsetWidth - 12 : tx) + "px";
    tipEl.style.top = "8px";
  };
  const leave = () => { if (charts[id]) charts[id].hover = null; tipEl.style.display = "none"; drawCharts(); };
  cv.addEventListener("mousemove", move);
  cv.addEventListener("mouseleave", leave);
}
["c-lat", "c-conf", "c-score"].forEach(hookHover);

export function drawCharts() {
  if (!S) return;
  const mv = S.moves;
  const lat = mv.map((m) => ({ x: m.n, y: m.ms, m }));
  const latMax = niceMax(Math.max(1, ...lat.map((p) => p.y)) * 1.1);
  lineChart("c-lat", lat, {
    yMax: latMax, yFmt: (v) => v >= 1000 ? (v / 1000).toFixed(1) + "s" : Math.round(v) + "",
    ref: S.totals.decisions ? { v: S.totals.avg_ms, label: "- - avg " + fmtMs(S.totals.avg_ms) } : null,
    tip: (i) => { const m = lat[i].m; return `<div class="t">Decision #${m.n} · round ${m.round}</div><b class="num">${fmtMs(m.ms)}</b> · ${m.chosen}${m.intervened ? " (shield)" : ""}`; },
  });
  const conf = mv.map((m) => ({ x: m.n, y: Math.max(...m.probs), m }));
  lineChart("c-conf", conf, {
    yMax: 1, yFmt: (v) => v.toFixed(2),
    ref: { v: 0.25, label: "- - uniform 0.25" },
    tip: (i) => { const m = conf[i].m; return `<div class="t">Decision #${m.n}</div>top p <b class="num">${conf[i].y.toFixed(3)}</b> · ${m.model_top}<br><span class="t">${DIRS.map((d, k) => d[0].toUpperCase() + " " + m.probs[k].toFixed(2)).join("  ")}</span>`; },
  });
  const rounds = S.rounds.slice(0, 30).reverse();
  const bars = rounds.map((r) => ({ y: r.score, label: String(r.round), r }));
  barChart("c-score", bars, {
    yMax: Math.max(4, Math.ceil(Math.max(...bars.map((b) => b.y), 1) * 1.1 / 4) * 4), yFmt: (v) => Math.round(v) + "",
    empty: "no finished rounds yet",
    tip: (i) => { const r = bars[i].r; return `<div class="t">Round ${r.round} · ${r.size}</div>score <b class="num">${r.score}</b> · ${r.decisions} moves · ${r.cause}`; },
  });
}

/* ---------- render ---------- */
// Sets an element's text only when it changed.
export function setText(id, v) { const el = $(id); if (el.textContent !== v) el.textContent = v; }

// Fills the board-size picker, keeping the current size selectable even when it is custom.
function sizeOptions(s) {
  const cur = `${s.width}x${s.height}`, sel = $("size-sel");
  const list = SIZES.filter((z) => { const [w, h] = z.split("x").map(Number); return w >= s.limits.min_w && h >= s.limits.min_h && w <= s.limits.max_w && h <= s.limits.max_h; });
  if (!list.includes(cur)) list.push(cur);
  list.sort((a, b) => parseInt(a) - parseInt(b));
  const html = list.map((z) => `<option value="${z}">${z.replace("x", " × ")}</option>`).join("");
  if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
  sel.value = cur;
}

// Applies one server state to every HTML panel of the page (never the 3D scene).
export function renderDashboard(s) {
  S = s;

  const st = $("status");
  st.className = "pill " + s.status;
  setText("status-text", s.status === "live" ? "LIVE" : s.status === "paused" ? "PAUSED" : "GAME OVER");
  setText("hdr-speed", `L${s.speed_level}/5`);
  setText("hdr-size", `${s.width}x${s.height}`);
  setText("clock", clock(s.uptime_s));
  setText("round-clock", clock(s.round_s));
  $("mock-banner").classList.toggle("show", !!s.info.mock);

  setText("round", "ROUND " + String(s.round).padStart(2, "0"));
  setText("score", pad3(s.score));
  setText("length", pad3(s.length));
  setText("best", pad3(s.best));
  setText("fill-v", (s.fill * 100).toFixed(1) + "%");
  $("fill").style.width = (s.fill * 100).toFixed(2) + "%";
  const trail = s.trail.slice(-10).map((d) => `<span>${TRAIL[d]}</span>`).join("");
  if ($("trail").innerHTML !== trail) $("trail").innerHTML = trail;

  $("thinking").classList.toggle("on", s.thinking);
  setText("thinking-text", s.thinking ? `thinking… ${fmtMs(s.think_ms)}` : s.paused ? "paused" : s.game_over ? "round over" : "waiting for pacing");

  $("overlay").classList.toggle("show", s.game_over);
  const last = s.rounds[0];
  const hit = { self: "its own body", wall: "the fence", rock: "a rock" };
  setText("overlay-text", s.game_over ? `${last && last.round === s.round ? (hit[last.cause] ? "Hit " + hit[last.cause] : last.cause) + " · " : ""}score ${s.score}${s.auto_restart && !s.paused ? " · next round starting" : " · press R"}` : "");

  // settings
  setText("pause-label", s.paused ? "Resume" : "Pause");
  $("btn-pause").classList.toggle("on", s.paused);
  if (document.activeElement !== $("speed")) $("speed").value = s.speed_level;
  setText("speed-v", `L${s.speed_level}/5`);
  sizeOptions(s);
  setText("food-v", String(s.rules.foods));
  $("food-dec").disabled = s.rules.foods <= 1;
  $("food-inc").disabled = s.rules.foods >= s.rules.max_foods;
  $("sw-obst").classList.toggle("on", s.rules.obstacles);
  $("sw-shield").classList.toggle("on", s.shield);
  $("sw-auto").classList.toggle("on", s.auto_restart);

  // decision
  const d = s.decision;
  DIRS.forEach((dir, i) => {
    const row = probRows[dir], p = d ? d.probs[i] : 0;
    row.classList.toggle("exec", !!d && d.chosen === dir);
    row.classList.toggle("top", !!d && d.model_top === dir);
    row.querySelector(".bar i").style.width = (p * 100).toFixed(1) + "%";
    row.querySelector(".val").textContent = p.toFixed(2);
    dpad[dir].classList.toggle("on", !!d && d.chosen === dir);
    dpad[dir].classList.toggle("top", !!d && d.model_top === dir);
  });
  setText("exec", d && d.chosen ? d.chosen : "-");
  const tag = $("exec-tag");
  if (!d) { tag.className = "tag"; tag.textContent = s.thinking ? "first decision…" : "waiting"; }
  else if (d.intervened) { tag.className = "tag warn"; tag.textContent = `shield override (model: ${d.model_top})`; }
  else { tag.className = "tag good"; tag.textContent = `model pick · ${fmtMs(d.ms)}`; }

  // safety
  setText("risk-v", s.decision ? s.risk.toFixed(2) : "-");
  const ri = $("risk");
  ri.style.width = (s.risk * 100) + "%";
  ri.className = s.risk > 0.5 ? "bad" : s.risk > 0.15 ? "warn" : "";
  setText("reach-v", s.decision ? (s.reachable >= 1 ? "yes" : "no") : "-");
  $("reach").style.width = s.decision ? (s.reachable * 100) + "%" : "0";
  $("shield-v").innerHTML = s.shield ? '<span class="tag good">ON</span>' : '<span class="tag bad">OFF · raw model</span>';
  setText("ovr-round", String(s.round_interventions));
  setText("ovr-total", String(s.totals.interventions));

  // engine
  setText("e-model", "Laya · " + s.info.checkpoint);
  setText("e-engine", s.info.engine);
  setText("e-device", s.info.device + " · local");
  setText("e-last", d ? fmtMs(d.ms) : "-");
  setText("e-dps", s.dps.toFixed(2));
  setText("e-err", String(s.totals.model_errors));
  setText("e-error", s.error ? "Model error: " + s.error : "");

  // tiles
  const t = s.totals;
  setText("t-dec", t.decisions.toLocaleString());
  setText("t-dec-s", `${s.round_decisions} this round`);
  setText("t-avg", t.decisions ? fmtMs(t.avg_ms) : "-");
  setText("t-range", t.decisions ? `range ${Math.round(t.min_ms)}–${Math.round(t.max_ms)} ms` : "range -");
  setText("t-p50", t.window ? fmtMs(t.p50_ms) : "-");
  setText("t-p95", t.window ? fmtMs(t.p95_ms) : "-");
  setText("t-win", `last ${t.window} moves`);
  setText("t-food", String(t.food));
  setText("t-food-s", t.decisions ? `1 per ${(t.decisions / Math.max(1, t.food)).toFixed(1)} moves` : "-");
  setText("t-rounds", String(t.rounds_played));
  setText("t-deaths", `fence ${t.deaths_wall} · rock ${t.deaths_rock} · self ${t.deaths_self}`);
  setText("t-shield", t.decisions ? (100 * t.interventions / t.decisions).toFixed(1) + "%" : "-");
  const done = s.rounds;
  setText("t-mean", done.length ? (done.reduce((a, r) => a + r.score, 0) / done.length).toFixed(1) : "-");
  setText("t-mean-s", done.length ? `best ${Math.max(...done.map((r) => r.score))} of ${done.length} rounds` : "finished rounds");
  setText("c-lat-aside", t.window ? `last ${t.window} decisions` : "ms per decision");
  setText("c-score-aside", done.length ? `last ${Math.min(30, done.length)} rounds` : "last rounds");

  // tables
  setText("rounds-aside", done.length ? `${done.length} finished` : "");
  $("rounds").innerHTML = done.length ? done.map((r) => `<tr>
      <td class="num">${String(r.round).padStart(2, "0")}</td><td class="num">${r.size}</td><td class="num"><b>${r.score}</b></td>
      <td class="num">${r.length}</td><td class="num">${r.decisions}</td><td class="num">${r.interventions}</td>
      <td class="num">${r.avg_ms.toFixed(0)}</td><td class="num">${fmtDur(r.duration_s)}</td>
      <td class="l"><span class="tag ${r.cause === "reset" ? "" : r.cause === "board full" ? "good" : "bad"}">${r.cause === "wall" ? "fence" : r.cause}</span></td></tr>`).join("")
    : `<tr><td class="empty" colspan="9">Rounds appear here when they end</td></tr>`;
  const log = s.moves.slice(-40).reverse();
  $("log").innerHTML = log.length ? log.map((m) => {
    const top = Math.max(...m.probs);
    return `<tr>
      <td class="num">${m.n}</td><td class="l">${ARROW[m.model_top]} ${m.model_top}</td>
      <td class="l">${ARROW[m.chosen]} ${m.chosen}${m.intervened ? ' <span class="tag warn">shield</span>' : ""}${m.ate ? ' <span class="tag good">food</span>' : ""}</td>
      <td class="num">${top.toFixed(2)}</td>
      <td><span class="mini-probs" title="${DIRS.map((d, k) => d + " " + m.probs[k].toFixed(3)).join(", ")}">${m.probs.map((p) => `<i><b style="height:${(p * 100).toFixed(0)}%"></b></i>`).join("")}</span></td>
      <td class="num">${m.risk.toFixed(2)}</td><td class="num">${m.ms.toFixed(0)}</td><td class="num">${m.score}</td></tr>`;
  }).join("") : `<tr><td class="empty" colspan="8">Waiting for the first decision</td></tr>`;

  drawCharts();
}

