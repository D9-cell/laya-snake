import { game, applyState, subscribe } from "./gameState.js";
import { trackMotion } from "./snakeMotion.js";
import { renderDashboard, drawCharts, setText } from "./dashboard.js";
import { CanvasRenderer } from "./renderer2d.js";

const $ = (id) => document.getElementById(id);
const VIEWS = ["2d", "3d", "cinematic"];

/* ---------- viewer preferences (this browser only) ---------- */
const prefs = { view: "3d", tod: "golden", grid: true, follow: false };
try { Object.assign(prefs, JSON.parse(localStorage.getItem("laya-snake-view") || "{}")); } catch (_) {}
if (!VIEWS.includes(prefs.view)) prefs.view = "3d";
const debug = new URLSearchParams(location.search).has("debug");

// Saves the viewer's scene preferences and applies them.
function savePrefs() {
  try { localStorage.setItem("laya-snake-view", JSON.stringify(prefs)); } catch (_) {}
  showPrefs();
  selectRenderer();
}

// Reflects the local scene preferences in the settings panel.
function showPrefs() {
  [...$("view-seg").children].forEach((b) => b.classList.toggle("on", b.dataset.v === prefs.view));
  $("tod-sel").value = prefs.tod;
  $("sw-grid").classList.toggle("on", prefs.grid);
  $("sw-follow").classList.toggle("on", prefs.follow);
  setText("hdr-view", { "2d": "2D view", "3d": "3D view", cinematic: "Realistic view" }[prefs.view]);
}

/* ---------- commands (unchanged server API) ---------- */
// Sends one control command to the game server and applies the state it returns.
async function cmd(name) {
  try {
    const r = await fetch("/api/cmd/" + name, { method: "POST" });
    if (r.ok) applyState(await r.json());
  } catch (_) {}
}
$("btn-pause").onclick = () => cmd("pause");
$("btn-reset").onclick = () => cmd("reset");
$("sw-shield").onclick = () => cmd("shield");
$("sw-auto").onclick = () => cmd("auto");
$("sw-obst").onclick = () => cmd("obstacles");
$("food-dec").onclick = () => game.state && cmd("foods/" + (game.state.rules.foods - 1));
$("food-inc").onclick = () => game.state && cmd("foods/" + (game.state.rules.foods + 1));
$("speed").oninput = (e) => cmd("speed/" + e.target.value);
$("size-sel").onchange = (e) => cmd("size/" + e.target.value);
$("tod-sel").onchange = (e) => { prefs.tod = e.target.value; savePrefs(); };
$("sw-grid").onclick = () => { prefs.grid = !prefs.grid; savePrefs(); };
$("sw-follow").onclick = () => { prefs.follow = !prefs.follow; savePrefs(); };
[...$("view-seg").children].forEach((b) => b.onclick = () => { prefs.view = b.dataset.v; savePrefs(); });
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || /SELECT|INPUT/.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  const local = {
    v: () => { prefs.view = VIEWS[(VIEWS.indexOf(prefs.view) + 1) % VIEWS.length]; },
    g: () => { prefs.grid = !prefs.grid; },
    f: () => { prefs.follow = !prefs.follow; },
  };
  if (local[k]) { e.preventDefault(); local[k](); savePrefs(); return; }
  const map = { " ": "pause", r: "reset", s: "shield", a: "auto", o: "obstacles",
    arrowup: "faster", arrowdown: "slower", "+": "grow", "=": "grow", "-": "shrink", "_": "shrink" };
  if (map[k]) { e.preventDefault(); cmd(map[k]); }
});
const navLinks = [...document.querySelectorAll(".nav a")];
const spy = new IntersectionObserver((entries) => {
  entries.forEach((en) => {
    if (en.isIntersecting) navLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id));
  });
}, { rootMargin: "-30% 0px -60% 0px" });
["game", "stats", "model", "charts", "history"].forEach((id) => spy.observe($(id)));

/* ---------- renderers ---------- */
const canvas2d = new CanvasRenderer($("stage"), prefs);
let three = null, threeState = "idle";
let current = null;

// Loads the Three.js world the first time a 3D view is chosen; afterwards it is reused as is.
function ensureThree() {
  if (threeState !== "idle") return;
  threeState = "loading";
  $("load3d").classList.add("show");
  import("./renderer3d.js")
    .then(async (m) => {
      three = new m.ThreeRenderer($("stage3d"), prefs);
      if (debug) window.__laya3d = three;
      if (game.state) three.update(game.state);
      await three.init();
      if (debug) { $("debug3d").classList.add("show"); await three.enableDebug($("debug3d")); }
      threeState = "ready";
    })
    .catch((err) => {
      console.error("[3d] could not start the 3D view:", err);
      threeState = "failed";
      $("load3d").textContent = "3D view unavailable (WebGL). Showing 2D.";
      setTimeout(() => $("load3d").classList.remove("show"), 4000);
    })
    .finally(() => { if (threeState === "ready") $("load3d").classList.remove("show"); selectRenderer(); });
}

// Picks the renderer for the current view; the game itself is never touched by a switch.
function selectRenderer() {
  const wants3d = prefs.view !== "2d";
  if (wants3d) ensureThree();
  const next = wants3d && threeState === "ready" ? three : canvas2d;
  if (next === three) three.setMode(prefs.view);
  $("stage-box").dataset.mode = next === three ? prefs.view : "2d";
  if (next !== current) {
    if (current) current.deactivate();
    current = next;
    current.resize();
  }
}

new ResizeObserver(() => { canvas2d.resize(); if (three) three.resize(); }).observe($("stage-box"));
window.addEventListener("resize", () => { canvas2d.resize(); drawCharts(); });

/* ---------- state flow: server update -> state -> dashboard + both renderers ---------- */
subscribe((s, prev) => {
  trackMotion(prev, s);
  canvas2d.update(s);
  if (three) three.update(s);
  renderDashboard(s);
});

// One animation loop for the whole page; only the visible renderer draws.
function loop(now) {
  requestAnimationFrame(loop);
  if (game.state && current) current.render(now);
}

// Subscribes to the server's state stream (Server-Sent Events, unchanged).
function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => applyState(JSON.parse(e.data));
  es.onerror = () => {
    $("status").className = "pill down";
    setText("status-text", "DISCONNECTED");
  };
}

showPrefs();
selectRenderer();
fetch("/api/state").then((r) => r.json()).then(applyState).catch(() => {});
connect();
requestAnimationFrame(loop);
