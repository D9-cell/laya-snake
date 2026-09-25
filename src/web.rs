//! `--web`: the same live game, served as a browser dashboard.
//!
//! One thread runs the game loop exactly like the terminal UI does. A tiny std-only HTTP server
//! serves the page, streams the state over Server-Sent Events, and takes commands as POSTs.

use crate::{
    ai::Decision,
    app::{App, BrainInfo, InferenceWorker, MIN_H, MIN_W, SPEED_LEVELS_MS},
    game::{Dir, GameSnapshot, MAX_FOODS},
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

const INDEX_HTML: &str = include_str!("web/index.html");

/// Page files compiled into the binary, by URL path.
const STATIC_FILES: &[(&str, &str, &str)] = &[
    ("/css/dashboard.css", "text/css; charset=utf-8", include_str!("web/css/dashboard.css")),
    ("/js/app.js", JS, include_str!("web/js/app.js")),
    ("/js/gameState.js", JS, include_str!("web/js/gameState.js")),
    ("/js/snakeMotion.js", JS, include_str!("web/js/snakeMotion.js")),
    ("/js/dashboard.js", JS, include_str!("web/js/dashboard.js")),
    ("/js/renderer2d.js", JS, include_str!("web/js/renderer2d.js")),
    ("/js/renderer3d.js", JS, include_str!("web/js/renderer3d.js")),
    ("/js/scene3d.js", JS, include_str!("web/js/scene3d.js")),
    ("/js/snake3d.js", JS, include_str!("web/js/snake3d.js")),
    ("/js/rabbit3d.js", JS, include_str!("web/js/rabbit3d.js")),
    ("/js/textures3d.js", JS, include_str!("web/js/textures3d.js")),
    ("/js/util.js", JS, include_str!("web/js/util.js")),
    ("/vendor/three/build/three.module.min.js", JS, include_str!("web/vendor/three/build/three.module.min.js")),
    ("/vendor/three/addons/loaders/GLTFLoader.js", JS, include_str!("web/vendor/three/addons/loaders/GLTFLoader.js")),
    ("/vendor/three/addons/utils/BufferGeometryUtils.js", JS, include_str!("web/vendor/three/addons/utils/BufferGeometryUtils.js")),
    ("/vendor/three/addons/utils/SkeletonUtils.js", JS, include_str!("web/vendor/three/addons/utils/SkeletonUtils.js")),
    ("/vendor/three/addons/controls/OrbitControls.js", JS, include_str!("web/vendor/three/addons/controls/OrbitControls.js")),
];
const JS: &str = "text/javascript; charset=utf-8";
const MAX_W: usize = 64;
const MAX_H: usize = 40;
/// How long a finished game stays on screen before auto-restart starts the next round.
const RESTART_AFTER: Duration = Duration::from_millis(2500);
/// How many recent moves go out with each state update.
const MOVES_SENT: usize = 160;

struct Web {
    app: App,
    auto_restart: bool,
    over_since: Option<Instant>,
    /// Bumped on every visible change; the event streams send only when it moves.
    version: u64,
}

type SharedWeb = Arc<Mutex<Web>>;

fn lock(web: &SharedWeb) -> std::sync::MutexGuard<'_, Web> {
    web.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn run<F>(
    decide: F,
    info: BrainInfo,
    width: usize,
    height: usize,
    host: &str,
    port: u16,
) -> Result<()>
where
    F: Fn(&GameSnapshot) -> Result<Decision, String> + Send + 'static,
{
    let listener =
        TcpListener::bind((host, port)).with_context(|| format!("binding {host}:{port}"))?;
    let worker = InferenceWorker::new(decide);
    let app = App::new(
        worker,
        info,
        width.clamp(MIN_W, MAX_W),
        height.clamp(MIN_H, MAX_H),
    );
    let web = Arc::new(Mutex::new(Web {
        app,
        auto_restart: true,
        over_since: None,
        version: 1,
    }));

    {
        let web = Arc::clone(&web);
        thread::Builder::new()
            .name("laya-game".into())
            .spawn(move || game_loop(web))?;
    }

    let shown = if host == "0.0.0.0" { "localhost" } else { host };
    println!("Dashboard: http://{shown}:{port}/   (Ctrl+C to quit)");

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let web = Arc::clone(&web);
        let _ = thread::Builder::new()
            .name("laya-http".into())
            .spawn(move || {
                let _ = handle(stream, &web);
            });
    }
    Ok(())
}

fn game_loop(web: SharedWeb) {
    loop {
        let wait = {
            let mut w = lock(&web);
            let mut changed = w.app.tick();
            if w.app.game.game_over {
                let since = *w.over_since.get_or_insert_with(Instant::now);
                if w.auto_restart && !w.app.paused && since.elapsed() >= RESTART_AFTER {
                    w.app.reset();
                    w.over_since = None;
                    changed = true;
                }
            } else {
                w.over_since = None;
            }
            if changed {
                w.version += 1;
            }
            if w.app.animating() || w.app.pending.is_some() {
                5
            } else {
                25
            }
        };
        thread::sleep(Duration::from_millis(wait));
    }
}

fn command(w: &mut Web, path: &str) -> bool {
    let mut parts = path.split('/');
    let name = parts.next().unwrap_or("");
    let arg = parts.next();
    match name {
        "pause" => w.app.paused = !w.app.paused,
        "reset" => {
            w.app.reset();
            w.over_since = None;
        }
        "shield" => w.app.shield_enabled = !w.app.shield_enabled,
        "obstacles" => {
            let on = !w.app.game.rules.obstacles;
            w.app.set_obstacles(on);
            w.over_since = None;
        }
        "foods" => match arg.and_then(|a| a.parse::<usize>().ok()) {
            Some(n) if (1..=MAX_FOODS).contains(&n) => w.app.set_food_count(n),
            _ => return false,
        },
        "auto" => w.auto_restart = !w.auto_restart,
        "faster" => w.app.speed_level = (w.app.speed_level + 1).min(4),
        "slower" => w.app.speed_level = w.app.speed_level.saturating_sub(1),
        "speed" => match arg.and_then(|a| a.parse::<usize>().ok()) {
            Some(l) if (1..=5).contains(&l) => w.app.speed_level = l - 1,
            _ => return false,
        },
        "grow" | "shrink" => {
            let d: i32 = if name == "grow" { 1 } else { -1 };
            let nw = (w.app.game.width as i32 + 4 * d).clamp(MIN_W as i32, MAX_W as i32);
            let nh = (w.app.game.height as i32 + 2 * d).clamp(MIN_H as i32, MAX_H as i32);
            w.app.resize(nw as usize, nh as usize);
        }
        "size" => {
            let Some((a, b)) = arg.and_then(|a| a.split_once('x')) else {
                return false;
            };
            let (Ok(nw), Ok(nh)) = (a.parse::<usize>(), b.parse::<usize>()) else {
                return false;
            };
            w.app.resize(nw.clamp(MIN_W, MAX_W), nh.clamp(MIN_H, MAX_H));
        }
        _ => return false,
    }
    w.version += 1;
    true
}

fn dir_name(d: Dir) -> &'static str {
    d.name()
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

fn state_json(w: &Web) -> Value {
    let a = &w.app;
    let g = &a.game;
    let status = if a.paused {
        "paused"
    } else if g.game_over {
        "over"
    } else {
        "live"
    };
    let decision = a.last_decision.as_ref().map(|d| {
        json!({
            "probs": d.probs,
            "model_top": dir_name(d.top),
            "chosen": a.last_chosen.map(dir_name),
            "intervened": a.last_intervened,
            "ms": d.inference_ms,
            "confidence": d.confidence,
        })
    });

    let recent: Vec<f64> = a.moves.iter().map(|m| m.inference_ms).collect();
    let mut sorted = recent.clone();
    sorted.sort_by(f64::total_cmp);
    let t = &a.totals;

    let skip = a.moves.len().saturating_sub(MOVES_SENT);
    let moves: Vec<Value> = a
        .moves
        .iter()
        .skip(skip)
        .map(|m| {
            json!({
                "n": m.n,
                "round": m.round,
                "probs": m.probs,
                "model_top": dir_name(m.model_top),
                "chosen": dir_name(m.chosen),
                "intervened": m.intervened,
                "ms": m.inference_ms,
                "risk": m.risk,
                "reachable": m.reachable,
                "score": m.score,
                "ate": m.ate,
            })
        })
        .collect();
    let rounds: Vec<Value> = a
        .rounds
        .iter()
        .rev()
        .map(|r| {
            json!({
                "round": r.round,
                "size": format!("{}x{}", r.width, r.height),
                "score": r.score,
                "length": r.length,
                "decisions": r.decisions,
                "interventions": r.interventions,
                "avg_ms": r.avg_ms,
                "duration_s": r.duration_s,
                "cause": r.cause,
            })
        })
        .collect();

    json!({
        "version": w.version,
        "status": status,
        "paused": a.paused,
        "game_over": g.game_over,
        "auto_restart": w.auto_restart,
        "speed_level": a.speed_level + 1,
        "speed_ms": SPEED_LEVELS_MS[a.speed_level],
        "width": g.width,
        "height": g.height,
        "limits": { "min_w": MIN_W, "min_h": MIN_H, "max_w": MAX_W, "max_h": MAX_H },
        "round": a.round_no,
        "score": g.score,
        "length": g.snake.len(),
        "best": a.best,
        "fill": g.board_fill_ratio(),
        "snake": g.snake.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        "foods": g.foods.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        "obstacles": g.obstacles.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        "rules": { "foods": g.rules.foods, "obstacles": g.rules.obstacles, "max_foods": MAX_FOODS },
        "direction": dir_name(g.direction),
        "info": {
            "checkpoint": a.info.checkpoint,
            "engine": a.info.engine,
            "device": a.info.device,
            "mock": a.info.checkpoint == "mock",
        },
        "thinking": a.awaiting_result && !a.paused,
        "think_ms": if a.awaiting_result { a.think_start.elapsed().as_secs_f64() * 1000.0 } else { 0.0 },
        "decision": decision,
        "risk": a.last_risk,
        "reachable": a.last_reachable,
        "shield": a.shield_enabled,
        "round_interventions": a.shield_count,
        "round_decisions": a.decisions_count,
        "round_s": a.round_start.elapsed().as_secs_f64(),
        "dps": a.decisions_per_sec(),
        "trail": a.trail.iter().map(|d| dir_name(*d)).collect::<Vec<_>>(),
        "error": a.error_message,
        "uptime_s": a.start_time.elapsed().as_secs_f64(),
        "totals": {
            "decisions": t.decisions,
            "avg_ms": if t.decisions > 0 { t.inference_ms / t.decisions as f64 } else { 0.0 },
            "min_ms": t.min_ms.unwrap_or(0.0),
            "max_ms": t.max_ms,
            "p50_ms": percentile(&sorted, 0.5),
            "p95_ms": percentile(&sorted, 0.95),
            "window": sorted.len(),
            "interventions": t.interventions,
            "food": t.food,
            "model_errors": t.model_errors,
            "deaths_wall": t.deaths_wall,
            "deaths_rock": t.deaths_rock,
            "deaths_self": t.deaths_self,
            "dir_counts": t.dir_counts,
            "rounds_played": a.rounds.len(),
        },
        "moves": moves,
        "rounds": rounds,
    })
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

fn handle(mut stream: TcpStream, web: &SharedWeb) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0).min(4096);
            }
        }
    }
    if content_length > 0 {
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body)?;
    }
    let path = path.split('?').next().unwrap_or("/");

    match (method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        ),
        ("GET", "/api/state") => {
            let body = state_json(&lock(web)).to_string();
            respond(&mut stream, "200 OK", "application/json", body.as_bytes())
        }
        ("GET", "/api/events") => stream_events(stream, web),
        ("POST", p) if p.starts_with("/api/cmd/") => {
            let mut w = lock(web);
            if command(&mut w, &p["/api/cmd/".len()..]) {
                let body = state_json(&w).to_string();
                drop(w);
                respond(&mut stream, "200 OK", "application/json", body.as_bytes())
            } else {
                drop(w);
                respond(
                    &mut stream,
                    "400 Bad Request",
                    "text/plain",
                    b"unknown command",
                )
            }
        }
        ("GET", "/favicon.ico") => respond(&mut stream, "204 No Content", "text/plain", b""),
        ("GET", p) if p.starts_with("/assets/") => match read_asset(&p["/assets/".len()..]) {
            Some((body, ctype)) => respond(&mut stream, "200 OK", ctype, &body),
            None => respond(&mut stream, "404 Not Found", "text/plain", b"no such asset"),
        },
        ("GET", p) => match STATIC_FILES.iter().find(|(path, _, _)| *path == p) {
            Some((_, ctype, body)) => respond(&mut stream, "200 OK", ctype, body.as_bytes()),
            None => respond(&mut stream, "404 Not Found", "text/plain", b"not found"),
        },
        _ => respond(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
}

/// Optional 3D assets (models, textures) read from disk so they can be swapped without a rebuild:
/// `$LAYA_WEB_ASSETS`, else `src/web/assets` under the working directory, else `assets` beside the
/// binary. The page falls back to built-in geometry when a file is missing.
fn read_asset(rel: &str) -> Option<(Vec<u8>, &'static str)> {
    let safe = !rel.is_empty()
        && rel
            .split('/')
            .all(|part| !part.is_empty() && !part.starts_with('.'))
        && rel
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/".contains(c));
    if !safe {
        return None;
    }
    let ctype = match rel.rsplit('.').next()? {
        "glb" => "model/gltf-binary",
        "gltf" => "model/gltf+json",
        "bin" => "application/octet-stream",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ktx2" => "image/ktx2",
        _ => return None,
    };
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("LAYA_WEB_ASSETS") {
        roots.push(std::path::PathBuf::from(dir));
    }
    roots.push(std::path::PathBuf::from("src/web/assets"));
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("assets"))) {
        roots.push(dir);
    }
    roots
        .into_iter()
        .find_map(|root| std::fs::read(root.join(rel)).ok())
        .map(|body| (body, ctype))
}

/// Server-Sent Events: one full state whenever it changes, at most every 40 ms, and at least
/// every 250 ms so timers keep moving.
fn stream_events(mut stream: TcpStream, web: &SharedWeb) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n",
    )?;
    let mut sent_version = 0;
    let mut sent_at = Instant::now() - Duration::from_secs(1);
    loop {
        let body = {
            let w = lock(web);
            let thinking = w.app.awaiting_result && !w.app.paused;
            let due = if thinking {
                Duration::from_millis(100)
            } else {
                Duration::from_millis(250)
            };
            if w.version != sent_version || sent_at.elapsed() >= due {
                sent_version = w.version;
                Some(state_json(&w).to_string())
            } else {
                None
            }
        };
        if let Some(body) = body {
            stream.write_all(b"data: ")?;
            stream.write_all(body.as_bytes())?;
            stream.write_all(b"\n\n")?;
            stream.flush()?;
            sent_at = Instant::now();
        }
        thread::sleep(Duration::from_millis(40));
    }
}
