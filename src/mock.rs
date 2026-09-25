//! `--mock`: a stand-in for the model, for working on the UI on a machine without the checkpoint.
//! It is labelled as such everywhere it shows up. Nothing here is Laya.

use crate::{
    ai::Decision,
    app::BrainInfo,
    game::{estimate_risk_and_reachability, Dir, GameSnapshot, SnakeGame},
};
use rand::Rng;
use std::time::{Duration, Instant};

pub fn info() -> BrainInfo {
    BrainInfo {
        checkpoint: "mock".into(),
        engine: "MOCK heuristic (no model)".into(),
        device: "CPU".into(),
    }
}

/// Softmax over a hand-written score: toward the food, away from walls and dead ends. Sleeps a
/// little so the UI sees a latency; the reported time is the real elapsed time of this call.
pub fn decide(snap: &GameSnapshot) -> Result<Decision, String> {
    let t0 = Instant::now();
    let mut rng = rand::rng();
    std::thread::sleep(Duration::from_millis(rng.random_range(70..140)));

    let food = snap.food.ok_or("snake has no food")?;
    let mut game = SnakeGame::new(snap.width, snap.height);
    game.snake = snap.snake.iter().copied().collect();
    game.food = snap.food;

    let head = snap.head();
    let dist = |x: i32, y: i32| (food.x - x).abs() + (food.y - y).abs();
    let mut logits = [0.0f64; 4];
    for d in Dir::ALL {
        let (dx, dy) = d.delta();
        let mut l = if dist(head.x + dx, head.y + dy) < dist(head.x, head.y) {
            1.6
        } else {
            0.0
        };
        if snap.is_fatal(d) {
            l -= 5.0;
        } else {
            let (risk, reachable) = estimate_risk_and_reachability(&game, d);
            l += reachable - 3.0 * risk;
        }
        logits[d.index()] = l + rng.random_range(-0.4..0.4);
    }
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut probs = [0.0; 4];
    for i in 0..4 {
        probs[i] = exps[i] / sum;
    }
    let top = Dir::ALL
        .into_iter()
        .max_by(|a, b| probs[a.index()].total_cmp(&probs[b.index()]))
        .unwrap_or(Dir::Up);
    Ok(Decision {
        probs,
        top,
        confidence: Some(probs[top.index()]),
        inference_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}
