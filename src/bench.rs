//! `--bench` and `--verify`: measure and cross-check the model backends without the terminal UI.

use crate::{
    ai::{EnginePref, LayaBrain},
    game::{Dir, GameSnapshot, SnakeGame},
};
use anyhow::Result;
use std::{path::Path, time::Instant};

/// Realistic, reproducible positions: a greedy safe-move player on a 24x14 board, snapshotted at
/// every step, each with a fixed key-to-direction shuffle.
fn scenarios(n: usize) -> Vec<(GameSnapshot, [Dir; 4])> {
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }
    let mut rng = Lcg(42);
    let mut game = SnakeGame::new(24, 14);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let snap = GameSnapshot::from_game(&game);
        let mut order = Dir::ALL;
        for i in (1..4).rev() {
            order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        out.push((snap.clone(), order));

        let (h, f) = (snap.head(), snap.food.expect("food"));
        let mut best: Option<(Dir, i32)> = None;
        for d in Dir::ALL {
            if snap.is_fatal(d) {
                continue;
            }
            let (dx, dy) = d.delta();
            let score =
                ((f.x - (h.x + dx)).abs() + (f.y - (h.y + dy)).abs()) * 4 + (rng.next() % 3) as i32;
            if best.is_none_or(|(_, s)| score < s) {
                best = Some((d, score));
            }
        }
        match best {
            Some((d, _)) => game.step(d),
            None => game.reset(),
        }
        if game.game_over {
            game.reset();
        }
    }
    out
}

fn summarize(label: &str, times: &[f64]) {
    let mut s = times.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    println!(
        "{label}: mean {mean:.0} ms   p50 {:.0} ms   min {:.0} ms   max {:.0} ms   (n={})",
        s[s.len() / 2],
        s[0],
        s[s.len() - 1],
        s.len()
    );
}

/// Time `runs` decisions after a warm-up. Long runs on a laptop show thermal throttling: compare
/// the first and last decisions.
pub fn bench(
    model_dir: &Path,
    pref: EnginePref,
    threads: Option<usize>,
    runs: usize,
) -> Result<()> {
    let t = Instant::now();
    let brain = LayaBrain::load(model_dir, pref, threads)?;
    println!(
        "engine: {}   load: {:.2}s",
        brain.engine_str(),
        t.elapsed().as_secs_f64()
    );
    println!("warm-up: {:.0} ms", brain.warm_up()?);
    let _ = brain.take_profile();

    let sc = scenarios(runs.clamp(1, 512));
    let mut times = Vec::with_capacity(runs);
    for i in 0..runs {
        let (snap, order) = &sc[i % sc.len()];
        let t = Instant::now();
        brain.decide_ordered(snap, *order)?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    summarize("decision latency", &times);
    if runs >= 20 {
        let k = (runs / 4).max(5);
        summarize("  first quarter   ", &times[..k]);
        summarize("  last quarter    ", &times[runs - k..]);
    }
    if let Some(p) = brain.take_profile().filter(|p| p.lines().count() > 1) {
        println!("per-op time over {runs} decisions:\n{p}");
    }
    Ok(())
}

/// Run the fast engine and the reference candle implementation on identical prompts and compare.
pub fn verify(model_dir: &Path, threads: Option<usize>, runs: usize) -> Result<()> {
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (model_dir, threads, runs);
        println!("the fast engine is x86-64 only; nothing to verify on this platform");
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    {
        let fast = LayaBrain::load(model_dir, EnginePref::Auto, threads)?;
        if fast.engine_str() == "Rust/Candle" {
            println!("the fast engine is not in use on this build/CPU; nothing to verify");
            return Ok(());
        }
        let reference = LayaBrain::load(model_dir, EnginePref::Candle, None)?;
        fast.warm_up()?;

        let sc = scenarios(runs.clamp(1, 512));
        let (mut worst, mut same, mut t_fast, mut t_ref) = (0f64, 0usize, Vec::new(), Vec::new());
        for (snap, order) in &sc {
            let a = fast.decide_ordered(snap, *order)?;
            let b = reference.decide_ordered(snap, *order)?;
            t_fast.push(a.inference_ms);
            t_ref.push(b.inference_ms);
            worst = worst.max(
                a.probs
                    .iter()
                    .zip(&b.probs)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max),
            );
            same += (a.top == b.top) as usize;
        }
        summarize("fast      ", &t_fast);
        summarize("reference ", &t_ref);
        println!(
            "same chosen move: {same}/{}   worst probability difference: {worst:.4}",
            sc.len()
        );
        if same == sc.len() && worst <= 0.001 {
            println!("OK: outputs match the reference implementation");
            Ok(())
        } else {
            anyhow::bail!("outputs differ from the reference implementation")
        }
    }
}
