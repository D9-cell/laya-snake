use crate::game::{Dir, GameSnapshot, SnakeGame};
use anyhow::{Context, Result};
use laya::{Agent, Options, Question};
use rand::rng;
use rand::seq::SliceRandom;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

#[cfg(target_arch = "x86_64")]
use crate::engine::Engine;

const NEUTRAL_KEYS: [&str; 4] = ["k0", "k1", "k2", "k3"];

#[derive(Debug, Clone)]
pub struct Decision {
    /// Model probability per direction, indexed by `Dir::index`.
    pub probs: [f64; 4],
    pub top: Dir,
    #[allow(dead_code)]
    pub confidence: Option<f64>,
    pub inference_ms: f64,
}

/// Which implementation runs the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePref {
    /// The fast CPU engine when this build/CPU supports it, otherwise candle.
    Auto,
    /// Always the reference candle implementation from the `laya` crate.
    Candle,
}

enum Backend {
    #[cfg(target_arch = "x86_64")]
    Fast(Box<Engine>),
    Candle(Box<Agent>),
}

pub struct LayaBrain {
    backend: Backend,
    checkpoint: String,
}

/// The question the model sees for one position, with the direction behind each neutral key.
struct Prompt {
    instructions: String,
    state: String,
    key_to_dir: [Dir; 4],
}

fn build_prompt(game: &GameSnapshot, order: [Dir; 4]) -> Result<Prompt> {
    let h = game.head();
    let food = game.nearest_food().context("snake has no food")?;
    let closer = game.toward_food();

    let mut lines = Vec::with_capacity(4);
    for (key, direction) in NEUTRAL_KEYS.iter().zip(order) {
        let toward = closer[direction.index()];
        let danger = if game.is_fatal(direction) {
            "UNSAFE: ends the game immediately."
        } else {
            "Safe."
        };
        let progress = if toward {
            "moves closer to the food"
        } else {
            "moves farther from the food"
        };
        lines.push(format!("{key}: {progress}. {danger}"));
    }

    Ok(Prompt {
        instructions: format!(
            "Pick the option that moves closer to the food and is not marked UNSAFE.\n{}",
            lines.join("\n")
        ),
        state: format!(
            "Snake head at ({},{}). Food at ({},{}). Snake length={}.",
            h.x,
            h.y,
            food.x,
            food.y,
            game.snake.len()
        ),
        key_to_dir: order,
    })
}

fn threads_from_env() -> usize {
    std::env::var("LAYA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
}

impl LayaBrain {
    pub fn load(model_dir: &Path, pref: EnginePref, threads: Option<usize>) -> Result<Self> {
        let checkpoint = model_dir
            .file_name()
            .and_then(|x| x.to_str())
            .unwrap_or("root")
            .to_string();

        #[cfg(target_arch = "x86_64")]
        if pref == EnginePref::Auto && !cfg!(any(feature = "cuda", feature = "metal")) {
            match Engine::load(model_dir, threads.unwrap_or_else(threads_from_env)) {
                Ok(engine) => {
                    return Ok(Self {
                        backend: Backend::Fast(Box::new(engine)),
                        checkpoint,
                    })
                }
                Err(e) => {
                    eprintln!("[laya-snake] fast CPU engine unavailable ({e:#}); using candle")
                }
            }
        }
        let _ = (pref, threads);

        let agent =
            Agent::from_dir(model_dir, Options::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            backend: Backend::Candle(Box::new(agent)),
            checkpoint,
        })
    }

    pub fn checkpoint(&self) -> &str {
        &self.checkpoint
    }

    pub fn device_str(&self) -> String {
        // The current laya crate selects the runtime device through its Options.
        // Keep this display stable across crate versions.
        if cfg!(feature = "cuda") {
            "CUDA".into()
        } else if cfg!(feature = "metal") {
            "METAL".into()
        } else {
            "CPU".into()
        }
    }

    /// Short name of the implementation in use, for the UI.
    pub fn engine_str(&self) -> String {
        match &self.backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Fast(e) => format!("Rust/AVX2 f16 x{}", e.threads()),
            Backend::Candle(_) => "Rust/Candle".into(),
        }
    }

    /// One throw-away inference so page faults and thread start-up happen before play. Returns ms.
    pub fn warm_up(&self) -> Result<f64> {
        match &self.backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Fast(e) => e.warm_up(),
            Backend::Candle(_) => {
                let snap = GameSnapshot::from_game(&SnakeGame::new(24, 14));
                let t = Instant::now();
                self.decide(&snap)?;
                Ok(t.elapsed().as_secs_f64() * 1e3)
            }
        }
    }

    /// Per-op timing breakdown of the fast engine (`LAYA_PROFILE=1`), if any.
    pub fn take_profile(&self) -> Option<String> {
        match &self.backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Fast(e) => Some(e.take_profile()),
            Backend::Candle(_) => None,
        }
    }

    pub fn decide(&self, game: &GameSnapshot) -> Result<Decision> {
        // The option keys are neutral and re-mapped to directions at random on every request, so the
        // model cannot lean on a fixed key-to-direction habit.
        let mut order = Dir::ALL;
        order.shuffle(&mut rng());
        self.decide_ordered(game, order)
    }

    /// [`decide`](Self::decide) with an explicit key order (`order[i]` is the direction behind `k{i}`).
    pub fn decide_ordered(&self, game: &GameSnapshot, order: [Dir; 4]) -> Result<Decision> {
        let prompt = build_prompt(game, order)?;
        let t0 = Instant::now();

        let (probs_by_key, best_key, confidence) = match &self.backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Fast(engine) => {
                let c = engine.choose(&prompt.instructions, &prompt.state, &NEUTRAL_KEYS)?;
                (c.probs, c.best, Some(c.confidence))
            }
            Backend::Candle(agent) => {
                let question = Question::choice(&prompt.instructions, NEUTRAL_KEYS.iter().copied());
                let questions = vec![("move".to_string(), question)];
                let response = agent
                    .system_one(&json!(prompt.state), &questions)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let answer = response
                    .answers
                    .get("move")
                    .context("Laya response has no answer")?;
                let probabilities = answer["probabilities"]
                    .as_object()
                    .context("Laya response has no choice probabilities")?;
                let mut probs = Vec::with_capacity(4);
                for key in NEUTRAL_KEYS {
                    probs.push(
                        probabilities
                            .get(key)
                            .and_then(|v| v.as_f64())
                            .with_context(|| format!("missing probability for {key}"))?,
                    );
                }
                let choice = answer["choice"]
                    .as_str()
                    .context("Laya response has no choice")?;
                let best = NEUTRAL_KEYS
                    .iter()
                    .position(|k| *k == choice)
                    .with_context(|| format!("unknown selected Laya key: {choice}"))?;
                (probs, best, answer["confidence"].as_f64())
            }
        };
        let inference_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut probs = [0.0; 4];
        for (key, dir) in prompt.key_to_dir.iter().enumerate() {
            probs[dir.index()] = probs_by_key[key];
        }
        Ok(Decision {
            probs,
            top: prompt.key_to_dir[best_key],
            confidence,
            inference_ms,
        })
    }
}

pub fn checkpoint_dir(checkpoint: &str) -> PathBuf {
    PathBuf::from("models").join(match checkpoint {
        "root" => "laya-root",
        "multilingual" => "laya-multilingual",
        "typed" => "laya-typed-decisions",
        _ => "laya-root",
    })
}

pub fn download_checkpoint_if_needed(model: &str, checkpoint: &str) -> Result<PathBuf> {
    let local = PathBuf::from(model);
    if local.is_dir() && local.join("model.safetensors").exists() {
        return Ok(local);
    }

    let dir = checkpoint_dir(checkpoint);
    if dir.join("model.safetensors").exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)?;
    let subfolder = match checkpoint {
        "root" => String::new(),
        "multilingual" => "multilingual/".into(),
        "typed" => "typed-decisions/".into(),
        _ => anyhow::bail!("invalid checkpoint"),
    };

    let base = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model.trim_end_matches('/'),
        subfolder
    );

    let files = [
        ("model.safetensors", "model.safetensors"),
        ("encoder/config.json", "encoder/config.json"),
        ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json"),
        (
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer_config.json",
        ),
        ("rl_agent_config.json", "rl_agent_config.json"),
    ];

    for (remote, relative) in files {
        let destination = dir.join(relative);
        if destination.exists() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let url = format!("{base}{remote}");
        println!("[setup] downloading {url}");

        // Download beside the destination and rename on success, so an interrupted transfer is never
        // mistaken for a finished file on the next launch.
        let mut part = destination.clone().into_os_string();
        part.push(".part");
        let part = PathBuf::from(part);

        let status = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Invoke-WebRequest -UseBasicParsing -Uri '{}' -OutFile '{}'",
                        url,
                        part.display()
                    ),
                ])
                .status()?
        } else {
            Command::new("curl")
                .args(["-fL", "--retry", "3", "-o"])
                .arg(&part)
                .arg(&url)
                .status()?
        };

        if !status.success() {
            let _ = std::fs::remove_file(&part);
            anyhow::bail!(
                "failed to download {url}. Install curl or use --model <local-directory>."
            );
        }
        std::fs::rename(&part, &destination)?;
    }

    Ok(dir)
}
