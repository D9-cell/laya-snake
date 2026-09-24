//! Fast CPU inference for Laya (ModernBERT-large encoder + decision head).
//!
//! The model, the weights and the arithmetic are the ones the `laya` crate uses; what differs is how
//! the CPU executes them: weights stay f16 in a memory map and are widened inside the matmul
//! (halving the memory traffic that dominates a 108-token forward pass), and the ops candle runs as
//! chains of generic tensor kernels (attention, layer norm, GeGLU) are fused and parallelised.
//!
//! Only `choice` questions on a single sequence are supported, which is all the game asks.

// Kernel signatures take raw pointers and strides (many arguments); the Cephes exp coefficients are
// quoted to their published precision; `% 8` is kept for compatibility with older toolchains.
#![allow(
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]

pub mod kernels;
mod prompt;
mod weights;

use anyhow::{bail, Context, Result};
use kernels::*;
use laya::{AgentConfig, EncoderConfig};
use std::{path::Path, sync::Mutex, time::Instant};
use tokenizers::Tokenizer;
use weights::{F16Mat, Weights};

const LN_EPS_HEAD: f32 = 1e-5;
/// Question-type embedding row for `choice`.
const QTYPE_CHOICE: usize = 0;

struct EncLayer {
    attn_norm: Option<Vec<f32>>,
    wqkv: F16Mat,
    wo: F16Mat,
    mlp_norm: Vec<f32>,
    wi: F16Mat,
    mlp_wo: F16Mat,
    local: bool,
}

struct HeadLayer {
    norm1: (Vec<f32>, Vec<f32>),
    in_proj: F16Mat,
    in_proj_b: Vec<f32>,
    out_proj: F16Mat,
    out_proj_b: Vec<f32>,
    norm2: (Vec<f32>, Vec<f32>),
    lin1: F16Mat,
    lin1_b: Vec<f32>,
    lin2: F16Mat,
    lin2_b: Vec<f32>,
}

struct Scorer {
    norm: (Vec<f32>, Vec<f32>),
    fc1: F16Mat,
    fc1_b: Vec<f32>,
    fc2: F16Mat,
    fc2_b: f32,
}

/// Rotate-half RoPE tables `[positions, head_dim / 2]`, built the way the reference builds them.
struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn new(theta: f64, positions: usize) -> Self {
        let half = HEAD_DIM / 2;
        let inv: Vec<f32> = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect();
        let mut cos = vec![0f32; positions * half];
        let mut sin = vec![0f32; positions * half];
        for t in 0..positions {
            for j in 0..half {
                let f = t as f32 * inv[j];
                sin[t * half + j] = f.sin();
                cos[t * half + j] = f.cos();
            }
        }
        Self { cos, sin }
    }
}

/// Reusable activation buffers, sized once for the longest sequence.
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    qh: Vec<f32>,
    kh: Vec<f32>,
    vh: Vec<f32>,
    ao: Vec<f32>,
    wi: Vec<f32>,
    g: Vec<f32>,
}

impl Scratch {
    fn new(max_len: usize, d: usize, inter: usize) -> Self {
        let z = |n: usize| vec![0f32; n];
        Self {
            x: z(max_len * d),
            h: z(max_len * d),
            qkv: z(max_len * 3 * d),
            qh: z(max_len * d),
            kh: z(max_len * d),
            vh: z(max_len * d),
            ao: z(max_len * d),
            wi: z(max_len * 2 * inter.max(2 * d)),
            g: z(max_len * inter),
        }
    }
}

/// Optional per-op wall-clock breakdown (`LAYA_PROFILE=1`), summed over layers.
#[derive(Default)]
pub struct Profile {
    enabled: bool,
    acc: Vec<(&'static str, f64)>,
}

impl Profile {
    fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        if !self.enabled {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match self.acc.iter_mut().find(|(n, _)| *n == name) {
            Some((_, v)) => *v += ms,
            None => self.acc.push((name, ms)),
        }
        r
    }

    pub fn report(&self) -> String {
        let total: f64 = self.acc.iter().map(|(_, v)| v).sum();
        let mut rows: Vec<_> = self.acc.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rows.iter()
            .map(|(n, v)| format!("  {n:<22}{v:8.1} ms {:5.1}%\n", v / total * 100.0))
            .collect::<String>()
            + &format!("  {:<22}{total:8.1} ms\n", "total (timed)")
    }
}

/// A `choice` answer, post-processed exactly as `laya::Agent::system_one` does.
#[derive(Debug, Clone)]
pub struct Choice {
    /// Calibrated probability per option, rounded to 4 decimals.
    pub probs: Vec<f64>,
    pub best: usize,
    /// `1 - normalised entropy`, rounded to 4 decimals.
    pub confidence: f64,
    /// Encoder sequence length, for diagnostics.
    #[allow(dead_code)]
    pub input_tokens: usize,
}

pub struct Engine {
    d: usize,
    nh: usize,
    inter: usize,
    ln_eps: f32,
    local_window: usize,
    max_len: usize,
    head_max_len: usize,
    tok_emb: F16Mat,
    emb_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<EncLayer>,
    rope_global: Rope,
    rope_local: Rope,
    type_emb: Vec<f32>,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    agent_cfg: AgentConfig,
    tokenizer: Tokenizer,
    specials: prompt::Specials,
    pool: rayon::ThreadPool,
    scratch: Mutex<Scratch>,
    profile: Mutex<Profile>,
}

/// Why the fast engine cannot be used on this machine, if it cannot.
pub fn unavailable_reason() -> Option<&'static str> {
    if !supported() {
        Some("CPU lacks AVX2/FMA/F16C")
    } else {
        None
    }
}

impl Engine {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        if let Some(r) = unavailable_reason() {
            bail!("{r}");
        }
        let agent_cfg = AgentConfig::load(dir.join("rl_agent_config.json"))?;
        let enc = EncoderConfig::load(dir.join("encoder/config.json"))?;
        let w = Weights::open(&dir.join("model.safetensors"))?;

        let d = enc.hidden_size;
        let inter = enc.intermediate_size;
        let nh = enc.num_attention_heads;
        if d != nh * HEAD_DIM {
            bail!("unsupported head size: hidden {d} / {nh} heads != {HEAD_DIM}");
        }
        if d % 8 != 0 || inter % 8 != 0 {
            bail!("hidden/intermediate sizes must be multiples of 8");
        }

        let (theta_global, theta_local) = match &enc.rope_parameters {
            Some(r) => (r.full_attention.rope_theta, r.sliding_attention.rope_theta),
            None => (
                enc.global_rope_theta.unwrap_or(160_000.0),
                enc.local_rope_theta.unwrap_or(10_000.0),
            ),
        };
        let max_len = agent_cfg.max_len;

        let mut layers = Vec::with_capacity(enc.num_hidden_layers);
        for i in 0..enc.num_hidden_layers {
            let p = format!("encoder.layers.{i}");
            layers.push(EncLayer {
                attn_norm: if w.has(&format!("{p}.attn_norm.weight")) {
                    Some(w.vec(&format!("{p}.attn_norm.weight"), d)?)
                } else {
                    None
                },
                wqkv: w.mat(&format!("{p}.attn.Wqkv.weight"), 3 * d, d)?,
                wo: w.mat(&format!("{p}.attn.Wo.weight"), d, d)?,
                mlp_norm: w.vec(&format!("{p}.mlp_norm.weight"), d)?,
                wi: w.mat(&format!("{p}.mlp.Wi.weight"), 2 * inter, d)?,
                mlp_wo: w.mat(&format!("{p}.mlp.Wo.weight"), d, inter)?,
                local: i % enc.global_attn_every_n_layers != 0,
            });
        }

        let head_layers = if agent_cfg.head_layers == 0 {
            2
        } else {
            agent_cfg.head_layers
        };
        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            let ln = |n: &str| -> Result<(Vec<f32>, Vec<f32>)> {
                Ok((
                    w.vec(&format!("{p}.{n}.weight"), d)?,
                    w.vec(&format!("{p}.{n}.bias"), d)?,
                ))
            };
            head.push(HeadLayer {
                norm1: ln("norm1")?,
                in_proj: w.mat(&format!("{p}.self_attn.in_proj_weight"), 3 * d, d)?,
                in_proj_b: w.vec(&format!("{p}.self_attn.in_proj_bias"), 3 * d)?,
                out_proj: w.mat(&format!("{p}.self_attn.out_proj.weight"), d, d)?,
                out_proj_b: w.vec(&format!("{p}.self_attn.out_proj.bias"), d)?,
                norm2: ln("norm2")?,
                lin1: w.mat(&format!("{p}.linear1.weight"), 4 * d, d)?,
                lin1_b: w.vec(&format!("{p}.linear1.bias"), 4 * d)?,
                lin2: w.mat(&format!("{p}.linear2.weight"), d, 4 * d)?,
                lin2_b: w.vec(&format!("{p}.linear2.bias"), d)?,
            });
        }

        let scorer = Scorer {
            norm: (w.vec("scorer.0.weight", d)?, w.vec("scorer.0.bias", d)?),
            fc1: w.mat("scorer.1.weight", d, d)?,
            fc1_b: w.vec("scorer.1.bias", d)?,
            fc2: w.mat("scorer.3.weight", 1, d)?,
            fc2_b: w.vec("scorer.3.bias", 1)?[0],
        };

        let tok_path = dir.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tok_path.display()))?;
        let specials = prompt::Specials::resolve(&tokenizer)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("laya-{i}"))
            .build()
            .context("building inference thread pool")?;

        Ok(Self {
            d,
            nh,
            inter,
            ln_eps: enc.layer_norm_eps as f32,
            local_window: enc.local_attention / 2,
            max_len,
            head_max_len: agent_cfg.head_max_len,
            tok_emb: w.mat(
                "encoder.embeddings.tok_embeddings.weight",
                enc.vocab_size,
                d,
            )?,
            emb_norm: w.vec("encoder.embeddings.norm.weight", d)?,
            final_norm: w.vec("encoder.final_norm.weight", d)?,
            layers,
            rope_global: Rope::new(theta_global, max_len),
            rope_local: Rope::new(theta_local, max_len),
            type_emb: w.row_f32("type_emb.weight", 3, d, QTYPE_CHOICE)?,
            head,
            scorer,
            agent_cfg,
            tokenizer,
            specials,
            pool,
            scratch: Mutex::new(Scratch::new(max_len, d, inter)),
            profile: Mutex::new(Profile {
                enabled: std::env::var_os("LAYA_PROFILE").is_some(),
                acc: Vec::new(),
            }),
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Answer one `choice` question about `state`.
    pub fn choose(&self, instructions: &str, state: &str, options: &[&str]) -> Result<Choice> {
        let enc = prompt::build_choice_sequence(
            &self.tokenizer,
            &self.specials,
            instructions,
            options,
            state,
            self.max_len,
            self.head_max_len,
        )?;
        let logits = {
            let mut guard = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            let scratch: &mut Scratch = &mut guard;
            let mut prof = self.profile.lock().unwrap_or_else(|e| e.into_inner());
            let prof: &mut Profile = &mut prof;
            self.pool
                .install(|| self.forward(&enc.ids, &enc.markers, scratch, prof))
        };

        // Post-processing identical to `laya::Agent::system_one` for a choice question.
        let k = enc.markers.len();
        let t = self.agent_cfg.temperature_for(QTYPE_CHOICE, k);
        let z: Vec<f32> = logits.iter().map(|v| v / t).collect();
        let p = stable_softmax(&z);
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let confidence = round4(confidence_from_probs(&p, k));
        Ok(Choice {
            probs: p.iter().map(|v| round4(*v)).collect(),
            best,
            confidence,
            input_tokens: enc.ids.len(),
        })
    }

    /// Full forward pass; returns one logit per marker (uncalibrated).
    fn forward(
        &self,
        ids: &[u32],
        markers: &[usize],
        s: &mut Scratch,
        pr: &mut Profile,
    ) -> Vec<f32> {
        let (d, l) = (self.d, ids.len());
        let Scratch {
            x,
            h,
            qkv,
            qh,
            kh,
            vh,
            ao,
            wi,
            g,
        } = s;

        // Embeddings + norm.
        pr.time("embed", || {
            for (r, &id) in ids.iter().enumerate() {
                widen_f16(self.tok_emb.row(id as usize), &mut h[r * d..(r + 1) * d]);
            }
            layer_norm(h, l, d, &self.emb_norm, None, self.ln_eps, x);
        });

        for layer in &self.layers {
            let (rope, window) = if layer.local {
                (&self.rope_local, Some(self.local_window))
            } else {
                (&self.rope_global, None)
            };
            let input: &[f32] = match &layer.attn_norm {
                Some(w) => {
                    pr.time("layer_norm", || {
                        layer_norm(x, l, d, w, None, self.ln_eps, h)
                    });
                    h
                }
                None => x,
            };
            pr.time("mm.wqkv", || {
                linear(input, l, d, layer.wqkv.data(), 3 * d, None, qkv)
            });
            pr.time("split_qkv+rope", || {
                split_qkv(
                    qkv,
                    l,
                    self.nh,
                    Some((&rope.cos, &rope.sin)),
                    (HEAD_DIM as f32).powf(-0.5),
                    qh,
                    kh,
                    vh,
                )
            });
            pr.time("attention", || {
                attention(qh, kh, vh, l, self.nh, window, ao)
            });
            pr.time("mm.wo", || linear(ao, l, d, layer.wo.data(), d, None, h));
            pr.time("residual", || add_assign(&mut x[..l * d], &h[..l * d]));

            pr.time("layer_norm", || {
                layer_norm(x, l, d, &layer.mlp_norm, None, self.ln_eps, h)
            });
            pr.time("mm.wi", || {
                linear(h, l, d, layer.wi.data(), 2 * self.inter, None, wi)
            });
            pr.time("geglu", || geglu(wi, l, self.inter, g));
            pr.time("mm.mlp_wo", || {
                linear(g, l, self.inter, layer.mlp_wo.data(), d, None, h)
            });
            pr.time("residual", || add_assign(&mut x[..l * d], &h[..l * d]));
        }
        pr.time("layer_norm", || {
            layer_norm(x, l, d, &self.final_norm, None, self.ln_eps, h)
        });
        x[..l * d].copy_from_slice(&h[..l * d]);

        // Decision head: question-type embedding, then pre-norm transformer layers over the
        // whole sequence (full attention, ReLU feed-forward).
        add_row_broadcast(&mut x[..l * d], d, &self.type_emb);
        let last = self.head.len().saturating_sub(1);
        for hl in self.head.iter().take(last) {
            pr.time("head", || {
                layer_norm(x, l, d, &hl.norm1.0, Some(&hl.norm1.1), LN_EPS_HEAD, h);
                linear(h, l, d, hl.in_proj.data(), 3 * d, Some(&hl.in_proj_b), qkv);
                split_qkv(
                    qkv,
                    l,
                    self.nh,
                    None,
                    (HEAD_DIM as f32).powf(-0.5),
                    qh,
                    kh,
                    vh,
                );
                attention(qh, kh, vh, l, self.nh, None, ao);
                linear(ao, l, d, hl.out_proj.data(), d, Some(&hl.out_proj_b), h);
                add_assign(&mut x[..l * d], &h[..l * d]);

                layer_norm(x, l, d, &hl.norm2.0, Some(&hl.norm2.1), LN_EPS_HEAD, h);
                linear(h, l, d, hl.lin1.data(), 4 * d, Some(&hl.lin1_b), wi);
                relu(&mut wi[..l * 4 * d]);
                linear(wi, l, 4 * d, hl.lin2.data(), d, Some(&hl.lin2_b), h);
                add_assign(&mut x[..l * d], &h[..l * d]);
            });
        }

        // The last head layer is only ever read at the marker rows, so keys/values are computed for
        // every position but queries, the output projection and the feed-forward only for the
        // markers. Identical result, a fraction of the work.
        let k = markers.len();
        let mut xm = vec![0f32; k * d];
        if let Some(hl) = self.head.last() {
            pr.time("head(last,pruned)", || {
                layer_norm(x, l, d, &hl.norm1.0, Some(&hl.norm1.1), LN_EPS_HEAD, h);
                // K and V for all rows: in_proj rows [d, 3d).
                linear(
                    h,
                    l,
                    d,
                    &hl.in_proj.data()[d * d..],
                    2 * d,
                    Some(&hl.in_proj_b[d..]),
                    qkv,
                );
                to_head_major(qkv, 2 * d, 0, l, self.nh, 1.0, kh);
                to_head_major(qkv, 2 * d, d, l, self.nh, 1.0, vh);
                // Q for the marker rows only: in_proj rows [0, d).
                let mut hm = vec![0f32; k * d];
                for (r, &m) in markers.iter().enumerate() {
                    hm[r * d..(r + 1) * d].copy_from_slice(&h[m * d..(m + 1) * d]);
                    xm[r * d..(r + 1) * d].copy_from_slice(&x[m * d..(m + 1) * d]);
                }
                let mut q = vec![0f32; k * d];
                linear(
                    &hm,
                    k,
                    d,
                    &hl.in_proj.data()[..d * d],
                    d,
                    Some(&hl.in_proj_b[..d]),
                    &mut q,
                );
                to_head_major(&q, d, 0, k, self.nh, (HEAD_DIM as f32).powf(-0.5), qh);
                attention_rows(qh, k, kh, vh, l, self.nh, None, ao);
                let mut o = vec![0f32; k * d];
                linear(
                    ao,
                    k,
                    d,
                    hl.out_proj.data(),
                    d,
                    Some(&hl.out_proj_b),
                    &mut o,
                );
                add_assign(&mut xm, &o);

                layer_norm(
                    &xm,
                    k,
                    d,
                    &hl.norm2.0,
                    Some(&hl.norm2.1),
                    LN_EPS_HEAD,
                    &mut hm,
                );
                linear(&hm, k, d, hl.lin1.data(), 4 * d, Some(&hl.lin1_b), wi);
                relu(&mut wi[..k * 4 * d]);
                linear(wi, k, 4 * d, hl.lin2.data(), d, Some(&hl.lin2_b), &mut o);
                add_assign(&mut xm, &o);
            });
        } else {
            for (r, &m) in markers.iter().enumerate() {
                xm[r * d..(r + 1) * d].copy_from_slice(&x[m * d..(m + 1) * d]);
            }
        }

        // Score the marker rows.
        let sc = &self.scorer;
        layer_norm(&xm, k, d, &sc.norm.0, Some(&sc.norm.1), LN_EPS_HEAD, x);
        linear(x, k, d, sc.fc1.data(), d, Some(&sc.fc1_b), h);
        gelu_erf_inplace(&mut h[..k * d]);
        (0..k)
            .map(|r| dot_f16(&h[r * d..(r + 1) * d], sc.fc2.data()) + sc.fc2_b)
            .collect()
    }

    /// Per-op timings accumulated since the last call (empty unless `LAYA_PROFILE` is set).
    pub fn take_profile(&self) -> String {
        let mut p = self.profile.lock().unwrap_or_else(|e| e.into_inner());
        let r = p.report();
        p.acc.clear();
        r
    }

    /// One throw-away inference so page faults, thread start-up and caches are paid before play.
    pub fn warm_up(&self) -> Result<f64> {
        let t = Instant::now();
        self.choose(
            "Pick the option that moves closer to the food.\nk0: a\nk1: b\nk2: c\nk3: d",
            "Snake head at (5,5). Food at (7,5). Snake length=3.",
            &["k0", "k1", "k2", "k3"],
        )?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }
}

/// Softmax in f32 with the max subtracted, as in the reference.
fn stable_softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Round to 4 decimals in f32, then widen through the shortest decimal form, exactly as the
/// reference serialises it (so 0.2657 stays 0.2657 rather than 0.26570001).
fn round4(v: f32) -> f64 {
    let r = (v * 10_000.0).round() / 10_000.0;
    format!("{r}").parse::<f64>().unwrap_or(r as f64)
}

fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p[..k]
        .iter()
        .map(|v| v * v.clamp(1e-12, 1.0).ln())
        .sum::<f32>();
    1.0 - ent / (k as f32).ln()
}
