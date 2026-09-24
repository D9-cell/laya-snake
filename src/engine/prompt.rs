//! Builds the encoder input for a `choice` question.
//!
//! This is a faithful port of `build_sequence` from the `laya` crate (which keeps it crate-private):
//! `[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`.
//! The `[MASK]` before each option is the marker position the decision head scores.

use anyhow::{bail, Result};
use tokenizers::Tokenizer;

/// Per-option text budget in tokens (matches the reference's `[:48]`).
const MAX_OPTION_TOKENS: usize = 48;

pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
}

impl Specials {
    pub fn resolve(tok: &Tokenizer) -> Result<Self> {
        let id = |t: &str| {
            tok.token_to_id(t)
                .ok_or_else(|| anyhow::anyhow!("tokenizer has no {t} token"))
        };
        Ok(Self {
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            mask: id("[MASK]")?,
        })
    }
}

pub struct Encoded {
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]` marker.
    pub markers: Vec<usize>,
}

fn encode_plain(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tok.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
}

pub fn build_choice_sequence(
    tok: &Tokenizer,
    sp: &Specials,
    instructions: &str,
    options: &[&str],
    state: &str,
    max_len: usize,
    head_max_len: usize,
) -> Result<Encoded> {
    if options.is_empty() {
        bail!("a choice question needs at least one option");
    }
    // A literal [MASK] in user text would be read as a marker, so neutralise it.
    let scrub = |s: &str| s.replace("[MASK]", " ");

    let mut head_ids = encode_plain(tok, &format!("choice question: {}", scrub(instructions)))?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
    for o in options {
        let mut ids = vec![sp.mask];
        let mut body = encode_plain(tok, &format!(" {}", scrub(o)))?;
        body.truncate(MAX_OPTION_TOKENS);
        ids.extend(body);
        opt_ids.push(ids);
    }

    let used: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as isize - used as isize;
    if opt_budget < 16 {
        // Too many or too long options: shrink every option evenly so the instructions still fit.
        let per = std::cmp::max(
            4,
            head_max_len.saturating_sub(16) / std::cmp::max(1, opt_ids.len()),
        );
        for o in opt_ids.iter_mut() {
            o.truncate(per);
        }
        let used: usize = opt_ids.iter().map(Vec::len).sum();
        opt_budget = head_max_len as isize - used as isize;
    }
    head_ids.truncate(std::cmp::max(8, opt_budget.max(0) as usize));

    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls);
    ids.extend_from_slice(&head_ids);
    ids.push(sp.sep);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(sp.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let mut st = encode_plain(tok, &scrub(state))?;
    st.truncate(room);
    ids.extend_from_slice(&st);
    ids.push(sp.sep);
    ids.truncate(max_len);

    let markers: Vec<usize> = markers.into_iter().filter(|m| *m < max_len).collect();
    if markers.len() != options.len() {
        bail!(
            "options do not fit in head_max_len={head_max_len} tokens ({} of {} markers kept)",
            markers.len(),
            options.len()
        );
    }
    Ok(Encoded { ids, markers })
}
