//! Memory-mapped safetensors access. Weights are used exactly as stored (f16); nothing is copied
//! or up-cast except the small vectors (norms, biases) the kernels want as f32.

use super::kernels::{f16_slice_to_f32, f16_to_f32};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

struct Info {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct Weights {
    mmap: Arc<Mmap>,
    base: usize,
    index: HashMap<String, Info>,
}

/// A row-major `[n, k]` f16 matrix living inside the mapped checkpoint.
pub struct F16Mat {
    mmap: Arc<Mmap>,
    off: usize,
    pub n: usize,
    pub k: usize,
}

impl F16Mat {
    pub fn data(&self) -> &[u16] {
        // SAFETY: `off` is 2-byte aligned (checked at construction) inside a live mapping that this
        // struct keeps alive, and `n * k * 2` bytes were bounds-checked against the tensor extent.
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(self.off) as *const u16,
                self.n * self.k,
            )
        }
    }

    pub fn row(&self, r: usize) -> &[u16] {
        &self.data()[r * self.k..(r + 1) * self.k]
    }
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is treated as read-only for the life of the process.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        if mmap.len() < 8 {
            bail!("{} is too small to be a safetensors file", path.display());
        }
        let hlen = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let base = 8 + hlen;
        if base > mmap.len() {
            bail!("corrupt safetensors header length in {}", path.display());
        }
        let header: serde_json::Map<String, Value> =
            serde_json::from_slice(&mmap[8..base]).context("parsing safetensors header")?;
        let mut index = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"]
                .as_str()
                .context("tensor without dtype")?
                .to_string();
            let shape = v["shape"]
                .as_array()
                .context("tensor without shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .context("tensor without data_offsets")?;
            let (start, end) = (
                off.first()
                    .and_then(Value::as_u64)
                    .context("bad data_offsets")? as usize,
                off.get(1)
                    .and_then(Value::as_u64)
                    .context("bad data_offsets")? as usize,
            );
            if base + end > mmap.len() || start > end {
                bail!("tensor {name} points outside the file");
            }
            index.insert(
                name,
                Info {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        let mmap = Arc::new(mmap);
        // Ask the kernel to start paging the weights in; the warm-up pass finishes the job.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        Ok(Self { mmap, base, index })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn info(&self, name: &str) -> Result<&Info> {
        self.index
            .get(name)
            .with_context(|| format!("checkpoint has no tensor `{name}`"))
    }

    /// A 2-D f16 matrix `[n, k]`. `k` must be a multiple of 8 (the kernels' vector width).
    pub fn mat(&self, name: &str, n: usize, k: usize) -> Result<F16Mat> {
        let i = self.info(name)?;
        if i.dtype != "F16" {
            bail!(
                "tensor `{name}` is {} but the fast engine needs F16",
                i.dtype
            );
        }
        if i.shape != [n, k] {
            bail!(
                "tensor `{name}` has shape {:?}, expected [{n}, {k}]",
                i.shape
            );
        }
        let off = self.base + i.start;
        if off % 2 != 0 || i.end - i.start != n * k * 2 {
            bail!("tensor `{name}` is misaligned or has an unexpected size");
        }
        Ok(F16Mat {
            mmap: Arc::clone(&self.mmap),
            off,
            n,
            k,
        })
    }

    /// A 1-D tensor of `len` elements, widened to f32 (F16 and F32 checkpoints both work).
    pub fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let i = self.info(name)?;
        if i.shape != [len] {
            bail!("tensor `{name}` has shape {:?}, expected [{len}]", i.shape);
        }
        let bytes = &self.mmap[self.base + i.start..self.base + i.end];
        match i.dtype.as_str() {
            "F16" => Ok(f16_slice_to_f32(
                &bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>(),
            )),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            other => bail!("tensor `{name}` has unsupported dtype {other}"),
        }
    }

    /// One row of a 2-D f16 tensor widened to f32.
    pub fn row_f32(&self, name: &str, rows: usize, cols: usize, row: usize) -> Result<Vec<f32>> {
        let m = self.mat(name, rows, cols)?;
        Ok(m.row(row).iter().map(|&h| f16_to_f32(h)).collect())
    }
}
