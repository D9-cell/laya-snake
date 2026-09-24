use serde_json::Value;
use std::{fs, path::PathBuf};

fn scores_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".laya_snake").join("best_scores.json")
}

fn load_all() -> serde_json::Map<String, Value> {
    let path = scores_path();
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => serde_json::Map::new(),
    }
}

pub fn get_best(width: usize, height: usize) -> u32 {
    load_all()
        .get(&format!("{width}x{height}"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32
}

pub fn set_best(width: usize, height: usize, score: u32) -> u32 {
    let mut data = load_all();
    let key = format!("{width}x{height}");
    let current = data.get(&key).and_then(Value::as_u64).unwrap_or(0) as u32;
    if score <= current {
        return current;
    }
    data.insert(key, Value::from(score));
    let path = scores_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&data) {
        let _ = fs::write(path, json);
    }
    score
}
