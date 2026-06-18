use std::path::{Path, PathBuf};

use candle_nn::VarMap;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainState {
    pub global_step: usize,
    pub epoch: usize,
    pub best_eval_loss: f64,
    pub algorithm: String,
}

pub fn save_checkpoint(
    varmap: &VarMap,
    state: &TrainState,
    output_dir: &Path,
    step: usize,
) -> Result<PathBuf, CheckpointError> {
    std::fs::create_dir_all(output_dir)?;

    let weights_path = output_dir.join(format!("checkpoint-{}.safetensors", step));
    varmap.save(&weights_path)?;

    let state_path = output_dir.join(format!("checkpoint-{}-state.json", step));
    let state_json = serde_json::to_string_pretty(state)?;
    std::fs::write(&state_path, state_json)?;

    tracing::info!("saved checkpoint at step {} to {}", step, weights_path.display());

    Ok(weights_path)
}

pub fn load_train_state(checkpoint_dir: &Path, step: usize) -> Result<TrainState, CheckpointError> {
    let state_path = checkpoint_dir.join(format!("checkpoint-{}-state.json", step));
    let content = std::fs::read_to_string(&state_path)?;
    let state: TrainState = serde_json::from_str(&content)?;
    Ok(state)
}

pub fn find_latest_checkpoint(output_dir: &Path) -> Option<(usize, PathBuf)> {
    let entries = std::fs::read_dir(output_dir).ok()?;

    let mut latest: Option<(usize, PathBuf)> = None;

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("checkpoint-") && name.ends_with(".safetensors") {
            let step_str = name
                .trim_start_matches("checkpoint-")
                .trim_end_matches(".safetensors");
            if let Ok(step) = step_str.parse::<usize>() {
                if latest.as_ref().map_or(true, |(s, _)| step > *s) {
                    latest = Some((step, entry.path()));
                }
            }
        }
    }

    latest
}
