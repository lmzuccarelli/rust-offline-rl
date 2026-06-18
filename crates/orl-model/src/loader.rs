use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as qwen;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("HF Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

pub fn download_model(model_id: &str) -> Result<ModelFiles, LoadError> {
    tracing::info!("downloading model {} from HuggingFace Hub", model_id);
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    // Try single file first, then sharded
    let weight_paths = if let Ok(path) = repo.get("model.safetensors") {
        vec![path]
    } else {
        let index_path = repo.get("model.safetensors.index.json")?;
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| LoadError::Json(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::Other, "no weight_map in index")
            )))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let mut paths = Vec::new();
        for shard in &shard_files {
            paths.push(repo.get(shard)?);
        }
        paths
    };

    tracing::info!("downloaded {} weight files", weight_paths.len());

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        weight_paths,
    })
}

pub fn load_config(config_path: &std::path::Path) -> Result<qwen::Config, LoadError> {
    let content = std::fs::read_to_string(config_path)?;
    let config: qwen::Config = serde_json::from_str(&content)?;
    Ok(config)
}

pub fn load_tokenizer(tokenizer_path: &std::path::Path) -> Result<Tokenizer, LoadError> {
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e: tokenizers::Error| LoadError::Tokenizer(e.to_string()))
}

pub fn load_weights_vb<'a>(
    weight_paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'a>, LoadError> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, device)?
    };
    Ok(vb)
}

pub fn parse_dtype(dtype_str: &str) -> DType {
    match dtype_str.to_lowercase().as_str() {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        _ => DType::BF16,
    }
}
