use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::post;
use axum::Router;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3 as qwen;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use orl_model::loader;
use orl_train::checkpoint;
use orl_train::trainer;

// ---------------------------------------------------------------------------
// OpenAI-compatible request/response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_max_tokens() -> usize {
    256
}

fn default_seed() -> u64 {
    42
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct AppState {
    model: Mutex<qwen::ModelForCausalLM>,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    model_name: String,
    eos_tokens: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

fn apply_chat_template(messages: &[ChatMessage], tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let mut prompt = String::new();

    let has_system = messages.iter().any(|m| m.role == "system");
    if !has_system {
        prompt.push_str("<|im_start|>system\n/no_think<|im_end|>\n");
    }

    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        if msg.role == "system" {
            prompt.push_str("\n/no_think");
        }
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");

    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenization failed");
    encoding.get_ids().to_vec()
}

fn resolve_eos_tokens(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let vocab = tokenizer.get_vocab(true);
    let mut eos = Vec::new();

    for token_str in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(&id) = vocab.get(token_str) {
            eos.push(id);
        }
    }

    if eos.is_empty() {
        if let Some(&id) = vocab.get("</s>") {
            eos.push(id);
        }
    }

    eos
}

// ---------------------------------------------------------------------------
// Autoregressive generation
// ---------------------------------------------------------------------------

struct GenerationOutput {
    output_tokens: Vec<u32>,
    prompt_tokens: usize,
    finish_reason: String,
}

fn run_generation(
    model: &mut qwen::ModelForCausalLM,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_tokens: &[u32],
    sampler: &mut LogitsProcessor,
    device: &Device,
) -> anyhow::Result<GenerationOutput> {
    let prompt_len = prompt_tokens.len();

    model.clear_kv_cache();

    // Prefill: forward the full prompt
    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;

    let mut next_token = sampler.sample(&logits)?;
    let mut output_tokens = vec![next_token];

    if eos_tokens.contains(&next_token) {
        return Ok(GenerationOutput {
            output_tokens: vec![],
            prompt_tokens: prompt_len,
            finish_reason: "stop".to_string(),
        });
    }

    // Decode loop
    for i in 1..max_tokens {
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_len + i - 1)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        next_token = sampler.sample(&logits)?;

        if eos_tokens.contains(&next_token) {
            return Ok(GenerationOutput {
                output_tokens,
                prompt_tokens: prompt_len,
                finish_reason: "stop".to_string(),
            });
        }

        output_tokens.push(next_token);
    }

    Ok(GenerationOutput {
        output_tokens,
        prompt_tokens: prompt_len,
        finish_reason: "length".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let prompt_tokens = apply_chat_template(&req.messages, &state.tokenizer);

    if prompt_tokens.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "empty prompt after tokenization"})),
        )
            .into_response();
    }

    let mut model = state.model.lock().await;
    let mut sampler = LogitsProcessor::new(req.seed, req.temperature, req.top_p);

    let result = run_generation(
        &mut model,
        &prompt_tokens,
        req.max_tokens,
        &state.eos_tokens,
        &mut sampler,
        &state.device,
    );

    drop(model);

    match result {
        Ok(output) => {
            let output_text = state
                .tokenizer
                .decode(&output.output_tokens, true)
                .unwrap_or_default();

            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", created),
                object: "chat.completion".to_string(),
                created,
                model: state.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: output_text,
                    },
                    finish_reason: output.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: output.prompt_tokens,
                    completion_tokens: output.output_tokens.len(),
                    total_tokens: output.prompt_tokens + output.output_tokens.len(),
                },
            };

            Json(serde_json::to_value(response).unwrap()).into_response()
        }
        Err(e) => {
            tracing::error!("generation failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("generation failed: {}", e)})),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

pub async fn run_server(
    config_path: &PathBuf,
    checkpoint_dir: &PathBuf,
    bind_addr: &str,
) -> anyhow::Result<()> {
    let config = trainer::load_config(config_path)?;

    let latest = checkpoint::find_latest_checkpoint(checkpoint_dir)
        .ok_or_else(|| anyhow::anyhow!("no checkpoints found in {}", checkpoint_dir.display()))?;
    let (step, checkpoint_path) = latest;
    tracing::info!("loading checkpoint step {} from {}", step, checkpoint_path.display());

    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;
    tracing::info!("using device: {:?}", device);

    let model_files = loader::download_model(&config.model.model_id)?;
    let model_config = loader::load_config(&model_files.config_path)?;
    let tokenizer = loader::load_tokenizer(&model_files.tokenizer_path)?;

    tracing::info!("loading checkpoint weights...");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[checkpoint_path],
            dtype,
            &device,
        )?
    };
    let model = qwen::ModelForCausalLM::new(&model_config, vb)?;
    tracing::info!("model loaded successfully");

    let eos_tokens = resolve_eos_tokens(&tokenizer);
    tracing::info!("eos tokens: {:?}", eos_tokens);

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        device,
        model_name: config.model.model_id.clone(),
        eos_tokens,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    tracing::info!("starting server on {}", bind_addr);
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
