use candle_core::{Device, Tensor, DType};
use candle_nn::ops::log_softmax;
use rand::rngs::StdRng;
use serde::Deserialize;

use orl_data::replay_buffer::ReplayBuffer;
use orl_model::causal_lm::TrainableCausalLM;

use crate::traits::{Metrics, OfflineRLAlgorithm};

#[derive(Debug, Clone, Deserialize)]
pub struct RwSftConfig {
    #[serde(default = "default_temp")]
    pub temperature: f64,
    #[serde(default = "default_norm")]
    pub normalization: String,
    #[serde(default = "default_thresh")]
    pub reward_threshold: f64,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
}

fn default_temp() -> f64 { 1.0 }
fn default_norm() -> String { "exponential".to_string() }
fn default_thresh() -> f64 { 0.5 }
fn default_bs() -> usize { 1 }

impl Default for RwSftConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            normalization: "exponential".to_string(),
            reward_threshold: 0.5,
            batch_size: 1,
        }
    }
}

pub struct RewardWeightedSft {
    config: RwSftConfig,
}

impl RewardWeightedSft {
    pub fn new(config: RwSftConfig) -> Self {
        Self { config }
    }
}

impl OfflineRLAlgorithm for RewardWeightedSft {
    fn name(&self) -> &str {
        "rw_sft"
    }

    fn compute_loss(
        &mut self,
        model: &mut TrainableCausalLM,
        rng: &mut StdRng,
        buffer: &ReplayBuffer,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Metrics)> {
        let batch = buffer.sample_weighted_sft_batch(self.config.batch_size, rng, device)?;

        let logits = model.forward_train(&batch.input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let logits = logits.to_dtype(DType::F32)?;

        let loss = weighted_cross_entropy(&logits, &batch.labels, &batch.weights, u32::MAX)?;

        let loss_val = loss.to_scalar::<f32>()? as f64;
        let mut metrics = Metrics::new();
        metrics.insert("loss".to_string(), loss_val);

        Ok((loss, metrics))
    }
}

fn weighted_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    weights: &Tensor,
    ignore_index: u32,
) -> candle_core::Result<Tensor> {
    let (_batch, seq_len, vocab_size) = logits.dims3()?;

    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_labels = labels.narrow(1, 1, seq_len - 1)?;

    let log_probs = log_softmax(&shift_logits, 2)?;

    let mask = shift_labels.ne(ignore_index)?.to_dtype(DType::F32)?;

    let safe_labels = shift_labels.clamp(0u32, (vocab_size - 1) as u32)?;
    let safe_labels = safe_labels.unsqueeze(2)?;

    let gathered = log_probs.gather(&safe_labels.to_dtype(DType::U32)?, 2)?;
    let gathered = gathered.squeeze(2)?;

    let masked_loss = gathered.mul(&mask)?;

    let token_counts = mask.sum(1)?.clamp(1.0f32, f32::MAX)?;
    let per_example_loss = masked_loss.sum(1)?.neg()?;
    let per_example_loss = per_example_loss.div(&token_counts)?;

    let weighted_loss = per_example_loss.mul(weights)?;

    weighted_loss.mean(0)
}
