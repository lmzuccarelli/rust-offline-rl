use candle_core::{Device, Tensor, DType};
use candle_nn::ops::log_softmax;
use rand::rngs::StdRng;
use serde::Deserialize;

use orl_data::replay_buffer::ReplayBuffer;
use orl_model::causal_lm::{FrozenCausalLM, TrainableCausalLM};

use crate::traits::{Metrics, OfflineRLAlgorithm};

#[derive(Debug, Clone, Deserialize)]
pub struct DpoConfig {
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default)]
    pub label_smoothing: f64,
    #[serde(default = "default_margin")]
    pub reward_margin: f64,
    #[serde(default = "default_strategy")]
    pub pairing_strategy: String,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
}

fn default_beta() -> f64 { 0.1 }
fn default_margin() -> f64 { 0.1 }
fn default_strategy() -> String { "same_technique".to_string() }
fn default_bs() -> usize { 1 }

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
            reward_margin: 0.1,
            pairing_strategy: "same_technique".to_string(),
            batch_size: 1,
        }
    }
}

pub struct Dpo {
    config: DpoConfig,
    ref_model: Option<FrozenCausalLM>,
}

impl Dpo {
    pub fn new(config: DpoConfig, ref_model: Option<FrozenCausalLM>) -> Self {
        Self { config, ref_model }
    }
}

impl OfflineRLAlgorithm for Dpo {
    fn name(&self) -> &str {
        "dpo"
    }

    fn compute_loss(
        &mut self,
        model: &mut TrainableCausalLM,
        rng: &mut StdRng,
        buffer: &ReplayBuffer,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Metrics)> {
        let batch = buffer.sample_dpo_batch(self.config.batch_size, rng, device)?;

        // Policy log probs for chosen and rejected
        let chosen_logits = model.forward_train(&batch.chosen_input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let chosen_logits = chosen_logits.to_dtype(DType::F32)?;
        let policy_chosen_lp = sequence_log_probs(&chosen_logits, &batch.chosen_labels, u32::MAX)?;

        let rejected_logits = model.forward_train(&batch.rejected_input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let rejected_logits = rejected_logits.to_dtype(DType::F32)?;
        let policy_rejected_lp = sequence_log_probs(&rejected_logits, &batch.rejected_labels, u32::MAX)?;

        // Reference log probs
        let (ref_chosen_lp, ref_rejected_lp) = if let Some(ref mut ref_model) = self.ref_model {
            let ref_ch = ref_model.forward_logprobs(&batch.chosen_input_ids)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let ref_ch = ref_ch.to_dtype(DType::F32)?;
            let ref_ch_lp = sequence_log_probs(&ref_ch, &batch.chosen_labels, u32::MAX)?;

            let ref_rj = ref_model.forward_logprobs(&batch.rejected_input_ids)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let ref_rj = ref_rj.to_dtype(DType::F32)?;
            let ref_rj_lp = sequence_log_probs(&ref_rj, &batch.rejected_labels, u32::MAX)?;

            (ref_ch_lp, ref_rj_lp)
        } else {
            let zeros = Tensor::zeros_like(&policy_chosen_lp)?;
            (zeros.clone(), zeros)
        };

        // DPO loss
        let log_ratio_chosen = policy_chosen_lp.sub(&ref_chosen_lp)?;
        let log_ratio_rejected = policy_rejected_lp.sub(&ref_rejected_lp)?;
        let diff = log_ratio_chosen.sub(&log_ratio_rejected)?;

        let beta_scalar = Tensor::new(&[self.config.beta as f32], device)?
            .broadcast_as(diff.shape())?;
        let scaled = diff.mul(&beta_scalar)?;

        // -log(sigmoid(x)) = softplus(-x)
        let neg_scaled = scaled.neg()?;
        let loss = softplus(&neg_scaled)?;

        // Optional label smoothing
        let loss = if self.config.label_smoothing > 0.0 {
            let smooth = self.config.label_smoothing as f32;
            let reverse = softplus(&scaled)?;
            let one_minus = 1.0f32 - smooth;
            let a = loss.affine(one_minus as f64, 0.0)?;
            let b = reverse.affine(smooth as f64, 0.0)?;
            a.add(&b)?
        } else {
            loss
        };

        let loss = loss.mean(0)?;

        let loss_val = loss.to_scalar::<f32>()? as f64;
        let accuracy = scaled
            .gt(&Tensor::zeros_like(&scaled)?)?
            .to_dtype(DType::F32)?
            .mean(0)?
            .to_scalar::<f32>()? as f64;

        let mut metrics = Metrics::new();
        metrics.insert("loss".to_string(), loss_val);
        metrics.insert("accuracy".to_string(), accuracy);

        Ok((loss, metrics))
    }
}

fn sequence_log_probs(
    logits: &Tensor,
    labels: &Tensor,
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

    let masked = gathered.mul(&mask)?;
    let token_counts = mask.sum(1)?.clamp(1.0f32, f32::MAX)?;
    masked.sum(1)?.div(&token_counts)
}

fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    // softplus(x) = log(1 + exp(x)), numerically stable
    let zeros = Tensor::zeros_like(x)?;
    let pos = x.maximum(&zeros)?;
    let neg_abs = x.abs()?.neg()?;
    let exp_neg = neg_abs.exp()?;
    let log_term = exp_neg.affine(1.0, 1.0)?.log()?;
    pos.add(&log_term)
}
