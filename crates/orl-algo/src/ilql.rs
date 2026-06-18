use candle_core::{Device, Tensor, DType};
use candle_nn::{ops::log_softmax, VarMap};
use rand::rngs::StdRng;
use serde::Deserialize;

use orl_data::replay_buffer::ReplayBuffer;
use orl_model::causal_lm::TrainableCausalLM;
use orl_model::value_head::ValueHeads;

use crate::traits::{Metrics, OfflineRLAlgorithm};

#[derive(Debug, Clone, Deserialize)]
pub struct IlqlConfig {
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_cql_alpha")]
    pub cql_alpha: f64,
    #[serde(default = "default_expectile")]
    pub expectile_tau: f64,
    #[serde(default = "default_target_tau")]
    pub target_update_tau: f64,
    #[serde(default = "default_sft_weight")]
    pub sft_weight: f64,
    #[serde(default = "default_q_layers")]
    pub q_head_layers: usize,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
}

fn default_gamma() -> f64 { 0.99 }
fn default_cql_alpha() -> f64 { 1.0 }
fn default_expectile() -> f64 { 0.7 }
fn default_target_tau() -> f64 { 0.005 }
fn default_sft_weight() -> f64 { 0.1 }
fn default_q_layers() -> usize { 2 }
fn default_bs() -> usize { 1 }

impl Default for IlqlConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            cql_alpha: 1.0,
            expectile_tau: 0.7,
            target_update_tau: 0.005,
            sft_weight: 0.1,
            q_head_layers: 2,
            batch_size: 1,
        }
    }
}

pub struct Ilql {
    config: IlqlConfig,
    value_heads: ValueHeads,
    varmap: VarMap,
}

impl Ilql {
    pub fn new(
        config: IlqlConfig,
        hidden_size: usize,
        vocab_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, candle_core::Error> {
        let varmap = VarMap::new();
        let value_heads = ValueHeads::new(hidden_size, vocab_size, &varmap, dtype, device)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok(Self {
            config,
            value_heads,
            varmap,
        })
    }

    pub fn value_varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn value_vars(&self) -> Vec<candle_core::Var> {
        self.varmap.all_vars()
    }
}

impl OfflineRLAlgorithm for Ilql {
    fn name(&self) -> &str {
        "ilql"
    }

    fn compute_loss(
        &mut self,
        model: &mut TrainableCausalLM,
        rng: &mut StdRng,
        buffer: &ReplayBuffer,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Metrics)> {
        let batch = buffer.sample_ilql_batch(self.config.batch_size, rng, device)?;

        let hidden_states = model.forward_hidden(&batch.input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let hidden_states = hidden_states.to_dtype(DType::F32)?;

        let logits = model.forward_train(&batch.input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let logits = logits.to_dtype(DType::F32)?;

        let q_values = self.value_heads.q_values(&hidden_states)?;
        let v_values = self.value_heads.v_values(&hidden_states)?;
        let target_v_values = self.value_heads.target_v_values(&hidden_states.detach())?;

        let (_batch_size, seq_len, vocab_size) = q_values.dims3()?;

        // ---- Q-learning loss ----
        let shift_q = q_values.narrow(1, 0, seq_len - 1)?;
        let shift_labels = batch.labels.narrow(1, 1, seq_len - 1)?;
        let shift_mask = batch.action_mask.narrow(1, 1, seq_len - 1)?;
        let shift_rewards = batch.rewards.narrow(1, 1, seq_len - 1)?;

        let safe_labels = shift_labels.clamp(0u32, (vocab_size - 1) as u32)?;
        let safe_labels_unsq = safe_labels.unsqueeze(2)?;

        let q_at_action = shift_q
            .gather(&safe_labels_unsq.to_dtype(DType::U32)?, 2)?
            .squeeze(2)?;

        // TD target: r + gamma * V_target(s')
        let shift_target_v = target_v_values.narrow(1, 1, seq_len - 1)?;

        let gamma_tensor = Tensor::new(&[self.config.gamma as f32], device)?
            .broadcast_as(shift_target_v.shape())?;
        let discounted_v = shift_target_v.mul(&gamma_tensor)?;
        let td_target = shift_rewards.add(&discounted_v)?;

        let q_error = q_at_action.sub(&td_target.detach())?;
        let q_loss_per_token = q_error.sqr()?;
        let shift_mask_f32 = shift_mask.to_dtype(DType::F32)?;
        let q_loss = masked_mean(&q_loss_per_token.mul(&shift_mask_f32)?, &shift_mask_f32)?;

        // ---- Value loss (expectile regression) ----
        let shift_v = v_values.narrow(1, 0, seq_len - 1)?;
        let advantage = q_at_action.detach().sub(&shift_v)?;
        let v_loss_raw = expectile_loss(&advantage, self.config.expectile_tau as f32, device)?;
        let v_loss = masked_mean(&v_loss_raw.mul(&shift_mask_f32)?, &shift_mask_f32)?;

        // ---- CQL conservative penalty ----
        let logsumexp_q = shift_q.log_sum_exp(2)?;
        let cql_penalty = logsumexp_q.sub(&q_at_action)?;
        let cql_loss = masked_mean(&cql_penalty.mul(&shift_mask_f32)?, &shift_mask_f32)?;

        // ---- SFT regularization loss ----
        let sft_loss = sft_cross_entropy(&logits, &batch.labels, &batch.action_mask, u32::MAX)?;

        // ---- Total loss ----
        let total_loss = q_loss
            .add(&v_loss)?
            .add(&cql_loss.affine(self.config.cql_alpha, 0.0)?)?
            .add(&sft_loss.affine(self.config.sft_weight, 0.0)?)?;

        let mut metrics = Metrics::new();
        metrics.insert("q_loss".to_string(), q_loss.to_scalar::<f32>()? as f64);
        metrics.insert("v_loss".to_string(), v_loss.to_scalar::<f32>()? as f64);
        metrics.insert("cql_loss".to_string(), cql_loss.to_scalar::<f32>()? as f64);
        metrics.insert("sft_loss".to_string(), sft_loss.to_scalar::<f32>()? as f64);
        metrics.insert("total_loss".to_string(), total_loss.to_scalar::<f32>()? as f64);

        Ok((total_loss, metrics))
    }

    fn post_step(&mut self) -> candle_core::Result<()> {
        self.value_heads.update_targets(self.config.target_update_tau, &self.varmap)
    }
}

fn expectile_loss(diff: &Tensor, tau: f32, device: &Device) -> candle_core::Result<Tensor> {
    // |tau - 1(diff < 0)| * diff^2
    let zeros = Tensor::zeros_like(diff)?;
    let indicator = diff.lt(&zeros)?.to_dtype(DType::F32)?;

    let tau_tensor = Tensor::new(&[tau], device)?.broadcast_as(diff.shape())?;
    let _one_minus_tau = Tensor::new(&[1.0f32 - tau], device)?.broadcast_as(diff.shape())?;

    // weight = indicator * (1 - tau) + (1 - indicator) * tau
    //        = tau + indicator * (1 - 2*tau)
    let two_tau_minus_one = Tensor::new(&[1.0f32 - 2.0 * tau], device)?.broadcast_as(diff.shape())?;
    let weight = tau_tensor.add(&indicator.mul(&two_tau_minus_one)?)?;

    weight.mul(&diff.sqr()?)
}

fn masked_mean(values: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    let masked = values.mul(mask)?;
    let sum = masked.sum_all()?;
    let count = mask.sum_all()?.clamp(1.0f32, f32::MAX)?;
    sum.div(&count)
}

fn sft_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    action_mask: &Tensor,
    ignore_index: u32,
) -> candle_core::Result<Tensor> {
    let (_batch, seq_len, vocab_size) = logits.dims3()?;

    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_labels = labels.narrow(1, 1, seq_len - 1)?;
    let shift_mask = action_mask.narrow(1, 1, seq_len - 1)?;

    let log_probs = log_softmax(&shift_logits, 2)?;

    let valid_mask = shift_labels.ne(ignore_index)?.to_dtype(DType::F32)?;
    let combined_mask = valid_mask.mul(&shift_mask.to_dtype(DType::F32)?)?;

    let safe_labels = shift_labels.clamp(0u32, (vocab_size - 1) as u32)?;
    let safe_labels = safe_labels.unsqueeze(2)?;

    let gathered = log_probs.gather(&safe_labels.to_dtype(DType::U32)?, 2)?;
    let gathered = gathered.squeeze(2)?.neg()?;

    let masked = gathered.mul(&combined_mask)?;
    masked_mean(&masked, &combined_mask)
}
