use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, VarBuilder, VarMap};
use candle_transformers::models::qwen3 as qwen;

use crate::loader::{LoadError, ModelFiles};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("load error: {0}")]
    Load(#[from] LoadError),
}

pub struct TrainableCausalLM {
    base: qwen::Model,
    lm_head: candle_nn::Linear,
    config: qwen::Config,
    varmap: VarMap,
    device: Device,
    dtype: DType,
}

impl TrainableCausalLM {
    pub fn from_varmap(
        config: &qwen::Config,
        varmap: &VarMap,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let vb = VarBuilder::from_varmap(varmap, dtype, device);
        let base = qwen::Model::new(config, vb.clone())?;

        let lm_head = if config.tie_word_embeddings {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            base,
            lm_head,
            config: config.clone(),
            varmap: varmap.clone(),
            device: device.clone(),
            dtype,
        })
    }

    fn reset_kv_cache(&mut self) -> Result<(), ModelError> {
        let vb = VarBuilder::from_varmap(&self.varmap, self.dtype, &self.device);
        self.base = qwen::Model::new(&self.config, vb)?;
        Ok(())
    }

    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
        self.reset_kv_cache()?;
        let hidden_states = self.base.forward(input_ids, 0)?;
        let logits = hidden_states.apply(&self.lm_head)?;
        Ok(logits)
    }

    pub fn forward_hidden(&mut self, input_ids: &Tensor) -> Result<Tensor, ModelError> {
        self.reset_kv_cache()?;
        let hidden_states = self.base.forward(input_ids, 0)?;
        Ok(hidden_states)
    }

    pub fn forward_train_with_hidden(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<(Tensor, Tensor), ModelError> {
        self.reset_kv_cache()?;
        let hidden_states = self.base.forward(input_ids, 0)?;
        let logits = hidden_states.apply(&self.lm_head)?;
        Ok((hidden_states, logits))
    }

    pub fn config(&self) -> &qwen::Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

pub struct FrozenCausalLM {
    model: qwen::ModelForCausalLM,
    device: Device,
}

impl FrozenCausalLM {
    pub fn load(
        files: &ModelFiles,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let config = crate::loader::load_config(&files.config_path)?;
        let vb = crate::loader::load_weights_vb(&files.weight_paths, dtype, device)?;
        let model = qwen::ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    pub fn forward_logprobs(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        self.model.clear_kv_cache();
        let logits = self.model.forward(input_ids, 0)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

pub fn compute_sequence_log_probs(
    logits: &Tensor,
    labels: &Tensor,
    ignore_index: u32,
) -> Result<Tensor, candle_core::Error> {
    let (_batch, seq_len, vocab_size) = logits.dims3()?;

    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_labels = labels.narrow(1, 1, seq_len - 1)?;

    let log_probs = candle_nn::ops::log_softmax(&shift_logits, 2)?;

    let mask = shift_labels
        .ne(ignore_index)?
        .to_dtype(DType::F32)?;

    let safe_labels = shift_labels.clamp(0u32, (vocab_size - 1) as u32)?;
    let safe_labels_unsq = safe_labels.unsqueeze(2)?;

    let gathered = log_probs.gather(&safe_labels_unsq.to_dtype(DType::U32)?, 2)?;
    let gathered = gathered.squeeze(2)?;

    let masked = gathered.mul(&mask)?;

    // Per-sequence sum of log probs, normalized by token count
    let token_counts = mask.sum(1)?.clamp(1.0f32, f32::MAX)?;
    let per_seq = masked.sum(1)?.div(&token_counts)?;

    Ok(per_seq)
}
