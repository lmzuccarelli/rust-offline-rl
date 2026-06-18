use candle_core::{Device, DType};
use candle_nn::ops::log_softmax;

use orl_data::replay_buffer::ReplayBuffer;
use orl_model::causal_lm::TrainableCausalLM;

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvalMetrics {
    pub eval_loss: f64,
    pub perplexity: f64,
    pub num_examples: usize,
}

pub fn evaluate(
    model: &mut TrainableCausalLM,
    buffer: &ReplayBuffer,
    device: &Device,
) -> candle_core::Result<EvalMetrics> {
    let num_eval = buffer.eval_len();
    if num_eval == 0 {
        return Ok(EvalMetrics {
            eval_loss: 0.0,
            perplexity: 1.0,
            num_examples: 0,
        });
    }

    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut rng = rand::SeedableRng::seed_from_u64(42);

    // Evaluate on all eval examples one at a time
    for _i in 0..num_eval.min(50) {
        let batch = buffer.sample_sft_batch(1, &mut rng, device)?;

        let logits = model.forward_train(&batch.input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let logits = logits.to_dtype(DType::F32)?;

        let (_, seq_len, vocab_size) = logits.dims3()?;
        let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
        let shift_labels = batch.labels.narrow(1, 1, seq_len - 1)?;

        let log_probs = log_softmax(&shift_logits, 2)?;

        let mask = shift_labels.ne(u32::MAX)?.to_dtype(DType::F32)?;
        let safe_labels = shift_labels.clamp(0u32, (vocab_size - 1) as u32)?;
        let safe_labels = safe_labels.unsqueeze(2)?;

        let gathered = log_probs.gather(&safe_labels.to_dtype(DType::U32)?, 2)?;
        let gathered = gathered.squeeze(2)?.neg()?;

        let masked = gathered.mul(&mask)?;
        let batch_loss = masked.sum_all()?.to_scalar::<f32>()? as f64;
        let batch_tokens = mask.sum_all()?.to_scalar::<f32>()? as f64;

        total_loss += batch_loss;
        total_tokens += batch_tokens as usize;
    }

    let avg_loss = if total_tokens > 0 {
        total_loss / total_tokens as f64
    } else {
        0.0
    };

    let perplexity = avg_loss.exp();

    Ok(EvalMetrics {
        eval_loss: avg_loss,
        perplexity,
        num_examples: num_eval.min(50),
    })
}
