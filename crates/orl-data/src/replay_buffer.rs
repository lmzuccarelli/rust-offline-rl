use std::collections::HashMap;

use candle_core::{Device, Tensor, DType};
use rand::prelude::*;
use tokenizers::Tokenizer;

use crate::trajectory::Dataset;
use crate::preference::PreferencePair;

#[derive(Debug, Clone)]
pub struct TokenizedExample {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub reward: f64,
    pub trajectory_id: usize,
    pub step_id: usize,
    pub technique: String,
}

#[derive(Debug, Clone)]
pub struct DataConfig {
    pub max_prompt_length: usize,
    pub max_response_length: usize,
    pub min_reward: f64,
    pub train_split: f64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            max_prompt_length: 2048,
            max_response_length: 4096,
            min_reward: -1.0,
            train_split: 0.9,
        }
    }
}

pub struct ReplayBuffer {
    pub train_examples: Vec<TokenizedExample>,
    pub eval_examples: Vec<TokenizedExample>,
    pub train_preference_pairs: Vec<PreferencePair>,
    pub eval_preference_pairs: Vec<PreferencePair>,
    pub reward_weights: Vec<f64>,
    config: DataConfig,
}

impl ReplayBuffer {
    pub fn new(
        dataset: &Dataset,
        tokenizer: &Tokenizer,
        config: DataConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut all_examples = Vec::new();

        for step in dataset.all_steps() {
            if step.reward < config.min_reward {
                continue;
            }

            let prompt_ids = tokenize_text(tokenizer, &step.prompt_text, config.max_prompt_length)?;
            let response_ids = tokenize_text(tokenizer, &step.response_text, config.max_response_length)?;

            if prompt_ids.is_empty() || response_ids.is_empty() {
                continue;
            }

            all_examples.push(TokenizedExample {
                prompt_ids,
                response_ids,
                reward: step.reward,
                trajectory_id: step.trajectory_id,
                step_id: step.step_id,
                technique: step.technique.clone(),
            });
        }

        let split_idx = (all_examples.len() as f64 * config.train_split) as usize;
        let eval_examples = all_examples.split_off(split_idx);
        let train_examples = all_examples;

        let train_preference_pairs = build_preference_pairs(&train_examples, 0.1);
        let eval_preference_pairs = build_preference_pairs(&eval_examples, 0.1);

        let reward_weights = compute_reward_weights(&train_examples, 1.0);

        Ok(Self {
            train_examples,
            eval_examples,
            train_preference_pairs,
            eval_preference_pairs,
            reward_weights,
            config,
        })
    }

    pub fn sample_sft_batch(
        &self,
        batch_size: usize,
        rng: &mut StdRng,
        device: &Device,
    ) -> candle_core::Result<SftBatch> {
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.train_examples.len()))
            .collect();

        self.build_sft_batch(&indices, device)
    }

    pub fn sample_weighted_sft_batch(
        &self,
        batch_size: usize,
        rng: &mut StdRng,
        device: &Device,
    ) -> candle_core::Result<SftBatch> {
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.train_examples.len()))
            .collect();

        let mut batch = self.build_sft_batch(&indices, device)?;

        let weights: Vec<f32> = indices
            .iter()
            .map(|&i| self.reward_weights[i] as f32)
            .collect();
        batch.weights = Tensor::new(weights, device)?;

        Ok(batch)
    }

    pub fn sample_dpo_batch(
        &self,
        batch_size: usize,
        rng: &mut StdRng,
        device: &Device,
    ) -> candle_core::Result<DpoBatch> {
        if self.train_preference_pairs.is_empty() {
            return Err(candle_core::Error::Msg("no preference pairs available".to_string()));
        }

        let pairs: Vec<&PreferencePair> = (0..batch_size)
            .map(|_| {
                let idx = rng.random_range(0..self.train_preference_pairs.len());
                &self.train_preference_pairs[idx]
            })
            .collect();

        build_dpo_batch(&pairs, &self.config, device)
    }

    pub fn sample_ilql_batch(
        &self,
        batch_size: usize,
        rng: &mut StdRng,
        device: &Device,
    ) -> candle_core::Result<IlqlBatch> {
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.train_examples.len()))
            .collect();

        build_ilql_batch(&self.train_examples, &indices, &self.config, device)
    }

    pub fn train_len(&self) -> usize {
        self.train_examples.len()
    }

    pub fn eval_len(&self) -> usize {
        self.eval_examples.len()
    }

    pub fn preference_pairs_len(&self) -> usize {
        self.train_preference_pairs.len()
    }

    fn build_sft_batch(
        &self,
        indices: &[usize],
        device: &Device,
    ) -> candle_core::Result<SftBatch> {
        let max_len = indices
            .iter()
            .map(|&i| {
                let ex = &self.train_examples[i];
                ex.prompt_ids.len() + ex.response_ids.len()
            })
            .max()
            .unwrap_or(1);

        let batch_size = indices.len();
        let mut input_ids_flat = vec![0u32; batch_size * max_len];
        let mut labels_flat = vec![u32::MAX; batch_size * max_len]; // u32::MAX as ignore index
        let mut attention_mask_flat = vec![0f32; batch_size * max_len];

        for (b, &idx) in indices.iter().enumerate() {
            let ex = &self.train_examples[idx];
            let prompt_len = ex.prompt_ids.len();
            let response_len = ex.response_ids.len();
            let total = prompt_len + response_len;

            for (i, &tok) in ex.prompt_ids.iter().enumerate() {
                input_ids_flat[b * max_len + i] = tok;
                attention_mask_flat[b * max_len + i] = 1.0;
            }
            for (i, &tok) in ex.response_ids.iter().enumerate() {
                input_ids_flat[b * max_len + prompt_len + i] = tok;
                attention_mask_flat[b * max_len + prompt_len + i] = 1.0;
            }
            // Labels: predict response tokens (shifted by 1 internally by the loss)
            for i in prompt_len..total {
                labels_flat[b * max_len + i] = input_ids_flat[b * max_len + i];
            }
        }

        let input_ids = Tensor::new(input_ids_flat, device)?.reshape((batch_size, max_len))?;
        let labels = Tensor::new(labels_flat, device)?.reshape((batch_size, max_len))?;
        let attention_mask = Tensor::new(attention_mask_flat, device)?.reshape((batch_size, max_len))?;

        let weights = Tensor::ones((batch_size,), DType::F32, device)?;

        Ok(SftBatch {
            input_ids,
            labels,
            attention_mask,
            weights,
        })
    }
}

pub struct SftBatch {
    pub input_ids: Tensor,
    pub labels: Tensor,
    pub attention_mask: Tensor,
    pub weights: Tensor,
}

pub struct DpoBatch {
    pub chosen_input_ids: Tensor,
    pub chosen_labels: Tensor,
    pub chosen_attention_mask: Tensor,
    pub rejected_input_ids: Tensor,
    pub rejected_labels: Tensor,
    pub rejected_attention_mask: Tensor,
}

pub struct IlqlBatch {
    pub input_ids: Tensor,
    pub labels: Tensor,
    pub attention_mask: Tensor,
    pub rewards: Tensor,
    pub action_mask: Tensor,
    pub is_terminal: Tensor,
}

fn tokenize_text(
    tokenizer: &Tokenizer,
    text: &str,
    max_length: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("tokenization error: {}", e))?;

    let mut ids: Vec<u32> = encoding.get_ids().to_vec();

    // Tail truncation: keep the last max_length tokens
    if ids.len() > max_length {
        ids = ids[ids.len() - max_length..].to_vec();
    }

    Ok(ids)
}

fn build_preference_pairs(
    examples: &[TokenizedExample],
    min_margin: f64,
) -> Vec<PreferencePair> {
    let mut by_technique: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, ex) in examples.iter().enumerate() {
        by_technique.entry(&ex.technique).or_default().push(i);
    }

    let mut pairs = Vec::new();

    // Same-technique pairing
    for indices in by_technique.values() {
        if indices.len() < 2 {
            continue;
        }
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let a = &examples[indices[i]];
                let b = &examples[indices[j]];
                let diff = (a.reward - b.reward).abs();
                if diff >= min_margin {
                    let (chosen, rejected) = if a.reward > b.reward {
                        (indices[i], indices[j])
                    } else {
                        (indices[j], indices[i])
                    };
                    pairs.push(PreferencePair {
                        chosen_idx: chosen,
                        rejected_idx: rejected,
                        chosen_prompt_ids: examples[chosen].prompt_ids.clone(),
                        chosen_response_ids: examples[chosen].response_ids.clone(),
                        rejected_prompt_ids: examples[rejected].prompt_ids.clone(),
                        rejected_response_ids: examples[rejected].response_ids.clone(),
                        reward_chosen: examples[chosen].reward,
                        reward_rejected: examples[rejected].reward,
                    });
                }
            }
        }
    }

    // Cross-technique pairing for examples with large reward differences
    let large_margin = 0.3;
    for i in 0..examples.len() {
        for j in (i + 1)..examples.len() {
            if examples[i].technique == examples[j].technique {
                continue;
            }
            let diff = (examples[i].reward - examples[j].reward).abs();
            if diff >= large_margin {
                let (chosen, rejected) = if examples[i].reward > examples[j].reward {
                    (i, j)
                } else {
                    (j, i)
                };
                pairs.push(PreferencePair {
                    chosen_idx: chosen,
                    rejected_idx: rejected,
                    chosen_prompt_ids: examples[chosen].prompt_ids.clone(),
                    chosen_response_ids: examples[chosen].response_ids.clone(),
                    rejected_prompt_ids: examples[rejected].prompt_ids.clone(),
                    rejected_response_ids: examples[rejected].response_ids.clone(),
                    reward_chosen: examples[chosen].reward,
                    reward_rejected: examples[rejected].reward,
                });
            }
        }
    }

    pairs
}

fn compute_reward_weights(examples: &[TokenizedExample], temperature: f64) -> Vec<f64> {
    if examples.is_empty() {
        return Vec::new();
    }

    let scores: Vec<f64> = examples.iter().map(|e| e.reward / temperature).collect();
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();

    exp_scores.iter().map(|s| s / sum * examples.len() as f64).collect()
}

fn build_dpo_batch(
    pairs: &[&PreferencePair],
    _config: &DataConfig,
    device: &Device,
) -> candle_core::Result<DpoBatch> {
    let batch_size = pairs.len();

    let max_chosen_len = pairs
        .iter()
        .map(|p| p.chosen_prompt_ids.len() + p.chosen_response_ids.len())
        .max()
        .unwrap_or(1);

    let max_rejected_len = pairs
        .iter()
        .map(|p| p.rejected_prompt_ids.len() + p.rejected_response_ids.len())
        .max()
        .unwrap_or(1);

    let mut chosen_ids = vec![0u32; batch_size * max_chosen_len];
    let mut chosen_labels = vec![u32::MAX; batch_size * max_chosen_len];
    let mut chosen_mask = vec![0f32; batch_size * max_chosen_len];

    let mut rejected_ids = vec![0u32; batch_size * max_rejected_len];
    let mut rejected_labels = vec![u32::MAX; batch_size * max_rejected_len];
    let mut rejected_mask = vec![0f32; batch_size * max_rejected_len];

    for (b, pair) in pairs.iter().enumerate() {
        let cp_len = pair.chosen_prompt_ids.len();
        let _cr_len = pair.chosen_response_ids.len();
        for (i, &tok) in pair.chosen_prompt_ids.iter().enumerate() {
            chosen_ids[b * max_chosen_len + i] = tok;
            chosen_mask[b * max_chosen_len + i] = 1.0;
        }
        for (i, &tok) in pair.chosen_response_ids.iter().enumerate() {
            chosen_ids[b * max_chosen_len + cp_len + i] = tok;
            chosen_mask[b * max_chosen_len + cp_len + i] = 1.0;
            chosen_labels[b * max_chosen_len + cp_len + i] = tok;
        }

        let rp_len = pair.rejected_prompt_ids.len();
        let _rr_len = pair.rejected_response_ids.len();
        for (i, &tok) in pair.rejected_prompt_ids.iter().enumerate() {
            rejected_ids[b * max_rejected_len + i] = tok;
            rejected_mask[b * max_rejected_len + i] = 1.0;
        }
        for (i, &tok) in pair.rejected_response_ids.iter().enumerate() {
            rejected_ids[b * max_rejected_len + rp_len + i] = tok;
            rejected_mask[b * max_rejected_len + rp_len + i] = 1.0;
            rejected_labels[b * max_rejected_len + rp_len + i] = tok;
        }
    }

    Ok(DpoBatch {
        chosen_input_ids: Tensor::new(chosen_ids, device)?.reshape((batch_size, max_chosen_len))?,
        chosen_labels: Tensor::new(chosen_labels, device)?.reshape((batch_size, max_chosen_len))?,
        chosen_attention_mask: Tensor::new(chosen_mask, device)?.reshape((batch_size, max_chosen_len))?,
        rejected_input_ids: Tensor::new(rejected_ids, device)?.reshape((batch_size, max_rejected_len))?,
        rejected_labels: Tensor::new(rejected_labels, device)?.reshape((batch_size, max_rejected_len))?,
        rejected_attention_mask: Tensor::new(rejected_mask, device)?.reshape((batch_size, max_rejected_len))?,
    })
}

fn build_ilql_batch(
    examples: &[TokenizedExample],
    indices: &[usize],
    _config: &DataConfig,
    device: &Device,
) -> candle_core::Result<IlqlBatch> {
    let batch_size = indices.len();

    let max_len = indices
        .iter()
        .map(|&i| examples[i].prompt_ids.len() + examples[i].response_ids.len())
        .max()
        .unwrap_or(1);

    let mut input_ids_flat = vec![0u32; batch_size * max_len];
    let mut labels_flat = vec![u32::MAX; batch_size * max_len];
    let mut attention_mask_flat = vec![0f32; batch_size * max_len];
    let mut rewards_flat = vec![0f32; batch_size * max_len];
    let mut action_mask_flat = vec![0f32; batch_size * max_len];
    let is_terminal = vec![1f32; batch_size]; // default terminal

    for (b, &idx) in indices.iter().enumerate() {
        let ex = &examples[idx];
        let prompt_len = ex.prompt_ids.len();
        let response_len = ex.response_ids.len();
        let total = prompt_len + response_len;

        for (i, &tok) in ex.prompt_ids.iter().enumerate() {
            input_ids_flat[b * max_len + i] = tok;
            attention_mask_flat[b * max_len + i] = 1.0;
        }
        for (i, &tok) in ex.response_ids.iter().enumerate() {
            input_ids_flat[b * max_len + prompt_len + i] = tok;
            attention_mask_flat[b * max_len + prompt_len + i] = 1.0;
            labels_flat[b * max_len + prompt_len + i] = tok;
            action_mask_flat[b * max_len + prompt_len + i] = 1.0;
        }
        // Assign reward to last response token
        if response_len > 0 {
            rewards_flat[b * max_len + total - 1] = ex.reward as f32;
        }
    }

    Ok(IlqlBatch {
        input_ids: Tensor::new(input_ids_flat, device)?.reshape((batch_size, max_len))?,
        labels: Tensor::new(labels_flat, device)?.reshape((batch_size, max_len))?,
        attention_mask: Tensor::new(attention_mask_flat, device)?.reshape((batch_size, max_len))?,
        rewards: Tensor::new(rewards_flat, device)?.reshape((batch_size, max_len))?,
        action_mask: Tensor::new(action_mask_flat, device)?.reshape((batch_size, max_len))?,
        is_terminal: Tensor::new(is_terminal, device)?,
    })
}
