use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;

use orl_algo::dpo::{Dpo, DpoConfig};
use orl_algo::ilql::{Ilql, IlqlConfig};
use orl_algo::rw_sft::{RewardWeightedSft, RwSftConfig};
use orl_algo::traits::OfflineRLAlgorithm;
use orl_data::ingest;
use orl_data::replay_buffer::{DataConfig, ReplayBuffer};
use orl_model::causal_lm::{FrozenCausalLM, TrainableCausalLM};
use orl_model::loader;

use crate::checkpoint::{self, TrainState};
use crate::eval;
use crate::scheduler::{LrScheduler, SchedulerConfig};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub dtype: String,
    pub max_prompt_length: usize,
    pub max_response_length: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: String,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub save_every: usize,
    pub eval_every: usize,
    pub output_dir: String,
    pub seed: u64,
    pub adamw: AdamWConfig,
    pub scheduler: SchedulerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AdamWConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataSectionConfig {
    pub logs_dir: String,
    pub train_split: f64,
    pub prompt_truncation: String,
    pub min_reward: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FullConfig {
    pub model: ModelConfig,
    pub data: DataSectionConfig,
    pub training: TrainingConfig,
    #[serde(default)]
    pub dpo: DpoConfig,
    #[serde(default)]
    pub rw_sft: RwSftConfig,
    #[serde(default)]
    pub ilql: IlqlConfig,
}

pub fn load_config(path: &Path) -> anyhow::Result<FullConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: FullConfig = toml::from_str(&content)?;
    Ok(config)
}

pub fn train(config: FullConfig) -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    tracing::info!("using device: {:?}", device);

    let dtype = loader::parse_dtype(&config.model.dtype);
    tracing::info!("using dtype: {:?}", dtype);

    // Download and load model files
    tracing::info!("downloading model: {}", config.model.model_id);
    let model_files = loader::download_model(&config.model.model_id)?;

    // Load tokenizer
    let tokenizer = loader::load_tokenizer(&model_files.tokenizer_path)?;
    tracing::info!("loaded tokenizer with {} vocab", tokenizer.get_vocab_size(true));

    // Load model config
    let model_config = loader::load_config(&model_files.config_path)?;
    tracing::info!(
        "model config: {} layers, {} hidden, {} heads",
        model_config.num_hidden_layers,
        model_config.hidden_size,
        model_config.num_attention_heads
    );

    // Ingest trajectory data
    let logs_dir = PathBuf::from(&config.data.logs_dir);
    let dataset = ingest::ingest_logs(&logs_dir)?;
    tracing::info!(
        "loaded {} experiments, {} trajectories, {} steps",
        dataset.experiments.len(),
        dataset.total_trajectories(),
        dataset.total_steps()
    );

    // Build replay buffer
    let data_config = DataConfig {
        max_prompt_length: config.model.max_prompt_length,
        max_response_length: config.model.max_response_length,
        min_reward: config.data.min_reward,
        train_split: config.data.train_split,
    };
    let buffer = ReplayBuffer::new(&dataset, &tokenizer, data_config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    tracing::info!(
        "replay buffer: {} train, {} eval, {} preference pairs",
        buffer.train_len(),
        buffer.eval_len(),
        buffer.preference_pairs_len()
    );

    // Create trainable model with VarMap
    let mut varmap = VarMap::new();
    let mut model = TrainableCausalLM::from_varmap(&model_config, &varmap, dtype, &device)?;

    // Load pretrained weights
    tracing::info!("loading pretrained weights...");
    for path in &model_files.weight_paths {
        varmap.load(path)?;
    }
    tracing::info!("pretrained weights loaded");

    // Setup optimizer
    let params = ParamsAdamW {
        lr: config.training.learning_rate,
        beta1: config.training.adamw.beta1,
        beta2: config.training.adamw.beta2,
        eps: config.training.adamw.eps,
        weight_decay: config.training.adamw.weight_decay,
    };

    let mut all_vars = varmap.all_vars();

    // Create algorithm
    let mut algorithm: Box<dyn OfflineRLAlgorithm> = match config.training.algorithm.as_str() {
        "dpo" => {
            let ref_model = FrozenCausalLM::load(&model_files, dtype, &device)?;
            tracing::info!("loaded frozen reference model for DPO");
            let mut dpo_config = config.dpo.clone();
            dpo_config.batch_size = config.training.batch_size;
            Box::new(Dpo::new(dpo_config, Some(ref_model)))
        }
        "rw_sft" => {
            let mut rw_config = config.rw_sft.clone();
            rw_config.batch_size = config.training.batch_size;
            Box::new(RewardWeightedSft::new(rw_config))
        }
        "ilql" => {
            let mut ilql_config = config.ilql.clone();
            ilql_config.batch_size = config.training.batch_size;
            let ilql = Ilql::new(
                ilql_config,
                model_config.hidden_size,
                model_config.vocab_size,
                dtype,
                &device,
            )?;
            // Add value head vars to optimizer
            all_vars.extend(ilql.value_vars());
            Box::new(ilql)
        }
        other => anyhow::bail!("unknown algorithm: {}", other),
    };

    let mut optimizer = AdamW::new(all_vars, params)?;

    // LR scheduler
    let total_steps = config.training.max_steps;
    let mut scheduler = LrScheduler::new(
        config.training.scheduler.clone(),
        config.training.learning_rate,
        config.training.warmup_steps,
        total_steps,
    );

    let mut rng = StdRng::seed_from_u64(config.training.seed);
    let output_dir = PathBuf::from(&config.training.output_dir);

    let mut best_eval_loss = f64::INFINITY;

    // Progress bar
    let progress = indicatif::ProgressBar::new(total_steps as u64);
    progress.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:50} {pos}/{len} loss={msg}")
            .unwrap(),
    );

    tracing::info!("starting training with algorithm: {}", algorithm.name());

    for step in 0..total_steps {
        let lr = scheduler.step();
        optimizer.set_learning_rate(lr);

        // Gradient accumulation
        let mut accum_loss = 0.0f64;
        let mut accum_metrics = std::collections::HashMap::new();

        for _micro in 0..config.training.gradient_accumulation_steps {
            let (loss, metrics) = algorithm.compute_loss(&mut model, &mut rng, &buffer, &device)?;

            // Scale loss by accumulation steps
            let scaled_loss = (&loss / config.training.gradient_accumulation_steps as f64)?;
            optimizer.backward_step(&scaled_loss)?;

            accum_loss += loss.to_scalar::<f32>()? as f64;
            for (k, v) in metrics {
                *accum_metrics.entry(k).or_insert(0.0) += v;
            }
        }

        let avg_loss = accum_loss / config.training.gradient_accumulation_steps as f64;

        // Post-step updates (e.g., ILQL target network update)
        algorithm.post_step()?;

        progress.set_position(step as u64);
        progress.set_message(format!("{:.4}", avg_loss));

        if step % 10 == 0 {
            tracing::info!(
                "step={}, loss={:.4}, lr={:.2e}",
                step, avg_loss, lr
            );
        }

        // Evaluation
        if step > 0 && step % config.training.eval_every == 0 {
            let eval_metrics = eval::evaluate(&mut model, &buffer, &device)?;
            tracing::info!(
                "eval: loss={:.4}, perplexity={:.2}, examples={}",
                eval_metrics.eval_loss,
                eval_metrics.perplexity,
                eval_metrics.num_examples
            );

            if eval_metrics.eval_loss < best_eval_loss {
                best_eval_loss = eval_metrics.eval_loss;
                let state = TrainState {
                    global_step: step,
                    epoch: step / buffer.train_len().max(1),
                    best_eval_loss,
                    algorithm: algorithm.name().to_string(),
                };
                checkpoint::save_checkpoint(&varmap, &state, &output_dir, step)?;
            }
        }

        // Periodic checkpoint
        if step > 0 && step % config.training.save_every == 0 {
            let state = TrainState {
                global_step: step,
                epoch: step / buffer.train_len().max(1),
                best_eval_loss,
                algorithm: algorithm.name().to_string(),
            };
            checkpoint::save_checkpoint(&varmap, &state, &output_dir, step)?;
        }
    }

    progress.finish();

    // Final checkpoint
    let state = TrainState {
        global_step: total_steps,
        epoch: total_steps / buffer.train_len().max(1),
        best_eval_loss,
        algorithm: algorithm.name().to_string(),
    };
    checkpoint::save_checkpoint(&varmap, &state, &output_dir, total_steps)?;

    tracing::info!("training complete. best eval loss: {:.4}", best_eval_loss);

    Ok(())
}
