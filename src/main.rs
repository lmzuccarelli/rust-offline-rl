use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use clap::{Parser, Subcommand};

use orl_data::ingest;
use orl_data::replay_buffer::{DataConfig, ReplayBuffer};
use orl_model::causal_lm::TrainableCausalLM;
use orl_model::loader;
use orl_train::trainer;

mod serve;

#[derive(Parser)]
#[command(
    name = "orl",
    about = "Offline Reinforcement Learning for LLM fine-tuning"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect trajectory data and print statistics
    Inspect {
        /// Path to logs directory
        #[arg(short, long, default_value = "logs")]
        data_dir: PathBuf,
    },

    /// Run offline RL training
    Train {
        /// Path to config file
        #[arg(short, long, default_value = "config/default.toml")]
        config: PathBuf,

        /// Override algorithm (dpo, rw_sft, ilql)
        #[arg(short, long)]
        algorithm: Option<String>,

        /// Override learning rate
        #[arg(long)]
        learning_rate: Option<f64>,

        /// Override max training steps
        #[arg(long)]
        max_steps: Option<usize>,
    },

    /// Evaluate a checkpoint
    Eval {
        /// Path to checkpoint directory
        #[arg(short = 'k', long)]
        checkpoint_dir: PathBuf,

        /// Path to config file
        #[arg(short, long, default_value = "config/default.toml")]
        config: PathBuf,
    },

    /// Serve a checkpoint via OpenAI-compatible API
    Serve {
        /// Path to checkpoint directory
        #[arg(short = 'k', long)]
        checkpoint_dir: PathBuf,

        /// Path to config file
        #[arg(short, long, default_value = "config/default.toml")]
        config: PathBuf,

        /// Bind address (host:port)
        #[arg(short, long, default_value = "0.0.0.0:8080")]
        bind: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { data_dir } => cmd_inspect(&data_dir),
        Commands::Train {
            config,
            algorithm,
            learning_rate,
            max_steps,
        } => cmd_train(&config, algorithm, learning_rate, max_steps),
        Commands::Eval {
            checkpoint_dir,
            config,
        } => cmd_eval(&checkpoint_dir, &config),
        Commands::Serve {
            checkpoint_dir,
            config,
            bind,
        } => cmd_serve(&checkpoint_dir, &config, &bind),
    }
}

fn cmd_serve(checkpoint_dir: &PathBuf, config_path: &PathBuf, bind_addr: &str) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve::run_server(config_path, checkpoint_dir, bind_addr))
}

fn cmd_inspect(data_dir: &PathBuf) -> Result<()> {
    let dataset = ingest::ingest_logs(data_dir)?;

    println!("=== Dataset Statistics ===\n");
    println!("Experiments: {}", dataset.experiments.len());
    println!("Total trajectories: {}", dataset.total_trajectories());
    println!("Total steps: {}", dataset.total_steps());

    let (min_r, max_r, mean_r, std_r) = dataset.reward_stats();
    println!("\nReward statistics:");
    println!("  min:  {:.4}", min_r);
    println!("  max:  {:.4}", max_r);
    println!("  mean: {:.4}", mean_r);
    println!("  std:  {:.4}", std_r);

    for exp in &dataset.experiments {
        println!("\n--- {} ({}) ---", exp.problem_name, exp.model_name);
        println!("  Level: {}", exp.level);
        println!("  Trajectories: {}", exp.trajectories.len());
        println!("  Total steps: {}", exp.total_steps());

        let techniques = exp.unique_techniques();
        println!("  Unique techniques: {}", techniques.len());
        for tech in &techniques {
            let count = exp.all_steps().filter(|s| s.technique == *tech).count();
            let avg_reward: f64 = exp
                .all_steps()
                .filter(|s| s.technique == *tech)
                .map(|s| s.reward)
                .sum::<f64>()
                / count as f64;
            println!("    {}: {} uses, avg reward {:.4}", tech, count, avg_reward);
        }

        println!("\n  Per-trajectory summary:");
        for traj in &exp.trajectories {
            println!(
                "    trajectory_{} ({}): {} steps, best={:.4}, mean={:.4}",
                traj.id,
                traj.hash,
                traj.steps.len(),
                traj.best_reward(),
                traj.mean_reward()
            );
        }
    }

    Ok(())
}

fn cmd_train(
    config_path: &PathBuf,
    algorithm: Option<String>,
    learning_rate: Option<f64>,
    max_steps: Option<usize>,
) -> Result<()> {
    let mut config = trainer::load_config(config_path)?;

    if let Some(algo) = algorithm {
        config.training.algorithm = algo;
    }
    if let Some(lr) = learning_rate {
        config.training.learning_rate = lr;
    }
    if let Some(steps) = max_steps {
        config.training.max_steps = steps;
    }

    tracing::info!(
        "training with algorithm={}, lr={:.2e}, max_steps={}",
        config.training.algorithm,
        config.training.learning_rate,
        config.training.max_steps
    );

    trainer::train(config)?;

    Ok(())
}

fn cmd_eval(checkpoint_dir: &PathBuf, config_path: &PathBuf) -> Result<()> {
    let config = trainer::load_config(config_path)?;

    let latest = orl_train::checkpoint::find_latest_checkpoint(checkpoint_dir);
    let (step, checkpoint_path) = match latest {
        Some((step, path)) => {
            println!("=== Checkpoint Info ===\n");
            println!("latest checkpoint: step {} at {}", step, path.display());
            let state = orl_train::checkpoint::load_train_state(checkpoint_dir, step)?;
            println!("algorithm: {}", state.algorithm);
            println!("epoch: {}", state.epoch);
            println!("best eval loss: {:.4}", state.best_eval_loss);
            (step, path)
        }
        None => {
            println!("no checkpoints found in {}", checkpoint_dir.display());
            return Ok(());
        }
    };

    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F32;
    tracing::info!("using device: {:?}", device);

    // Load model config and tokenizer from HuggingFace cache
    let model_files = loader::download_model(&config.model.model_id)?;
    let model_config = loader::load_config(&model_files.config_path)?;
    let tokenizer = loader::load_tokenizer(&model_files.tokenizer_path)?;

    // Load checkpoint weights into a VarMap
    let varmap = VarMap::new();
    loader::load_pretrained_weights(&varmap, &[checkpoint_path.clone()], dtype, &device)?;
    tracing::info!("loaded checkpoint weights");

    // Print weight tensor statistics
    println!("\n=== Weight Statistics (sample) ===\n");
    print_varmap_stats(&varmap, 8)?;

    // Build model from checkpoint
    let mut model = TrainableCausalLM::from_varmap(&model_config, &varmap, dtype, &device)?;

    // Load replay buffer for evaluation
    let logs_dir = PathBuf::from(&config.data.logs_dir);
    let dataset = ingest::ingest_logs(&logs_dir)?;
    let data_config = DataConfig {
        max_prompt_length: config.model.max_prompt_length,
        max_response_length: config.model.max_response_length,
        min_reward: config.data.min_reward,
        train_split: config.data.train_split,
    };
    let buffer = ReplayBuffer::new(&dataset, &tokenizer, data_config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Run evaluation
    println!("\n=== Evaluation ===\n");
    let eval_metrics = orl_train::eval::evaluate(&mut model, &buffer, &device)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("eval loss:    {:.4}", eval_metrics.eval_loss);
    println!("perplexity:   {:.2}", eval_metrics.perplexity);
    println!("num examples: {}", eval_metrics.num_examples);

    // Check for value-head auxiliary checkpoint
    if orl_train::checkpoint::has_auxiliary_checkpoint(checkpoint_dir, step) {
        let aux_path = orl_train::checkpoint::auxiliary_checkpoint_path(checkpoint_dir, step);
        println!("\n=== Value Head Checkpoint ===\n");
        println!("path: {}", aux_path.display());

        let aux_varmap = VarMap::new();
        loader::load_pretrained_weights(&aux_varmap, &[aux_path], dtype, &device)?;
        print_varmap_stats(&aux_varmap, 20)?;
    } else {
        println!("\n(no value-head checkpoint found for step {})", step);
    }

    Ok(())
}

fn print_varmap_stats(varmap: &VarMap, max_entries: usize) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();

    let total = names.len();
    let show = names.len().min(max_entries);

    println!("{} tensors total, showing {}:\n", total, show);
    println!(
        "{:<50} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "name", "shape", "mean", "std", "min", "max"
    );
    println!("{}", "-".repeat(102));

    for name in names.iter().take(show) {
        let var = &data[*name];
        let t = var.as_tensor().to_dtype(DType::F32)?;
        let numel = t.elem_count();
        if numel == 0 {
            continue;
        }

        let flat = t.flatten_all()?;
        let mean = flat.mean_all()?.to_scalar::<f32>()?;
        let var_val = flat
            .broadcast_sub(&flat.mean_all()?)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let std = var_val.sqrt();
        let min = flat.min(0)?.to_scalar::<f32>()?;
        let max = flat.max(0)?.to_scalar::<f32>()?;

        let shape_str = format!("{:?}", t.dims());
        let name_display = if name.len() > 48 {
            format!("...{}", &name[name.len() - 45..])
        } else {
            name.to_string()
        };

        println!(
            "{:<50} {:>12} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            name_display, shape_str, mean, std, min, max
        );
    }

    if total > show {
        println!("... and {} more tensors", total - show);
    }

    Ok(())
}
