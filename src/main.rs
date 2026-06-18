use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use orl_data::ingest;
use orl_train::trainer;

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
    }
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
    let _config = trainer::load_config(config_path)?;

    let latest = orl_train::checkpoint::find_latest_checkpoint(checkpoint_dir);
    match latest {
        Some((step, path)) => {
            println!("latest checkpoint: step {} at {}", step, path.display());
            let state = orl_train::checkpoint::load_train_state(checkpoint_dir, step)?;
            println!("algorithm: {}", state.algorithm);
            println!("best eval loss: {:.4}", state.best_eval_loss);
        }
        None => {
            println!("no checkpoints found in {}", checkpoint_dir.display());
        }
    }

    Ok(())
}
