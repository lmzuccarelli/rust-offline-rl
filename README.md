 # Overview 

Complete Offline RL Solution

## Project structure — Cargo workspace with 4 crates:

| Crate | Purpose | Key files |
|:-----:|:-------:|:---------:|
| orl-data | Data ingestion, replay buffer, tokenization | ingest.rs, trajectory.rs, replay_buffer.rs, preference.rs |
| orl-model | Model loading, training wrappers, value heads | loader.rs, causal_lm.rs, value_head.rs |
| orl-algo  | Three RL algorithms | traits.rs, rw_sft.rs, dpo.rs, ilql.rs |
| orl-train | Training loop, checkpointing, evaluation | trainer.rs, checkpoint.rs, eval.rs, scheduler.rs |

##  Three offline RL algorithms:
  - Reward-Weighted SFT — cross-entropy weighted by exp(reward/temp), simplest baseline
  - DPO — preference pairs from same-technique trajectories, sigmoid loss on log-ratio differences vs frozen reference model
  - ILQL — token-level Q-learning with Q/V heads, TD targets, CQL conservative penalty, expectile regression, plus SFT regularization

  Data pipeline correctly ingests your logs: 10 trajectories, 85 valid steps, 22 optimization techniques, reward range 0.22-0.96.

##  CLI commands:
  - cargo run -- inspect --data-dir logs — dataset statistics (works now)
  - cargo run -- train -c config/default.toml -a rw_sft — training (requires --features cuda and a GPU)
  - cargo run -- eval -k checkpoints — checkpoint evaluation

  To train on a CUDA machine: cargo run --features cuda -- train --algorithm rw_sft
