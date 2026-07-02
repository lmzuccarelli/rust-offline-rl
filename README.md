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

## CLI commands:

```bash
# Dataset statistics
cargo run -- inspect --data-dir logs

# Training (requires --features cuda and a GPU)
cargo run --features cuda -- train -c config/default.toml -a rw_sft

# Checkpoint evaluation — loads weights, runs eval on replay buffer, prints tensor stats
cargo run -- eval -k checkpoints -c config/default.toml

# Inference server — OpenAI-compatible /v1/chat/completions endpoint
cargo run -- serve -k checkpoints -c config/default.toml -b 0.0.0.0:8080
```

## Checkpointing

Training saves checkpoints at regular intervals and on best eval loss:

| File | Contents |
|:-----|:---------|
| `checkpoint-{step}.safetensors` | Base model weights (F32) |
| `checkpoint-{step}-state.json` | Training metadata (step, epoch, best eval loss, algorithm) |
| `checkpoint-{step}-value-heads.safetensors` | ILQL Q/V head weights (only when training with ILQL) |

The `eval` command loads the latest checkpoint, reconstructs the model, runs evaluation on the replay buffer, and reports loss, perplexity, and per-tensor statistics. If a value-head checkpoint exists, its tensors are reported as well.

## Inference server

The `serve` command loads a fine-tuned checkpoint and exposes an OpenAI-compatible HTTP API.

```bash
cargo run -- serve -k checkpoints -c config/default.toml -b 0.0.0.0:8080
```

### Endpoint

`POST /v1/chat/completions`

### Request

```json
{
  "model": "qwen3",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain loop tiling for CUDA kernels."}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "seed": 42
}
```

### Response

```json
{
  "id": "chatcmpl-1751452800",
  "object": "chat.completion",
  "created": 1751452800,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Loop tiling partitions iteration spaces into smaller blocks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 48,
    "total_tokens": 80
  }
}
```

### Implementation details

- Uses `ModelForCausalLM` from `candle_transformers` with KV-cache for efficient autoregressive decoding
- Applies Qwen3 chat template (`<|im_start|>role\ncontent<|im_end|>`)
- Sampling via `LogitsProcessor` (temperature, top-p, greedy)
- Stops on `<|im_end|>` / `<|endoftext|>` or `max_tokens`
- Built on axum; model is behind `Arc<Mutex<>>` for thread-safe request handling
