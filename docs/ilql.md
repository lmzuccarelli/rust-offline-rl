# ILQL: Implicit Language Q-Learning

ILQL adapts classical offline reinforcement learning to **token-level language model fine-tuning**. The core insight is: treat every token generation step as an RL action, attach lightweight Q and V heads to a frozen (or slowly-trained) language model backbone, and train those heads on a static dataset of (prompt, response, reward) tuples — no environment interaction needed.

## 1. The MDP Framing: Tokens as Actions

Standard RL has states, actions, and rewards. ILQL maps these onto autoregressive text generation:

| RL Concept | ILQL Mapping |
|---|---|
| **State s_t** | The hidden representation at position t (the LM's encoding of all tokens so far) |
| **Action a_t** | The token chosen at position t |
| **Action space** | The full vocabulary (e.g. 151,936 tokens for Qwen3) |
| **Reward r_t** | 0 for all tokens except the final response token, which gets the trajectory reward |
| **Transition** | Deterministic — appending token a_t to the sequence yields s_{t+1} |

In the codebase, `IlqlBatch` (`orl-data/src/replay_buffer.rs`) materializes this: `rewards` is a `(batch, seq_len)` tensor that is zero everywhere except the last response token, and `action_mask` marks which positions are in the response (i.e., which tokens are "actions" the agent chose vs. prompt tokens it was given).

## 2. Architecture: Q-Head and V-Head

On top of the LM backbone's hidden states `h_t` (dimension `hidden_size`), two small heads are attached (see `orl-model/src/value_head.rs`):

### Q-Head (2-layer MLP: hidden -> GELU -> vocab)

```
Q(s_t, .) = W2 . GELU(W1 . h_t)    (output dimension: vocab_size)
```

This produces a Q-value for *every possible token* at position t. To get the Q-value for the token that was actually chosen, you gather: `Q(s_t, a_t) = Q_all[a_t]`.

### V-Head (single linear layer: hidden -> 1)

```
V(s_t) = w_v . h_t    (scalar output)
```

The value head estimates the expected return from state s_t *regardless* of which action is taken — it is the baseline.

### Target Networks

Both heads have frozen copies (`target_q_head`, `target_v_head`) that are updated via exponential moving average after each training step. This is the standard trick from DQN/SAC to stabilize training — the targets you regress against move slowly, preventing oscillation.

## 3. The Four Loss Terms

The total loss is a weighted sum of four components. Each serves a distinct purpose.

### 3a. Q-Learning Loss (TD Error)

The Bellman equation says the Q-value of taking action a_t in state s_t should equal the immediate reward plus the discounted future value:

```
Q(s_t, a_t) ~ r_t + gamma . V_target(s_{t+1})
```

The loss minimizes the squared TD error:

```
L_Q = E[ (Q(s_t, a_t) - [r_t + gamma . V_target(s_{t+1})])^2 ]
```

In the implementation (`orl-algo/src/ilql.rs`, lines 112-135), this is computed with shifted tensors — `shift_q` covers positions 0..T-1, `shift_target_v` covers positions 1..T (the "next states"), and `shift_rewards` provides r_t. The `action_mask` ensures only response tokens contribute.

**Why use V_target(s') instead of max_a Q_target(s', a)?** This is the key "implicit" part of ILQL. In classic DQN you would take `max_a Q(s', a)`, but with a vocabulary of 150K+ tokens, that max is noisy and overestimates. Instead, ILQL learns a separate V-head and uses expectile regression (below) to implicitly extract the optimal value without an explicit argmax.

### 3b. Value Loss (Expectile Regression)

The V-head must approximate the *optimal* value — not the average Q, but something closer to the maximum. Expectile regression achieves this with an asymmetric loss:

```
L_V = E[ |tau - 1(A < 0)| . A^2 ]
```

where `A = Q(s_t, a_t) - V(s_t)` is the advantage, and tau (default `expectile_tau = 0.7`) controls the asymmetry.

The implementation (`ilql.rs`, lines 172-186):

```
weight = tau + indicator(A < 0) . (1 - 2*tau)
loss = weight . A^2
```

When tau = 0.7:
- If A > 0 (action better than value baseline): weight = 0.7 — penalize *heavily* for underestimating V
- If A < 0 (action worse than baseline): weight = 0.3 — penalize *lightly* for overestimating V

This biases V upward toward the better actions in the dataset. As tau approaches 1.0, V approaches max_a Q(s, a). The value tau = 0.7 is a practical sweet spot — aggressive enough to learn from the best trajectories, conservative enough to avoid overfitting to outliers.

**This is what makes ILQL "implicit"**: instead of explicitly computing the max over 150K tokens (expensive, noisy), the V-head *implicitly* tracks the optimistic quantile of Q through asymmetric regression. The TD target then uses this optimistic V, which guides Q toward high-value actions without ever computing an argmax.

### 3c. CQL Conservative Penalty

Conservative Q-Learning (CQL) addresses the fundamental problem of offline RL: **overestimation of out-of-distribution actions**. Since you never interact with the environment, Q-values for unseen actions can drift arbitrarily high — the model might learn to assign high Q to token sequences it has never observed.

The CQL penalty pushes down Q-values for all actions while pushing up Q-values for actions actually taken in the dataset:

```
L_CQL = E[ log sum_a exp(Q(s_t, a)) - Q(s_t, a_t) ]
```

The first term (`logsumexp` over the vocabulary, `ilql.rs` line 144) is a soft-max over all possible tokens — it penalizes high Q-values broadly. The second term (Q at the actual action) rewards the action that was taken. The net effect: Q-values are pushed down everywhere *except* for actions that appear in the data, creating a conservative lower bound on the true Q-function.

The weight `cql_alpha = 1.0` controls how conservative the penalty is. Higher alpha means more pessimistic Q-values, which makes the policy stick closer to the data distribution.

### 3d. SFT Regularization

The final term is plain supervised fine-tuning cross-entropy on the language modeling objective:

```
L_SFT = -E[ log P(a_t | s_t) ]    (over response tokens)
```

This is computed in `sft_cross_entropy` (`ilql.rs`, lines 195-220): standard next-token prediction loss, masked to only count response tokens.

**Why include this?** Two reasons:

1. **Prevents catastrophic forgetting**: The RL objectives can distort the LM's output distribution. SFT loss keeps the model coherent as a language model.
2. **Stabilizes early training**: Before the Q/V heads have learned meaningful values, the SFT signal provides a useful gradient. It acts as a warm prior — the model stays close to "reasonable text" while the RL heads figure out what "good text" means.

The weight `sft_weight = 0.1` keeps this term subordinate to the RL objectives.

## 4. Total Loss and Training

```
L_total = L_Q + L_V + alpha_CQL . L_CQL + w_SFT . L_SFT
```

After each gradient step, `post_step()` calls `update_targets()` which performs the soft EMA update on all target network parameters:

```
theta_target <- tau_update . theta_current + (1 - tau_update) . theta_target
```

With `target_update_tau = 0.005`, targets move at 0.5% per step — very slowly, which stabilizes training.

## 5. Default Hyperparameters

| Parameter | Default | Purpose |
|---|---|---|
| `gamma` | 0.99 | Discount factor for future rewards |
| `cql_alpha` | 1.0 | Weight of the CQL conservative penalty |
| `expectile_tau` | 0.7 | Expectile asymmetry (higher = more optimistic V) |
| `target_update_tau` | 0.005 | Soft target EMA rate |
| `sft_weight` | 0.1 | Weight of the SFT regularization term |
| `q_head_layers` | 2 | Number of layers in the Q-head MLP |
| `batch_size` | 1 | Training batch size |

## 6. Advantages for CUDA Kernel Optimization Trajectories

**Offline-only**: Trains entirely from a fixed dataset of (prompt, kernel code, performance reward) tuples without needing to run kernels during training (no environment interaction).

**Token-level credit assignment**: A trajectory-level reward ("this kernel is 2x faster") gets propagated back through every token via the Bellman equation. The Q-head can learn that, say, choosing `__shared__` at position 47 is high-value while an unnecessary `__syncthreads()` at position 82 is low-value — even though the reward was only given once at the end.

**Conservative guarantees**: CQL ensures the model does not hallucinate "amazing" kernel patterns it has never seen. It stays grounded in what worked in the training data.

**Implicit optimization avoids the vocab-size problem**: With 150K+ tokens, explicitly computing `argmax Q` or sampling from a softmax over Q is expensive. The expectile-regression V-head sidesteps this entirely.

**SFT regularization preserves code fluency**: Without it, aggressive RL training could produce token sequences that have high Q-values but are syntactically broken. The SFT term keeps the model writing valid code.

## 7. Comparison with Alternatives

| Property | ILQL | PPO/RLHF | DPO | Best-of-N |
|---|---|---|---|---|
| Needs environment | No | Yes (online rollouts) | No | Yes (inference-time) |
| Credit assignment | Token-level | Token-level (but noisy) | Trajectory-level only | None |
| Conservatism | Built-in (CQL) | Clip ratio + KL penalty | Implicit (reference model) | N/A |
| Computational cost | One forward + head passes | Multiple rollouts + critic | One forward pass | N forward passes |
| Handles mixed-quality data | Yes (expectile selects best) | Poorly (on-policy only) | Yes (pairwise) | N/A |

ILQL's main advantage over DPO for this setting: DPO only learns "this trajectory is better than that one" — it cannot learn *which tokens* made the difference. ILQL's token-level Q-values can, which matters for long code sequences where a single design decision (memory layout, loop tiling, warp divergence) drives the reward.
