# Offline Reinforcement Learning Strategies

Strategies for fine-tuning language models on trajectory data with per-step rewards and detailed actions, evaluated in the context of CUDA kernel optimization.

---

## 1. Reward-Weighted Supervised Fine-Tuning (RW-SFT)

**How it works:** Weight each trajectory (or each step) proportionally to its reward during supervised fine-tuning. High-reward trajectories contribute more to the gradient, low-reward ones contribute less (or are filtered out). The simplest variant uses a binary filter (keep top-k% trajectories); the continuous variant weights the cross-entropy loss by `exp(R / temperature)`.

**Loss:** `L = -Sigma w(r_i) * log pi(a_i | s_i)` where `w(r_i)` is the reward weight.

**Pros:**
- Extremely simple to implement -- just weighted SFT
- Stable training; no divergence risk from value function bootstrapping
- Works well when you have clear reward signal separation between good and bad trajectories
- Low compute overhead -- single forward/backward pass per sample
- Naturally handles per-step reward structure by weighting individual steps

**Cons:**
- Ignores the sequential/temporal structure within trajectories -- treats steps independently
- No credit assignment across steps; a bad step in a good trajectory still gets upweighted
- Wastes data -- low-reward trajectories are downweighted or discarded entirely
- Cannot learn to improve beyond the best trajectory in the dataset
- Sensitive to reward scale and temperature hyperparameter

---

## 2. Direct Preference Optimization (DPO)

**How it works:** Given pairs of trajectories (or steps) where one is preferred over another, DPO directly optimizes the policy to assign higher likelihood to the preferred completion. It reformulates the RLHF objective (KL-constrained reward maximization) into a classification loss over preference pairs, eliminating the need for an explicit reward model.

**Loss:** `L = -log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))` where `y_w` is preferred, `y_l` is dispreferred.

**Pros:**
- No reward model or value function needed -- simpler pipeline
- Theoretically equivalent to RLHF under Bradley-Terry preference model
- Stable training with the KL constraint anchoring to the reference policy
- Well-suited for LLM fine-tuning (proven at scale with language models)
- Directly learns relative quality, which is robust when absolute reward scales are noisy

**Cons:**
- Requires constructing preference pairs -- need to pair trajectories/steps by reward and this pairing strategy matters a lot
- Operates on full completions; applying it per-step within a trajectory requires careful chunking
- `beta` (temperature) is sensitive -- too low collapses to best-of-N, too high ignores preferences
- Cannot leverage absolute reward magnitudes -- a trajectory with reward 0.99 vs 0.98 is treated the same as 0.9 vs 0.1
- Reference policy must remain fixed (or periodically refreshed), adding memory overhead

---

## 3. Implicit Language Q-Learning (ILQL / IQL for LLMs)

**How it works:** Learns a Q-function and value function on top of the language model's token representations. For each token position, it estimates the expected return of taking that token given the current state. At inference time, it reweights the LM's next-token distribution by the advantage `Q(s,a) - V(s)`, guiding generation toward higher-value completions without modifying the base model weights (or with light fine-tuning).

**Loss:** Combines:
- Value loss: expectile regression `L_V = |tau - 1(Q - V < 0)| * (Q - V)^2`
- Q loss: temporal difference `L_Q = (r + gamma*V(s') - Q(s,a))^2`
- Optional policy extraction via advantage-weighted regression

**Pros:**
- Best credit assignment -- learns per-token value estimates, so it knows which specific actions (tokens/steps) drove the reward
- Fully exploits per-step reward structure; can learn that a specific step was the critical optimization
- Can generalize beyond the dataset -- learns a value landscape, not just imitation
- Decouples value estimation from policy; can use the Q-values to guide any base model
- Handles multi-step sequential decision-making natively (this is what it was designed for)

**Cons:**
- Most complex to implement -- need Q-head, V-head, target networks, and careful loss balancing
- Bootstrapping (TD learning) can be unstable, especially in high-dimensional token spaces
- Expectile parameter `tau` and discount `gamma` are additional hyperparameters to tune
- Higher memory footprint -- multiple heads on top of the LM
- Slower convergence than supervised methods
- Offline Q-learning is prone to overestimation of OOD (out-of-distribution) actions

---

## 4. Decision Transformer (DT)

**How it works:** Frames RL as sequence modeling. The model is conditioned on desired return-to-go (sum of future rewards), past states, and past actions, then predicts the next action autoregressively. At inference time, you condition on a high return-to-go to elicit optimal behavior.

**Input sequence:** `(R_hat_1, s_1, a_1, R_hat_2, s_2, a_2, ...)`

**Pros:**
- Elegant -- turns RL into a supervised sequence modeling problem, which aligns perfectly with LLM architectures
- No value function, no bootstrapping, no instability
- Naturally handles variable-length trajectories
- Can condition on different reward levels at inference time to control quality
- Trajectory data with per-step rewards and actions maps directly to DT's input format

**Cons:**
- Cannot "stitch" -- it can only reproduce trajectory segments it has seen, not combine the best parts of different trajectories
- Performance bounded by the best full trajectories in the dataset
- Needs careful reward-to-go normalization
- Does not truly do credit assignment; it memorizes reward-conditioned behavior patterns
- Struggles when the reward signal is sparse or delayed (less of an issue with per-step rewards)

---

## 5. Step-Level DPO (StepDPO / Token-Level DPO)

**How it works:** An extension of DPO that constructs preference pairs at the step level within trajectories rather than comparing entire trajectories. For each decision point, it compares the action taken against alternative actions (from other trajectories at similar states), creating fine-grained preference signal.

**Pros:**
- Combines DPO's stability with step-level granularity
- Directly leverages per-step reward structure for constructing preferences
- Better credit assignment than trajectory-level DPO
- Still no reward model or value function needed

**Cons:**
- Constructing meaningful step-level preference pairs is non-trivial -- need to align "similar states" across trajectories
- For CUDA kernel optimization, defining state similarity is domain-specific and may require embedding-based matching
- Increases the number of preference pairs dramatically, raising compute cost
- Relatively new; less established than trajectory-level DPO

---

## 6. Conservative Q-Learning (CQL)

**How it works:** Adds a regularizer to standard Q-learning that penalizes Q-values for out-of-distribution actions, pushing the learned policy to stay close to the data distribution. This directly addresses the overestimation problem in offline RL.

**Loss:** Standard TD loss + `alpha * (E_{a~pi}[Q(s,a)] - E_{a~D}[Q(s,a)])`

**Pros:**
- Strong theoretical guarantees -- provably learns a lower bound on the true Q-function
- Handles distributional shift well
- Can outperform behavioral cloning when dataset quality is mixed

**Cons:**
- Can be overly conservative -- underestimates good novel actions
- `alpha` regularization weight is tricky to tune
- Same complexity concerns as ILQL (Q-heads, target networks)
- Designed for continuous control; adaptation to token-level LLM actions adds engineering burden

---

## Comparison Table

| Strategy | Credit Assignment | Complexity | Data Efficiency | Can Exceed Dataset | Stability |
|----------|------------------|------------|-----------------|-------------------|-----------|
| RW-SFT | Poor (trajectory) | Very Low | Low | No | High |
| DPO | Moderate (pair) | Low | Moderate | No | High |
| ILQL | **Excellent** (token) | High | **High** | **Yes** | Moderate |
| Decision Transformer | Poor (pattern) | Low | Moderate | No | High |
| Step-Level DPO | Good (step) | Moderate | Moderate | No | High |
| CQL | Excellent (token) | High | High | Yes | Low-Moderate |

---

## Recommendation: ILQL as Primary, with RW-SFT as Baseline

For trajectory data with per-step rewards and detailed actions in CUDA kernel optimization, **ILQL is the strongest choice**:

1. **Per-step reward exploitation.** ILQL is the only method that fully exploits this structure -- it learns which specific optimization steps (loop tiling, memory coalescing, register allocation) actually drove the performance gain, rather than treating the whole trajectory as good or bad.

2. **Generalization beyond the dataset.** RW-SFT and DPO can only reproduce or interpolate between existing trajectories. ILQL can learn a value landscape and potentially discover novel optimization sequences by stitching together good fragments from different trajectories.

3. **Sequential decision-making.** CUDA kernel optimization is inherently sequential -- the order and combination of optimizations matters. ILQL's temporal difference learning captures these dependencies.

**Practical approach:** Start with RW-SFT as a quick baseline, then compare ILQL against it. Use DPO as a middle-ground if ILQL proves too unstable. These three algorithms cover the spectrum from simple-but-limited to complex-but-powerful.
