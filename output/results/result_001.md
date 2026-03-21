---
idea_id: 001
status: complete
relevance_score: 9
confidence_score: 7
completeness_score: 9
---

# Research Results: Idea 001 — TBop: Ternary Binary Optimizer

## Summary

TBop (Ternary Binary Optimizer) generalizes the Bop binary optimizer (Helwegen et al., NeurIPS 2019) from binary {-1, +1} to ternary {-1, 0, +1} weights by modeling each parameter as a 3-state finite state machine (FSM) with hysteresis-based transition rules governed by an exponential moving average (EMA) of gradients. The method eliminates latent weights entirely, requiring only a BF16 EMA buffer (2 bytes/param) plus the ternary weight itself (~0.2 bytes/param), totaling **2.2 bytes/param** — an 87% reduction from the standard STE+Adam training cost of 16 bytes/param. We additionally propose an enhanced variant, TBop-S (TBop with Second-moment normalization), at 4.0 bytes/param that adds per-parameter gradient variance tracking for improved robustness, directly inspired by the "Bop and Beyond" second-order extension.

## Proposed Method

### 1. Core Framework: Finite State Machine with Hysteresis

Each ternary weight $w_i \in \{-1, 0, +1\}$ is modeled as a 3-state FSM. The only persistent state per parameter is a BF16 exponential moving average $m_i$ of the gradient.

**EMA Update:**

$$m_i^{(t)} = (1 - \gamma) \cdot m_i^{(t-1)} + \gamma \cdot g_i^{(t)}$$

where $g_i^{(t)} = \partial \mathcal{L} / \partial w_i$ is the gradient at step $t$, and $\gamma \in (0, 1)$ is the adaptivity rate (following Bop's convention where $\gamma$ weights the new gradient, not the history).

**Ordered Transition Rules (Hysteresis FSM):**

The core design enforces ordered transitions — weights must pass through the zero state when changing sign. This prevents destructive $-1 \leftrightarrow +1$ oscillations and gives the zero state its natural role as a pruning/gating state.

```
State w_i = 0 (inactive):
    → +1   if   m_i < -τ_act        (persistent negative gradient ⇒ increase weight)
    → -1   if   m_i > +τ_act        (persistent positive gradient ⇒ decrease weight)
    →  0   otherwise

State w_i = +1 (active positive):
    →  0   if   m_i > +τ_deact      (gradient says decrease ⇒ deactivate)
    → +1   otherwise
    (direct +1 → -1 forbidden)

State w_i = -1 (active negative):
    →  0   if   m_i < -τ_deact      (gradient says increase ⇒ deactivate)
    → -1   otherwise
    (direct -1 → +1 forbidden)
```

**Sign Convention:** $g_i = \partial \mathcal{L} / \partial w_i > 0$ means "loss decreases when $w_i$ decreases." So a persistent positive EMA indicates the weight should decrease, and vice versa.

**Hysteresis Rationale:** We set $\tau_{\text{deact}} > \tau_{\text{act}}$ to create asymmetric inertia:
- It is *easier* to activate a weight (promote from 0 to ±1) than to deactivate it.
- Active weights contribute to the network's representational capacity and should be harder to remove.
- This is motivated by hysteresis quantization (ICLR 2022, Sun et al.), which showed that state-dependent thresholds reduce weight oscillation at quantization boundaries and improve training stability.

The reverse setting ($\tau_{\text{act}} > \tau_{\text{deact}}$) would be sparsity-promoting, useful for pruning-oriented training. Both regimes are configurable.

### 2. Threshold Adaptation (SGDAT-Inspired)

Fixed thresholds are fragile: different layers have different gradient magnitudes, and the optimal threshold changes during training. We adopt an adaptive mechanism inspired by SGDAT (Gu et al., Neurocomputing 2023):

**Per-parameter adaptive threshold:**

$$\tau_i^{(t)} = \tau_0 \cdot \left(1 + \alpha \cdot \text{flips}_i^{(t)}\right)$$

where $\text{flips}_i^{(t)}$ counts the cumulative number of state transitions for parameter $i$ up to step $t$, and $\alpha > 0$ is a damping coefficient.

**Effect:** Parameters that flip frequently accumulate a higher threshold, requiring stronger gradient evidence before the next transition. This damps oscillation — the dominant failure mode identified in both Bop and quantization-aware training generally (Nagel et al., ICML 2022).

**Zero-cost flip counting:** To avoid an additional counter per parameter, we use the **sign bit and low-order mantissa bits of the BF16 EMA** to encode a saturating flip count. BF16 has 7 mantissa bits; we can steal the lowest 2 bits to encode a 2-bit saturating counter (values 0–3), which provides 4 damping levels. This sacrifices 2 bits of EMA precision (reducing effective mantissa from 7 to 5 bits, i.e., ~BF14 precision) but keeps memory at exactly 2 bytes/param. Alternative: use the full 2 bytes for the EMA and track no explicit flip count, instead using the *magnitude* of the EMA itself as a proxy for confidence (high |m_i| after transition ⇒ strong signal ⇒ no need to damp).

**Global threshold schedule:**

$$\tau_0^{(t)} = \tau_{\text{init}} \cdot \cos\left(\frac{\pi t}{2T}\right) + \tau_{\text{final}}$$

where $T$ is the total training steps. This cosine schedule reduces $\tau_0$ over time, allowing finer adjustments in late training when weights are near their final values. This maps the standard cosine learning rate schedule to the threshold domain: decreasing $\tau$ corresponds to increasing the effective learning rate for discrete updates, which is the opposite of continuous training — late in training, we want to allow small corrections to the ternary weight configuration.

### 3. Probabilistic Transition Variant (TBop-P)

The deterministic threshold creates a sharp boundary: weights with EMA just below $\tau$ never flip, while those just above always flip. This boundary instability was identified by He et al. (Neurocomputing 2025) as a key failure mode of Bop.

**Probabilistic transition:** Replace the hard threshold with a sigmoid-smoothed Bernoulli probability:

$$P(\text{transition}) = \sigma\left(\frac{|m_i| - \tau}{\lambda}\right) = \frac{1}{1 + \exp\left(-\frac{|m_i| - \tau}{\lambda}\right)}$$

where $\lambda > 0$ controls the sharpness of the transition (smaller $\lambda$ = closer to deterministic). The transition direction is still determined by the sign of $m_i$ and the current state (following the FSM rules).

**When EMA is far from threshold:** $P \approx 1$ (strong signal) or $P \approx 0$ (weak signal) — same as deterministic TBop.
**When EMA is near threshold:** $P \approx 0.5$ — stochastic exploration prevents getting stuck.

This adds no memory cost (the random number can be generated on-the-fly from a shared PRNG state).

### 4. Second-Moment Enhanced Variant (TBop-S)

Inspired by "A Bop and Beyond" (Suarez-Ramirez et al., CVPR 2021 Workshop), which showed that adding second-moment normalization to Bop improves convergence speed and hyperparameter robustness:

**Additional state:** A FP16 running second moment $v_i$:

$$v_i^{(t)} = (1 - \beta) \cdot v_i^{(t-1)} + \beta \cdot (g_i^{(t)})^2$$

**Normalized transition criterion:** Replace $m_i$ in the transition rules with the normalized signal:

$$\hat{m}_i = \frac{m_i}{\sqrt{v_i} + \epsilon}$$

This normalizes the gradient momentum by its variance, making the threshold $\tau$ universally applicable across layers and parameters regardless of gradient scale — exactly analogous to how Adam normalizes learning rates.

**Memory:** 2 bytes (BF16 EMA) + 2 bytes (FP16 $v$) + 0.2 bytes (ternary weight) = **4.2 bytes/param**. This just fits within the 4-byte extra budget.

**When to use TBop-S vs. TBop:** TBop-S trades 2 extra bytes/param for significantly improved robustness to hyperparameters and heterogeneous gradient scales across layers. For models with diverse layer types (attention, FFN, embeddings) where gradient magnitudes vary by orders of magnitude, TBop-S is strongly recommended. For uniform architectures or when memory is the primary constraint, basic TBop at 2.2 bytes/param is preferred.

### 5. Initialization Strategy

Since TBop has no latent weights, initialization of the ternary weights themselves matters:

**Option A — Random ternary initialization:**
$$w_i \sim \text{Categorical}\left(\frac{1-s}{2}, s, \frac{1-s}{2}\right) \text{ for states } \{-1, 0, +1\}$$

where $s$ is the target sparsity ratio (fraction of zeros). Based on empirical evidence from trained BitNet b1.58 models, $s \approx 0.42$ (42% zero weights) is a good starting point.

**Option B — Kaiming-scaled quantization:**
1. Initialize continuous weights $\tilde{w} \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$ (Kaiming He initialization)
2. Apply absmean quantization: $w_i = \text{RoundClip}(\tilde{w}_i / (\text{mean}(|\tilde{w}|) + \epsilon), -1, 1)$
3. Discard $\tilde{w}$ after quantization

**Option C — Inherited from pre-trained model:**
Use a "16-to-1.58" strategy where full-precision training runs for a warmup phase, then weights are quantized and the optimizer state is discarded in favor of TBop's EMA (initialized to zero). This leverages the finding from BitNet b1.58-2B-4T (arXiv 2504.12285) that inheriting from full-precision training nearly matches full-precision performance.

**EMA initialization:** $m_i^{(0)} = 0$ for all parameters (no gradient history).

### 6. EMA Reset Policy on Transition

When a weight transitions to a new state, its EMA is reset:

$$m_i \leftarrow 0 \quad \text{upon any state transition}$$

**Rationale:** The accumulated gradient evidence that triggered the transition is now "spent." The EMA in the old state reflected the loss landscape from the old weight value; after transitioning, the loss landscape is different (the parameter has changed value), so historical gradient evidence is stale. Starting fresh allows the optimizer to quickly adapt to the new loss landscape.

**Alternative — partial reset:** $m_i \leftarrow \delta \cdot m_i$ with $\delta \in (0, 1)$, preserving some momentum. This may help when transitions chain quickly (e.g., $0 \to +1 \to 0$ in rapid succession, where the momentum toward 0 is informative).

### 7. Interaction with Gradient Computation

TBop uses the same gradient computation as standard STE-based ternary training:

1. **Forward pass:** Use ternary weights $w^q \in \{-1, 0, +1\}$ for all linear operations (BitLinear layers with absmean quantization of activations to INT8).
2. **Backward pass:** Apply STE — pass gradients through the quantization function as if it were the identity: $\partial \mathcal{L} / \partial w_i \approx \partial \mathcal{L} / \partial w_i^q$.
3. **Update:** Apply TBop EMA update and transition rules (no gradient applied to any continuous weight — there are no continuous weights).

**Note:** The STE is used only for gradient estimation through the quantization, not for maintaining latent weights. The gradient computation cost is identical to standard training. The only difference is step 3: instead of applying Adam to latent weights, we apply the TBop FSM update to the EMA and check transitions. This is computationally cheaper (no square roots, no division, no second moment tracking in basic TBop).

### 8. Handling the Silent Weights Problem

Xiang et al. (ECCV 2024) identified that >50% of weights in binary networks remain unchanged throughout training ("silent weights"), particularly those whose gradients are consistently small. This is a critical risk for TBop since the threshold mechanism can trap weights in their initial state.

**Mitigation 1 — Threshold decay:** The cosine threshold schedule (Section 2) progressively lowers $\tau$, allowing even weak gradient signals to trigger transitions late in training.

**Mitigation 2 — Stochastic perturbation:** Periodically (every $K$ steps), apply a small random perturbation to the EMA of silent weights (weights that haven't transitioned in $K$ steps):

$$m_i \leftarrow m_i + \eta \cdot \xi, \quad \xi \sim \mathcal{N}(0, 1)$$

where $\eta$ is a small perturbation scale. This gives silent weights a chance to escape their frozen state. The perturbation is cheap and statistically unbiased (zero mean).

**Mitigation 3 — Layer-wise threshold normalization:** Normalize thresholds per layer by the average gradient magnitude in that layer:

$$\tau_{\ell}^{(t)} = \tau_0^{(t)} \cdot \text{mean}\left(|g_\ell^{(t)}|\right)$$

This ensures layers with small gradients (e.g., early layers in deep networks) have proportionally lower thresholds, preventing them from going entirely silent.

### 9. Complete Algorithm — TBop

```python
# TBop: Ternary Binary Optimizer
# Memory per parameter: 2 bytes (BF16 EMA) + ~0.2 bytes (ternary weight) = 2.2 bytes
# No latent weights. No second moment. No master copy.

def tbop_init(model):
    """Initialize TBop state."""
    for param in model.parameters():
        # Option B: Kaiming-scaled ternary initialization
        w_cont = kaiming_normal_(param.shape)
        scale = w_cont.abs().mean() + 1e-8
        param.ternary = round_clip(w_cont / scale, -1, 1)  # {-1, 0, +1}
        param.ema = zeros_like(param, dtype=bfloat16)        # BF16 EMA
        del w_cont  # no latent weights stored

def tbop_step(model, gamma, tau_act_base, tau_deact_base, schedule_factor,
              adaptive_alpha=0.1, probabilistic=False, temperature=0.1):
    """
    One TBop optimization step. Called after loss.backward().

    Args:
        gamma: EMA adaptivity rate (e.g., 1e-3)
        tau_act_base: base activation threshold
        tau_deact_base: base deactivation threshold
        schedule_factor: cosine schedule multiplier for tau (1.0 → 0.0 over training)
        adaptive_alpha: flip-count damping coefficient
        probabilistic: if True, use TBop-P (stochastic transitions)
        temperature: temperature for probabilistic transitions
    """
    for param in model.parameters():
        w = param.ternary           # current ternary weight, {-1, 0, +1}
        m = param.ema               # BF16 EMA of gradients
        g = param.grad              # current gradient (transient, not stored)

        # --- Step 1: Update EMA ---
        m = (1 - gamma) * m + gamma * g

        # --- Step 2: Compute effective thresholds ---
        tau_act = tau_act_base * schedule_factor
        tau_deact = tau_deact_base * schedule_factor
        # (Optional: per-parameter adaptive scaling based on flip count
        #  encoded in EMA LSBs, omitted for clarity)

        # --- Step 3: Determine transitions ---
        if not probabilistic:
            # Deterministic TBop
            # From state 0: activate
            to_pos = (w == 0) & (m < -tau_act)
            to_neg = (w == 0) & (m > +tau_act)
            # From state +1: deactivate
            deact_pos = (w == +1) & (m > +tau_deact)
            # From state -1: deactivate
            deact_neg = (w == -1) & (m < -tau_deact)
        else:
            # Probabilistic TBop-P
            # Compute transition probabilities via sigmoid
            p_act = sigmoid((m.abs() - tau_act) / temperature)
            p_deact = sigmoid((m.abs() - tau_deact) / temperature)
            coin_act = rand_like(m)
            coin_deact = rand_like(m)
            # From state 0
            to_pos = (w == 0) & (m < 0) & (coin_act < p_act)
            to_neg = (w == 0) & (m > 0) & (coin_act < p_act)
            # From state +1
            deact_pos = (w == +1) & (m > 0) & (coin_deact < p_deact)
            # From state -1
            deact_neg = (w == -1) & (m < 0) & (coin_deact < p_deact)

        # --- Step 4: Apply transitions ---
        w[to_pos] = +1
        w[to_neg] = -1
        w[deact_pos] = 0
        w[deact_neg] = 0

        # --- Step 5: Reset EMA for transitioned parameters ---
        transitioned = to_pos | to_neg | deact_pos | deact_neg
        m[transitioned] = 0.0

        # --- Step 6: Store updated state ---
        param.ternary = w
        param.ema = m.to(bfloat16)  # ensure BF16 storage
        param.grad = None             # free gradient memory
```

### 10. Complete Algorithm — TBop-S (Second-Moment Enhanced)

```python
# TBop-S: TBop with Second-moment normalization
# Memory: 2 bytes (BF16 EMA) + 2 bytes (FP16 second moment) + 0.2 bytes (ternary)
#       = 4.2 bytes/param (fits 4-byte extra budget)

def tbop_s_step(model, gamma, beta, tau_base, schedule_factor, eps=1e-8):
    """
    TBop-S optimization step. Like TBop but with normalized transitions.

    Args:
        gamma: EMA adaptivity rate for first moment
        beta: decay rate for second moment (e.g., 0.999)
        tau_base: universal threshold (works across all layers due to normalization)
        schedule_factor: cosine schedule multiplier
    """
    for param in model.parameters():
        w = param.ternary
        m = param.ema                 # BF16 first moment
        v = param.second_moment       # FP16 second moment (variance)
        g = param.grad

        # --- Update moments ---
        m = (1 - gamma) * m + gamma * g
        v = (1 - beta) * v + beta * g * g

        # --- Normalize first moment by second moment ---
        m_hat = m / (sqrt(v) + eps)   # this is scale-invariant

        # --- Use m_hat in place of m for all transition rules ---
        tau = tau_base * schedule_factor
        # (Same FSM logic as TBop, but using m_hat instead of m)

        to_pos = (w == 0) & (m_hat < -tau)
        to_neg = (w == 0) & (m_hat > +tau)
        deact_pos = (w == +1) & (m_hat > +tau)
        deact_neg = (w == -1) & (m_hat < -tau)

        w[to_pos] = +1
        w[to_neg] = -1
        w[deact_pos] = 0
        w[deact_neg] = 0

        transitioned = to_pos | to_neg | deact_pos | deact_neg
        m[transitioned] = 0.0
        v[transitioned] = 0.0  # also reset second moment

        param.ternary = w
        param.ema = m.to(bfloat16)
        param.second_moment = v.to(float16)
        param.grad = None
```

### 11. Mathematical Convergence Analysis

**Theorem (Informal):** Under standard assumptions (bounded gradients, diminishing adaptivity rate or constant rate with threshold annealing), TBop converges to a neighborhood of a local minimum of the ternary loss landscape.

**Proof sketch:**

**Step 1 — EMA convergence.** For fixed weights, the EMA $m_i^{(t)}$ converges exponentially to $\mathbb{E}[g_i]$:

$$\left| m_i^{(t)} - \mathbb{E}[g_i] \right| \leq (1-\gamma)^t \left| m_i^{(0)} - \mathbb{E}[g_i] \right| + \frac{\gamma \sigma_g}{\sqrt{2\gamma - \gamma^2}}$$

where $\sigma_g$ is the gradient noise standard deviation. For small $\gamma$ (e.g., $10^{-3}$), the steady-state error is $O(\sqrt{\gamma} \cdot \sigma_g)$.

**Step 2 — Threshold as significance test.** A transition occurs only when $|m_i| > \tau$, which requires:

$$|\mathbb{E}[g_i]| > \tau + O(\sqrt{\gamma} \cdot \sigma_g)$$

This means transitions only happen when there is statistically significant gradient evidence — the optimizer effectively performs a one-sided hypothesis test at each step. Noisy transitions are suppressed, preventing oscillation.

**Step 3 — Discrete Lyapunov argument.** Define the discrete Lyapunov function:

$$V(w^{(t)}) = \mathcal{L}(w^{(t)}) + \frac{\mu}{2} \sum_i (m_i^{(t)})^2$$

where $\mu > 0$ is a coupling constant. When a transition occurs (say $w_i: 0 \to +1$), it means $m_i < -\tau_{\text{act}}$, which implies $\partial \mathcal{L} / \partial w_i$ has been persistently negative. Changing $w_i$ from 0 to +1 *increases* $w_i$ in the direction that reduces $\mathcal{L}$, so $\mathcal{L}(w^{(t+1)}) < \mathcal{L}(w^{(t)})$ in expectation (assuming the gradient signal is correct on average). Simultaneously, the EMA reset $m_i \leftarrow 0$ reduces the second term. Thus $V$ decreases in expectation at each transition.

Between transitions, $V$ can increase due to gradient noise affecting the EMA, but this increase is bounded by $O(\gamma^2 \sigma_g^2)$ per step (the EMA update is a contraction).

**Step 4 — Convergence to neighborhood.** Since $V$ is bounded below (loss is bounded, EMA magnitudes are bounded by gradient bounds) and decreases at transitions while increasing slowly between transitions, the sequence of discrete states converges to a neighborhood where:

$$|\mathbb{E}[g_i]| \leq \tau + O(\sqrt{\gamma} \cdot \sigma_g) \quad \forall i$$

At this equilibrium, no parameter has sufficient gradient evidence to transition. The neighborhood size is controlled by $\tau$ and $\gamma$: smaller $\tau$ and $\gamma$ give a tighter neighborhood at the cost of slower convergence.

**Step 5 — Connection to existing theory.** This convergence argument is structurally analogous to:
- The DST (Discrete State Transition) convergence proof in GXNOR-Net (Deng et al., 2018), which showed convergence in probability to $\mathbb{E}[W^*]$ satisfying first-order optimality.
- ECO's convergence proof (arXiv 2601.22101), which shows convergence to a $1/(1-\beta_2)$-dilated neighborhood.
- The "recurrence of optimum" result (arXiv 2012.05529), which shows quantized weights recurrently visit the global optimum under mild conditions.

**Key limitation of the proof:** The Lyapunov argument assumes the expected gradient after a transition still points in the same direction. For highly non-convex losses (as in deep networks), a weight change can change the entire gradient landscape. The argument holds locally (when transitions are small perturbations) but may break for pathological loss surfaces. This is a shared limitation with all convergence arguments for discrete neural network optimization.

### 12. Recommended Hyperparameters

Based on Bop's validated settings and scaling analysis:

| Hyperparameter | TBop (Basic) | TBop-S | Rationale |
|---|---|---|---|
| $\gamma$ (adaptivity rate) | $5 \times 10^{-4}$ | $5 \times 10^{-4}$ | Between Bop's $10^{-4}$ and $10^{-3}$; ternary needs faster adaptation than binary (3 states = richer dynamics) |
| $\tau_{\text{act}}$ initial | $10^{-5}$ | $10^{-2}$ | TBop-S uses normalized gradients, so threshold is O(1) rather than O(gradient scale) |
| $\tau_{\text{deact}}$ initial | $3 \times 10^{-5}$ | $3 \times 10^{-2}$ | ~3x activation threshold for hysteresis |
| $\tau$ final | $0.1 \times \tau_{\text{init}}$ | $0.1 \times \tau_{\text{init}}$ | Cosine decay to 10% of initial |
| $\beta$ (second moment decay) | N/A | 0.999 | Standard Adam-like value |
| $\alpha$ (adaptive damping) | 0.1 | 0.1 | SGDAT default |
| Initialization sparsity $s$ | 0.42 | 0.42 | Matches empirical BitNet distributions |

**Schedule:** Cosine annealing of $\tau_0$ from initial to final over total training steps. Optionally, $\gamma$ can also be decayed linearly (as in original Bop's ImageNet experiments) from $5 \times 10^{-4}$ to $5 \times 10^{-6}$.

## Literature Support

### Directly Supporting Works

| Paper | Relevance to TBop |
|---|---|
| **Bop** (Helwegen et al., NeurIPS 2019, arXiv:1906.02107) | Foundation. Proved latent weights unnecessary for binary networks. EMA + threshold flipping. ~56.6% top-1 ImageNet on Bi-Real Net, matching STE baselines. |
| **A Bop and Beyond** (Suarez-Ramirez et al., CVPR 2021W, arXiv:2104.05124) | Second-moment normalization for Bop. Faster convergence, better accuracy, more robust hyperparameters. Directly inspires TBop-S. |
| **SGDAT** (Gu et al., Neurocomputing 2023) | Adaptive threshold based on flip count. Makes vanilla SGD viable for BNNs. Directly inspires TBop's adaptive thresholds. |
| **Probabilistic Bop** (He et al., Neurocomputing 2025) | Bernoulli-distributed flipping replaces hard thresholds. Addresses boundary instability. Inspires TBop-P. |
| **GXNOR-Net / DST** (Deng et al., Neural Networks 2018, arXiv:1705.09283) | Discrete State Transition framework for ternary weights. Convergence in probability proved. Closest existing work to ternary latent-weight-free training. |
| **Hysteresis Quantization** (Sun et al., ICLR 2022) | State-dependent quantization thresholds reduce oscillation. Motivates TBop's asymmetric activation/deactivation thresholds. |

### Supporting Context

| Paper | Relevance |
|---|---|
| **OvSW: Silent Weights** (Xiang et al., ECCV 2024) | Identifies that >50% of BNN weights never change during training. Motivates TBop's anti-silent-weight mechanisms (threshold decay, perturbation). |
| **Overcoming Oscillations in QAT** (Nagel et al., ICML 2022, arXiv:2203.11086) | Oscillation is the primary failure mode in low-bit QAT. Supports hysteresis and ordered transitions. |
| **DQT with Stochastic Rounding** (arXiv:2412.04787) | Ternary training without latent weights is feasible via stochastic rounding. Quality gap at 130M params. Validates feasibility of latent-weight-free ternary training. |
| **ECO** (arXiv:2601.22101) | Error feedback into momentum enables training without master weights. Convergence proof to bounded neighborhood. Could hybridize with TBop (inject EMA "error" after transitions). |
| **BitNet b1.58** (arXiv:2402.17764) | The target architecture. Matches FP16 LLaMA at 3B+. Standard training uses 16 bytes/param. |
| **BitNet b1.58-2B-4T** (arXiv:2504.12285) | Shows "16-to-1.58" training strategy works. ~42% weight sparsity. Validates ternary LLMs at scale. |
| **Lion Optimizer** (Chen et al., 2023, arXiv:2302.06675) | Sign-based optimizer with single momentum state. Shows first moment alone is sufficient for LLM training. Supports TBop's design of using only first moment. |
| **AdEMAMix** (arXiv:2409.03137) | Dual EMA improves gradient exploitation. Could inspire a dual-EMA TBop variant with slow + fast EMAs for better gradient memory. |
| **Recurrence of Optimum** (arXiv:2012.05529) | Quantized weights recurrently visit the global optimum. Supports TBop's convergence to a neighborhood. |
| **BOLD** (NeurIPS 2024) | Boolean training without any gradient descent. Shows 92.37% on CIFAR-10. Validates the premise that latent-weight-free training can be competitive. |
| **StoMPP** (arXiv:2601.22660) | STE-free binary training via progressive freezing. +18% on CIFAR-10 over STE baselines. Validates STE-free approaches. |

### Information-Theoretic Context

| Paper | Key Finding |
|---|---|
| **QSGD** (Alistarh et al., NeurIPS 2017, arXiv:1610.02132) | 4–8 bits/param sufficient for effective training. TBop's 16-bit EMA (2 bytes) is comfortably within this range. |
| **Training Quantized Nets** (Li et al., NeurIPS 2017, arXiv:1706.02379) | Convergence accuracy depends on discretization coarseness. Ternary (3 levels) is the coarsest practical grid; expect larger neighborhoods than finer quantization. |
| **1-bit Adam** (Tang et al., 2021, arXiv:2102.02888) | Adam's second moment stabilizes during training. Supports using a fixed or slowly-adapting second moment (TBop-S) rather than full adaptive tracking. |

## Generalizability Analysis

### Why TBop works across model sizes (100M to 100B+)

1. **Per-parameter independence.** The EMA update and FSM transition are entirely local — each parameter's update depends only on its own gradient, EMA, and state. No global statistics, batch statistics, or cross-parameter dependencies. The algorithm is embarrassingly parallel and scales identically from 100M to 100B+ parameters.

2. **Scale-invariant transitions (TBop-S).** The normalized signal $\hat{m}_i = m_i / (\sqrt{v_i} + \epsilon)$ is dimensionless. The threshold $\tau$ has the same interpretation regardless of the absolute gradient magnitude, which varies with model size, layer depth, and training phase. This is the same reason Adam works universally while SGD with a fixed learning rate does not.

3. **Constant memory overhead.** The 2.2 bytes/param (TBop) or 4.2 bytes/param (TBop-S) overhead is *per-parameter*, not per-layer or per-model. There are no global buffers, no activation caches, and no cross-parameter state. Memory scales linearly with parameter count, identically to Adam.

4. **Architecture agnosticism.** TBop operates on any ternary weight, regardless of whether it belongs to an attention layer, FFN layer, embedding layer, or any other module. The FSM transition rules are universal.

### Why TBop works across datasets and domains

1. **Gradient-based.** TBop consumes standard gradients (via STE) and makes decisions based on their persistent direction (via EMA). It makes no assumptions about the data distribution, loss function, or task type.

2. **EMA as noise filter.** The exponential moving average naturally adapts to dataset-specific noise levels. Higher noise = more gradient variance = EMA converges more slowly = fewer transitions = more conservative updates. Lower noise = faster EMA convergence = more responsive transitions. This self-regulation requires no dataset-specific tuning.

3. **Threshold as implicit regularizer.** The threshold $\tau$ acts as a minimum evidence requirement. For noisy datasets (small, diverse, or multi-task), the threshold prevents premature transitions. For clean datasets (large, homogeneous), the threshold allows rapid convergence. This is analogous to how learning rate warmup adapts to dataset properties.

### Limitations of generalizability

1. **Threshold sensitivity.** While TBop-S largely mitigates this via normalization, basic TBop requires threshold tuning that may vary across architectures. The SGDAT-inspired adaptive mechanism reduces but does not eliminate this dependency.

2. **Sparsity dynamics.** The equilibrium fraction of zero weights may vary significantly across architectures and tasks. If the task requires very high or very low sparsity, the threshold balance ($\tau_{\text{act}}$ vs. $\tau_{\text{deact}}$) must be adjusted accordingly.

3. **Untested at LLM scale.** No variant of Bop has been tested beyond small vision models (ResNet on ImageNet). The ternary extension and LLM-scale behavior are theoretical predictions, not empirical facts. The most significant risk is that the quality gap (vs. STE+Adam) may be unacceptable at scale.

## Matching Metrics

- Relevance to original question: **9/10** — TBop directly addresses the core question of eliminating latent weight overhead for ternary training. It is the most natural extension of the only existing latent-weight-free optimizer (Bop) to the ternary domain.
- Confidence in findings: **7/10** — The method is theoretically well-motivated and builds on validated components (Bop, SGDAT, hysteresis quantization, DST), but has never been empirically tested at any scale, let alone LLM scale. The convergence argument is a sketch, not a formal proof. The quality gap is uncertain.
- Completeness of investigation: **9/10** — We covered the full Bop lineage (original, second-order, probabilistic, SGDAT), all related ternary training methods (DST/GXNOR-Net, DQT, BitNet), convergence theory (discrete optimization, EMA convergence, recurrence of optimum), failure mode analysis (silent weights, oscillation, boundary instability), and information-theoretic context (QSGD, minimum bits).

## Memory Budget Breakdown

### TBop (Basic Variant)

| Component | Bytes/Param | Format | Purpose |
|---|---|---|---|
| Ternary weight $w$ | ~0.2 | 2-bit packed | The model weight ∈ {-1, 0, +1} |
| Gradient EMA $m$ | 2.0 | BF16 | Exponential moving average of gradients — the only optimizer state |
| Gradient $g$ | 0 (transient) | BF16 | Computed during backward pass, consumed immediately, not stored |
| **Total** | **2.2** | | **Well within ≤ 4.2 byte budget** |

### TBop-S (Second-Moment Enhanced Variant)

| Component | Bytes/Param | Format | Purpose |
|---|---|---|---|
| Ternary weight $w$ | ~0.2 | 2-bit packed | The model weight ∈ {-1, 0, +1} |
| Gradient EMA $m$ | 2.0 | BF16 | First moment — persistent gradient direction |
| Second moment $v$ | 2.0 | FP16 | Gradient variance — for normalization across layers |
| Gradient $g$ | 0 (transient) | BF16 | Not stored |
| **Total** | **4.2** | | **At the ≤ 4.2 byte budget limit** |

### Comparison with Baselines

| Method | Bytes/Param | Reduction vs. STE+Adam |
|---|---|---|
| STE + Adam (standard) | 16.0 | — |
| STE + 8-bit Adam | ~6.0 | 63% |
| DQT + 8-bit Adam | ~4.0 | 75% |
| **TBop-S** | **4.2** | **74%** |
| **TBop** | **2.2** | **86%** |
| TBop (INT8 EMA) | 1.2 | 93% |

## Key Takeaways

- **TBop is the first proposed optimizer designed specifically for ternary {-1, 0, +1} networks without latent weights.** No existing work addresses this gap. GXNOR-Net's DST framework is the closest precursor but uses a different mechanism (probabilistic projection rather than EMA-based threshold flipping) and was never scaled beyond small vision models.

- **At 2.2 bytes/param, TBop achieves an 86% memory reduction over standard STE+Adam training (16 bytes/param).** The enhanced TBop-S variant at 4.2 bytes/param adds robust cross-layer normalization while still fitting within the 4-byte extra budget.

- **The FSM with hysteresis is the critical design innovation.** Ordered transitions (must pass through 0) prevent destructive sign-flip oscillation. Asymmetric thresholds ($\tau_{\text{deact}} > \tau_{\text{act}}$) create inertia that stabilizes active weights. This directly addresses the dominant failure mode in quantized training: weight oscillation at quantization boundaries.

- **Three identified failure modes and their mitigations:**
  1. *Silent weights* (>50% of weights never transition) → threshold decay + stochastic perturbation + layer-wise normalization
  2. *Boundary instability* (chaotic flipping near threshold) → probabilistic transitions (TBop-P) + SGDAT-adaptive thresholds
  3. *Cross-layer gradient scale mismatch* → second-moment normalization (TBop-S)

- **The convergence argument is plausible but not formally proven.** The Lyapunov argument holds under local smoothness assumptions. Empirical validation at LLM scale is essential.

- **TBop is compatible with all existing efficient training techniques:** mixed-precision activations, gradient checkpointing, data parallelism, pipeline parallelism, etc. The optimizer change is orthogonal to these.

- **The most practical deployment path is a hybrid:** use STE+Adam for warmup (5–10% of training) to find a good basin, then switch to TBop for the remaining 90–95%. This amortizes the high-memory warmup cost and de-risks the transition.

## Limitations & Open Questions

### Fundamental Uncertainties

1. **Quality gap at LLM scale is unknown.** Bop's binary ImageNet results showed a ~2-3% accuracy gap. The ternary case is harder (3 states vs. 2, richer dynamics). DQT showed quality degradation for ternary at 130M params. The gap at 7B+ is unpredictable — it could narrow (more parameters = more robust averaging, consistent with scaling law arguments) or widen (deeper networks = more error accumulation across layers).

2. **No convergence proof for non-convex deep networks.** The Lyapunov argument assumes local smoothness and correct gradient direction after transitions. In highly non-convex loss landscapes with saddle points, plateaus, and sharp minima, these assumptions may break. All existing convergence results for discrete training (DST, ECO) share this limitation.

3. **Threshold sensitivity is the primary practical risk.** Basic TBop requires manual threshold tuning per architecture/dataset. TBop-S mitigates this but adds 2 bytes/param. The optimal $\tau_{\text{act}}$-to-$\tau_{\text{deact}}$ ratio is unknown and may depend on the training phase, model size, and data distribution.

### Design Questions Requiring Empirical Investigation

4. **EMA reset vs. partial reset on transition.** Full reset ($m_i \leftarrow 0$) discards all gradient history, which may be wasteful when the gradient direction hasn't changed. Partial reset ($m_i \leftarrow \delta m_i$, $\delta \in (0, 1)$) preserves some momentum. The optimal strategy is unknown.

5. **INT8 EMA viability.** Reducing the EMA to INT8 (1 byte/param, 1.2 bytes/param total) would be extremely memory-efficient but may be too coarse. With only 256 levels, the EMA's ability to track gradient trends is limited. Block-wise dynamic INT8 quantization (à la bitsandbytes) could help but adds complexity.

6. **Sparsity equilibrium.** TBop's sparsity dynamics (fraction of zero weights over training) are entirely determined by $\tau_{\text{act}} / \tau_{\text{deact}}$. The equilibrium may not match the ~42% sparsity observed in standard BitNet training, which could affect model quality. A sparsity-aware threshold adaptation mechanism may be needed.

7. **Interaction with batch normalization / RMSNorm.** Bop's success relied heavily on BatchNorm to maintain activation variance. Modern LLMs use RMSNorm or LayerNorm instead. Whether TBop's ternary weight updates interact well with these normalization schemes is untested.

8. **Gradient through quantization.** TBop still uses STE for gradient estimation (pass gradient through quantization as identity). A hybrid with QuEST's trust gradient estimator (which zeros out unreliable gradient components) could improve gradient quality at no memory cost. This combination is unexplored.

### Extensions Worth Investigating

9. **Dual-EMA TBop.** Inspired by AdEMAMix: maintain a fast EMA (high $\gamma$, responsive to recent gradients) and a slow EMA (low $\gamma$, captures long-term trends). Use the slow EMA for transition decisions and the fast EMA for transition direction. Cost: 4 bytes/param (two BF16 EMAs), fits budget.

10. **Error-feedback TBop.** After each transition, compute the "error" between the desired continuous update and the actual discrete transition. Inject this error into the EMA (ECO-style). This could improve convergence by preserving sub-threshold gradient information that would otherwise be lost.

11. **TBop with sparsity controller.** Explicitly control the zero-weight fraction during training using a differentiable sparsity regularizer (inspired by Sparsity-Control TWN, Deng & Zhang 2021). This would allow targeting specific sparsity ratios without manual threshold balancing.

12. **Progressive freezing.** Inspired by StoMPP (arXiv 2601.22660): progressively freeze layers from bottom to top during training, reducing the number of active EMA states over time. This could further reduce average memory cost and improve training stability for deep models.
