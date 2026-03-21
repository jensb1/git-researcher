---
idea_id: 006
status: complete
relevance_score: 7
confidence_score: 8
completeness_score: 8
---

# Research Results: Idea 006 — Hybrid Phase Training: STE Warm-Start + Low-Memory Finish

## Summary

This investigation designs a two-phase training method for ternary {-1, 0, 1} networks: Phase 1 uses standard STE+Adam (16 bytes/param) for a short initial period to establish stable weight patterns, then Phase 2 transitions to a low-memory optimizer (2-4 bytes/param) for the remaining majority of training. The key contribution is a rigorous framework for **when** to transition (data-driven flip-rate criterion), **how** to transition (warm initialization of Phase 2 from Adam states), and **which Phase 2 method to pair with** (ECO-Ternary or TBop). Literature strongly supports the approach: the Continual QAT Pre-Training paper (Nielsen et al., 2025) empirically demonstrates that an optimal transition point exists for BitNet 16-to-1.58-bit training, 1-bit Adam proves the "warmup-then-compress" paradigm works at LLM scale, and the Transition Rate Scheduling paper (Lee et al., 2024) provides the theory for controlling weight flip dynamics during quantized training.

## Proposed Method

### 1. Core Architecture

The method partitions training into two phases with a carefully designed transition:

```
Phase 1: STE + Adam (standard BitNet training)
  Memory: 16 bytes/param
  Duration: F fraction of total steps (target F = 0.05-0.20)
  Purpose: Navigate to a good basin, establish weight sign patterns and sparsity structure

Transition: Smooth state transfer (not abrupt drop)
  Transfer Adam momentum -> Phase 2 optimizer state
  Compute quantization error budget for ECO-style initialization
  Apply transition smoothing over Delta_T steps

Phase 2: Low-memory optimizer (TBop, ECO-Ternary, or DQT+SGDM)
  Memory: 2-4 bytes/param
  Duration: (1-F) fraction of total steps
  Purpose: Continue training with per-parameter memory within budget
```

### 2. Mathematical Formulation

#### Phase 1: Standard STE + Adam

Standard BitNet b1.58 training as described in the original paper:

```
w_latent^(t+1) = Adam_update(w_latent^(t), g^(t))
w_ternary^(t) = RoundClip(w_latent^(t) / (|w_latent^(t)|_mean + epsilon), -1, 1)
```

where g^(t) = nabla_w L(w_ternary^(t)) via STE (treating quantization as identity in the backward pass).

#### Transition Criterion: Adaptive Flip-Rate Monitor

Define the **transition ratio** (TR), inspired by Lee et al. (2024):

```
TR(t) = (1/N) * sum_{i=1}^{N} 1[w_ternary_i^(t) != w_ternary_i^(t-1)]
```

where N is the total number of parameters. This measures the fraction of ternary weights that change between consecutive steps.

**Switch condition:** Transition when BOTH of the following are met:
1. `TR(t) < tau_flip` (flip rate below threshold, e.g., tau_flip = 0.005)
2. `t > t_min` (minimum steps completed, e.g., t_min = 0.05 * T_total)

**Exponentially smoothed TR** (to avoid premature switching on transient dips):

```
TR_ema(t) = beta_tr * TR_ema(t-1) + (1 - beta_tr) * TR(t),  beta_tr = 0.99
```

Switch when TR_ema(t) < tau_flip AND t > t_min.

**Theoretical justification:** The TR directly measures how "settled" the ternary weight pattern is. When TR is high, many weights are actively changing discrete states — the network is making structural decisions about which weights should be {-1, 0, +1}. When TR drops, the structure has largely been determined and remaining optimization is refinement within the chosen configuration. This aligns with the information-theoretic argument: early training gradients carry high mutual information I(g_t; w*) about the optimal configuration, while late gradients are dominated by noise.

#### Transition: Warm Initialization of Phase 2

The critical innovation is **not** discarding Phase 1 optimizer states, but **transferring** them to initialize Phase 2. This prevents the loss spike that would result from cold-starting a new optimizer.

**For ECO-Ternary (recommended Phase 2):**

```
# Extract final ternary weights
w_ternary = quantize_ternary(w_latent^(T_switch))

# Initialize momentum from Adam's first moment
m_hat_0 = m_adam^(T_switch)  (or compress to FP16: m_hat_0 = BF16(m_adam))

# Compute initial quantization error
e_0 = w_latent^(T_switch) - w_ternary.float() * alpha_layer

# Seed error into momentum (ECO-style)
m_hat_0 = m_hat_0 + alpha_eco * e_0
where alpha_eco = (1/eta_phase2) * (1 - 1/beta_phase2)

# Discard: w_latent, v_adam, master_copy  (frees 10-14 bytes/param)
```

**For TBop (Ternary Bop):**

```
# Extract final ternary weights
w_ternary = quantize_ternary(w_latent^(T_switch))

# Initialize EMA from Adam's momentum
ema_0 = m_adam^(T_switch)

# Set initial thresholds from Adam's variance
tau_up_i = c * sqrt(v_adam_i^(T_switch))   per-parameter adaptive threshold
tau_down_i = c' * sqrt(v_adam_i^(T_switch))

# Discard: w_latent, v_adam, master_copy
```

#### Transition Smoothing

To avoid discontinuity, apply a linear interpolation of learning rates over a smoothing window Delta_T (e.g., Delta_T = 1000 steps):

```
For t in [T_switch, T_switch + Delta_T]:
  alpha_blend = (t - T_switch) / Delta_T
  eta(t) = (1 - alpha_blend) * eta_phase1(T_switch) + alpha_blend * eta_phase2_initial
```

This smoothing is inspired by the WSD (Warmup-Stable-Decay) schedule used in Falcon-Edge training. The learning rate ramp for Phase 2 prevents the sudden change in optimizer dynamics from causing a loss spike.

#### Phase 2: ECO-Ternary Training (Primary Recommendation)

Using ECO's error-compensated momentum with ternary quantization:

```
# Phase 2 update (ECO-style, per step t > T_switch):

1. Compute gradient: g_t = nabla_w L(w_ternary^(t))  via STE
2. Update momentum: m_tilde^(t+1) = beta * m_hat^(t) + (1-beta) * g_t
3. Compute candidate: w_tilde^(t+1) = w_ternary^(t).float() * alpha - eta * m_tilde^(t+1)
4. Re-quantize: w_ternary^(t+1) = ternary_quantize(w_tilde^(t+1))
5. Compute error: e^(t+1) = w_tilde^(t+1) - w_ternary^(t+1).float() * alpha_new
6. Inject error: m_hat^(t+1) = m_tilde^(t+1) + alpha_eco * e^(t+1)
```

where alpha_eco = (1/eta)(1 - 1/beta) as derived in ECO (Nikdan et al., 2026).

**Alternatively, Phase 2 can use TBop:**

```
# Phase 2 update (TBop-style, per step t > T_switch):

1. Compute gradient: g_t = nabla_w L(w_ternary^(t))  via STE
2. Update EMA: ema^(t+1) = gamma * ema^(t) + (1-gamma) * g_t
3. For each weight w_i:
   if w_i == 0:
     if ema_i > tau_up:   w_i <- +1,  reset ema_i = 0
     if ema_i < -tau_up:  w_i <- -1,  reset ema_i = 0
   if w_i == +1:
     if ema_i > tau_down:  w_i <- 0,  reset ema_i = 0  (gradient says decrease)
   if w_i == -1:
     if ema_i < -tau_down: w_i <- 0,  reset ema_i = 0  (gradient says increase)
```

### 3. Pseudocode

```python
def hybrid_phase_train(model, data, T_total, F_min=0.05, F_max=0.20,
                       tau_flip=0.005, phase2_type='eco_ternary'):
    """
    Hybrid Phase Training for Ternary Networks

    Phase 1: STE + Adam (16 bytes/param)
    Phase 2: ECO-Ternary or TBop (2-4 bytes/param)

    Args:
        T_total: total training steps
        F_min: minimum fraction for Phase 1
        F_max: maximum fraction for Phase 1
        tau_flip: flip rate threshold for switching
        phase2_type: 'eco_ternary' or 'tbop'
    """

    # === PHASE 1: Standard STE + Adam ===
    optimizer = AdamW(model.latent_weights, lr=lr_phase1, betas=(0.9, 0.999))
    T_switch = int(F_max * T_total)  # default max
    TR_ema = 1.0  # start high
    w_ternary_prev = None

    for t in range(int(F_max * T_total)):
        # Standard BitNet forward
        w_ternary = quantize_ternary(model.latent_weights)
        loss = forward(model, data[t], w_ternary)
        grads = backward(loss)  # STE through quantization
        optimizer.step(grads)

        # Monitor transition ratio
        if w_ternary_prev is not None:
            TR = (w_ternary != w_ternary_prev).float().mean().item()
            TR_ema = 0.99 * TR_ema + 0.01 * TR

            # Check switch condition
            if TR_ema < tau_flip and t >= int(F_min * T_total):
                T_switch = t
                break

        w_ternary_prev = w_ternary.clone()

    # === TRANSITION ===
    w_ternary = quantize_ternary(model.latent_weights)

    if phase2_type == 'eco_ternary':
        # ECO-Ternary: transfer Adam momentum, inject quantization error
        m_phase2 = bf16(optimizer.state['m'])  # compress to FP16
        e_0 = model.latent_weights - w_ternary.float() * layer_scale
        alpha_eco = (1.0 / lr_phase2) * (1.0 - 1.0 / 0.9)
        m_phase2 = m_phase2 + alpha_eco * e_0

    elif phase2_type == 'tbop':
        # TBop: transfer Adam momentum as EMA, variance as thresholds
        ema_phase2 = bf16(optimizer.state['m'])
        tau_up = C_UP * sqrt(optimizer.state['v'])
        tau_down = C_DOWN * sqrt(optimizer.state['v'])

    # Free Phase 1 memory
    del model.latent_weights, optimizer  # frees 10-14 bytes/param

    # Learning rate transition smoothing
    Delta_T = 1000

    # === PHASE 2: Low-memory training ===
    for t in range(T_switch, T_total):
        # Smooth LR transition
        if t < T_switch + Delta_T:
            alpha_blend = (t - T_switch) / Delta_T
            eta_t = (1 - alpha_blend) * lr_phase1_final + alpha_blend * lr_phase2
        else:
            eta_t = lr_schedule_phase2(t)

        # Forward and backward with ternary weights
        loss = forward(model, data[t], w_ternary)
        grads = backward(loss)  # STE through quantization

        if phase2_type == 'eco_ternary':
            w_ternary, m_phase2 = eco_ternary_step(
                w_ternary, m_phase2, grads, eta_t, beta=0.9
            )
        elif phase2_type == 'tbop':
            w_ternary, ema_phase2 = tbop_step(
                w_ternary, ema_phase2, grads, tau_up, tau_down, gamma=0.999
            )

    return w_ternary


def eco_ternary_step(w_ternary, m_hat, grads, eta, beta=0.9):
    """ECO-style error-compensated step for ternary weights."""
    alpha_layer = w_ternary.abs().float().mean()  # absmean scaling

    # Momentum update
    m_tilde = beta * m_hat + (1 - beta) * grads

    # Candidate weight (in continuous space)
    w_candidate = w_ternary.float() * alpha_layer - eta * m_tilde

    # Re-quantize to ternary
    alpha_new = w_candidate.abs().mean()
    w_ternary_new = ternary_quantize(w_candidate, alpha_new)

    # Compute quantization error
    e = w_candidate - w_ternary_new.float() * alpha_new

    # Inject error into momentum (ECO formula)
    alpha_eco = (1.0 / eta) * (1.0 - 1.0 / beta)
    m_hat_new = m_tilde + alpha_eco * e

    return w_ternary_new, bf16(m_hat_new)
```

### 4. Memory Budget Analysis

#### Phase 1 (duration: F fraction)

| Component | Bytes/Param | Purpose |
|---|---|---|
| BF16 latent weights | 2 | Gradient accumulator |
| BF16 gradients | 2 | Backprop (transient) |
| FP32 master copy | 4 | Adam precision |
| FP32 first moment (m) | 4 | Adam momentum |
| FP32 second moment (v) | 4 | Adam variance |
| **Total Phase 1** | **16** | Standard STE+Adam |

#### Phase 2 with ECO-Ternary (duration: 1-F fraction)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight | ~0.2 | The model weight (2 bits, packed) |
| BF16 gradients | 2 | Backprop (transient) |
| BF16 momentum (m_hat) | 2 | Momentum + error compensation |
| **Total Phase 2** | **~4.2** | Fits the 4-byte extra budget |

#### Phase 2 with TBop

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight | ~0.2 | The model weight |
| BF16 EMA | 2 | Gradient accumulator |
| **Total Phase 2** | **~2.2** | Well within budget |

#### Amortized Memory

```
M_amortized = F * 16 + (1-F) * M_phase2

For F=0.10, ECO-Ternary (4.2 B/param):
  M_amortized = 0.1 * 16 + 0.9 * 4.2 = 1.6 + 3.78 = 5.38 bytes/param

For F=0.05, ECO-Ternary (4.2 B/param):
  M_amortized = 0.05 * 16 + 0.95 * 4.2 = 0.8 + 3.99 = 4.79 bytes/param

For F=0.10, TBop (2.2 B/param):
  M_amortized = 0.1 * 16 + 0.9 * 2.2 = 1.6 + 1.98 = 3.58 bytes/param
```

**Critical caveat:** Peak memory during Phase 1 is still 16 bytes/param. The 4-byte budget is satisfied during Phase 2 (the vast majority of training) but NOT during Phase 1. This is the fundamental limitation of the hybrid approach.

### 5. Convergence Argument

The convergence argument rests on three pillars:

**Pillar 1: Phase 1 convergence is established.** STE+Adam for BitNet is proven to work at scale (BitNet b1.58-2B-4T, Falcon-Edge 3B). The convergence theory for STE (Yin et al., 2019; beyond-discreteness, 2025) shows that STE produces a coarse gradient that correlates positively with the true gradient, and gradient descent converges to a critical point.

**Pillar 2: Phase 2 inherits a good initialization.** The Continual QAT Pre-Training paper (Nielsen et al., 2025, ACL Findings) empirically demonstrates that models initialized from a short period of 16-bit training achieve significantly better final quality than models trained from scratch at 1.58-bit. Their finding: even 2K 16-bit steps (a tiny fraction of total training) provides substantial benefit. This validates our Phase 1.

**Pillar 3: Phase 2 methods have convergence guarantees from a good initialization.**

For ECO-Ternary: ECO (Nikdan et al., 2026) proves convergence to a bounded neighborhood:
```
min_t E[||grad f(w_hat_t)||^2] <= 4(f(w_0) - f*) / (eta*T) + sigma^2_quant
```

When initialized from a Phase 1 solution, f(w_0) is already close to f*, so the first term is small. The noise floor sigma^2_quant = O(L^2 * sigma^2 / (1 - beta^2)) is the intrinsic cost of ternary quantization. Importantly, ECO proves that WITHOUT error compensation, the noise floor grows as 1/eta -> infinity as learning rate anneals, making convergence impossible. Error compensation is essential.

For TBop: While Bop lacks formal convergence theory, the warm initialization from Adam's momentum provides a strong starting point. The EMA inherits Adam's gradient history, so the flip decisions are informed by the full Phase 1 gradient signal.

**Pillar 4: Transition smoothing prevents loss spikes.**

The 1-bit Adam paper (Tang et al., 2021) demonstrates that a warmup-then-compress strategy works at LLM scale when the transition is handled correctly. Their key insight — that Adam's variance stabilizes after an initial phase and can be frozen — directly supports our approach. The loss spike risk at transition is mitigated by:
- Warm initialization of Phase 2 state from Adam states
- Learning rate smoothing over Delta_T steps
- The flip-rate criterion ensuring weights are already stable

### 6. The Flip-Rate Criterion: When to Switch

The flip-rate criterion is the most important practical contribution. It provides a **data-driven, model-adaptive** transition point rather than a fixed schedule.

**Why fixed F is suboptimal:**
- Different model sizes stabilize at different rates (larger models may stabilize earlier in terms of fraction of training due to implicit regularization from width)
- Different datasets induce different convergence dynamics
- The optimal F depends on learning rate schedule, batch size, and architecture

**Why flip-rate works:**
The transition ratio TR(t) directly measures what we care about: whether the ternary weight pattern has stabilized. When TR drops below a threshold, the discrete structure is no longer changing rapidly, and a simpler optimizer can maintain/refine it.

**Expected behavior of TR(t):**
Based on observations from Bop (Helwegen et al., 2019), binary/ternary weight patterns stabilize after 10-30% of training. For LLMs:
- During warmup (first 1-2% of steps): TR is very high (> 0.1), many weights changing
- During rapid learning (2-15% of steps): TR gradually decreases as patterns form
- Stabilization (15-30% of steps): TR drops below 0.01, most weights settled
- Late training (30-100% of steps): TR is very low (< 0.001), only occasional refinements

The TR schedule from Lee et al. (2024) supports this view: they show that controlling weight transition rates explicitly (rather than learning rates) is the correct abstraction for quantized training, and that transition rates should follow a decaying schedule similar to learning rate decay.

**Recommended threshold:** tau_flip = 0.005 (0.5% of weights flip per step). This balances switching early enough for memory savings vs. late enough for quality.

## Literature Support

### Directly Supporting Papers

1. **Continual QAT Pre-Training (Nielsen, Schneider-Kamp, Galke, 2025, ACL Findings, arXiv 2502.11895):** Directly investigates the question "when to transition from 16-bit to 1.58-bit training for BitNet?" Found that even a short 16-bit warmup (2K steps) significantly improves final 1.58-bit model quality. The paper concludes that a data-optimal transition point exists and that the 16-to-1.58-bit strategy is preferable over pure 1.58-bit training. **This is the strongest empirical validation of our hybrid approach.**

2. **1-bit Adam (Tang et al., 2021, arXiv 2102.02888):** Proves the "warmup-then-compress" paradigm works at LLM scale (BERT-Large on 256 GPUs). Adam's variance stabilizes after 15-25% of training, at which point it can be frozen and the remaining optimization becomes linear in gradients, enabling aggressive compression. The two-phase design and the insight about non-linear optimizer operations being incompatible with error compensation are directly applicable.

3. **1-bit LAMB (Li et al., 2021, arXiv 2104.06069):** Confirms the generality of the warmup-then-compress pattern across different optimizers (LAMB for large-batch training). Same convergence speed as uncompressed LAMB with 4.6x communication reduction.

4. **Scheduling Weight Transitions for QAT (Lee et al., 2024, arXiv 2404.19248):** Introduces the transition ratio as a fundamental metric for quantized training. Shows that controlling weight flip rates explicitly (rather than learning rates) is a superior optimization strategy for QAT. Provides theoretical framework for our flip-rate switching criterion.

5. **ECO (Nikdan et al., 2026, arXiv 2601.22101):** Provides the Phase 2 optimizer with formal convergence guarantees. Proves that error injection into momentum eliminates master weights without divergence. The convergence-to-neighborhood result sets the quality bound for Phase 2.

6. **Falcon-Edge (TII, 2025):** Demonstrates a practical variant of hybrid training: simultaneously produces BF16 and BitNet models from a single training run with ~20% overhead. Uses WSD learning rate scheduling, which informs our transition smoothing design.

### Supporting Theory

7. **Why Warmup the Learning Rate? (NeurIPS 2024, arXiv 2406.09405):** Shows that warmup forces the network into well-conditioned regions of the loss landscape, enabling tolerance of larger learning rates subsequently. Directly supports the rationale: Phase 1 conditions the loss landscape for Phase 2's simpler optimizer.

8. **Overcoming Oscillations in QAT (Nagel et al., ICML 2022, arXiv 2203.11086):** Identifies weight oscillations as a key failure mode in low-bit QAT. Proposes oscillation dampening and weight freezing. The Phase 1 warm start naturally reduces oscillation risk in Phase 2 by starting from stable ternary patterns.

9. **Momentum Provably Improves Error Feedback (Fatkhullin et al., NeurIPS 2023, arXiv 2305.15155):** Proves that momentum acts as a stabilizing force on error feedback mechanisms, preventing exponential divergence of compression errors. Directly validates using momentum-based Phase 2 optimizers.

10. **SPAM: Spike-Aware Adam (ICLR 2025):** Identifies loss spikes in Adam as arising from preconditioner dynamics. Informs our transition design: we must smoothly transition the effective preconditioner, not abruptly switch optimizers.

### Negative/Cautionary Evidence

11. **DQT (Zhao et al., 2024, arXiv 2412.04787):** Shows that direct ternary training without latent weights has a quality gap at 130M parameters. This motivates the hybrid approach: pure low-memory training from scratch is risky, but starting from a warm initialization may close the gap.

12. **STE Convergence Theory (High-Dim Dynamics, 2025, arXiv 2510.10693):** In low-bit regimes, STE dynamics become non-monotonic, leading to slower convergence. This suggests that the STE-based backward pass in Phase 2 may benefit from the more stable loss landscape inherited from Phase 1.

## Generalizability Analysis

### Model Size Scaling (100M to 100B+)

**Phase 1:** Standard STE+Adam is proven at scale from 100M (TernaryLM) to 3B (Falcon-Edge, BitNet b1.58-2B-4T). No known fundamental barrier at larger scales.

**Transition point:** The flip-rate criterion is a per-parameter metric that scales naturally. Larger models may stabilize faster in terms of training fraction because:
- More parameters provide stronger implicit regularization (the ternary pattern is overdetermined by the data)
- BitNet b1.58-2B-4T shows ~42% weight sparsity, suggesting the ternary structure is well-determined by the data

**Phase 2:** Both ECO-Ternary and TBop are per-parameter update rules with no inter-parameter dependencies. They scale linearly with model size. ECO's convergence proof makes no model-size assumptions.

**Prediction:** The optimal switch fraction F should *decrease* with model size (larger models stabilize faster), making the hybrid approach *more* memory-efficient at scale. At 100B+, F=0.05 may suffice, giving 95% of training at 2-4 bytes/param.

### Architecture Generality

The method is architecture-agnostic — it applies to any network with BitLinear layers. The transition criterion (flip rate) is computed per-parameter and works regardless of layer type (attention, FFN, embedding).

**Layer-specific considerations:** Middle transformer layers show highest compatibility with ternary quantization (TernaryLM finding). A refinement could use per-layer flip rates to transition different layers at different times, but this adds complexity and is not necessary for the base method.

### Dataset/Domain Independence

No dataset-specific assumptions. The flip-rate criterion is data-adaptive: it measures weight stability regardless of the data distribution. The same method works for language modeling (WikiText, C4), instruction tuning, and other domains.

### Hardware Practicality

Phase 1 is standard STE+Adam — existing code, existing GPU/TPU kernels. Phase 2 (ECO-Ternary or TBop) requires only momentum updates, ternary quantization, and error injection — all elementwise operations implementable with standard GPU primitives. No exotic hardware required.

The transition requires one synchronization point to reallocate memory (free Adam states, allocate Phase 2 state). This is a one-time cost.

### Limitation: Peak Memory

The fundamental limitation is that **peak** memory during Phase 1 is 16 bytes/param. For settings where peak memory is the hard constraint (e.g., training on edge devices with fixed RAM), this approach does not qualify. It is designed for data-center training where:
- Phase 1 can use gradient checkpointing to reduce peak memory
- The GPU/TPU has enough memory for 16 bytes/param initially
- The memory freed after Phase 1 can be repurposed (larger batch size, more data parallelism)

## Matching Metrics

- Relevance to original question: **7/10** — Addresses the memory problem pragmatically but does NOT meet the strict "4 bytes at all times" constraint. Meets it for 80-95% of training.
- Confidence in findings: **8/10** — Strong literature support (Continual QAT, 1-bit Adam, Falcon-Edge). The warmup-then-compress paradigm is well-validated. Specific ternary-to-ternary transition is novel but built on proven components.
- Completeness of investigation: **8/10** — Comprehensive design with transition criterion, warm initialization, convergence argument, and memory budget. The exact optimal F and tau_flip values require empirical validation at LLM scale.

## Memory Budget Breakdown

### Phase 2 with ECO-Ternary (steady-state)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight | ~0.2 | Model weight (2 bits packed) |
| BF16 momentum (m_hat) | 2.0 | Optimizer state + error compensation |
| BF16 gradients | 2.0 | Transient during backprop |
| Per-channel scale | ~0.001 | BF16 absmean scale (amortized) |
| **Total** | **~4.2** | **4.0 bytes extra beyond ternary weight** |

### Phase 2 with TBop

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight | ~0.2 | Model weight (2 bits packed) |
| BF16 EMA | 2.0 | Gradient accumulator for flip decisions |
| Thresholds | ~0.001 | Per-channel, amortized |
| **Total** | **~2.2** | **2.0 bytes extra beyond ternary weight** |

### Phase 1 (temporary peak)

| Component | Bytes/Param | Purpose |
|---|---|---|
| BF16 latent weights | 2 | STE gradient accumulator |
| BF16 gradients | 2 | Backprop |
| FP32 master copy | 4 | Adam precision |
| FP32 first moment | 4 | Adam momentum |
| FP32 second moment | 4 | Adam variance |
| **Total** | **16** | Standard STE+Adam |

## Key Takeaways

- **The "warmup-then-compress" paradigm is well-validated at LLM scale.** 1-bit Adam, 1-bit LAMB, Continual QAT Pre-Training, and Falcon-Edge all demonstrate that a short high-precision phase followed by aggressive compression/quantization works in practice. This is not speculative — it is an established pattern.

- **The optimal Phase 1 duration is short (5-20% of training).** The Continual QAT paper found that even 2K steps of 16-bit training provides substantial benefit. Our flip-rate criterion provides a data-driven way to determine the exact switch point.

- **Transition design is critical — not the method, but the engineering.** Warm initialization of Phase 2 from Adam states (transferring momentum, seeding with quantization error) prevents loss spikes. The 1-bit Adam and SPAM papers show that abrupt optimizer transitions cause instability; smooth handoff is essential.

- **ECO-Ternary is the recommended Phase 2 optimizer.** It has formal convergence guarantees, naturally uses the momentum buffer as error compensation (no extra memory), and is compatible with the warm initialization from Adam's momentum. TBop is a viable alternative with even lower memory (2 bytes/param extra) but lacks convergence theory.

- **This approach is most valuable as a safe baseline and experimental framework.** Even if a pure low-memory method (Ideas 001-005) eventually works from scratch, hybrid training provides: (a) a guaranteed-quality fallback, (b) a controlled comparison framework to evaluate which Phase 2 method works best, and (c) practical value for real training runs where some peak memory overhead is acceptable.

- **The flip-rate switching criterion is a novel and generalizable contribution.** It applies beyond this specific method — any system that transitions between training regimes for quantized networks can use it. The Transition Rate Scheduling paper provides the theoretical foundation.

## Limitations & Open Questions

1. **Peak memory is still 16 bytes/param during Phase 1.** This is the fundamental limitation. For strict per-parameter memory budgets, this approach does not qualify. Possible mitigation: use gradient checkpointing or a reduced model in Phase 1 (e.g., train a smaller model, then expand; or use layer-wise training).

2. **Optimal F is unknown at LLM scale (>7B).** The Continual QAT paper tested at small scale only. Whether the optimal F is model-size-dependent (and in which direction) is an open empirical question.

3. **Loss spike at transition is a risk.** Despite warm initialization and smoothing, the abrupt change in optimizer dynamics could cause instability, especially if the ternary weight pattern is not fully stabilized. The severity scales with model depth (deeper models are more sensitive to perturbation cascades).

4. **ECO's noise floor for ternary is untested.** ECO's convergence guarantee includes a noise floor sigma^2_quant that depends on quantization coarseness. For ternary (3 levels), this floor is much larger than for FP8 (256 levels) or INT4 (16 levels). Whether the floor is acceptable for LLM-quality training is an open question. The 1/(1-beta^2) amplification factor (5.26x for beta=0.9) applied to ternary's large sigma^2 could make the noise floor too high.

5. **Interaction between Phase 2 learning rate schedule and quantization dynamics.** The effective learning rate in discrete optimization is fundamentally different from continuous optimization — a weight can only change by discrete amounts. The relationship between the continuous learning rate eta and the actual rate of weight flips is mediated by the quantization grid and error compensation, and this relationship is poorly understood theoretically.

6. **No existing codebase implements this.** While each component (STE+Adam, ECO, TBop) has implementations, the integrated system with transition monitoring, warm handoff, and Phase 2 optimizer has not been built. Engineering effort is moderate but non-trivial.

7. **Gradient memory (2 bytes/param) is counted in the budget.** If gradients are computed and consumed layer-by-layer during backprop (as in standard practice), they are transient and don't need to be stored for all parameters simultaneously. This would bring ECO-Ternary Phase 2 from 4.2 to 2.2 bytes/param. The budget interpretation matters.

8. **This approach does not address the *training from scratch on memory-constrained devices* use case.** For that use case, a pure low-memory method (Ideas 001-005, 007) is necessary. This approach is best suited for data-center training where the goal is to maximize model size for a given GPU memory budget — Phase 2 allows scaling the model ~4x larger than Phase 1 could support, with a brief initial training period at reduced model size or with gradient checkpointing.
