---
idea_id: 005
status: complete
relevance_score: 7
confidence_score: 6
completeness_score: 8
---

# Research Results: Idea 005 — Gradient Sign Accumulator with Adaptive Thresholds (GSA-AT)

## Summary

GSA-AT is an ultra-memory-efficient optimizer for ternary {-1, 0, 1} weight networks that uses only an INT8 counter per parameter (1 byte) to accumulate gradient sign information and trigger discrete weight transitions when sufficient evidence accumulates. By discarding gradient magnitudes and relying solely on directional consensus — analogous to a per-parameter "ballot box" — it achieves the lowest possible optimizer memory footprint (1.25–2.25 bytes/param total) among all proposed methods. This document provides a rigorous mathematical formulation, convergence analysis grounded in signSGD theory, and a detailed assessment of the information-magnitude tradeoff.

## Proposed Method

### 1. Mathematical Formulation

#### 1.1 State Variables

For each parameter $i$:
- $w_i \in \{-1, 0, +1\}$: ternary weight (2 bits, packed ~0.25 bytes)
- $c_i \in [-128, +127]$: INT8 counter (1 byte)

Optional (Approach A — 2-byte variant):
- $f_i \in [0, 255]$: INT8 flip counter tracking transition history (1 byte)

#### 1.2 Core Update Rule

At each training step $t$, given stochastic gradient $g_t^{(i)}$:

**Step 1 — Counter update with exponential decay:**

$$c_i \leftarrow \text{clip}\!\Big(\!\left\lfloor \gamma \cdot c_i + s_l \cdot \text{sign}(g_t^{(i)}) \right\rceil, -128, 127\Big)$$

where:
- $\gamma \in (0, 1)$ is the decay factor (e.g., 0.97–0.99), analogous to momentum coefficient
- $s_l = \min\!\left(\left\lfloor \frac{\text{mean}(|g_t^{(\text{layer } l)}|)}{\text{median}_{\text{layers}}(\text{mean}(|g_t|))} \cdot s_{\text{base}} \right\rceil, 8\right)$ is a per-layer integer scaling factor
- $s_{\text{base}} \in \{1, 2, 4\}$ is a global base scale
- $\lfloor \cdot \rceil$ denotes rounding to nearest integer

**Step 2 — State transition check:**

Define activation threshold $\tau_a$ and deactivation threshold $\tau_d$ (with $\tau_a \geq \tau_d$ typically):

| Current State $w_i$ | Condition | New State | Counter Reset |
|---|---|---|---|
| $0$ | $c_i < -\tau_a$ | $+1$ | $c_i \leftarrow 0$ |
| $0$ | $c_i > +\tau_a$ | $-1$ | $c_i \leftarrow 0$ |
| $+1$ | $c_i > +\tau_d$ | $0$ | $c_i \leftarrow 0$ |
| $-1$ | $c_i < -\tau_d$ | $0$ | $c_i \leftarrow 0$ |

**Interpretation of signs:** The gradient $g_t^{(i)} = \partial L / \partial w_i$ indicates the direction that *increases* loss. Therefore:
- Persistent positive gradient → weight should *decrease* → counter accumulates positive → triggers transition toward lower state ($+1 \to 0$ or $0 \to -1$)
- Persistent negative gradient → weight should *increase* → counter accumulates negative → triggers transition toward higher state ($-1 \to 0$ or $0 \to +1$)

#### 1.3 Adaptive Threshold Mechanism (Approach A — 2 bytes)

The flip counter $f_i$ tracks total transitions. Thresholds increase with flip count to suppress oscillation:

$$\tau_a(f_i) = \tau_{a,\text{base}} + \alpha \cdot \min(f_i, f_{\max})$$
$$\tau_d(f_i) = \tau_{d,\text{base}} + \alpha \cdot \min(f_i, f_{\max})$$

where $\alpha$ is the threshold growth rate (e.g., $\alpha = 1$) and $f_{\max}$ caps the growth (e.g., $f_{\max} = 50$).

After each transition: $f_i \leftarrow \min(f_i + 1, 255)$.

This is directly inspired by SGDAT (Gu et al., 2023), which demonstrated that adaptive thresholds based on flip history significantly improve BNN training by suppressing oscillation while encouraging diverse weight participation.

#### 1.4 Adaptive Threshold Mechanism (Approach B — 1 byte, implicit)

Use the decay factor $\gamma$ to create implicit adaptivity without storing $f_i$. The counter with decay acts as an exponential moving average of gradient signs in INT8:

At steady state with consistent gradient sign $s \in \{-1, +1\}$:
$$c_\infty = \frac{s \cdot s_l}{1 - \gamma}$$

For $\gamma = 0.98, s_l = 1$: $c_\infty = \pm 50$ (well within INT8 range)
For $\gamma = 0.99, s_l = 1$: $c_\infty = \pm 100$ (near INT8 boundary)
For $\gamma = 0.95, s_l = 1$: $c_\infty = \pm 20$ (very responsive)

The effective threshold is thus $\tau / c_\infty$ fraction of the steady-state — the counter must reach $\tau / c_\infty$ of its maximum before triggering a transition. For ambiguous weights (gradient direction frequently changes), the decay prevents counter saturation, naturally requiring more evidence before flipping.

### 2. Pseudocode

```python
# GSA-AT: Gradient Sign Accumulator with Adaptive Thresholds
# Memory: 1.25 bytes/param (Approach B) or 2.25 bytes/param (Approach A)

def gsa_at_init(model):
    for layer in model.layers:
        layer.w = init_ternary(layer.shape)     # {-1, 0, +1}
        layer.counter = zeros_int8(layer.shape)  # INT8
        # Approach A only:
        # layer.flip_count = zeros_uint8(layer.shape)

def gsa_at_step(model, loss_fn, data, gamma, tau_a, tau_d, s_base):
    # Forward pass with ternary weights (no quantization needed — weights ARE ternary)
    loss = loss_fn(model, data)

    # Backward pass — compute gradients (transient, not stored)
    grads = backward(loss)

    # Compute per-layer gradient magnitude scales (one scalar per layer)
    layer_scales = []
    for layer in model.layers:
        layer_scales.append(mean(abs(grads[layer])))
    median_scale = median(layer_scales)

    for layer in model.layers:
        g = grads[layer]                    # transient gradient
        w = layer.w                         # ternary weight
        c = layer.counter                   # INT8 counter

        # Per-layer integer scale factor
        s_l = min(round(layer_scales[layer] / (median_scale + eps) * s_base), 8)
        s_l = max(s_l, 1)                  # at least 1

        # Step 1: Update counter with decay + gradient sign
        sign_g = sign(g)                    # in {-1, 0, +1}
        c_new = clip(round(gamma * c + s_l * sign_g), -128, 127)

        # Step 2: Check transitions
        # From state 0: activate
        act_pos = (w == 0) & (c_new < -tau_a)   # strong negative gradient → set w = +1
        act_neg = (w == 0) & (c_new > tau_a)    # strong positive gradient → set w = -1

        # From state +1: deactivate (gradient says decrease)
        deact_pos = (w == +1) & (c_new > tau_d)

        # From state -1: deactivate (gradient says increase)
        deact_neg = (w == -1) & (c_new < -tau_d)

        # Apply transitions
        w[act_pos] = +1
        w[act_neg] = -1
        w[deact_pos] = 0
        w[deact_neg] = 0

        # Reset counters for transitioned weights
        transitioned = act_pos | act_neg | deact_pos | deact_neg
        c_new[transitioned] = 0

        # Approach A: update flip counters
        # layer.flip_count[transitioned] = min(layer.flip_count[transitioned] + 1, 255)
        # tau_a = tau_a_base + alpha * min(layer.flip_count, f_max)
        # tau_d = tau_d_base + alpha * min(layer.flip_count, f_max)

        layer.w = w
        layer.counter = c_new

    return loss
```

### 3. Enhanced Variant: GSA-AT with Error Feedback (GSA-AT-EF)

The primary weakness of GSA-AT is discarding gradient magnitude. We can partially compensate by incorporating a lightweight error feedback mechanism inspired by ECO (Nikdan et al., 2026) and EF-SGD (Karimireddy et al., 2019).

**Key insight:** When a weight transitions (e.g., $0 \to +1$), the "ideal" continuous update may have been to some value $v \in (0.5, 1.5)$. The quantization error $e = v - 1$ is lost. We can encode this error into the counter after reset:

```python
# After transition from 0 to +1:
# The counter was at c_trigger (e.g., -55)
# The "overshoot" beyond threshold encodes magnitude information
overshoot = abs(c_trigger) - tau_a  # how far past threshold
c_new[transitioned] = clip(round(-overshoot * error_retention), -128, 127)
# where error_retention in (0, 1) controls how much overshoot is preserved
```

This preserves some gradient history across transitions, analogous to ECO injecting quantization error into the momentum buffer. The overshoot encodes how "confident" the optimizer was about the transition — larger overshoot means the gradient was highly consistent, and this information carries forward.

**Memory cost:** Zero additional bytes — reuses the same INT8 counter.

### 4. Connection to Lion Optimizer

The Lion optimizer (Chen et al., 2023, NeurIPS) — discovered via automated program search — uses only momentum and element-wise sign operations:

$$\text{update} = \text{sign}(\beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t)$$
$$m_t = \beta_2 \cdot m_{t-1} + (1 - \beta_2) \cdot g_t$$

Lion matches Adam performance for LLM training while requiring only 1 momentum buffer (no variance). GSA-AT can be viewed as an **extreme quantization of Lion to INT8**: the counter $c$ plays the role of the momentum $m$, the decay $\gamma$ corresponds to $\beta_2$, and the threshold-based transition replaces the continuous sign-based update. Lion's success at LLM scale with sign-only updates provides strong empirical evidence that gradient magnitude is less critical than gradient direction for optimization.

### 5. Convergence Analysis

#### 5.1 Grounding in signSGD Theory

**Theorem (Bernstein et al., 2018):** For L-smooth, bounded-below objective functions, signSGD achieves:

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\!\left[\|\nabla f(w_t)\|_1\right] \leq \mathcal{O}\!\left(\frac{1}{\sqrt{T}}\right)$$

under the assumption that gradient noise has bounded variance per coordinate.

**Theorem (Sun et al., 2023 — ICML):** signSGD *with momentum* (SIGNUM) converges at rate $\mathcal{O}(1/\sqrt{T})$ under weaker assumptions — no bounded gradient requirement, works with small batch sizes, and achieves improved rates under second-order smoothness.

GSA-AT's counter-with-decay is mathematically equivalent to an INT8-quantized SIGNUM momentum buffer. The transition thresholds add a hysteresis layer that prevents the oscillation problems identified by Nagel et al. (2022, ICML) for low-bit QAT.

#### 5.2 Convergence Argument for GSA-AT

**Claim:** GSA-AT converges to a neighborhood of a stationary point, with the neighborhood size depending on: (a) the threshold $\tau$, (b) the quantization noise from INT8 counter rounding, and (c) the information loss from discarding gradient magnitude.

**Argument sketch:**

1. **Counter as noisy sign-momentum estimator:** The counter $c_t = \gamma \cdot c_{t-1} + \text{sign}(g_t)$ is a discretized exponential moving average of gradient signs. By SIGNUM theory, $\text{sign}(c_t)$ correctly identifies the descent direction with probability $\geq 1/2 + \delta$ whenever the true gradient $\nabla f(w)$ is nonzero, where $\delta$ depends on the gradient signal-to-noise ratio.

2. **Threshold as evidence accumulation:** The threshold $\tau$ requires the counter to reach a value that, under consistent gradient direction, takes approximately $\tau \cdot (1 - \gamma)$ steps. This is analogous to requiring $\Theta(\tau)$ i.i.d. gradient sign observations, providing a confidence level that scales as $1 - e^{-\Omega(\tau)}$ by Hoeffding's inequality.

3. **Transition correctness probability:** When a weight transition is triggered (counter crosses threshold), the probability that it is in the correct direction is:

   $$P(\text{correct transition}) \geq 1 - \exp\!\left(-\frac{2\tau^2 \cdot p_{\text{correct}}^2}{1}\right)$$

   where $p_{\text{correct}} = P(\text{sign}(g_t) = \text{sign}(\nabla f(w)))$ is the per-step probability that the gradient sign is correct. For $\tau = 30$ and $p_{\text{correct}} = 0.6$, this gives $P(\text{correct}) \geq 1 - e^{-2 \cdot 900 \cdot 0.04} = 1 - e^{-72} \approx 1$.

4. **Convergence to neighborhood:** Since transitions are correct with high probability but are discrete (step size is exactly 1 in ternary space), the method converges to a neighborhood where no weight wants to transition — i.e., where gradient signals are ambiguous. This neighborhood has radius proportional to the ternary quantization step.

5. **INT8 quantization noise:** The counter rounding introduces noise of magnitude $\leq 0.5$ per step. Over $T$ steps with decay $\gamma$, the accumulated noise has variance $\leq \frac{0.25}{1 - \gamma^2}$, which is $\leq 12.5$ for $\gamma = 0.99$. This is small relative to typical thresholds ($\tau \geq 20$).

#### 5.3 Convergence Speed Estimate

The expected number of steps to trigger a correct transition for a "decisive" weight (gradient consistently points one way) is:

$$T_{\text{transition}} \approx \frac{\tau \cdot (1 - \gamma)}{p_{\text{correct}} - 0.5}$$

For $\tau = 30, \gamma = 0.98, p_{\text{correct}} = 0.7$:
$$T_{\text{transition}} \approx \frac{30 \cdot 0.02}{0.2} = 3 \text{ steps}$$

For a "borderline" weight ($p_{\text{correct}} = 0.55$):
$$T_{\text{transition}} \approx \frac{30 \cdot 0.02}{0.05} = 12 \text{ steps}$$

This is remarkably fast for decisive weights, but slower for ambiguous ones — which is the desired behavior.

#### 5.4 Comparison with Full-Precision STE+Adam

| Aspect | STE + Adam | GSA-AT |
|---|---|---|
| Gradient information used | Full magnitude + direction | Direction only (1 bit/step) |
| Weight update granularity | Continuous → quantize | Discrete transitions only |
| Convergence rate | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T})$ (from SIGNUM theory), but effective rate slower due to thresholds |
| Convergence target | Continuous optimum → quantize | Directly optimizes in ternary space |
| Memory/param | 16 bytes | 1.25 bytes |
| Expected quality gap | Baseline | 10-25% perplexity degradation (estimated) |

### 6. Addressing the Gradient Magnitude Problem

Discarding gradient magnitude is the single most significant limitation. Three mitigations are proposed, in order of increasing complexity:

#### 6.1 Layer-wise Scaling (Zero extra memory per parameter)

Maintain one FP32 scalar per layer (negligible total memory):

$$s_l = \text{EMA}_\beta\!\left(\text{mean}(|g_t^{(\text{layer } l)}|)\right)$$

Scale the counter update by a discretized version of $s_l$ relative to a global median, as shown in the pseudocode. This reintroduces inter-layer gradient magnitude ratios while costing zero per-parameter memory.

**Theoretical justification:** The gradient magnitude ratio between layers captures the most important structural information (which layers need more/faster updates). Per-parameter magnitude variation within a layer is less critical for discrete weight decisions.

#### 6.2 Stochastic Multi-Increment (Zero extra memory)

Instead of always incrementing by $\pm 1$, use a probabilistic multi-step increment based on gradient magnitude:

$$\Delta c = \text{round}\!\left(\min\!\left(\frac{|g_t^{(i)}|}{\text{mean}(|g_t^{(\text{layer})}|)} \cdot s_{\text{base}},\; 8\right)\right) \cdot \text{sign}(g_t^{(i)})$$

Parameters with larger-than-average gradients increment the counter by more than 1, reaching the threshold faster. This encodes magnitude information in the increment size rather than storing it.

**Memory cost:** Zero — the computation uses only transient gradient values.

#### 6.3 INT4 Magnitude Tag (0.5 extra bytes/param)

Store a 4-bit magnitude indicator alongside the counter:

$$m_i = \text{clip}\!\left(\left\lfloor \log_2\!\left(\frac{|g_t^{(i)}|}{\text{mean}(|g_t^{(\text{layer})}|)}\right) + 8 \right\rfloor, 0, 15\right)$$

This uses 4 bits = 0.5 bytes extra per parameter, bringing total to 1.75 bytes/param. The magnitude tag modulates the effective threshold: weights with consistently large gradients get a lower effective threshold.

### 7. Memory Budget Analysis

#### 7.1 Approach B (1-byte optimizer, implicit adaptivity)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | The model parameter, encoded as 2 bits |
| INT8 counter $c$ | 1.0 | Gradient sign accumulator with decay (optimizer state) |
| Per-layer scale $s_l$ | ~0.00001 | Layer-wise gradient magnitude scaling (1 FP32 per layer; negligible) |
| **Total** | **1.25** | **3.75 bytes below 4-byte extra budget** |

#### 7.2 Approach A (2-byte optimizer, explicit adaptivity)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | The model parameter |
| INT8 counter $c$ | 1.0 | Gradient sign accumulator |
| UINT8 flip counter $f$ | 1.0 | Transition history for adaptive thresholds |
| Per-layer scale $s_l$ | ~0.00001 | Negligible |
| **Total** | **2.25** | **1.75 bytes below 4-byte extra budget** |

#### 7.3 Enhanced variant with INT4 magnitude tag

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | The model parameter |
| INT8 counter $c$ | 1.0 | Gradient sign accumulator |
| INT4 magnitude tag $m$ | 0.5 | Gradient magnitude indicator |
| UINT8 flip counter $f$ | 1.0 | Transition history (optional) |
| **Total** | **2.75** | **1.25 bytes below budget** |

#### 7.4 Gradient memory (transient)

Gradients ($g_t$) are computed during backpropagation and consumed immediately to update the counter. They are NOT stored persistently — each layer's gradient is computed, used to update counters, then discarded before computing the next layer's gradient. This is standard gradient-checkpointing-compatible and does not count toward the per-parameter budget.

**Critical note:** The per-parameter "extra" memory is 1.0 byte (Approach B) to 2.5 bytes (enhanced), all far below the 4-byte constraint. GSA-AT is the most memory-efficient proposal by a wide margin.

## Literature Support

### Directly Supporting Theories and Methods

1. **signSGD (Bernstein et al., 2018, ICML):** Proved that using only gradient signs achieves $\mathcal{O}(1/\sqrt{T})$ convergence for non-convex problems. Established the theoretical foundation that gradient magnitude can be discarded without losing convergence guarantees. [arXiv 1802.04434](https://arxiv.org/abs/1802.04434)

2. **SIGNUM / signSGD with Momentum (Bernstein et al., 2018; Sun et al., 2023, ICML):** The momentum variant of signSGD converges under *weaker* assumptions than vanilla signSGD — no bounded gradient requirement, works with small batch sizes. Key result: momentum remedies signSGD's fragility, making sign-based optimization practical. [PMLR v202](https://proceedings.mlr.press/v202/sun23l.html)

3. **signSGD with Majority Vote (Bernstein et al., 2018):** Proved that majority vote achieves the same variance reduction as full-precision averaging, with convergence rate $\mathcal{O}(d^{1/2} T^{-1/4})$. The per-parameter "voting" concept directly maps to GSA-AT's counter accumulation. [arXiv 1810.05291](https://arxiv.org/abs/1810.05291)

4. **Lion Optimizer (Chen et al., 2023, NeurIPS):** Discovered via automated search, Lion uses only sign of momentum-weighted gradients. Matches Adam for LLM training with only 1 momentum buffer. Demonstrates that sign-only updates are sufficient at scale. C-Lion variant achieves 1.28x sample efficiency speedup. [arXiv 2302.06675](https://arxiv.org/abs/2302.06675)

5. **Bop — Binary Optimizer (Helwegen et al., 2019, NeurIPS):** Proved that latent weights are unnecessary for binary networks; an EMA of gradients provides sufficient "inertia." The threshold-based flip mechanism is the direct precursor to GSA-AT's counter-based transitions. [arXiv 1906.02107](https://arxiv.org/abs/1906.02107)

6. **SGDAT (Gu et al., 2023, Neurocomputing):** Demonstrated that adaptive thresholds based on flip history suppress oscillation in BNNs and improve SGD performance to be comparable to Adam. Directly inspires GSA-AT's Approach A. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005544)

7. **Error Feedback Fixes SignSGD (Karimireddy et al., 2019, ICML):** Proved that error feedback restores convergence for biased compressors, achieving the same rate as uncompressed SGD. The GSA-AT-EF variant uses this principle. [arXiv 1901.09847](https://arxiv.org/abs/1901.09847)

8. **ECO (Nikdan et al., 2026):** Eliminates master weights via error injection into momentum. Convergence to bounded neighborhood proven for quantized training. Inspires the error retention mechanism in GSA-AT-EF. [arXiv 2601.22101](https://arxiv.org/abs/2601.22101)

9. **Overcoming Oscillations in QAT (Nagel et al., 2022, ICML):** Identified weight oscillation as a critical failure mode in low-bit QAT. GSA-AT's threshold mechanism directly prevents oscillation. [arXiv 2203.11086](https://arxiv.org/abs/2203.11086)

10. **Probabilistic Optimizer for BNNs (2025, Neurocomputing):** Interprets accumulated gradients as "potential energy" and uses Bernoulli probabilities for flips. Natural extension to categorical (ternary) distributions validates probabilistic discrete update approaches. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009695)

### Supporting Context

11. **GXNOR-Net (Deng et al., 2018):** Probabilistic discrete state transitions for ternary weights without latent weights. Only small-scale validation but establishes feasibility. [arXiv 1705.09283](https://arxiv.org/abs/1705.09283)

12. **DQT with Stochastic Rounding (Zhao et al., 2024):** Demonstrates ternary training without master weights is feasible, though with quality gap. [arXiv 2412.04787](https://arxiv.org/abs/2412.04787)

13. **EF21 (Richtarik et al., 2021, NeurIPS):** Modern error feedback achieving $\mathcal{O}(1/T)$ convergence for smooth nonconvex problems and linear convergence under PL condition. [arXiv 2106.05203](https://arxiv.org/abs/2106.05203)

14. **1-bit Adam (Tang et al., 2021):** Demonstrates error compensation works with 1-bit sign compression in Adam framework using warm-up + frozen variance trick. [arXiv 2102.02888](https://arxiv.org/abs/2102.02888)

## Generalizability Analysis

### Why GSA-AT Generalizes Across Model Sizes (100M to 100B+)

1. **Per-parameter independence:** Each weight's update depends only on its own counter and gradient sign. No inter-parameter dependencies, no matrix operations, no batch statistics. The method scales trivially — 100B parameters simply means 100B independent INT8 counters.

2. **Architecture agnosticism:** The update rule is defined per scalar weight. It applies identically to attention heads, FFN layers, embeddings, and any other linear layer in any Transformer variant. The only architecture-aware component is the per-layer scaling factor, which adapts automatically.

3. **Hardware universality:** INT8 add, compare, and clip operations are the simplest possible operations on any hardware platform (GPU, TPU, CPU, FPGA, custom ASIC). No floating-point arithmetic is required for the optimizer state update itself. This makes GSA-AT especially attractive for edge/mobile training scenarios.

4. **Gradient sign stability across scales:** A key insight from signSGD theory is that the *sign* of the gradient is more stable than the *magnitude* across mini-batches and across model scales. Bernstein et al. (2018) showed that the gradient sign carries the essential directional information regardless of absolute scale. This means GSA-AT's reliance on sign information does not degrade with model size.

5. **Layer-wise scaling adapts to depth:** Deeper models exhibit larger gradient magnitude variation across layers (gradient vanishing/exploding). The per-layer integer scaling factor $s_l$ automatically compensates: layers with larger gradients get larger counter increments, while layers with smaller gradients get smaller increments. This is analogous to the per-layer adaptive learning rates in Adam but at near-zero memory cost.

### Potential Scaling Risks

1. **Convergence speed at scale:** Larger models have more parameters to settle. If each parameter takes $\Theta(\tau / (1-\gamma))$ steps to make a decision, the total training time may scale unfavorably. However, this is offset by the fact that larger models tend to have more redundancy — many weights converge quickly while only a few are ambiguous.

2. **Loss landscape sharpness:** At extreme scale (100B+), the loss landscape may have sharper minima requiring more precise updates. GSA-AT's discrete transitions may "overshoot" narrow valleys. Mitigation: use higher thresholds $\tau$ at larger scale.

3. **Hyperparameter sensitivity:** The thresholds $\tau_a, \tau_d$ and decay $\gamma$ may need adjustment per model size. However, signSGD and Lion both demonstrate that sign-based methods are relatively robust to hyperparameter choices at scale, and the layer-wise scaling provides automatic per-layer adaptation.

### Dataset Generalizability

- Gradient sign is robust to data distribution shifts — the sign of a mini-batch gradient is more stable than its magnitude across different data batches and domains.
- The INT8 counter provides implicit noise filtering: transient gradient noise does not accumulate past the threshold because the decay factor $\gamma$ ensures old observations are forgotten.
- No dataset-specific components: the method uses standard backpropagation gradients as input and makes no assumptions about data distribution.

## Matching Metrics

- **Relevance to original question:** 7/10 — Directly addresses the memory constraint (far below 4 bytes/param) but trades quality for extreme efficiency. Best suited as a lower-bound explorer or component in hybrid approaches.
- **Confidence in findings:** 6/10 — Strong theoretical support from signSGD/SIGNUM/Lion literature for sign-based optimization, but no direct empirical validation at ternary LLM scale. The quality gap is the major uncertainty.
- **Completeness of investigation:** 8/10 — Thorough mathematical formulation, convergence analysis, multiple variants, error feedback enhancement, and comprehensive literature grounding. The missing piece is empirical data at scale.

## Memory Budget Breakdown

### Configuration: Approach B (1-byte, most efficient)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | Model parameter {-1, 0, +1} in 2 bits |
| INT8 counter $c$ | 1.0 | Gradient sign EMA / decision accumulator |
| Layer-scale $s_l$ | ~0 | 1 FP32 per layer (amortized ~0) |
| **Total** | **1.25** | **Extra beyond ternary: 1.0 byte** |

### Configuration: Approach A (2-byte, explicit adaptivity)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | Model parameter |
| INT8 counter $c$ | 1.0 | Gradient sign EMA |
| UINT8 flip counter $f$ | 1.0 | Adaptive threshold via transition history |
| **Total** | **2.25** | **Extra beyond ternary: 2.0 bytes** |

### Configuration: Enhanced (2.75-byte, magnitude-aware)

| Component | Bytes/Param | Purpose |
|---|---|---|
| Ternary weight $w$ | 0.25 | Model parameter |
| INT8 counter $c$ | 1.0 | Gradient sign EMA |
| INT4 magnitude tag $m$ | 0.5 | Per-param gradient magnitude indicator |
| UINT8 flip counter $f$ | 1.0 | Adaptive threshold |
| **Total** | **2.75** | **Extra beyond ternary: 2.5 bytes** |

All configurations are well within the 4-byte extra budget.

## Key Takeaways

- **Most memory-efficient optimizer proposed:** At 1.0 byte extra per parameter (Approach B), GSA-AT uses 4x less optimizer memory than the next most efficient proposal (TBop at ~2 bytes extra) and 16x less than standard STE+Adam.

- **Theoretically grounded in signSGD/SIGNUM:** The convergence of sign-based optimization is well-established (Bernstein et al., 2018; Sun et al., 2023). GSA-AT extends this to discrete ternary weights with threshold-based transitions, adding hysteresis to prevent oscillation.

- **Lion optimizer validates sign-only updates for LLMs:** The success of Lion (Chen et al., 2023) at LLM scale — matching Adam with sign-only updates and a single momentum buffer — provides the strongest empirical precedent that gradient magnitude information is not essential for optimization.

- **Counter-as-ballot-box is a natural abstraction:** Treating each weight's optimizer state as a running vote count maps the continuous optimization problem to a discrete decision problem with probabilistic guarantees. This is both interpretable and efficient.

- **The adaptive threshold mechanism (from SGDAT) is critical:** Without adaptive or decay-based thresholds, naive counter-based methods suffer from oscillation. SGDAT's flip-history-based adaptivity has been validated for BNNs and transfers directly to ternary.

- **Error feedback can be incorporated at zero extra cost:** The GSA-AT-EF variant reuses counter space to carry forward quantization error information across transitions, borrowing from ECO and EF-SGD theory.

- **Quality gap is the dominant risk:** Expected 10-25% perplexity degradation vs STE+Adam due to gradient magnitude loss. The layer-wise scaling and stochastic multi-increment mitigations may reduce this, but the gap is unlikely to close entirely at 1 byte/param.

- **Best used as a component, not standalone:** GSA-AT's extreme efficiency makes it ideal as the "Phase 2" optimizer in a hybrid scheme (Idea 005's relationship to Idea 005 from the hybrid phase training angle), or for layers that are less sensitive to precision, or as a baseline establishing the minimum information needed for ternary training.

## Limitations & Open Questions

### Critical Limitations

1. **No empirical validation at any scale.** GSA-AT is a purely theoretical design. While grounded in proven theories (signSGD, SIGNUM, Lion, Bop, SGDAT), the specific combination for ternary LLM training has never been tested. All quality estimates (10-25% degradation) are speculative.

2. **Gradient magnitude loss may be fatal for some architectures.** Attention layers and embedding layers may be more sensitive to gradient magnitude than FFN layers. A uniform 1-byte budget across all layers may be suboptimal — adaptive per-layer precision allocation could help.

3. **INT8 counter quantization introduces correlated errors.** The rounding in $\lfloor \gamma \cdot c + \text{sign}(g) \rceil$ is deterministic (nearest integer), not stochastic. Unlike stochastic rounding, this introduces bias. The bias is small (≤0.5 per step) but correlated across steps, which could cause systematic drift over millions of training steps.

4. **Threshold sensitivity.** While adaptive thresholds help, the base thresholds ($\tau_{a,\text{base}}, \tau_{d,\text{base}}$) and decay $\gamma$ must be tuned. No principled method exists for setting these — they likely depend on model size, learning rate schedule, and data distribution.

5. **No direct → ±1 transitions.** The current formulation only allows transitions through the zero state ($-1 \to 0 \to +1$ or $+1 \to 0 \to -1$). Direct jumps ($-1 \to +1$) require two consecutive transitions, which may slow convergence when a weight needs to completely reverse polarity.

### Open Questions

1. **What is the minimum information rate for ternary training?** GSA-AT uses ~8 bits of persistent state per parameter. Is this sufficient, or is there an information-theoretic lower bound that exceeds 8 bits? The answer would determine whether 1-byte optimization is fundamentally possible or doomed.

2. **Can stochastic rounding of the INT8 counter improve convergence?** Using $\text{SR}(\gamma \cdot c + \text{sign}(g))$ instead of nearest rounding would make the counter update unbiased. This adds negligible compute but could eliminate the correlated error problem.

3. **How does the quality gap scale with model size?** signSGD and Lion both show that sign-based methods *improve* relative to magnitude-based methods at larger scale. If this trend holds for GSA-AT, the quality gap may shrink at 7B+ parameters.

4. **Can GSA-AT be combined with knowledge distillation?** Training a ternary student with GSA-AT while distilling from a full-precision teacher could compensate for the quality gap. The teacher provides better gradient signals, which even through sign compression may be sufficient.

5. **What is the optimal decay schedule?** A fixed $\gamma$ may not be optimal. Early training (weights far from optimum) may benefit from lower decay (faster response), while late training (fine-tuning) may benefit from higher decay (more evidence required). An adaptive $\gamma$ schedule analogous to learning rate warmup could help.

6. **Can the zero-state sparsity be exploited?** BitNet b1.58-2B-4T reports ~42.3% of weights are zero. For these stable zero weights, the counter state is "wasted" (the counter hovers near zero, never reaching threshold). A compressed representation that doesn't allocate counters to stable zeros could reduce effective memory further.

7. **Does the $-1 \to 0 \to +1$ constraint hurt convergence?** Allowing direct $-1 \leftrightarrow +1$ transitions (triggered when $|c| > 2\tau$, for example) could speed convergence for polarity-reversing weights. The tradeoff is increased transition noise.
