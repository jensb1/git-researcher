---
idea_id: 002
status: complete
relevance_score: 9
confidence_score: 7
completeness_score: 9
---

# Research Results: Idea 002 — EC-DQT-T: Error-Compensated Direct Quantized Training for Ternary Networks

## Summary

We propose **EC-DQT-T**, a training method for BitNet-style ternary {-1, 0, 1} networks that eliminates latent weights entirely by combining Direct Quantized Training (DQT) with stochastic rounding, error compensation via a unified BF16 accumulator, and near-free adaptive preconditioning via Adafactor-style factored second moments. The method achieves **~2.25 bytes/param** total memory (well within the 4-byte budget), with convergence guarantees grounded in the error feedback framework (Karimireddy et al., 2019) and the LOTION smoothed-loss framework (Kwun et al., 2025). A key novelty is the **unified accumulator** that simultaneously serves as momentum buffer and quantization error residual, with a self-balancing reset mechanism triggered by ternary weight flips.

## Proposed Method

### 1. Core Insight: The Unified Accumulator

The central innovation is storing a single BF16 floating-point value per parameter — the **accumulator** `a` — that simultaneously encodes:
- **Momentum**: the exponentially-weighted moving average of past gradient signals
- **Quantization residual**: the error from stochastic rounding that should be compensated in the next step

This is possible because both momentum and error residual live in the same space: they represent "how far the continuous optimum has drifted from the current ternary weight." The accumulator tracks this drift. When it grows large enough, a ternary weight flip becomes likely (via stochastic rounding), and the flip resets the accumulator — a natural self-balancing mechanism.

### 2. Mathematical Formulation

**State per parameter:**
- `w_t ∈ {-1, 0, +1}` — the ternary weight (2-bit packed, 0.25 bytes)
- `a_t ∈ BF16` — the unified accumulator (2 bytes)

**State per weight matrix (shared):**
- `R_t ∈ FP32^{d_1}` — row-wise second-moment factor (Adafactor)
- `C_t ∈ FP32^{d_2}` — column-wise second-moment factor (Adafactor)

**Hyperparameters:** learning rate `η`, momentum decay `β` (e.g., 0.9), second-moment decay `ρ` (e.g., 0.999), warmup steps `T_w` (~20% of total training), stability constant `ε` (e.g., 1e-8).

**Update Rules:**

**Step 1 — Gradient computation:**

$$g_t = \nabla_w \mathcal{L}(w_t; x_t)$$

Computed in BF16, transient (not stored after this step).

**Step 2 — Adaptive preconditioning (Adafactor-style):**

For `t ≤ T_w` (warmup phase), update the factored second moment:

$$R_t = \rho \cdot R_{t-1} + (1 - \rho) \cdot \text{RowMean}(g_t^2 + \varepsilon)$$

$$C_t = \rho \cdot C_{t-1} + (1 - \rho) \cdot \text{ColMean}(g_t^2 + \varepsilon)$$

For `t > T_w`, freeze `R` and `C` (read-only, per 1-bit Adam's variance stabilization finding).

Reconstruct the preconditioner and scale the gradient:

$$\hat{V}_t = \frac{R_t \otimes C_t}{\mathbf{1}^\top R_t}, \quad \tilde{g}_t = \frac{g_t}{\sqrt{\hat{V}_t} + \varepsilon}$$

**Step 3 — Accumulator update (momentum):**

$$a_t = \beta \cdot a_{t-1} - (1 - \beta) \cdot \eta_t \cdot \tilde{g}_t$$

**Step 4 — Candidate weight and stochastic rounding:**

$$\tilde{w}_t = w_t + a_t \quad \text{(desired continuous weight)}$$

$$w_{t+1} = \text{SR}\left(\text{clip}(\tilde{w}_t, -1, 1)\right)$$

where SR is stochastic rounding to the ternary grid (see Section 3 below).

**Step 5 — Error compensation (accumulator adjustment):**

$$a_t \leftarrow a_t - (w_{t+1} - w_t)$$

This is the key step: if the weight flips (e.g., `w: 0 → 1`), the accumulator is reduced by the flip magnitude (e.g., `a ← a - 1`), resetting it toward zero. If the weight doesn't flip, the accumulator retains its value for the next iteration.

### 3. Stochastic Rounding for the Ternary Grid

For `x ∈ [-1, +1]`, stochastic rounding to `{-1, 0, +1}`:

```
If x ∈ [0, +1]:
    SR(x) = +1  with probability x
             0   with probability 1 - x

If x ∈ [-1, 0]:
    SR(x) = -1  with probability |x|
             0   with probability 1 - |x|

If x > +1:  SR(x) = +1  (clamped)
If x < -1:  SR(x) = -1  (clamped)
```

**Property:** `E[SR(x)] = x` for `x ∈ [-1, 1]` — the rounding is unbiased.

**Variance:** `Var[SR(x)] = x(1-x)` for `x ∈ [0,1]` (and symmetrically for negative x). Maximum variance is 1/4 at `x = 0.5`.

### 4. Self-Balancing Dynamics (Worked Example)

Consider a weight at `w = 0` with consistent gradient signal `g = -1` (pushing weight toward +1):

| Step | a (before SR) | w̃ | P(flip to 1) | E[a after SR] |
|------|--------------|-----|--------------|---------------|
| 1 | 0.001 | 0.001 | 0.1% | ≈0.001 |
| 10 | 0.007 | 0.007 | 0.7% | ≈0.007 |
| 100 | 0.010 | 0.010 | 1.0% | ≈0.010 |
| ... (steady state) | 0.010 | 0.010 | 1.0% | ≈0.010 |

(With β=0.9, η=0.001: steady-state |a| = η/(1-β) = 0.01)

When the weight eventually flips (`w: 0 → 1`), the accumulator resets: `a ← 0.01 - 1.0 = -0.99`. The large negative value resists further +1→+1 changes. Subsequent gradient updates erode this slowly. If the gradient is consistent, `a` climbs back toward 0, and the weight stays at 1 (SR(1 + a) for small a rounds to 1 with high probability).

This self-balancing ensures:
- Weights flip only when gradient evidence is sufficiently consistent
- Premature stochastic flips are self-correcting (accumulator compensates)
- The expected trajectory matches continuous SGD+momentum

### 5. Pseudocode

```python
def ec_dqt_t_train(model, dataloader, config):
    """
    EC-DQT-T: Error-Compensated Direct Quantized Training for Ternary
    Memory per parameter: ~2.25 bytes (BF16 accumulator + 2-bit ternary weight)
    """
    eta = config.lr              # learning rate
    beta = config.momentum       # momentum decay (e.g. 0.9)
    rho = config.rho             # 2nd moment decay (e.g. 0.999)
    T_w = config.warmup_steps    # ~20% of total steps
    eps = 1e-8

    # Initialize per-parameter state
    for layer in model.ternary_layers:
        W = layer.weight                     # ternary {-1, 0, +1}, 2-bit packed
        A = zeros_like(W, dtype=bfloat16)    # unified accumulator
        R = zeros(W.shape[0], dtype=float32) # row 2nd moment factor
        C = zeros(W.shape[1], dtype=float32) # col 2nd moment factor

    for t, batch in enumerate(dataloader):
        # === Forward pass (ternary weights, 8-bit activations) ===
        loss = model.forward(batch)

        # === Backward pass (compute BF16 gradients) ===
        grads = backward(loss)  # BF16, transient

        for layer in model.ternary_layers:
            W, A, R, C = layer.weight, layer.accum, layer.R, layer.C
            G = grads[layer]

            # --- Adaptive preconditioning (Adafactor-style) ---
            if t <= T_w:
                R = rho * R + (1 - rho) * row_mean(G**2 + eps)
                C = rho * C + (1 - rho) * col_mean(G**2 + eps)
            # Reconstruct per-parameter scaling
            V_inv_sqrt = 1.0 / sqrt(outer(R, C) / sum(R) + eps)
            G_scaled = G * V_inv_sqrt

            # --- Accumulator update (momentum) ---
            A = beta * A - (1 - beta) * eta_t * G_scaled

            # --- Stochastic rounding to ternary ---
            W_tilde = W.float() + A             # candidate continuous
            W_clipped = clip(W_tilde, -1.0, 1.0)
            W_new = stochastic_round_ternary(W_clipped)

            # --- Error compensation ---
            A = A - (W_new - W).to(bfloat16)    # adjust for realized flip
            W = W_new

            # Free gradient memory
            del G

    return model


def stochastic_round_ternary(x):
    """
    Stochastically round x ∈ [-1, +1] to {-1, 0, +1}.
    Unbiased: E[SR(x)] = x for x in [-1, 1].
    """
    u = uniform(0, 1, shape=x.shape)           # random draw
    result = zeros_like(x, dtype=int2)

    # Positive region: [0, 1] → {0, +1}
    pos = (x >= 0)
    result[pos] = where(u[pos] < x[pos], +1, 0)

    # Negative region: [-1, 0) → {-1, 0}
    neg = (x < 0)
    result[neg] = where(u[neg] < abs(x[neg]), -1, 0)

    return result
```

### 6. Convergence Analysis

**Theoretical Framework:** Our method can be analyzed through two complementary lenses:

**Lens 1: LOTION Smoothed Loss (Kwun et al., 2025)**

Define the smoothed loss:

$$\tilde{f}(\tilde{w}) = \mathbb{E}_{w \sim \text{SR}(\tilde{w})}[f(w)]$$

By LOTION's Lemma 1, `f̃` is continuous and differentiable almost everywhere. By Lemma 2, `f̃` preserves all global minima of the original quantized loss `f`. Our method is equivalent to running SGD+momentum on `f̃`, with the accumulator tracking the continuous variable `w̃ = w + a`.

Under standard assumptions (L-smooth `f̃`, bounded gradient variance `σ²`), SGD+momentum converges:

$$\frac{1}{T} \sum_{t=1}^{T} \mathbb{E}\left[\|\nabla \tilde{f}(\tilde{w}_t)\|^2\right] \leq \mathcal{O}\left(\sqrt{\frac{L(\tilde{f}_0 - \tilde{f}^*)\sigma^2}{T}}\right)$$

**Lens 2: Error Feedback Framework (Karimireddy et al., 2019)**

Our method applies an unbiased compressor (stochastic rounding) with error feedback (the accumulator stores the residual). By the error feedback theorem, this achieves the **same convergence rate as uncompressed SGD**, regardless of the compressor quality.

Formally, define the compression operator `C(x) = SR(clip(x, -1, 1))` and error `e_t = x_t - C(x_t)`. With error feedback, the effective update is:

$$x_{t+1} = x_t - \eta_t (g_t + e_{t-1})$$

The error feedback ensures that `Σ_t e_t` remains bounded (does not diverge), and the method converges at rate:

$$\mathbb{E}[f(\bar{x}_T) - f^*] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}} + \frac{\omega}{T}\right)$$

where `ω` is the compression ratio. For ternary SR, `ω ≤ 1/4` (the maximum variance).

**Lens 3: Variance Reduction via Error Compensation**

Without error compensation (pure DQT with SR), each step introduces variance `σ²_SR ≤ 1/4` per parameter. With error compensation, the effective variance is reduced because the accumulator carries the "intent":

After `k` steps without a flip, the accumulator `a_k` has accumulated gradient signal, making the candidate weight `w̃ = w + a_k` further from the nearest grid point. This makes the SR decision more deterministic (probability near 0 or 1), reducing the per-step variance:

$$\sigma^2_{\text{EC-SR}} \approx \mathbb{E}[|a_t \bmod 1| \cdot (1 - |a_t \bmod 1|)] < \frac{1}{4}$$

The inequality is strict because the accumulator spends most of its time growing monotonically (when gradient direction is consistent), only briefly visiting the high-variance midpoints.

**Why Momentum-Only May Suffice (Addressing Liu et al., 2021):**

Liu et al. (ICML 2021) showed Adam's second moment is crucial for BNN training. However, their analysis applies to **STE-based training** where the optimizer navigates a landscape distorted by the STE bias. In our DQT framework:

1. **No STE bias**: Stochastic rounding is unbiased (`E[SR(x)] = x`), unlike STE which systematically passes incorrect gradient information through the quantizer.
2. **Smoothed landscape**: The LOTION framework shows that SR creates a smooth, differentiable loss surface where standard first-order methods converge.
3. **1-bit Adam finding**: Tang et al. (ICML 2021) showed that Adam's variance term stabilizes after ~15-20% of training and can be frozen. This means per-parameter adaptivity is primarily needed during early training — exactly the phase where our Adafactor factored second moment is active.

We include Adafactor preconditioning as a safety measure at ~0.002 bytes/param cost, providing the adaptive benefit during warmup while adding essentially zero memory.

## Literature Support

### Directly Supporting Papers

| Paper | Key Contribution to Our Method |
|-------|-------------------------------|
| **DQT** (Zhao et al., ACML 2025; arXiv 2412.04787) | Proved stochastic rounding can replace STE+latent weights for ternary LLM training. Tested up to 1B params with LLaMA architecture. Quality gap vs BitNet b1.58 narrows with scale. |
| **Error Feedback** (Karimireddy et al., ICML 2019; arXiv 1901.09847) | Proved that error feedback enables any compressor (including SR) to achieve the same convergence rate as uncompressed SGD. Theoretical foundation for our error compensation. |
| **ECO** (Nikdan et al., Jan 2026; arXiv 2601.22101) | Demonstrated error feedback injected into momentum buffer eliminates master weights in FP8/INT4 training. Convergence proven to bounded neighborhood. We extend this idea to ternary. |
| **LOTION** (Kwun et al., Oct 2025; arXiv 2510.08757) | Proved that the expected loss under SR noise is a smooth function preserving global minima, enabling standard optimizer convergence theory. Tested at 150M-300M params with INT4. |
| **1-bit Adam** (Tang et al., ICML 2021; arXiv 2102.02888) | Showed Adam's variance (v_t) stabilizes after ~15-20% of training, justifying our warmup-then-freeze Adafactor schedule. |
| **Adafactor** (Shazeer & Stern, 2018; arXiv 1804.04235) | Factored second moment: row/col factors cost (d₁+d₂) FP32 values instead of d₁×d₂ — negligible per-parameter overhead. Validated at LLM scale (T5 11B). |
| **Stochastic Rounding for LLMs** (Ozkara et al., AISTATS 2025; arXiv 2502.20566) | Proved SR's quantization error can be subsumed by Adam's convergence bound with proper hyperparameters. BF16+SR outperformed mixed-precision at 6.7B scale. |

### Related Approaches

| Paper | Relationship |
|-------|-------------|
| **Bop** (Helwegen et al., NeurIPS 2019; arXiv 1906.02107) | Binary optimizer without latent weights using EMA+threshold. Our accumulator generalizes this: the "threshold" is implicit (SR fires when accumulator makes flip probable). Never extended to ternary. |
| **GXNOR-Net** (Deng et al., 2018; arXiv 1705.09283) | Discrete state transitions for ternary with tanh-based probabilistic flip. Our SR-based approach is simpler, unbiased, and theoretically grounded (LOTION). Only tested on small CNNs. |
| **8-bit Optimizers** (Dettmers et al., ICLR 2022; arXiv 2110.02861) | Block-wise INT8 quantization of Adam states. We apply similar block-wise quantization to our accumulator (INT8 variant). Validated at 1.5B+ scale. |
| **4-bit Optimizers** (Li et al., NeurIPS 2023; arXiv 2309.01507) | Row+column factored 4-bit optimizer states. Represents the frontier for compressed Adam. Our method is more aggressive — we eliminate the full second moment entirely. |
| **Spectra TriLMs** (Kaushal et al., 2024-2025; arXiv 2407.12327) | Ternary LLMs at 3.9B matching float at same parameter count. Confirms ternary architectures are viable at scale. Training still uses STE+Adam. |
| **FP4 Training** (Chmiel et al., NeurIPS 2025 Spotlight; arXiv 2505.19115) | Full FP4 training at 7B scale matching BF16. Uses SR for backward pass. Showed gradient norm threshold below which quantized training degrades. |
| **Training with Fewer Bits** (Liu et al., EMNLP 2025; arXiv 2511.00874) | 1-bit precision reduction compensated by ≤4x batch increase. Supports our recommendation of larger batch sizes for ternary DQT. |
| **Liu et al.** (ICML 2021) | Adam crucial for BNN training (STE-based). We address this by including Adafactor preconditioning during warmup, and argue the STE vs DQT distinction changes the landscape. |
| **Quartet II / MS-EDEN** (IST-DASLab, Jan 2026; arXiv 2601.22813) | Novel unbiased quantizer with 2x lower variance than SR. Potential future enhancement for our SR step, though currently only available for NVFP4 format. |
| **Q-Adam-mini** (2025; OpenReview) | INT8 first moment + FP32 second moment. 8x memory reduction at 8B scale. Confirms INT8 momentum is viable at LLM scale. |
| **CAME** (ACL 2023; arXiv 2307.02047) | Confidence-guided fix for Adafactor instability in decoder-only LLMs. Relevant if our Adafactor preconditioning shows instability. |

### Foundational Theory

| Paper | Contribution |
|-------|-------------|
| **Gupta et al.** (ICML 2015; arXiv 1502.02551) | Foundational SR paper. 16-bit fixed-point with SR trains CNNs with no loss. SR provides implicit "carry" mechanism for small updates. |
| **Li et al.** (NeurIPS 2017; arXiv 1706.02379) | SR converges to accuracy floor proportional to quantization step size Δ. For ternary (Δ=1), floor is larger than for fine-grained grids. |
| **El Arar et al.** (SIAM J. Sci. Comput. 2023; arXiv 2207.10321) | SR errors form a martingale. Probabilistic bounds via Bienaymé-Chebyshev are tighter than worst-case. Error growth is O(√n) vs O(n) for deterministic rounding. |
| **Connolly, Higham, Mary** (Royal Society 2022) | Comprehensive SR error analysis. SR promotes error cancellation and avoids stagnation (the "small update lost" problem of deterministic rounding). |
| **QSGD** (Alistarh et al., NeurIPS 2017; arXiv 1610.02132) | Near-optimal communication-variance tradeoff. With error feedback, even 1 bit per coordinate per iteration suffices for SGD convergence. |
| **Bernstein et al.** (ICML 2018; arXiv 1802.04434) | signSGD with momentum matches Adam on ImageNet. Sign-based updates are natural for discrete weight training. |

## Generalizability Analysis

### Model Size (100M to 100B+)

**All operations are per-parameter and size-independent:**
- Stochastic rounding: elementwise, O(1) per parameter, no cross-parameter dependencies
- Momentum update: elementwise, O(1) per parameter
- Error compensation: elementwise, O(1) per parameter
- Adafactor: per-row and per-column, scales as O(d₁+d₂) per matrix — sublinear in parameter count

**Scaling behavior favors larger models:**
- DQT empirically showed the quality gap vs STE+Adam narrows with scale (Zhao et al., 2024)
- The signal-to-noise ratio of accumulated gradients improves with more data per step (larger batch sizes become practical at scale)
- Spectra showed ternary LLMs at 3.9B match float performance; architecture is validated at scale
- The aggregated SR noise over d parameters has variance d/4, but the gradient signal also scales with d. The per-parameter signal-to-noise ratio is constant in d

**No scale-dependent hyperparameters:**
- β (momentum) is standard across scales (0.9-0.95)
- ρ (2nd moment decay) is standard (0.999)
- T_w (warmup fraction) is 15-20% regardless of total steps
- η (learning rate) follows standard scaling laws (e.g., linear scaling with batch size)

### Architecture Independence

The method requires only:
1. Weights quantized to the ternary grid {-1, 0, +1}
2. Differentiable loss function (for gradient computation)
3. Standard backward pass (no STE needed — gradients flow through the ternary weights via the LOTION smoothed-loss argument)

Works with: Transformers (self-attention, FFN, embeddings), CNNs, RNNs, MoE architectures — any architecture using linear layers that can be ternarized.

### Dataset Independence

- Stochastic rounding is unbiased regardless of data distribution
- Momentum and Adafactor adapt to gradient statistics automatically
- No dataset-specific hyperparameters beyond standard (lr, batch size, warmup)

### Hardware Compatibility

**GPU (NVIDIA):**
- Stochastic rounding: `cuRAND` for per-parameter random numbers (well-supported, ~ns per element)
- BF16 accumulator: native BF16 arithmetic on A100/H100/B200
- Ternary weights: stored as 2-bit packed INT, dequantized for matmul
- Block-wise INT8 (for accumulator variant): supported by `bitsandbytes` library
- No custom kernels required (all operations are standard elementwise or matmul)

**TPU (Google):**
- BF16 is the native format on TPU
- SR supported via `jax.random` primitives
- Adafactor is Google's preferred optimizer for TPU training (used for T5, PaLM)

**Intel Gaudi:**
- FP4 training validated on Gaudi2 (Chmiel et al., 2025) — our BF16 method is less demanding
- SR validated in the FP4 training pipeline

## Matching Metrics

- **Relevance to original question:** 9/10 — Directly addresses the core problem of eliminating latent weight overhead. Proposes a concrete method at 2.25 bytes/param, well below the 4-byte target.
- **Confidence in findings:** 7/10 — Each individual component (DQT, error feedback, Adafactor, SR) is validated in the literature. The combination is novel and untested. The main uncertainty is whether the quality gap closes sufficiently at LLM scale for ternary specifically.
- **Completeness of investigation:** 9/10 — Comprehensive literature survey (25+ papers), detailed mathematical formulation, pseudocode, convergence analysis from multiple theoretical angles, and thorough generalizability analysis.

## Memory Budget Breakdown

### Variant A: EC-DQT-T (Recommended)

| Component | Precision | Bytes/Param | Purpose |
|---|---|---|---|
| Ternary weight `w` | 2-bit packed | 0.25 | Model weight |
| Accumulator `a` | BF16 | 2.00 | Momentum + error residual |
| Factored 2nd moment `R, C` | FP32, shared | ~0.002 | Adaptive preconditioning |
| Gradient `g` | BF16, transient | 0* | Consumed during backward |
| **Total** | | **~2.25** | **Budget: ≤ 4.2** |

*Gradients are computed, used to update the accumulator, and discarded — not stored persistently.

**Factored 2nd moment detail:** For a `d₁ × d₂` weight matrix, we store `d₁ + d₂` FP32 values. For a typical 4096×4096 Transformer layer: (4096 + 4096) × 4 bytes / (4096 × 4096 params) = **0.002 bytes/param**. For 1D parameters (biases, norms), we store the full second moment, but these are <1% of total parameters.

### Variant B: EC-DQT-T-Lite (Maximum Compression)

| Component | Precision | Bytes/Param | Purpose |
|---|---|---|---|
| Ternary weight `w` | 2-bit packed | 0.25 | Model weight |
| Accumulator `a` | INT8 block-quantized | ~1.03 | Momentum + error residual |
| Factored 2nd moment | FP32, shared | ~0.002 | Adaptive preconditioning |
| **Total** | | **~1.28** | **Budget: ≤ 4.2** |

**Caution:** INT8 accumulator has resolution ~0.008 (for typical block scales). Gradient updates of magnitude η×|g| < 0.008 are lost. Viable only with gradient accumulation over K≥4 steps or larger learning rates.

### Variant C: EC-DQT-T-Full (Maximum Quality)

| Component | Precision | Bytes/Param | Purpose |
|---|---|---|---|
| Ternary weight `w` | 2-bit packed | 0.25 | Model weight |
| BF16 momentum `m` | BF16 | 2.00 | Gradient smoothing |
| BF16 error residual `e` | BF16 | 2.00 | SR error compensation |
| Factored 2nd moment | FP32, shared | ~0.002 | Adaptive preconditioning |
| **Total** | | **~4.25** | **Slightly over; use FP16 residual clipping to fit** |

This variant stores momentum and residual separately, avoiding the information-coupling of the unified accumulator. At 4.25 bytes/param, it slightly exceeds the 4.2 budget. Can be brought within budget by using INT8 for the error residual: 0.25 + 2.0 + 1.0 + 0.002 = **3.25 bytes/param**.

### Comparison to Existing Methods

| Method | Bytes/Param | Ternary Validated? | LLM Scale? |
|---|---|---|---|
| STE + Adam (baseline) | 16.0 | Yes (BitNet) | Yes (2B-4T) |
| STE + 8-bit Adam | ~6.0 | Partially | Yes |
| DQT + AdamW | ~10.0* | Yes (1B) | Partially |
| DQT + Adafactor | ~4.5* | Yes (1B) | Partially |
| **EC-DQT-T (ours)** | **~2.25** | **Theoretical** | **Designed for** |
| **EC-DQT-T-Lite (ours)** | **~1.28** | **Theoretical** | **Designed for** |

*DQT paper used high-precision optimizer states; memory savings were primarily from eliminating latent weights.

## Key Takeaways

- **2.25 bytes/param is achievable** by combining a unified BF16 accumulator (momentum + error residual) with Adafactor-style factored second moments and stochastic rounding. This is a **7× reduction** from the 16 bytes/param standard.

- **The unified accumulator is the key innovation**: a single BF16 value per parameter encodes both temporal gradient smoothing (momentum) and quantization error compensation (residual), with a self-balancing reset triggered by ternary weight flips.

- **Error compensation dramatically reduces the effective SR variance**: pure DQT suffers from high SR noise at the ternary level (variance up to 1/4 per parameter). Our error compensation mechanism ensures that the residual from each SR decision is preserved and applied at the next step, making the effective compressor nearly lossless over time.

- **Convergence is theoretically grounded** via three complementary frameworks: (1) LOTION's smoothed loss preserves global minima and enables standard optimizer convergence; (2) Karimireddy's error feedback theorem guarantees SGD-matching convergence rate; (3) the accumulator dynamics create a self-correcting system where premature stochastic flips are naturally compensated.

- **Adafactor preconditioning is nearly free** (~0.002 bytes/param) and addresses the concern from Liu et al. (2021) about the importance of adaptive learning rates for quantized training. Combined with the 1-bit Adam finding that variance stabilizes early, we warm up the factored second moment and then freeze it.

- **Larger batch sizes help disproportionately**: the stochastic rounding noise is independent of batch size, but gradient noise decreases with batch size. Effective batch sizes ≥2048 are recommended to ensure gradient signal dominates SR noise at the ternary level.

- **The method naturally generalizes** Bop (binary optimizer): where Bop uses EMA + threshold, our accumulator + SR provides a continuous-probability version of threshold flipping that is unbiased and has convergence guarantees. The zero state in ternary {-1, 0, +1} is handled naturally by SR (the zero state acts as a transition hub).

## Limitations & Open Questions

### Critical Unknowns

1. **Empirical quality gap at scale:** The DQT paper showed a gap for ternary at 130M-1B that narrowed with scale. EC-DQT-T should reduce this gap further (via error compensation), but the magnitude of improvement is unknown without experiments. The gap must be <5-10% perplexity degradation to meet success criteria.

2. **Accumulator precision sensitivity:** The BF16 accumulator has ~3.9 significant decimal digits. For very small learning rates (η < 1e-5) common in later training stages, gradient updates may approach BF16 precision limits. This could cause effective learning rate collapse, where updates become too small to register. Mitigation: use loss scaling or keep the accumulator in FP32 at the cost of 4 bytes/param total (still within budget).

3. **Clipping bias:** When the candidate weight `w̃ = w + a` falls outside [-1, 1], clipping introduces bias (the accumulator's "memory" is truncated). This occurs when momentum is very large (fast learning, early training). Analysis needed on whether warmup schedules sufficiently prevent this.

4. **Interaction with learning rate schedules:** Standard cosine/linear decay schedules were designed for continuous weights. The discrete ternary dynamics may need modified schedules — e.g., the effective learning rate (flip probability) decays faster than the nominal rate because the accumulator decays multiplicatively by β each step.

5. **Weight initialization:** Ternary initialization (random {-1, 0, 1}) may need careful design. The proportion of zero weights at init affects early training dynamics. Too many zeros = slow warmup; too few = early instability.

### Theoretical Gaps

6. **Tight convergence bound for ternary specifically:** Our convergence analysis uses general frameworks (LOTION, error feedback) that apply to any quantization level. A ternary-specific analysis might yield tighter bounds by exploiting the 3-state structure.

7. **Optimal momentum coefficient β for ternary:** The standard β=0.9-0.95 may not be optimal for ternary. Higher β provides more smoothing (good for reducing SR noise) but slower response to gradient changes. A ternary-specific analysis of the bias-variance tradeoff for β is needed.

8. **Information-theoretic lower bound:** What is the minimum bytes/param needed for ternary LLM training at a given quality level? Our 2.25 bytes/param may be close to optimal, or there may be room for further compression. The QSGD result (Alistarh et al., 2017) suggests that with error feedback, even 1 bit per coordinate suffices asymptotically — but this is for convergence rate, not final quality.

### Practical Considerations

9. **Wall-clock overhead:** Stochastic rounding requires one random number per parameter per step. At 7B parameters, this is 7 billion random draws per step. `cuRAND` generates ~10 billion FP32 randoms per second on an H100, so the overhead is ~0.7 seconds per step — potentially significant. Pseudorandom generators (e.g., counter-based Philox) or approximate SR methods may be needed.

10. **Gradient flow through ternary weights:** In standard DQT, gradients are computed w.r.t. the ternary weights directly, using the STE for the rounding operation. In our method, the LOTION framework suggests gradients should be computed w.r.t. the smoothed loss. The practical implementation (which STE variant to use for the backward pass) needs careful specification.

11. **Distributed training:** The accumulator `a` is per-parameter state that must be maintained on each device. In data-parallel training, accumulators are not synchronized (each device has its own stochastic rounding outcomes). The gradient is synchronized (all-reduce), but the post-gradient accumulator update is local. This is correct and parallels standard optimizer state handling, but introduces subtle divergence between devices' ternary weights. The impact of this divergence on training quality is unclear.

12. **Comparison to DQT + Adafactor baseline:** The DQT paper already tested Adafactor, achieving ~4.5 bytes/param with "minimal performance degradation." Our method's primary advantage is the unified accumulator + error compensation, which should improve quality at a lower memory cost. A rigorous A/B comparison is needed.
