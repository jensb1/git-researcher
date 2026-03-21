# Literature Search Results: Memory-Efficient Training for Ternary Networks

Date: 2026-03-21

---

## 1. FP4 Fully Quantized Training (arXiv 2505.19115)

**Paper:** "FP4 All the Way: Fully Quantized Training of LLMs" (May 2025)

**How it works:**
- First demonstration of fully quantized training (FQT) using predominantly FP4 (4-bit floating point) for weights, activations, AND gradients simultaneously.
- Uses the NVFP4 format: E2M1 data (4 bits) with block size 16, sharing an E4M3 scale factor per block. Effectively ~4.5 bits per value including the amortized scale overhead.
- Stochastic rounding for backward and update passes; round-to-nearest for forward pass.
- Successfully trained a 7B parameter model on 200B tokens across 256 Intel Gaudi2 accelerators, matching BF16 downstream task performance.

**Memory per parameter (estimated breakdown for FP4 training):**
The paper itself does not provide an explicit bytes-per-parameter table. However, based on the approach:
- FP4 weights: 0.5 bytes/param
- FP4 gradients: 0.5 bytes/param
- Optimizer states: The paper does NOT quantize optimizer states to FP4. A mixed-precision strategy is used: gradients and first-order moments in FP8 (~1 byte each), second-order moments in FP16 (~2 bytes). So optimizer states are still ~3-4 bytes/param at minimum.
- **Total estimate: ~4.5-5.5 bytes/param** (much less than BF16's 16 bytes, but optimizer states remain the bottleneck).

**Key limitation:** A theoretical threshold exists -- when gradient norm falls below ~sqrt(3) times the quantization noise, FP4 training becomes unreliable. This matters for late-stage fine-tuning.

**Relevance to ternary networks (4-byte budget):**
- FP4 training reduces weight/gradient memory but still needs FP8/FP16 optimizer states. The approach cannot directly be adapted for ternary networks since ternary weights are discrete {-1,0,1}, not continuous FP4 values.
- The stochastic rounding insight is transferable: it shows that random rounding preserves gradient information in expectation even at extremely low precision.
- The total memory is close to but likely exceeds the 4-byte-extra budget when optimizer states are included.

**Related paper -- Quartet (arXiv 2505.14669):**
- Another FP4 training paper achieving ~2x speedup over FP8 on NVIDIA Blackwell (RTX 5090).
- Uses MXFP4 format with Hadamard transforms before quantization.
- Also does not quantize optimizer states to 4-bit; focuses on compute speedup rather than memory.
- No explicit memory-per-parameter comparison provided.

Sources:
- [FP4 All the Way (arXiv 2505.19115)](https://arxiv.org/abs/2505.19115)
- [Quartet (arXiv 2505.14669)](https://arxiv.org/abs/2505.14669)
- [FP4 Training HTML](https://arxiv.org/html/2505.19115v2)

---

## 2. 8-bit Optimizers (bitsandbytes)

**Paper:** "8-bit Optimizers via Block-wise Quantization" (arXiv 2110.02861, Dettmers et al.)

**How it works:**
- Block-wise dynamic quantization of Adam optimizer states (momentum and variance) from FP32 to INT8.
- Input tensors divided into independent blocks; each block quantized separately with its own normalization constant.
- Dynamic exponent data types map more values to small magnitudes (common in optimizer states).
- Maintains full 32-bit performance with 8-bit states on all tested benchmarks.

**Memory per parameter:**

| Configuration | Weights | Gradients | Momentum | Variance | Master Copy | Total |
|---|---|---|---|---|---|---|
| Standard BF16 + FP32 Adam | 2 | 2 | 4 | 4 | 4 | **16 bytes** |
| BF16 + 8-bit Adam (bnb) | 2 | 2 | 1 | 1 | 4 | **10 bytes** |
| BF16 + 8-bit Adam (no master) | 2 | 2 | 1 | 1 | 0 | **6 bytes** |

- The key savings: optimizer states drop from 8 bytes to 2 bytes per parameter.
- The FP32 master copy of weights (4 bytes) is still typically maintained for mixed-precision stability.
- Total savings: 37.5% reduction (16 -> 10 bytes) or 62.5% (16 -> 6 bytes without master copy).

**4-bit optimizer states (arXiv 2309.01507):**
- Pushes optimizer states from 8-bit to 4-bit using row-wise and column-wise quantization with smaller block sizes.
- Achieves comparable accuracy to full-precision optimizers on NLU, MT, image classification, and instruction tuning.
- Estimated total: ~2+2+0.5+0.5+4 = **9 bytes/param** (with FP32 master copy) or **5 bytes/param** (without).

**Relevance to ternary networks (4-byte budget):**
- 8-bit Adam alone does NOT fit the 4-byte budget because it still needs latent weights + master copy.
- However, if we could eliminate the latent weight / master copy requirement (i.e., update ternary weights directly), then 8-bit momentum (1 byte) + 8-bit variance (1 byte) = 2 bytes of optimizer state, which fits comfortably within 4 bytes.
- The question is whether Adam makes sense for discrete ternary weights at all, or if a simpler optimizer (like momentum-only SGD) suffices.

Sources:
- [8-bit Optimizers (arXiv 2110.02861)](https://arxiv.org/abs/2110.02861)
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
- [Memory Efficient Optimizers with 4-bit States (arXiv 2309.01507)](https://arxiv.org/abs/2309.01507)
- [8-bit Adam on Kaggle](https://www.kaggle.com/code/nbroad/8-bit-adam-optimization)

---

## 3. BitNet b1.58 2B4T Training Details (arXiv 2504.12285)

**Paper:** "BitNet b1.58 2B4T Technical Report" (April 2025, Microsoft Research)

**Architecture:**
- 2B parameters trained from scratch on 4 trillion tokens.
- Standard Transformer with BitLinear layers replacing nn.Linear.
- Ternary weights {-1, 0, 1} via absmean quantization during forward pass.
- 8-bit activations via absmax per-token quantization.

**Training approach:**
- Master weights maintained in **BF16** precision throughout training.
- Straight-Through Estimator (STE) used for backpropagation through quantization.
- Latent/shadow BF16 weights accumulate gradient updates; ternary weights computed on-the-fly per forward pass.
- The BF16 master weight is published as "bitnet-b1.58-2B-4T (bf16): used for training only."

**Training memory per parameter (estimated):**

| Component | Bytes/Param |
|---|---|
| BF16 master/latent weights | 2 |
| BF16 gradients | 2 |
| FP32 optimizer master copy | 4 |
| FP32 first moment (m) | 4 |
| FP32 second moment (v) | 4 |
| **Total** | **~16** |

**What the paper does NOT disclose:**
- Specific optimizer used (likely AdamW but not stated).
- Optimizer state precision (likely FP32 standard).
- Training infrastructure details, compute cost, wall-clock time.
- Any memory optimization techniques used during training.
- The paper heavily focuses on inference benchmarks, not training efficiency.

**Key insight for our problem:**
- This confirms the 16 bytes/param training cost for state-of-the-art ternary LLMs.
- The model achieves competitive performance vs FP16 models of similar size, validating that ternary networks CAN match full-precision at scale.
- The training approach is entirely standard STE + Adam; no innovation on training memory efficiency.

Sources:
- [BitNet b1.58 2B4T (arXiv 2504.12285)](https://arxiv.org/abs/2504.12285)
- [BitNet b1.58 2B4T HTML](https://arxiv.org/html/2504.12285v1)
- [Model on HuggingFace](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

---

## 4. Discrete Optimization / Gradient Accumulation for Quantized Weights

### 4a. Direct Quantized Training (DQT) with Stochastic Rounding (arXiv 2412.04787)

**How it works:**
- Eliminates high-precision shadow/latent weights entirely.
- Directly updates quantized low-precision weights during backpropagation.
- Uses stochastic rounding (SR) to preserve gradient information: `SR(x) = floor(x)` with probability `ceil(x)-x`, else `ceil(x)`.
- When optimizer produces W', DQT applies: `W_tilde = SR(W')`, keeping weights in n-bit format without separate re-quantization.
- Stochastic rounding is unbiased: in expectation, it preserves the true gradient signal even when individual updates are too small to change the quantized value.

**Performance:**
- Ternary DQT: converges, but performance gap vs higher-precision models exists.
- 8-bit DQT: matches BitNet b1.58 accuracy with only ~5% relative degradation.
- 8-bit DQT with ternary inference: performance slightly decreases vs 8-bit inference but remains on par with BitNet.
- WikiText-2 (1B, FineWeb): DQT 8-bit = 25.43 ppl vs BitNet = 28.20 ppl; ternary inference = 27.32 ppl.

**Memory per parameter:**
- DQT ternary weights: ~0.2 bytes/param (the weights themselves).
- Gradients: precision depends on environment (BF16 = 2 bytes, FP8 = 1 byte).
- Optimizer states: standard AdamW still maintains "two high-precision states (momentum and variance)." Experiments also tried Adafactor which eliminates full state storage.
- GPU measurements (1B model): FP32 environment = 76,533 MB; BF16 environment = 58,345 MB.
- **Critical insight:** DQT eliminates the 2-4 byte master weight copy, but if you still use Adam, you still need 8 bytes for optimizer states. The real savings come from dropping the master copy.

**Estimated memory with DQT + 8-bit Adam:**
| Component | Bytes/Param |
|---|---|
| 8-bit quantized weights | 1 |
| BF16 gradients | 2 |
| 8-bit momentum | 1 |
| 8-bit variance | 1 |
| **Total** | **~5** |

**With DQT ternary + SGD-momentum (no Adam):**
| Component | Bytes/Param |
|---|---|
| Ternary weights | 0.2 |
| BF16 gradients | 2 |
| FP16 momentum accumulator | 2 |
| **Total** | **~4.2** |

This is very close to the 4-byte budget target!

### 4b. Zeroth-Order Optimization for Quantized Networks (arXiv 2505.13430)

- Fine-tuning quantized NNs using zeroth-order (gradient-free) methods.
- Avoids backpropagation entirely; estimates gradients via function evaluations.
- Potentially relevant for discrete weights since it doesn't require differentiability.
- However, convergence is much slower than gradient-based methods; impractical for pretraining LLMs from scratch.

Sources:
- [DQT with Stochastic Rounding (arXiv 2412.04787)](https://arxiv.org/abs/2412.04787)
- [DQT HTML version](https://arxiv.org/html/2412.04787v2)
- [Fine-tuning Quantized NNs with Zeroth-order (arXiv 2505.13430)](https://arxiv.org/html/2505.13430)

---

## 5. Sign-Based Optimization for Discrete Weight Networks

### 5a. signSGD with Momentum

**How it works:**
- Only uses the SIGN of the stochastic gradient to update weights.
- Update rule: `w_{t+1} = w_t - lr * sign(g_t)`
- With momentum: `m_t = beta * m_{t-1} + (1-beta) * g_t; w_{t+1} = w_t - lr * sign(m_t)`
- Compresses gradient communication to 1 bit per parameter.
- With momentum, matches Adam accuracy and convergence speed on deep ImageNet models.
- Convergence proven under weaker assumptions when momentum is used (no bounded gradient requirement, works with small batches).

**Memory per parameter:**
- signSGD (no momentum): 0 bytes extra (just compute sign and apply).
- signSGD with momentum: 4 bytes (FP32 momentum) or 2 bytes (FP16 momentum) or 1 byte (INT8 momentum).
- No second moment (variance) needed, unlike Adam.

**Relevance to ternary networks:**
- signSGD is naturally aligned with discrete weight updates. The sign of accumulated gradients is exactly the type of signal needed to decide whether to flip a ternary weight up or down.
- The momentum buffer acts as a "gradient accumulator" that remembers the direction of recent gradients.
- A ternary-adapted signSGD would: accumulate gradients in momentum, then flip weights when momentum exceeds a threshold (combining signSGD ideas with Bop's threshold mechanism).

### 5b. Binary Optimizer (Bop) -- arXiv 1906.02107

**How it works:**
- First optimizer designed specifically for binary neural networks.
- Maintains an exponential moving average (EMA) of gradients per parameter: `m_t = gamma * m_{t-1} + (1 - gamma) * g_t`
- Flips a binary weight when `|m_t| > threshold`.
- Key insight: latent weights in standard BNN training are NOT analogous to real weights. Their true role is providing **inertia** -- they resist weight flips until sufficient gradient evidence accumulates.
- Bop makes this explicit: the EMA IS the inertia, and flipping happens when evidence exceeds threshold.
- **"Latent-free"**: no full-precision weight copy maintained.

**Memory per parameter:**
- Binary weight: 1 bit (~0.125 bytes).
- EMA of gradients: FP32 = 4 bytes (or FP16 = 2 bytes, INT8 = 1 byte if quantized).
- No second moment (no variance estimate).
- **Total: ~4.125 bytes** (with FP32 EMA) or **~2.125 bytes** (with FP16 EMA).

**This fits the 4-byte budget!**

**Limitations:**
- Only validated on CIFAR-10 (91.3%) and ImageNet (Bi-Real Net: 56.6% top-1) -- small-scale vision, not LLMs.
- Designed for binary {-1, +1}, not ternary {-1, 0, +1}. Extending to ternary requires a transition mechanism for the zero state.
- Hyperparameter sensitivity: threshold and gamma require careful tuning.

### 5c. SGDAT (SGD with Adaptive Threshold)

**How it works:**
- Suppresses frequency of weight flipping via adaptive thresholds.
- Each parameter's threshold is adjusted based on its flip history.
- Weights that flip too frequently get higher thresholds (more conservative).
- Significantly improves vanilla SGD performance in BNNs to be comparable to Adam.

**Relevance:**
- The adaptive threshold idea could be combined with Bop-style EMA for ternary networks.
- Prevents oscillation between ternary states -- a likely failure mode for naive discrete optimizers.

### 5d. Probabilistic Optimizer for BNNs (2025)

**How it works:**
- Interprets accumulated gradients as "potential energy" for each neuron.
- Formulates a Bernoulli probability distribution based on gradient magnitude to decide weight flips.
- Probabilistic flipping (rather than deterministic threshold) adds beneficial stochasticity.
- Measures implemented to minimize sign reversals in weights that frequently flip.

**Relevance:**
- Probabilistic weight updates naturally extend to ternary: instead of binary Bernoulli, use a 3-state categorical distribution over {-1, 0, +1}.
- Could be combined with stochastic rounding ideas from DQT.
- Addresses the instability problem when accumulated gradients are near the flip threshold.

Sources:
- [signSGD (arXiv 1802.04434)](https://arxiv.org/abs/1802.04434)
- [Momentum Ensures Convergence of signSGD](https://proceedings.mlr.press/v202/sun23l.html)
- [Bop / Latent Weights Do Not Exist (arXiv 1906.02107)](https://arxiv.org/abs/1906.02107)
- [Bop in Larq](https://docs.larq.dev/larq/api/optimizers/)
- [SGDAT (Neurocomputing 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005544)
- [Probabilistic Optimizer for BNNs (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009695)
- [Bop implementation](https://github.com/plumerai/rethinking-bnn-optimization)

---

## Summary: Memory Budget Comparison

| Method | Weights | Gradients | Optimizer States | Total/Param | Fits 4-byte extra? |
|---|---|---|---|---|---|
| Standard STE + Adam (BF16) | 2 B | 2 B | 12 B (master+m+v) | **16 B** | NO |
| STE + 8-bit Adam | 2 B | 2 B | 6 B (master+m8+v8) | **10 B** | NO |
| DQT ternary + Adam | 0.2 B | 2 B | 8 B (m+v FP32) | **~10 B** | NO |
| DQT ternary + SGD-momentum (FP16) | 0.2 B | 2 B | 2 B (momentum) | **~4.2 B** | BORDERLINE |
| Bop-style EMA (FP32) | 0.125 B | 0* | 4 B (EMA) | **~4.1 B** | YES (barely) |
| Bop-style EMA (FP16) | 0.125 B | 0* | 2 B (EMA) | **~2.1 B** | YES |
| Ternary Bop (FP16 EMA) | 0.2 B | 0* | 2 B (EMA) | **~2.2 B** | YES |
| signSGD + INT8 momentum | 0.2 B | 0* | 1 B (momentum) | **~1.2 B** | YES |

*Bop/signSGD approaches compute gradients but don't need to store them persistently -- the EMA/momentum update happens in-place during backprop.

**Note:** Gradient memory (2 bytes) is technically transient -- it's computed and consumed layer-by-layer during backprop, so it doesn't need to persist for all parameters simultaneously. The "extra memory per parameter" that matters for the budget is primarily the optimizer state.

---

## Key Takeaways for the 4-Byte Budget

1. **Bop-style approaches are the most promising starting point.** A ternary extension of Bop with FP16 or INT8 EMA fits well within the 4-byte budget (~2 bytes extra). The challenge is scaling this to LLMs and adapting from binary to ternary.

2. **DQT's stochastic rounding is a critical technique.** It eliminates the master weight copy entirely by directly updating quantized weights. Combined with a lightweight optimizer (SGD+momentum instead of Adam), it approaches the 4-byte target.

3. **Adam is overkill for discrete weights.** The second moment (variance) is designed for adaptive learning rates in continuous optimization. For ternary weight flipping decisions, momentum alone (first moment) provides sufficient signal. This saves 4+ bytes/param.

4. **FP4 training is NOT directly applicable** to ternary networks. FP4 is about reducing precision of continuous values, not about discrete optimization. However, the stochastic rounding and low-precision gradient handling techniques are transferable.

5. **The gradient storage question is key.** If gradients can be consumed in-place (accumulated into momentum during backprop), we avoid the 2-byte gradient storage overhead entirely, making the budget much easier to meet.

6. **Hybrid approaches are unexplored.** No paper combines DQT's stochastic rounding with Bop's threshold-based flipping for ternary LLMs. This is the most promising research direction.

---

## 6. Error Feedback and Error Compensation in Optimization

### 6a. Error Feedback Fixes SignSGD (Karimireddy et al., ICML 2019)

**Paper:** "Error Feedback Fixes SignSGD and other Gradient Compression Schemes"
**Authors:** Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian U. Stich, Martin Jaggi (EPFL)
**Year:** 2019 (ICML)

**Key results:**

1. **SignSGD diverges on convex problems.** The paper provides simple convex counter-examples where signSGD (biased gradient compression without error feedback) does not converge to the optimum. Even when it does converge, it may generalize poorly compared to SGD.

2. **Error feedback (EF) fixes convergence.** The EF-SGD algorithm:
   - Compute gradient g_t
   - Form compensated gradient: p_t = g_t + e_t (add accumulated error)
   - Compress: c_t = C(p_t) (apply contractive compressor)
   - Update weights: w_{t+1} = w_t - eta * c_t
   - Update error: e_{t+1} = p_t - c_t (store what was lost)

3. **Contractive compressor definition:** An operator C is a delta-contraction if: `||x - C(x)||^2 <= (1 - delta)||x||^2` for all x, with delta in (0, 1]. This includes sign compression, top-k sparsification, and quantization.

4. **Main convergence theorem:** EF-SGD with *any* contractive compressor achieves the *same asymptotic convergence rate* as uncompressed SGD, without additional assumptions. The compression is essentially "free" in terms of convergence.

5. **Key insight for ternary training:** Error feedback works because the error term e_t is bounded (it can only be as large as the compression distortion) and is systematically corrected in subsequent iterations. The errors telescope: each step's lost information is recovered in the next.

**Relevance to ternary quantization:** The ternary quantization function Q(w) = round_to_{-1,0,+1}(w) is a contractive compressor. The quantization error e_t = w - Q(w) is bounded by the quantization step size. Error feedback theory says we can accumulate this error and inject it to maintain convergence. This is the theoretical foundation for approaches like ECO.

**Limitations noted:** The classical EF mechanism compresses (gradient + error), which is a potentially large vector that doesn't vanish even at convergence, because individual worker gradients need not be zero at the optimum. This leads to perpetual distortion.

Sources:
- [Error Feedback Fixes SignSGD (arXiv 1901.09847)](https://arxiv.org/abs/1901.09847)
- [ICML 2019 proceedings](https://proceedings.mlr.press/v97/karimireddy19a.html)

---

### 6b. Sparsified SGD with Memory (Stich et al., 2018/2019)

**Paper:** "Sparsified SGD with Memory"
**Authors:** Sebastian U. Stich, Jean-Baptiste Cordonnier, Martin Jaggi (EPFL)
**Year:** 2018 (arXiv), later published in JMLR

**Key results:**

1. **k-contraction operator:** `E[||x - comp(x)||^2] <= (1 - k/d)||x||^2` where k is the number of coordinates retained and d is dimension. Top-k and random-k are examples.

2. **Algorithm (sparsification with memory):**
   - Sparsify: g_t = comp_k(m_t + eta_t * grad_f(x_t))
   - Update: x_{t+1} = x_t - g_t
   - Accumulate error: m_{t+1} = m_t + eta_t * grad_f(x_t) - g_t

3. **Convergence rate (strongly convex):**
   `E[f(x_bar_T) - f*] = O(G^2 / (mu*T)) + O(d^2/k^2 * G^2*kappa / (mu*T^2)) + O(d^3/k^3 * G^2 / (mu*T^3))`

   The dominant term O(G^2/(mu*T)) matches vanilla SGD. After T = Omega(d/k * sqrt(kappa)) iterations, sparsification doesn't hurt the rate at all.

4. **Without error compensation:** Unbiased k-sparsification requires d/k times more iterations. The variance blows up by factor d/k. Error compensation recovers this loss.

**Relevance to ternary training:** This shows that even extreme compression (keeping only k out of d coordinates) can achieve the same convergence as full SGD when error compensation is used. Ternary quantization is another form of compression, and the same memory/error-feedback mechanism applies.

Sources:
- [Sparsified SGD with Memory (arXiv 1809.07599)](https://arxiv.org/abs/1809.07599)

---

### 6c. Error Compensated Quantized SGD (Wu et al., ICML 2018)

**Paper:** "Error Compensated Quantized SGD and its Applications to Large-scale Distributed Optimization"
**Authors:** Jiaxiang Wu et al.
**Year:** 2018 (ICML)

**Key results:**

1. **Error compensation update rule with decay:**
   - h_p^(t) = beta * h_p^(t-1) + (g_p^(t-1) - g_tilde_p^(t-1))   [error accumulation with decay beta]
   - g_tilde_p^(t) = Q(g_p^(t) + alpha * h_p^(t))   [quantize with compensation coefficient alpha]

2. **Contraction property for stability:** The key condition is:
   `lambda = alpha^2 * gamma + (beta - alpha)^2 < 1`
   This ensures accumulated error doesn't explode. The decay rate of historical quantization errors is:
   `nu = (beta - alpha) / (1 - eta * a_1) < 1`

3. **Key result (Lemma 4):** Historical quantization errors decay exponentially:
   `lim_{(t-t') -> inf} tau^(t') / tau_QSGD^(t') = 0`
   ECQ-SGD achieves strictly tighter error bounds than QSGD (quantized SGD without error compensation).

4. **Two hyperparameters for error control:** alpha (compensation strength) and beta (error decay). These provide control over how aggressively past errors are reinjected and how quickly they are forgotten.

**Relevance to ternary training:** This paper introduces the idea of *damped* error feedback (with decay factor beta), which is directly relevant to stabilizing error compensation for ternary weight quantization. The alpha/beta parameters give knobs to control the error injection strength, preventing the instability that can arise from naive error feedback.

Sources:
- [ECQ-SGD (arXiv 1806.08054)](https://arxiv.org/abs/1806.08054)
- [ICML 2018 proceedings](https://proceedings.mlr.press/v80/wu18d.html)

---

### 6d. EF21: A New, Simpler, Theoretically Better Error Feedback (Richtarik et al., NeurIPS 2021)

**Paper:** "EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback"
**Authors:** Peter Richtarik, Igor Sokolov, Ilyas Fatkhullin (KAUST)
**Year:** 2021 (NeurIPS)

**Key results:**

1. **Classical EF update rule:**
   ```
   e_i^{t+1} = e_i^t + gamma * grad_f_i(x^t) - w_i^t
   w_i^{t+1} = C(e_i^{t+1} + gamma * grad_f_i(x^{t+1}))
   ```
   Problem: Compresses (accumulated error + scaled gradient), a potentially large vector that doesn't vanish.

2. **EF21 update rule (the key innovation):**
   ```
   g_i^{t+1} = g_i^t + C(grad_f_i(x^{t+1}) - g_i^t)
   x^{t+1} = x^t - gamma * g^t
   ```
   EF21 compresses the *difference* between current and previous gradient estimates. This difference **progressively vanishes** as the method converges, reducing compression distortion over iterations.

3. **Contractive compressor:** `E[||C(x) - x||^2] <= (1 - alpha)||x||^2` for alpha in (0, 1].

4. **Error contraction lemma (Lemma 2):**
   `E[M_i^{t+1} | W^t] <= (1 - theta) * G_i^t + beta * ||grad_f_i(x^{t+1}) - grad_f_i(x^t)||^2`
   where theta = 1 - sqrt(1-alpha). The error contracts by factor (1-theta) at each step.

5. **Convergence rates:**
   - **Nonconvex (Theorem 1):** `E[||grad_f(x_hat^T)||^2] <= 2(f(x^0) - f^inf)/(gamma*T) + E[G^0]/(theta*T)` -- this is O(1/T), beating the previous O(1/T^{2/3}) bound.
   - **PL functions (Theorem 2):** Linear rate: `E[Psi^T] <= (1 - gamma*mu)^T * E[Psi^0]` where the Lyapunov function is Psi^t = f(x^t) - f(x*) + (gamma/theta)*G^t.

6. **Why EF21 is more stable than classical EF:**
   - Classical EF compresses vectors that don't necessarily converge to zero.
   - EF21 compresses (grad_f_i(x^{t+1}) - g_i^t), which vanishes as the method converges.
   - This achieves "progressively vanishing distortion."
   - EF21 admits larger stepsizes and avoids the bounded gradient assumption.

**Relevance to ternary training:** The EF21 framework suggests that for ternary weight error compensation, we should not inject the raw quantization error. Instead, we should inject the *change* in quantization error between steps. This naturally dampens the error as training converges and the weights stabilize.

Sources:
- [EF21 (arXiv 2106.05203)](https://arxiv.org/abs/2106.05203)
- [NeurIPS 2021 proceedings](https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)

---

### 6e. Error Feedback Reloaded (Richtarik et al., ICLR 2024)

**Paper:** "Error Feedback Reloaded: From Quadratic to Arithmetic Mean of Smoothness Constants"
**Authors:** Peter Richtarik, Elnur Gasanov, Konstantin Burlachenko
**Year:** 2024 (ICLR)

**Key results:**

1. Improves EF21's theoretical communication complexity from depending on the *quadratic mean* of smoothness parameters to the *arithmetic mean*, which is always smaller and can be substantially smaller in heterogeneous data regimes.

2. The approach applies EF21 to an equivalent reformulation of the problem and discovers a new *weighted* version of EF21 that naturally extends to stochastic gradients and partial participation.

**Relevance:** Shows that the EF21 framework continues to yield tighter bounds with refined analysis. The weighted variant could be relevant when different layers of a ternary network have different quantization sensitivity.

Sources:
- [Error Feedback Reloaded (arXiv 2402.10774)](https://arxiv.org/abs/2402.10774)

---

### 6f. Momentum Provably Improves Error Feedback (Fatkhullin et al., NeurIPS 2023)

**Paper:** "Momentum Provably Improves Error Feedback!"
**Authors:** Ilyas Fatkhullin, Alexander Tyurin, Peter Richtarik
**Year:** 2023 (NeurIPS)

**Key results:**

1. Applies Polyak's momentum to EF21, creating EF21-SGDM.

2. Improves both communication and sample complexities of error feedback algorithms under standard smoothness and bounded variance assumptions.

3. Does NOT require assumptions like bounded gradient dissimilarity.

4. A double-momentum version further improves complexities.

5. **Critical finding:** Momentum acts as a smoothing/stabilizing force on the error feedback mechanism. Without momentum, compression errors can propagate and cause exponential divergence. Momentum dampens these oscillations.

**Relevance to ternary training:** This is directly applicable. If we use error feedback to compensate ternary quantization error, adding momentum to the scheme provably improves convergence. The momentum buffer is already part of our memory budget (it serves double duty as both the optimizer state and the error-smoothing mechanism).

Sources:
- [Momentum Provably Improves Error Feedback (arXiv 2305.15155)](https://arxiv.org/abs/2305.15155)
- [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)

---

### 6g. Linearly Converging Error Compensated SGD (Gorbunov et al., NeurIPS 2020)

**Paper:** "Linearly Converging Error Compensated SGD"
**Authors:** Eduard Gorbunov, Dmitry Kovalev, Dmitry Makarenko, Peter Richtarik
**Year:** 2020 (NeurIPS)

**Key results:**

1. Unified analysis framework covering quantized SGD, error-compensated SGD, and SGD with delayed updates.

2. Proposes EC-SGD-DIANA: combines error feedback (for biased compression) with variance reduction (DIANA technique for quantized gradient differences). This converges to the *exact* optimum with constant learning rate for convex and strongly convex objectives.

3. EC-LSVRG-DIANA: first distributed stochastic method with error feedback and variance reduction converging to exact optimum with constant learning rate.

**Relevance:** Shows that error compensation can be combined with variance reduction to achieve exact convergence (not just convergence to a neighborhood). For ternary weight training, this suggests combining error feedback with techniques that reduce the variance of the quantization noise.

Sources:
- [Linearly Converging EC-SGD (arXiv 2010.12292)](https://arxiv.org/abs/2010.12292)
- [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/ef9280fbc5317f17d480e4d4f61b3751-Abstract.html)

---

### 6h. EF-BV: Unified Error Feedback and Variance Reduction (Condat et al., NeurIPS 2022)

**Paper:** "EF-BV: A Unified Theory of Error Feedback and Variance Reduction Mechanisms for Biased and Unbiased Compression in Distributed Optimization"
**Authors:** Laurent Condat, Kai Yi, Peter Richtarik
**Year:** 2022 (NeurIPS)

**Key results:**

1. Unifies DIANA (variance reduction for unbiased compressors) and EF21 (error feedback for biased/contractive compressors) into a single framework.

2. Proposes a general algorithm with a new, larger class of compressors parameterized by two values: bias and variance.

3. Proves linear convergence under appropriate conditions.

**Relevance:** Provides theoretical grounding for choosing between biased (top-k, sign) and unbiased (stochastic rounding) compression for ternary weights, and shows they can be analyzed within the same framework.

Sources:
- [EF-BV (arXiv 2205.04180)](https://arxiv.org/abs/2205.04180)
- [NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/6fb9ea5197c0b8ece8a64220fb82cdfe-Abstract-Conference.html)

---

### 6i. Contractive Error Feedback (ConEF) (2023)

**Paper:** "Contractive Error Feedback for Gradient Compression"
**Year:** 2023

**Key results:**

1. Addresses a practical problem: standard error feedback (EFSGD) stores a full-precision error buffer, which costs significant memory.

2. ConEF applies compression *to the error buffer itself*, saving 80%-90% of the extra memory in EFSGD with almost no loss on test performance.

3. Achieves 1.3x-5x speedup over SGD across image classification, language modeling, and machine translation.

**Relevance to ternary training:** This is highly relevant. If we use error feedback for ternary weight quantization, the error buffer e_t is a full-precision vector (4 bytes/param). ConEF shows we can compress the error buffer too, potentially to 1 byte or less, freeing memory budget for other uses.

Sources:
- [Contractive Error Feedback (arXiv 2312.08538)](https://arxiv.org/abs/2312.08538)

---

### 6j. Decentralized SGD with Compressed Communication (Koloskova et al., ICML 2019)

**Paper:** "Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication"
**Authors:** Anastasia Koloskova, Sebastian U. Stich, Martin Jaggi (EPFL)
**Year:** 2019 (ICML)

**Key results:**

1. **CHOCO-SGD convergence rate:** `O(1/(nT) + 1/(T * rho^2 * delta)^2)` for strongly convex objectives, where delta is compression quality and rho is the spectral gap of the connectivity matrix.

2. Despite compression affecting higher-order terms, the leading term O(1/(nT)) matches the centralized exact-communication baseline -- compression is asymptotically free.

3. Compression quality delta in (0, 1] where delta=1 means no compression. Supports both biased and unbiased compressors.

**Relevance:** Confirms the general principle that error-compensated compression preserves asymptotic convergence rates. The framework applies to our weight quantization setting.

Sources:
- [CHOCO-SGD (arXiv 1902.00340)](https://arxiv.org/abs/1902.00340)
- [ICML 2019 proceedings](https://proceedings.mlr.press/v97/koloskova19a.html)

---

## 7. 1-bit Adam and 1-bit LAMB

### 7a. 1-bit Adam (Tang et al., 2021)

**Paper:** "1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed"
**Authors:** Hanlin Tang, Shaoduo Gan, Shengtuo Hu, Xiangru Lian, Ji Liu, Ce Zhang, and others (Microsoft, University of Rochester)
**Year:** 2021

**Key results:**

1. **Why error compensation fails with Adam:** State-of-the-art error compensation techniques only work with optimizers that are *linearly dependent* on gradients (SGD, momentum SGD). Adam's variance term v_t is non-linearly dependent on gradient g_t. When basic error compensation is applied to Adam:
   - The compressed term becomes (g_t + delta_{t-1} - delta_t)^2
   - The quadratic error term (delta_{t-1} - delta_t)^2 doesn't cancel
   - This creates irrecoverable noise in the variance estimation, causing convergence failure

2. **Two-phase solution:**
   - **Warmup phase (T_w steps):** Run standard uncompressed Adam. The variance v_t stabilizes (empirically observed to stabilize after ~23K steps on BERT-Large).
   - **Compression phase:** Freeze v = v_{T_w} as a fixed preconditioner. Now the update is effectively momentum SGD with coordinate-wise scaling, which IS linearly dependent on gradients. Apply 1-bit compression with error compensation.

3. **Error compensation in compression phase:**
   - Compress momentum: m_hat_t = Compress(m_t + delta_{t-1})
   - Update error: delta_t = m_t + delta_{t-1} - m_hat_t
   - Model update: x_{t+1} = x_t - gamma * m_t / sqrt(v_{T_w})

4. **Convergence:** O(1/sqrt(nT)) rate, matching uncompressed distributed SGD.

5. **Practical results:** Up to 5x communication reduction and 3.3x throughput improvement on BERT-Large.

**Relevance to ternary training:** The key insight is that error compensation is fundamentally incompatible with non-linear optimizer operations. For ternary weight training:
- If using Adam, the variance term v_t interacts non-linearly with quantization errors, potentially causing divergence.
- Using momentum-only SGD (which IS linear in gradients) is more compatible with error compensation for ternary quantization.
- The warmup-then-freeze pattern could be adapted: train with full precision initially, then transition to error-compensated ternary training after optimizer states stabilize.

Sources:
- [1-bit Adam (arXiv 2102.02888)](https://arxiv.org/abs/2102.02888)
- [DeepSpeed blog post](https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html)

### 7b. 1-bit LAMB (Li et al., 2021)

**Paper:** "1-bit LAMB: Communication Efficient Large-Scale Large-Batch Training with LAMB's Convergence Speed"
**Authors:** Conglong Li et al.
**Year:** 2021

**Key results:**

1. Extends the 1-bit compression approach to the LAMB optimizer for large-batch training.

2. Same two-phase strategy: warmup (uncompressed LAMB) then compression (with frozen variance as preconditioner).

3. Achieves up to 4.6x communication volume reduction, 2.8x end-to-end speedup on BERT-Large with batch sizes 8K-64K.

4. Same sample-wise convergence speed as uncompressed LAMB.

**Relevance:** Confirms the generality of the warmup-then-compress pattern across different optimizers.

Sources:
- [1-bit LAMB (arXiv 2104.06069)](https://ar5iv.labs.arxiv.org/html/2104.06069)

---

## 8. DoReFa-Net and XNOR-Net: Early Quantized Training

### 8a. DoReFa-Net (Zhou et al., 2016)

**Paper:** "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
**Authors:** Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, Yuheng Zou
**Year:** 2016

**Key results:**

1. **Quantizes weights, activations, AND gradients** to low bitwidth during training.

2. **Quantization error handling:**
   - Weights: deterministic quantization to produce low bitwidth values.
   - Gradients: *stochastic* quantization is necessary for low bitwidth gradients to be effective. Gradients have unbounded range, so DoReFa first applies an affine transform to map them to [-1, 1], then quantizes, then inverts the transform.
   - Activations: deterministic quantization.

3. **Full-precision master weights are maintained** throughout training. The quantized weights are derived from master weights on-the-fly during the forward pass.

4. **STE is used** for backpropagation through the non-differentiable quantization functions.

**Relevance to ternary training:** DoReFa-Net demonstrates that stochastic rounding is essential for gradient quantization (deterministic fails). It also shows the standard pattern: maintain full-precision master weights, quantize on forward pass, STE on backward pass. Our goal is precisely to break this pattern for ternary networks.

Sources:
- [DoReFa-Net (arXiv 1606.06160)](https://arxiv.org/abs/1606.06160)

### 8b. XNOR-Net (Rastegari et al., ECCV 2016)

**Paper:** "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"
**Authors:** Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
**Year:** 2016 (ECCV)

**Key results:**

1. Binarizes both weights and activations to {-1, +1}.

2. **Filter-wise scaling factor:** To compensate for magnitude loss from binarization, XNOR-Net introduces a full-precision scaling factor alpha per filter, computed as the mean absolute value of the full-precision weights: alpha = (1/n)||W||_L1.

3. **Approximation:** W approx alpha * sign(W). The binary XNOR operations compute the sign, then the scaling factor restores magnitude.

4. **Activation binarization** with a separate per-channel scaling factor.

5. **STE used** for gradient computation through the sign function.

**Relevance to ternary training:** The per-filter scaling factor idea is exactly what BitNet b1.58 uses (absmean quantization). The key observation is that ternary/binary quantization destroys magnitude information, which must be restored by a continuous scaling factor. This scaling factor itself has negligible memory cost (one float per output channel, not per parameter).

Sources:
- [XNOR-Net (arXiv 1603.05279)](https://pjreddie.com/static/papers/xnor.pdf)
- [ECCV 2016 proceedings](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_32)

---

## 9. Trained Ternary Quantization (TTQ)

**Paper:** "Trained Ternary Quantization"
**Authors:** Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally
**Year:** 2017 (ICLR)

**Key results:**

1. Ternary quantization with *learned* scaling factors. Two separate scaling factors for positive and negative weights: w_q in {-W_n, 0, +W_p}.

2. **Threshold-based quantization:** A threshold Delta determines which weights become zero. Weights with |w| > Delta become +/-1 (scaled by W_p or W_n).

3. **Both Delta and the scaling factors are optimized** during training via backpropagation (using STE).

4. **Sparsity sweet spot:** As sparsity grows from 0 to 0.5, both training and validation error decrease. Beyond 50% sparsity, model capacity drops too much.

5. **Performance:** Ternary models outperform full-precision on ResNet-32/44/56 on CIFAR-10 by 0.04-0.36%. On ImageNet, TTQ AlexNet outperforms full-precision by 0.3%.

**Relevance to ternary training:** TTQ confirms that learned scaling factors and thresholds are important for ternary quantization quality. However, TTQ still maintains full-precision master weights. The insight about optimal sparsity (~50% zeros) is relevant for understanding the ternary weight distribution.

Sources:
- [TTQ (arXiv 1612.01064)](https://arxiv.org/abs/1612.01064)
- [ICLR 2017](https://openreview.net/forum?id=S1_pAu9xl)

---

## 10. Quantization Noise Models

**Mathematical characterization of quantization noise:**

1. **Additive noise model (classical):** Quantization error is modeled as additive white noise, independent of the signal, with uniform distribution over [-Delta/2, Delta/2] where Delta is the quantization step size. Variance = Delta^2/12.

2. **For uniform quantizers:** The additive noise model is valid when the input signal is smooth and spans many quantization levels. This is the standard model in signal processing.

3. **For ternary quantization {-1, 0, +1} with scaling factor alpha:**
   - The quantization error is: e = w - alpha * Q(w/alpha) where Q rounds to {-1, 0, +1}
   - The error is NOT uniformly distributed -- it is signal-dependent
   - Weights near the decision boundaries (e.g., near alpha/2 where Q transitions from 0 to 1) have larger errors than weights near quantization points
   - The error magnitude is bounded by alpha/2 (half the quantization step)
   - The variance depends on the weight distribution: for weights uniformly distributed over [-alpha, alpha], Var(e) = alpha^2/12. But real weight distributions are typically approximately Gaussian, making the actual variance distribution-dependent.

4. **BitNet b1.58 specific:** Absmean quantization sets alpha = mean(|w|). This makes the quantization error directly proportional to the weight distribution's scale. Ternary quantization destroys magnitude information (a weight of 0.003 and 0.847 both become +1), and the weight_scale tensor (one BF16 value per output channel) restores it.

5. **Key distinction from gradient compression:** In gradient compression (the setting of most error feedback papers), the quantized values change every iteration (because gradients change). In weight quantization for ternary training, the quantized values are the WEIGHTS themselves, which change slowly. This means the quantization error changes slowly too, which is favorable for error feedback (the "consecutive errors are similar" assumption holds well).

Sources:
- [Quantization (signal processing) - Wikipedia](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
- [BitNet b1.58 ternary math analysis](https://dev.to/ramr007/the-mathematics-that-make-158-bit-weights-work-how-bitnet-b158-survives-its-own-quantization-3901)

---

## 11. Thermometer and Alternative Codings for Ternary Weights

### Thermometer Encoding

**Concept:** Instead of representing a value as a single number, thermometer encoding represents it as a binary vector where progressively more bits are activated as the value increases. For a value x in [0, k], the encoding is a k-dimensional binary vector [1,1,...,1,0,...,0] with floor(x) ones.

**Applications in neural networks:**
1. **Generic Learned Thermometer (GLT):** Learns non-linear quantization thresholds for binary neural network inputs, improving data representation.
2. **Adversarial robustness:** Thermometer encoding makes models more robust by spreading information across bits, making it harder for adversarial perturbations to cause misclassification.
3. **Weightless Neural Networks:** Distributive thermometer encoding is the state-of-the-art for WNNs.

**For ternary weights specifically:**
The three states {-1, 0, +1} could be represented as 2-bit thermometer codes:
- -1 -> [0, 0]
- 0 -> [1, 0]
- +1 -> [1, 1]

This makes transitions between adjacent states differ by exactly one bit flip, potentially making gradient-based updates more stable (changing from 0 to +1 only requires flipping one bit, not changing a whole value).

**Relevance:** Thermometer coding could make error compensation more stable by ensuring that quantization errors from adjacent states are small and well-structured. However, no existing work applies this specifically to ternary weight training with error feedback.

Sources:
- [Thermometer Encoding (ICLR 2018)](https://colinraffel.com/publications/iclr2018thermometer.pdf)
- [Generic Learned Thermometer (arXiv 2505.13462)](https://arxiv.org/abs/2505.13462)
- [TBN: Ternary Inputs and Binary Weights (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Diwen_Wan_TBN_Convolutional_Neural_ECCV_2018_paper.pdf)

---

## 12. ECO: Error-Compensating Optimizer (Nikdan et al., 2026)

**Paper:** "ECO: Quantized Training without Full-Precision Master Weights"
**Authors:** Mahdi Nikdan, Amir Zandieh, Dan Alistarh, Vahab Mirrokni (Google Research, ISTA)
**Year:** January 2026 (arXiv 2601.22101)

This is the most directly relevant paper to our research problem.

**Core idea:** Eliminate master weights by applying updates directly to quantized parameters, then injecting the resulting quantization error into the optimizer momentum to form an error-feedback loop with no additional memory.

**ECO update rules (SGD with Momentum):**
```
m_tilde_{t+1} = beta * m_hat_t + (1-beta) * grad_f(theta_hat_t)    # standard momentum
theta_tilde_{t+1} = theta_hat_t - eta * m_tilde_{t+1}                # parameter update
theta_hat_{t+1} = q(theta_tilde_{t+1})                               # quantize
e_{t+1} = theta_tilde_{t+1} - theta_hat_{t+1}                        # quantization error
m_hat_{t+1} = m_tilde_{t+1} + alpha * e_{t+1}                        # inject error into momentum
```

**Error injection coefficient derivation:**
`alpha = (1/eta) * (1 - 1/beta)`

This coefficient is derived by requiring that the ECO trajectory matches the master-weight SGDM trajectory exactly. The exact matching requires injecting m <- m + (1/eta)*e_t - (1/(eta*beta))*e_{t+1}. By using the heuristic that consecutive quantization errors are similar (e_t approx e_{t+1}), this simplifies to the memory-efficient single-coefficient form.

**Convergence theorem (Theorem 3.8):**
```
min_{t in {0,...,T-1}} E[||grad_f(theta_hat_t)||^2] <= 4*(f(theta_0) - f*) / (eta*T) + sigma^2_quant
```

where the quantization noise floor is:
```
sigma^2_quant = 4*eta^2*beta^2*L^2*G^2 / (1-beta)^2 + 4*L^2*sigma^2 / (1-beta^2)
```

As eta -> 0: sigma^2_quant -> 4*L^2*sigma^2 / (1-beta^2)

**Critical lower-bound analysis (quadratic objective):**
- **Master weights:** lim_{eta->0} Loss_MW = L^2 * sigma^2
- **Naive removal (no error compensation):** Loss_Naive proportional to 1/eta -> INFINITY (diverges!)
- **ECO:** lim_{eta->0} Loss_ECO = L^2 * sigma^2 / (1-beta^2)

ECO prevents the catastrophic 1/eta divergence of naive master-weight removal while maintaining only a bounded constant factor (1/(1-beta^2)) worse than full precision.

**Momentum boundedness (Lemma 3.7):**
```
E[||m_hat_t||^2] <= 2*G^2 + 2*alpha^2*sigma^2 / (1-beta^2) = M^2
```
This ensures quantization errors injected into momentum don't explode.

**Consecutive error similarity (empirically validated):**
The heuristic e_t approx e_{t+1} is validated empirically: the relative norm ||e_t||/||e_{t+1}|| stays close to 1 during training, and cosine similarity between consecutive errors remains consistently high.

**Quantization levels tested:**
- FP8 (E4M3 format with row-wise scaling) for transformer pretraining (30M-800M, Gemma-3 1B, 2.1B Sparse MoE)
- INT4 (tensor-wise) for fine-tuning DeepSeek-MoE-16B

**Memory savings:** Up to 25% reduction in peak memory for sparse MoE models (12 -> 9 bytes per parameter when going from FP32 to FP8 master weights).

**Experimental highlights:**
- ECO with stochastic rounding nearly recovers master-weight baseline performance
- Naive master-weight removal *diverges* in multiple settings
- ECO maintains stability and matches accuracy benchmarks even at INT4

**CRITICAL RELEVANCE to our ternary training problem:**

ECO is exactly the pattern we need, but applied to FP8/INT4 quantization, not ternary. Key adaptations needed:

1. **ECO uses FP8/INT4 quantization; we need ternary {-1,0,+1}.** Ternary quantization is much more aggressive -- the quantization error is much larger relative to the weight values. The question is whether the error injection coefficient alpha = (1/eta)*(1-1/beta) remains stable for ternary.

2. **The 1/(1-beta^2) penalty.** ECO's convergence bound has a factor 1/(1-beta^2) compared to master weights. For beta=0.9 (typical), this is 1/(1-0.81) = 5.26x. For beta=0.99, it's 1/(1-0.9801) = 50.25x. With ternary quantization's larger sigma^2, this amplification could be problematic.

3. **The injection coefficient alpha = (1/eta)*(1-1/beta).** For small learning rates eta << 1, alpha can be very large (proportional to 1/eta). This means large quantization errors are injected into momentum, which could destabilize training for ternary where sigma^2 is large.

4. **Memory budget compatibility:** ECO's momentum buffer IS the error compensation mechanism (no separate error buffer needed). For ternary training: ternary weights (0.2 bytes) + FP16 momentum (2 bytes) = 2.2 bytes. This fits well within the 4-byte budget. With INT8 momentum, it's 1.2 bytes total.

5. **The "consecutive errors are similar" heuristic** may be even MORE valid for ternary weights than for FP8. Ternary weights change infrequently (they can only be -1, 0, or +1), so the quantization pattern is very stable between steps.

Sources:
- [ECO (arXiv 2601.22101)](https://arxiv.org/abs/2601.22101)
- [ECO HTML version](https://arxiv.org/html/2601.22101v1)

---

## 13. QuEST: Stable 1-bit Training via Trust Gradients (2025)

**Paper:** "QuEST: Stable Training of LLMs with 1-Bit Weights and Activations"
**Authors:** IST-DASLab (same group as ECO/GPTQ)
**Year:** 2025 (ICML)

**Key results:**

1. **Hadamard normalization** before quantization makes the weight/activation distributions more uniform, reducing worst-case quantization error.

2. **Trust gradient estimator:** Explicitly minimizes the error between the noisy gradient computed over quantized states and the "true" full-precision gradient. This replaces STE with a more principled estimator.

3. **Stable training as low as 1-bit weights and activations.** Shows stable scaling laws across all hardware-supported precisions.

4. **At 4-bit:** QuEST models outperform BF16 models nearly 4x their size when data and compute are scaled proportionally.

5. Still uses full-precision master weights during training (shadow weights in BF16).

**Relevance:** QuEST's trust gradient estimator is relevant because it addresses the fundamental problem with STE: STE passes gradients through as if quantization were identity, which introduces systematic bias. A trust gradient estimator that weights gradient components by their quantization reliability could be combined with error feedback.

Sources:
- [QuEST (arXiv 2502.05003)](https://arxiv.org/abs/2502.05003)
- [ICML 2025](https://icml.cc/virtual/2025/poster/45754)

---

## 14. BitNet b1.58 Quantization Error Characteristics

**Paper analysis from:** "The Mathematics That Make 1.58-bit Weights Work" (2025)

**Key findings on quantization error in ternary networks:**

1. **Magnitude destruction:** Ternary quantization to {-1, 0, +1} preserves sign and sparsity pattern but completely destroys magnitude. A weight of 0.003 and 0.847 both map to +1.

2. **Compensation cascade in BitNet:**
   - Absmean quantization: adapts threshold per layer to weight distribution
   - weight_scale tensors: BF16 per-output-channel, values 0.746-4.594 (mean 2.331)
   - sub_norm gains: progressive correction reaching means of 9.32 (FFN) and 6.14 (attention) by final layer

3. **Compounding error:** Quantization error accumulates across layers. Layer 29's attention sub_norm variance is 48.35 vs near-zero at layer 0.

4. **Sparsity pattern:** 42-51% of ternary weights are zero across layers.

5. **The architecture learns to correct:** "The architecture learned to continuously correct for the fact that 1.58 bits isn't enough" through cascading compensation mechanisms.

**Relevance to error feedback for ternary training:** The quantization error in ternary networks is:
- Signal-dependent (not white noise)
- Bounded by alpha/2 where alpha is the scaling factor
- Has a specific structure: it's the residual w - alpha * Q(w/alpha)
- Slowly varying between training steps (weights change gradually)

All of these properties are FAVORABLE for error feedback mechanisms.

Sources:
- [BitNet b1.58 math analysis](https://dev.to/ramr007/the-mathematics-that-make-158-bit-weights-work-how-bitnet-b158-survives-its-own-quantization-3901)

---

## Summary: Error Feedback Literature Landscape

### Chronological Development of Error Feedback Theory

| Year | Paper | Key Contribution |
|------|-------|-----------------|
| 2018 | ECQ-SGD (Wu et al.) | Error compensation with decay (alpha, beta parameters) |
| 2018 | Sparsified SGD with Memory (Stich et al.) | Error feedback matches SGD rate for k-sparsification |
| 2019 | EF fixes SignSGD (Karimireddy et al.) | Error feedback works for ANY contractive compressor |
| 2019 | CHOCO-SGD (Koloskova et al.) | Error feedback in decentralized setting |
| 2020 | Linearly Converging EC-SGD (Gorbunov et al.) | Error feedback + variance reduction = exact convergence |
| 2021 | EF21 (Richtarik et al.) | Compress gradient *differences*, not gradients+errors |
| 2021 | 1-bit Adam (Tang et al.) | Error compensation fails with non-linear optimizers |
| 2022 | EF-BV (Condat et al.) | Unified theory for biased and unbiased compressors |
| 2023 | Momentum improves EF (Fatkhullin et al.) | Momentum provably stabilizes error feedback |
| 2023 | ConEF | Compress the error buffer itself to save memory |
| 2024 | EF Reloaded (Richtarik et al.) | Arithmetic mean improvement, weighted EF21 |
| 2026 | ECO (Nikdan et al.) | Error feedback eliminates master weights for quantized training |

### Key Mathematical Conditions for Stable Error Feedback

1. **Contraction property of compressor:** ||x - C(x)||^2 <= (1-delta)||x||^2 for delta in (0,1]
2. **L-smoothness of objective:** ||grad_f(x) - grad_f(y)|| <= L||x-y||
3. **Bounded gradient variance:** E||grad_f_i(x)||^2 <= G^2
4. **For ECO:** Learning rate eta <= 1/(2L) and bounded momentum
5. **For damped error feedback (ECQ-SGD):** lambda = alpha^2*gamma + (beta-alpha)^2 < 1
6. **Error feedback is INCOMPATIBLE with non-linear optimizer operations** (e.g., Adam's variance term)

### Implications for Ternary Weight Training with Error Feedback

**Favorable factors:**
- Ternary quantization IS a contractive compressor (satisfies the theory)
- Quantization errors are bounded and slowly varying (good for "consecutive errors similar" heuristic)
- Momentum serves dual purpose: optimizer state AND error compensation (memory efficient)
- The EF21 idea of compressing differences is naturally suited to slowly-changing ternary weights

**Risk factors:**
- Ternary quantization is very aggressive (large sigma^2), amplifying the 1/(1-beta^2) penalty
- The error injection coefficient alpha = (1/eta)(1-1/beta) can be very large for small learning rates
- Non-linear optimizers (Adam) are incompatible with standard error feedback
- The error buffer may need compression (ConEF) to fit memory budget

**Most promising approach:** ECO-style error injection into momentum, adapted for ternary quantization, using SGD with momentum (not Adam) to avoid the non-linearity problem. Potentially combined with EF21-style differential error injection for additional stability.
