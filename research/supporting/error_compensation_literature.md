# Literature Search: Error-Compensated Optimization for Quantized Training

Date: 2026-03-21

---

## 1. ECO: Error-Compensating Optimizer (arXiv 2601.22101, January 2026)

**Title:** ECO: Quantized Training without Full-Precision Master Weights
**Authors:** Mahdi Nikdan, Amir Zandieh, Dan Alistarh, Vahab Mirrokni
**Year:** 2026 (submitted January 29)
**Venue:** arXiv preprint

### Key Contribution
ECO eliminates full-precision master weights by applying optimizer updates directly to quantized parameters. It injects quantization error into the optimizer's momentum buffer, forming an error-feedback loop with zero additional memory overhead. This is the most directly relevant paper to ternary error-compensated training.

### Error Feedback Mechanism (Exact Formulation)

**For SGDM:**
```
m <- m + (1/eta)(1 - 1/beta) * e_{t+1}
```
where `e_{t+1} = theta_tilde_{t+1} - theta_hat_{t+1}` is the quantization error between the high-precision temporary iterate and its quantized version.

**For Adam:**
```
m <- m + ((1 - beta_1^t) / eta)(1 - 1/beta_1) * (sqrt(v_{t+1}/(1-beta_2^t)) + epsilon) * e_{t+1}
```

**Algorithm pseudocode:**
```
Input: Quantized parameters theta_hat_t, optimizer state s_hat_t

1. theta_tilde_{t+1}, s_tilde_{t+1} <- OPTIM_STEP(theta_hat_t, s_hat_t, H)
2. theta_hat_{t+1} <- q(theta_tilde_{t+1})                    # quantize weights
3. e_{t+1} <- theta_tilde_{t+1} - theta_hat_{t+1}             # compute error
4. m_hat_{t+1} <- m_tilde_{t+1} + alpha * e_{t+1}             # inject into momentum
5. return theta_hat_{t+1}, s_hat_{t+1}
```

The key insight: rather than storing errors in a separate buffer (like classical EF), ECO reuses the optimizer's existing momentum buffer. Consecutive quantization errors are typically close, so the momentum buffer naturally serves as error memory.

### Convergence Proof

**Theorem 3.8:** For learning rate eta <= 1/(2L), ECO converges with guarantee:
```
min_{t in [0,T]} E[||grad f(theta_hat_t)||^2] <= 4(f(theta_0) - f*) / (eta*T) + sigma^2_quant
```

where the quantization noise floor is:
```
sigma^2_quant = 4*eta^2*beta^2*L^2*G^2/(1-beta)^2 + 4*L^2*sigma^2/(1-beta^2)
```

As eta -> 0: `lim sigma^2_quant = 4*L^2*sigma^2/(1-beta^2)`, proven tight up to constant 4.

**Key Assumption 3.2:** Quantization error must be zero-mean with bounded variance -- satisfied by stochastic rounding but NOT deterministic rounding (round-to-nearest).

**Proof technique:** Uses a "virtual sequence" `theta_t := theta_hat_t - (eta*beta/(1-beta))*m_hat_t` that evolves identically to standard gradient descent. The quantization error cancels perfectly through careful choice of the momentum injection coefficient.

**Critical comparison with naive approach:** Without error compensation, naive master-weight removal yields error proportional to 1/eta, which diverges as learning rate anneals. ECO's error stays bounded.

### Quantization Levels Tested
- **FP8 E4M3:** Primary experiments, row-wise scaling with max-absolute-value strategy
- **INT4:** Used for fine-tuning DeepSeek-MoE-16B with tensor-wise quantization
- **Stochastic rounding** performs best (required by theory); round-to-nearest also tested but inferior

### Results at Scale
| Model | Size | Quantization | Result |
|---|---|---|---|
| Small Transformers | 30M-800M | FP8 | Near-lossless vs master weights |
| Gemma-3 | 1B | FP8 | Matches master weight baseline |
| Sparse MoE | 2.1B | FP8 | Matches baseline, ~25% memory reduction |
| DeepSeek-MoE | 16B | INT4 fine-tuning | Near-lossless accuracy |

### Memory Analysis
- Eliminates 4 bytes/param of FP32 master weight storage
- No additional memory beyond existing optimizer states
- Particularly beneficial for Sparse MoE where all expert parameters reside in memory despite sparse activation

### Relevance to Ternary Error-Compensated Training
**HIGH RELEVANCE.** ECO demonstrates that:
1. Error feedback into momentum is sufficient to compensate for quantization -- no separate error buffer needed
2. The technique works with both SGDM and Adam
3. Stochastic rounding is theoretically required for convergence
4. The convergence to a noise floor (not exact optimum) is inherent to quantized training

**Gap:** ECO was tested at FP8/INT4, not at ternary precision. The quantization noise floor grows with coarser quantization. Whether the same technique extends to ternary (1.58-bit) with acceptable noise floor is an open question.

Sources:
- [ECO paper (arXiv 2601.22101)](https://arxiv.org/abs/2601.22101)
- [ECO HTML version](https://arxiv.org/html/2601.22101v1)

---

## 2. Foundational Error Feedback / Error Compensation Papers

### 2a. Error Feedback Fixes SignSGD (Karimireddy et al., 2019)

**Title:** Error Feedback Fixes SignSGD and other Gradient Compression Schemes
**Authors:** Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, Martin Jaggi
**Year:** 2019
**Venue:** ICML 2019 (PMLR 97:3252-3261)

**Key Contribution:**
- Proved that SignSGD (and other biased compressors) can FAIL to converge even on simple convex problems
- Showed that adding error feedback (EF-SGD) fixes convergence: accumulate compression error and add it back at the next step
- With error feedback, ANY compression operator achieves the SAME convergence rate as uncompressed SGD
- No additional assumptions needed beyond standard smoothness + bounded variance

**The EF-SGD Algorithm:**
```
e_0 = 0
For t = 0, 1, 2, ...:
    p_t = g_t + e_t                    # add accumulated error to gradient
    compressed_t = C(p_t)               # compress
    e_{t+1} = p_t - compressed_t        # new error = what was lost
    w_{t+1} = w_t - eta * compressed_t  # update with compressed signal
```

**Relevance:** This is the theoretical foundation showing error feedback restores convergence for biased compression. Directly applicable: ternary quantization of weights is a biased compressor.

Source: [Error Feedback Fixes SignSGD (arXiv 1901.09847)](https://arxiv.org/abs/1901.09847)

### 2b. Error Compensated Quantized SGD (Wu et al., 2018)

**Title:** Error Compensated Quantized SGD and its Applications to Large-scale Distributed Optimization
**Authors:** Jiaxiang Wu, Weidong Huang, Junzhou Huang, Tong Zhang
**Year:** 2018
**Venue:** ICML 2018

**Key Contribution:**
- Proposed ECQ-SGD: quantize local gradients + use accumulated quantization error to accelerate convergence
- Tighter worst-case error bound than plain QSGD with properly chosen hyperparameters
- Error feedback suppresses the quantization error's contribution to the error bound
- Achieved up to 100x gradient compression without performance degradation

**Convergence:** The error feedback scheme suppresses quantization error's contribution to the bound, leading to a smaller sub-optimality gap. SGD with compression + error compensation converges at the same rate as vanilla SGD.

Source: [ECQ-SGD (arXiv 1806.08054)](https://arxiv.org/abs/1806.08054)

### 2c. EF21: Modern Error Feedback (Richtarik et al., 2021)

**Title:** EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback
**Authors:** Peter Richtarik, Igor Sokolov, Ilyas Fatkhullin
**Year:** 2021
**Venue:** NeurIPS 2021

**Key Contribution:**
- Improved error feedback mechanism that works under standard assumptions only
- Achieves O(1/T) convergence for smooth nonconvex problems (previous EF methods: O(1/T^{2/3}))
- First linear convergence result for EF-type methods on PL (Polyak-Lojasiewicz) functions
- Works in distributed heterogeneous data settings
- Does NOT require bounded gradient assumption (unlike prior methods)
- Practically faster than all previous EF methods

**Convergence Rates:**
- Smooth nonconvex: O(1/T) -- matches SGD rate
- PL condition (generalization of strong convexity): Linear convergence

**Key insight for compression types:** Works for ANY compressor satisfying a contraction property, including both unbiased compressors (random sparsification) and biased compressors (TopK, sign compression, quantization).

Source: [EF21 (arXiv 2106.05203)](https://arxiv.org/abs/2106.05203)

---

## 3. Error Compensation Combined with Extreme Quantization

### 3a. 1-bit Adam (Tang et al., 2021)

**Title:** 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed
**Authors:** Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, Yuxiong He
**Year:** 2021
**Venue:** arXiv 2102.02888

**Key Contribution:**
- Standard error compensation works with SGD/momentum SGD (linearly dependent on gradients) but NOT directly with Adam (non-linear gradient dependence)
- Solution: Two-stage approach:
  1. **Warmup stage:** Run vanilla Adam until variance stabilizes
  2. **Compression stage:** Freeze variance as fixed preconditioner, apply error-compensated 1-bit compression to momentum updates

**Results (up to 256 GPUs):**
- Up to 5x less communication volume
- Up to 3.3x faster end-to-end throughput (BERT-Large pretraining)
- Same convergence behavior and final accuracy as uncompressed Adam

**Relevance:** Demonstrates that error compensation can work with 1-bit (sign) compression in the Adam setting, but requires the two-stage trick. The frozen variance insight could be useful for ternary training with Adam.

Source: [1-bit Adam (arXiv 2102.02888)](https://arxiv.org/abs/2102.02888)

### 3b. Quantized Adam with Error Feedback (Chen et al., 2021)

**Title:** Quantized Adam with Error Feedback
**Authors:** Chen, Shen, Huang, Liu
**Year:** 2021
**Venue:** arXiv 2004.14180

**Key Contribution:**
- Distributed Adam with TWO types of quantization: gradient quantization + weight quantization, both with error feedback
- Convergence in nonconvex setting:
  - Gradient quantization + EF: converges to first-order stationary point
  - Weight quantization + EF: converges to neighborhood related to quantization level

**Relevance:** Directly studies weight quantization with error feedback in the Adam framework. The convergence-to-neighborhood result for weight quantization aligns with ECO's findings.

Source: [Quantized Adam with EF (arXiv 2004.14180)](https://arxiv.org/abs/2004.14180)

### 3c. TernGrad (Wen et al., 2017)

**Title:** TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning
**Authors:** Wei Wen, Cong Xu, et al.
**Year:** 2017
**Venue:** NeurIPS 2017

**Key Contribution:**
- Compresses gradients to ternary values {-1, 0, 1} with scaling for distributed communication
- Convergence proven under bounded gradient assumption with layer-wise ternarization and gradient clipping
- No accuracy loss on AlexNet; <2% loss on GoogLeNet
- Deployed in production at Facebook for large-scale training

**Relevance:** Demonstrates that ternary compression with appropriate scaling can work in practice. However, this is gradient compression (communication), not weight quantization. The scaling techniques and convergence analysis are transferable.

Source: [TernGrad (arXiv 1705.07878)](https://arxiv.org/abs/1705.07878)

### 3d. QuEST: 1-Bit Weights and Activations (Panferov et al., 2025)

**Title:** QuEST: Stable Training of LLMs with 1-Bit Weights and Activations
**Authors:** Andrei Panferov, Jiale Chen, Soroush Tabesh, Roberto L. Castro, Mahdi Nikdan, Dan Alistarh
**Year:** 2025
**Venue:** ICML 2025

**Key Contribution:**
- Pushes QAT (quantization-aware training) to W1A1 (1-bit weights AND activations)
- Two key techniques: (1) Hadamard normalization for accurate quantization of weight/activation distributions, (2) a new "trust gradient estimator"
- W4A4 is Pareto-competitive with FP16 (better accuracy at lower model size)
- Stable training at 1-bit weights and activations

**Relevance:** Shows that extremely aggressive quantization (1-bit) can work for LLM training with the right techniques. However, QuEST still uses full-precision master weights during training -- it doesn't address the training memory problem.

Source: [QuEST (arXiv 2502.05003)](https://arxiv.org/abs/2502.05003)

### 3e. Direct Quantized Training with Stochastic Rounding (Zhao et al., 2024)

**Title:** Direct Quantized Training of Language Models with Stochastic Rounding
**Authors:** Kaiyan Zhao, Tsuguchika Tabaru, Kenichi Kobayashi, Takumi Honda, Masafumi Yamazaki, Yoshimasa Tsuruoka
**Year:** 2024
**Venue:** arXiv 2412.04787

**Key Contribution:**
- Directly updates quantized weights without high-precision master copy
- Uses stochastic rounding instead of straight-through estimator
- **Ternary training feasible** but with performance gap vs higher-bit models
- 8-bit DQT matches BitNet b1.58 accuracy

**Relevance:** Closest existing work to "ternary training without master weights." DQT uses stochastic rounding as an implicit form of error compensation (unbiased rounding preserves gradient information in expectation). However, it does NOT explicitly use the error feedback mechanism from ECO/EF-SGD.

Source: [DQT (arXiv 2412.04787)](https://arxiv.org/abs/2412.04787)

---

## 4. Convergence Theory of Error-Compensated Methods

### Summary of Known Convergence Guarantees

| Method | Setting | Rate | Assumptions | Year |
|---|---|---|---|---|
| EF-SGD (Karimireddy) | Smooth nonconvex | O(1/sqrt(T)) | L-smooth, bounded variance | 2019 |
| ECQ-SGD (Wu) | Smooth nonconvex | Same as SGD | L-smooth, bounded variance | 2018 |
| EF21 (Richtarik) | Smooth nonconvex | O(1/T) | L-smooth only | 2021 |
| EF21 | PL condition | Linear | L-smooth + PL | 2021 |
| ECO (Nikdan) | Smooth nonconvex | O(1/T) + noise floor | L-smooth, stochastic rounding | 2026 |
| 1-bit Adam | Smooth nonconvex | Same as Adam | Warm-up + frozen variance | 2021 |

### Standard Assumptions Required
1. **L-smoothness:** ||grad f(x) - grad f(y)|| <= L ||x - y|| for all x, y
2. **Bounded stochastic gradient variance:** E[||g - grad f(x)||^2] <= sigma^2
3. **Contraction property of compressor:** E[||C(x) - x||^2] <= (1 - delta) ||x||^2 for some delta > 0
4. **For ECO specifically:** Quantization error must be zero-mean (requires stochastic rounding)

### Key Theoretical Insights

**1. Error feedback eliminates bias from compression.** Without EF, biased compressors (like quantization) introduce systematic drift that prevents convergence. EF accumulates and re-injects this drift.

**2. Convergence to a noise floor, not the exact optimum.** For quantized weight training (as opposed to gradient compression), there is always a residual neighborhood related to quantization granularity. ECO proves this floor scales as O(L^2 * sigma^2 / (1 - beta^2)).

**3. Stochastic rounding is theoretically necessary.** Deterministic rounding (round-to-nearest) introduces bias that error feedback cannot fully correct. Stochastic rounding produces unbiased quantization errors, which is required by the convergence proofs.

**4. The noise floor grows with quantization coarseness.** Moving from FP8 to INT4 to ternary increases sigma^2 in the quantization noise term. The key open question is whether the noise floor at ternary precision is acceptable for practical training.

**5. Momentum injection coefficient must be precisely calibrated.** ECO's convergence proof depends on the exact coefficient alpha = (1/eta)(1 - 1/beta) for the error injection. Wrong coefficients break the virtual sequence cancellation and can cause divergence.

---

## 5. HALP: High-Accuracy Low-Precision Training (De Sa et al., 2018)

**Title:** High-Accuracy Low-Precision Training
**Authors:** Christopher De Sa, Megan Leszczynski, Jian Zhang, Alana Marzoev, Christopher R. Aberger, Kunle Olukotun, Christopher Re
**Year:** 2018
**Venue:** arXiv 1803.03383

### Key Contribution
HALP combines SVRG (variance reduction) with a novel "bit centering" technique to achieve full-precision convergence rates using low-precision arithmetic.

### The Bit Centering Technique
- As optimization approaches the optimum, gradients become smaller in magnitude
- Fixed-point representations waste bits on representing large ranges that are no longer needed
- Bit centering dynamically re-centers and re-scales the low-precision representation at each SVRG epoch
- This shrinks the quantization grid spacing as convergence progresses, reducing quantization noise proportionally
- Mathematically: for strongly convex with parameter mu, ||w - w*|| <= (1/mu)||grad f(w)||, so the range needed shrinks as gradients shrink

### Error Compensation Mechanism
HALP achieves error compensation through TWO complementary mechanisms:
1. **SVRG variance reduction:** Periodic full-precision gradient computation reduces stochastic gradient variance
2. **Bit centering:** Dynamic re-scaling reduces quantization noise as convergence progresses

Unlike classical EF methods that explicitly store and re-inject errors, HALP implicitly compensates by making the quantization progressively finer. The effect is similar but the mechanism is different.

### Convergence Rate
- **Linear convergence** matching full-precision SVRG: E[f(w_K+1) - f(w*)] <= gamma^K (f(w_1) - f(*))
- Where 0 < gamma < 1, measured per SVRG epoch K
- Achieves arbitrarily accurate solutions (no permanent noise floor) because bit centering makes quantization noise vanish as we converge

### Precision Levels Tested
- 8-bit and 16-bit fixed-point representations
- 8-bit HALP converges to high accuracy on synthetic regression
- Up to 4x faster than full-precision SVRG on CPU

### Memory Analysis
- All iterates stored in low precision (8 or 16 bits)
- Full gradient computation required periodically (SVRG overhead)
- Memory savings come from low-precision storage of model parameters and gradients

### Relevance to Ternary Error-Compensated Training
**MODERATE RELEVANCE.** HALP demonstrates that:
1. Error compensation + variance reduction can achieve exact convergence even with low precision
2. Dynamic rescaling of the quantization grid is powerful (analogous to adaptive scaling in ternary networks)
3. The SVRG requirement (periodic full gradient pass) is expensive for LLM-scale training

**Limitation:** HALP requires the ability to smoothly adjust quantization resolution, which is fundamentally incompatible with fixed ternary {-1, 0, 1} values. The bit centering idea doesn't directly transfer. However, the concept of "progressive refinement of the quantization scheme" could inspire adaptive threshold mechanisms for ternary training.

Source: [HALP (arXiv 1803.03383)](https://arxiv.org/abs/1803.03383)
Source: [HALP blog post](https://www.cs.cornell.edu/~cdesa/blog/2018-03-09-halp/halp.html)
Source: [HALP GitHub](https://github.com/HazyResearch/halp)

---

## 6. Synthesis: What This Means for Ternary Error-Compensated Training

### The Core Idea
Error compensation works by accumulating quantization residuals and feeding them back into subsequent optimization steps. For ternary training, this would mean:

1. Compute gradient update for a ternary weight
2. Apply update to get a temporary high-precision value
3. Quantize back to ternary
4. Compute the error (what was lost in quantization)
5. Inject that error into the optimizer's momentum buffer
6. The momentum buffer accumulates these micro-errors until they're large enough to trigger a ternary weight change

### What Is Known (Established)
- Error feedback restores convergence guarantees for any compressor satisfying contraction property
- Momentum buffer can serve as error memory (ECO) -- no separate buffer needed
- Stochastic rounding is required for unbiased error and theoretical convergence
- Convergence is to a noise-floor neighborhood, not exact optimum
- The technique works at FP8 and INT4 precision (ECO), and at 1-bit for gradient compression (TernGrad, 1-bit Adam)

### What Is Unknown (Open Questions)
1. **Does ECO's error feedback work at ternary precision?** The noise floor grows with quantization coarseness. At ternary (3 levels), sigma^2_quant may be too large for practical training.
2. **Can stochastic rounding between ternary values preserve enough gradient information?** The rounding granularity is extremely coarse -- rounding to {-1, 0, 1} with scaling.
3. **What is the optimal momentum injection coefficient for ternary?** ECO's coefficient alpha = (1/eta)(1 - 1/beta) was derived for continuous quantization schemes. It may need modification for discrete ternary grids.
4. **Can this be combined with Bop-style threshold mechanisms?** The error feedback could determine WHEN to flip a ternary weight, while the threshold prevents oscillation.
5. **Memory budget feasibility:** If using SGDM + error feedback (injected into momentum), the only extra state is the momentum buffer. At FP16, that's 2 bytes/param. With ternary weights (~0.2 bytes/param), total is ~2.2 bytes/param -- well within the 4-byte budget.

### Proposed Research Direction
Combine ECO's error feedback mechanism with DQT's stochastic rounding on ternary weights:
- Use SGDM (not Adam) to stay within memory budget
- Inject quantization error into momentum as ECO prescribes
- Use stochastic rounding for unbiased ternary quantization
- Potentially add adaptive thresholds (from Bop/SGDAT) to prevent oscillation
- Total memory: ternary weights (0.2B) + FP16 momentum (2B) = 2.2 bytes/param extra

This combination has NOT been explored in the literature.

---

## Complete Source List

- [ECO: Quantized Training without Master Weights (arXiv 2601.22101)](https://arxiv.org/abs/2601.22101)
- [Error Feedback Fixes SignSGD (arXiv 1901.09847)](https://arxiv.org/abs/1901.09847)
- [Error Compensated Quantized SGD (arXiv 1806.08054)](https://arxiv.org/abs/1806.08054)
- [EF21 (arXiv 2106.05203)](https://arxiv.org/abs/2106.05203)
- [1-bit Adam (arXiv 2102.02888)](https://arxiv.org/abs/2102.02888)
- [Quantized Adam with Error Feedback (arXiv 2004.14180)](https://arxiv.org/abs/2004.14180)
- [TernGrad (arXiv 1705.07878)](https://arxiv.org/abs/1705.07878)
- [QuEST: 1-Bit Training (arXiv 2502.05003)](https://arxiv.org/abs/2502.05003)
- [DQT with Stochastic Rounding (arXiv 2412.04787)](https://arxiv.org/abs/2412.04787)
- [HALP (arXiv 1803.03383)](https://arxiv.org/abs/1803.03383)
- [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764)
