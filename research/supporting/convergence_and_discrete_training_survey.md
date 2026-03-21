# Convergence Theory & Discrete Training Methods Survey

Date: 2026-03-21

This survey covers seven targeted research areas related to memory-efficient ternary weight training, focusing on convergence theory, stochastic rounding, discrete optimizers, and quantization error accumulation.

---

## 1. Convergence Theory for Quantized/Discrete Weight Training

### 1a. "Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets"
- **Authors:** Penghang Yin, Jiancheng Lyu, Shuai Zhang, Stanley Osher, Yingyong Qi, Jack Xin
- **Year:** 2019 (ICLR 2019)
- **Key Contribution:** First rigorous convergence analysis of STE for quantized neural networks. Proves that if the STE is properly chosen, the expected "coarse gradient" correlates positively with the population gradient, and its negation is a descent direction for minimizing the population loss.
- **Mathematical Results:**
  - The associated coarse gradient descent algorithm converges to a critical point of the population loss minimization problem.
  - Analysis is for a two-layer network with binarized ReLU activation and Gaussian input data.
  - A poor choice of STE leads to instability near certain local minima.
- **Limitation:** Restricted to two-layer networks; extension to deeper architectures remains open.
- **Source:** [OpenReview](https://openreview.net/forum?id=Skh4jRcKQ)

### 1b. "Beyond Discreteness: Finite-Sample Analysis of Straight-Through Estimator for Quantization"
- **Authors:** (arxiv 2505.18113)
- **Year:** 2025 (May)
- **Key Contribution:** First finite-sample convergence analysis of STE for neural network quantization. Provides sample complexity bounds.
- **Mathematical Results:**
  - For a two-layer binarized network with Gaussian inputs: **O(n^2) samples suffice for convergence of ergodic (averaged) iterates** to the optimal solution.
  - **O(n^4) samples** suffice for non-ergodic (last iterate) convergence.
  - The O(n^2) bound for ergodic convergence is tight (matching lower bound).
  - Employs dual STEs to handle discreteness in both objective and constraints.
- **Source:** [arXiv](https://arxiv.org/abs/2505.18113)

### 1c. "High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator"
- **Authors:** (arxiv 2510.10693)
- **Year:** 2025 (October)
- **Key Contribution:** Shows that in the high-dimensional limit, STE training dynamics converge to a deterministic ordinary differential equation (ODE).
- **Mathematical Results:**
  - STE training exhibits a **plateau phase followed by a sharp drop** in generalization error. Plateau length depends on the quantization range.
  - Fixed-point analysis quantifies the **asymptotic deviation** from the unquantized linear model.
  - Stability analysis identifies regimes where quantization preserves training stability, even at higher learning rates (quantization as implicit regularizer).
  - In the low-bit regime, STE dynamics become **non-monotonic**, leading to slower convergence.
- **Source:** [arXiv](https://arxiv.org/abs/2510.10693)

### 1d. "Bridging Discrete and Backpropagation: Straight-Through and Beyond"
- **Authors:** Liang Liu et al.
- **Year:** 2023 (NeurIPS 2023, Oral)
- **Key Contribution:** Demonstrates that STE works as a **first-order approximation** of the gradient through discrete operations. Proposes ReinMax, achieving second-order accuracy by integrating Heun's method with negligible computation overhead.
- **Mathematical Results:**
  - ST heuristic shown to be equivalent to first-order gradient approximation.
  - ReinMax achieves second-order accuracy without requiring Hessian computation.
- **Source:** [arXiv](https://arxiv.org/abs/2304.08612), [NeurIPS 2023](https://nips.cc/virtual/2023/oral/73827)

### 1e. "Overcoming Oscillations in Quantization-Aware Training"
- **Authors:** Markus Nagel, Marios Fournarakis, Yelysei Bondarenko, Tijmen Blankevoort
- **Year:** 2022 (ICML 2022)
- **Key Contribution:** Identifies and analyzes the phenomenon of **weight oscillations** in QAT, where quantized weights oscillate between two grid points during training.
- **Mathematical Results:**
  - Oscillations lead to wrongly estimated batch-normalization statistics and increased training noise.
  - Effects are particularly pronounced in low-bit (<=4-bit) quantization of efficient networks with depthwise separable layers.
  - Most previously proposed QAT algorithms cannot overcome oscillations.
  - Proposes **oscillation dampening** and **iterative weight freezing** as solutions.
- **Source:** [PMLR](https://proceedings.mlr.press/v162/nagel22a.html), [arXiv](https://arxiv.org/abs/2203.11086)

### 1f. "Learning Quantized Neural Nets by Coarse Gradient Method for Nonlinear Classification"
- **Authors:** Penghang Yin et al.
- **Year:** 2021 (Research in the Mathematical Sciences)
- **Key Contribution:** Extends coarse gradient descent convergence theory to nonlinear classification settings.
- **Source:** [Springer](https://link.springer.com/article/10.1007/s40687-021-00281-4)

### 1g. "Convergence of a Relaxed Variable Splitting Coarse Gradient Descent Method for Learning Sparse Weight Binarized Activation Neural Network"
- **Year:** 2020 (Frontiers in Applied Mathematics and Statistics)
- **Key Contribution:** Convergence analysis for a variable-splitting approach to sparse binary networks.
- **Source:** [Frontiers](https://www.frontiersin.org/articles/10.3389/fams.2020.00013/full)

---

## 2. Stochastic Rounding Convergence Guarantees

### 2a. "Deep Learning with Limited Numerical Precision" (Gupta et al., 2015)
- **Authors:** Suyog Gupta, Arun Agrawal, Kailash Gopalakrishnan, Pritish Narayanan
- **Year:** 2015 (ICML 2015)
- **Key Contribution:** Seminal work establishing that deep networks can be trained using only 16-bit fixed-point with stochastic rounding, with little to no accuracy degradation. Foundational theory for stochastic rounding in neural network training.
- **Mathematical Results:**
  - Stochastic rounding is **unbiased**: E[SR(x)] = x. This makes rounded weights an unbiased estimate of the precise weights.
  - Regardless of update magnitude, the expectation of rounded weights converges at the same speed as precise weights.
  - Proved feasibility with 16-bit and 8-bit fixed-point arithmetic.
- **Source:** [PMLR](https://proceedings.mlr.press/v37/gupta15.html), [arXiv](https://arxiv.org/abs/1502.02551)

### 2b. "Stochastic Rounding for LLM Training: Theory and Practice" (Ozkara et al., 2025)
- **Authors:** Kaan Ozkara, Tao Yu, Youngsuk Park (Amazon)
- **Year:** 2025 (AISTATS 2025)
- **Key Contribution:** Provides theoretical analysis of implicit regularization and convergence under the **Adam optimizer** when stochastic rounding is used.
- **Mathematical Results:**
  - SR shows **lower quantization error** compared to nearest rounding (NR) in the convergence bound.
  - With increased learning rates, the quantization error due to SR becomes negligible.
  - BF16 with SR outperforms (BF16, FP32) mixed precision, achieving 1.54x higher throughput and 30% lower memory.
  - Proved convergence bounds specific to SR under Adam dynamics.
- **Source:** [AISTATS 2025](https://proceedings.mlr.press/v258/ozkara25b.html), [arXiv](https://arxiv.org/abs/2502.20566)

### 2c. "Training with Fewer Bits: Unlocking Edge LLMs Training with Stochastic Rounding"
- **Year:** 2025 (arxiv 2511.00874)
- **Key Contribution:** Demonstrates SR for training LLMs in low-precision on edge devices.
- **Mathematical Results:**
  - Mini-batch SGD with SR shows that increased batch sizes can compensate for reduced precision.
  - Convergence degradation with SR is minimal compared to full-precision under worst-case analysis.
- **Source:** [arXiv](https://arxiv.org/abs/2511.00874)

### 2d. Stochastic Rounding + Momentum Interaction
- **Key Finding:** Quantization error in the momentum term persists until the momentum receives a non-zero gradient update. During this period, the momentum carrying quantization error is utilized at each iteration, leading to **progressive error accumulation**.
- When employing large momentum (large beta), the term (1-beta)*z_{t+1} becomes very small and is easily swamped in low-precision EMA updates, reducing adaptivity.
- **Mitigation:** Upper bounds on gradient noise variance can be derived for signed EMA updates in low-bit quantization, enabling regulation via momentum hyperparameter beta.
- **Source:** [arXiv 2505.00347](https://arxiv.org/html/2505.00347) ("Pushing the Limits of Low-Bit Optimizers: A Focus on EMA Dynamics")

### 2e. Classical Convergence Result
- **Key Result:** Stochastic rounding solves convex discrete problems up to a level of accuracy that depends on the quantization level. However, fully quantized training methods using SR **stall before training is complete** on non-convex problems, unlike methods that maintain floating-point representations.
- This is a fundamental limitation: SR preserves unbiasedness but cannot overcome the **discretization barrier** in non-convex landscapes without additional mechanisms.

---

## 3. Bop (Binary Optimizer) - Helwegen et al., 2019

### 3a. Original Paper: "Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization"
- **Authors:** Koen Helwegen, James Widdicombe, Lukas Geiger, Zechun Liu, Kwang-Ting Cheng, Roeland Nusselder
- **Year:** 2019 (NeurIPS 2019)
- **ArXiv:** 1906.02107
- **Key Contribution:** Introduces the Binary Optimizer (Bop), the first optimizer specifically designed for binary neural networks. Argues that latent weights in standard BNN training cannot be treated analogously to real-valued weights -- their main role is providing **inertia** (resistance to weight flips until sufficient gradient evidence accumulates).
- **Mechanism:**
  - Maintains exponential moving average (EMA) of gradients: `m_t = gamma * m_{t-1} + (1-gamma) * g_t`
  - Flips binary weight when the accumulated gradient signal exceeds a threshold tau: flip if `sign(w) != sign(m_t)` and `|m_t| > tau`
  - The adaptivity rate gamma controls how quickly the EMA adapts to changes.
- **Mathematical Results:**
  - No formal convergence proof is provided; the paper is primarily empirical.
  - Key theoretical insight: latent weights provide inertia, not a continuous approximation. Bop makes this explicit.
- **Memory:** 1 real-valued variable per weight (the EMA), versus 2-3 for standard approaches (latent weight + momentum + possibly variance). This is ~4 bytes with FP32 EMA, ~2 bytes with FP16.
- **Results:** CIFAR-10: 91.3%; ImageNet (Bi-Real Net): 56.6% top-1.
- **Limitation:** Only binary {-1, +1}, not ternary. Only small-scale vision models.
- **Source:** [arXiv](https://arxiv.org/abs/1906.02107), [GitHub](https://github.com/plumerai/rethinking-bnn-optimization)

### 3b. "A Bop and Beyond: A Second Order Optimizer for Binarized Neural Networks"
- **Authors:** Cuauhtemoc Daniel Suarez-Ramirez et al.
- **Year:** 2021 (CVPR 2021 Workshop - LXCV)
- **ArXiv:** 2104.05124
- **Key Contribution:** Extends Bop with a second-order mechanism (Bop2ndOrder). Takes an approach parallel to Adam: uses the **second raw moment estimate** to normalize the first raw moment before comparison with the threshold.
- **Mathematical Results:**
  - Two versions: biased and bias-corrected.
  - Converges faster than original Bop.
  - More robust to hyperparameter changes.
  - Achieves better accuracy on CIFAR-10 (BinaryNet), ImageNet (XnorNet, BiRealNet).
- **Ternary Extension:** No explicit ternary extension discussed in this paper.
- **Source:** [arXiv](https://arxiv.org/abs/2104.05124), [CVPR](https://openaccess.thecvf.com/content/CVPR2021W/LXCV/papers/Suarez-Ramirez_A_Bop_and_Beyond_A_Second_Order_Optimizer_for_Binarized_CVPRW_2021_paper.pdf), [GitHub](https://github.com/CuauSuarez/Bop2ndOrder)

### 3c. "A Comprehensive Study on Binary Optimizer and its Applicability"
- **Year:** 2020 (OpenReview)
- **Key Contribution:** Comprehensive empirical study of Bop's behavior, hyperparameter sensitivity, and applicability across architectures.
- **Source:** [OpenReview](https://openreview.net/pdf?id=y53aaSM5o)

### 3d. "A Probabilistic Optimizer for Binary Neural Networks"
- **Year:** 2025
- **Key Contribution:** Interprets accumulated gradients as "potential energy" for each neuron. Formulates a **Bernoulli probability distribution** based on gradient magnitude to decide weight flips, adding beneficial stochasticity.
- **Relevance to ternary:** The probabilistic framework naturally extends from Bernoulli (binary) to categorical (ternary) distributions over {-1, 0, +1}.
- **Source:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009695)

### 3e. Has Anyone Extended Bop to Ternary?
- **Finding:** No paper was found that directly extends Bop's threshold-based flipping mechanism to ternary {-1, 0, +1} weights. This appears to be an **open research gap**.
- The closest approaches are GXNOR-Net's discrete state transition (Section 7) and the probabilistic BNN optimizer (Section 3d), but neither directly adapts Bop's EMA+threshold framework for the ternary case.

---

## 4. Ternary Weight Training Without Latent Weights / Ternary Optimizers

### 4a. "Direct Quantized Training of Language Models with Stochastic Rounding"
- **Authors:** Kaiyan Zhao, Tsuguchika Tabaru, Kenichi Kobayashi, Takumi Honda, Masafumi Yamazaki, Yoshimasa Tsuruoka
- **Year:** 2024 (December)
- **ArXiv:** 2412.04787
- **Key Contribution:** Directly updates quantized low-precision weights without relying on STE during backpropagation. Uses stochastic rounding to minimize information loss.
- **Mathematical Results:**
  - Training with ternary-only weights is feasible and converges.
  - Extending to 8 bits achieves further performance gains.
  - Eliminates the need to maintain high-precision (unquantized) weights throughout training.
- **Memory:** Eliminates the 2-4 byte master weight copy. With SGD+momentum, approaches ~4.2 bytes/param total.
- **Source:** [arXiv](https://arxiv.org/abs/2412.04787)

### 4b. "Training Binary Neural Networks in a Binary Weight Space"
- **Year:** 2024 (ICLR 2024 submission, OpenReview)
- **Key Contribution:** Proposes training BNNs without holding any real-valued weights, saving memory. Defines an **update probability** for binary weights determined by current binary weights and real-valued gradients.
- **Mathematical Results:**
  - Provides a probabilistic framework for latent-free binary training.
  - Addresses the challenge that gradient-based optimization is difficult without real-valued weights.
- **Source:** [OpenReview](https://openreview.net/forum?id=Dm4qrBuFKH)

### 4c. ECO: Error-Compensating Optimizer (Quantized Training without Full-Precision Master Weights)
- **Year:** 2025 (January)
- **ArXiv:** 2601.22101
- **Key Contribution:** Eliminates master weights entirely. Applies updates directly to quantized parameters and injects quantization error into the optimizer momentum, forming an **error-feedback loop** with no additional memory.
- **Mathematical Results:**
  - Under standard assumptions and decaying learning rate, ECO converges to a **constant-radius neighborhood** of the optimum.
  - Proved that naive master-weight removal (without error compensation) **cannot converge**.
  - Tested on Transformers 30M-800M, Gemma-3 1B, and 2.1B Sparse MoE with FP8 quantization; fine-tuning DeepSeek-MoE-16B in INT4.
  - Reduces static memory by up to 25%.
- **Relevance:** The error-feedback mechanism is directly applicable to ternary training. The quantization error from rounding to {-1, 0, +1} can be accumulated in the momentum buffer.
- **Source:** [arXiv](https://arxiv.org/abs/2601.22101)

### 4d. Ternary Weight Networks (TWN)
- **Authors:** Fengfu Li, Bo Zhang, Bin Liu
- **Year:** 2016
- **ArXiv:** 1605.04711
- **Key Contribution:** Original ternary weight network paper. Constrains weights to {-1, 0, +1} with a scaling factor. Uses threshold-based ternary function and minimizes Euclidean distance between full-precision and ternary weights.
- **Mathematical Results:**
  - Threshold-based quantization: w_q = +1 if w > delta, -1 if w < -delta, 0 otherwise.
  - Optimal threshold delta ~ 0.7 * E[|w|] (analytically derived under normality assumption).
  - Scaling factor alpha optimized per layer.
  - 16x compression; near full-precision accuracy on MNIST/CIFAR-10.
- **Note:** Still uses full-precision latent weights during training.
- **Source:** [arXiv](https://arxiv.org/abs/1605.04711)

### 4e. Sparsity-Control Ternary Weight Networks
- **Year:** 2021 (Neural Networks journal)
- **Key Contribution:** Uses a weight discretization regularizer (WDR) and sparsity-control approach (SCA) to train ternary networks. Can control sparsity of ternary weights through a controller parameter without relying on gradient estimators.
- **Source:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004093), [arXiv](https://arxiv.org/abs/2011.00580)

---

## 5. Quantization Error Accumulation

### 5a. "A White Paper on Neural Network Quantization" (Nagel et al., 2021)
- **Authors:** Markus Nagel et al.
- **Year:** 2021
- **Key Contribution:** Comprehensive survey of quantization errors in neural networks.
- **Mathematical Results:**
  - Quantization error arises from mapping high-precision floats to low-bit fixed-point: two components are **clipping error** and **rounding error**.
  - Rounding errors have upper limit of +/- 1/(2s) where s is the scale factor.
  - Trade-off between clipping and rounding errors.
  - Errors compound across layers: quantization errors from early layers propagate and grow through subsequent layers.
- **Source:** [arXiv](https://arxiv.org/abs/2106.08295)

### 5b. "Quantization Error Propagation" (QEP, 2025)
- **Year:** 2025 (arxiv 2504.09629)
- **Key Contribution:** Lightweight framework that explicitly propagates quantization error to compensate for accumulated errors in layer-wise post-training quantization.
- **Mathematical Results:**
  - Quantization errors do not remain localized but grow and propagate through subsequent layers.
  - Error compensation through explicit propagation significantly reduces accumulated error.
- **Source:** [arXiv](https://arxiv.org/abs/2504.09629)

### 5c. Error Accumulation in Iterative/Training Processes
- **Key Finding:** In iterative processes (including training), stepwise quantization errors accumulate progressively. The iterative nature amplifies error across successive steps.
- **For training specifically:**
  - Accumulating gradients in quantized precision can result in **zero-gradient or high-error gradients**, especially at low precision.
  - Standard mitigation: maintain weights in full precision (floating-point), update smoothly, requantize each step.
  - **STE overlooks discretization errors** between latent and discrete values, which become significant at low bit-widths.
  - Quantization error in momentum persists until a non-zero gradient update arrives, leading to progressive accumulation during sparse-gradient periods.

### 5d. "Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models"
- **Year:** 2025 (arxiv 2508.12094)
- **Key Contribution:** Shows that in diffusion models (iterative denoising), quantization errors accumulate across sampling steps and compromise output fidelity. Analogous to training error accumulation.
- **Source:** [arXiv](https://arxiv.org/abs/2508.12094)

### 5e. Key Insight for Ternary Training
- **The fundamental tension:** Ternary quantization (to {-1, 0, 1}) produces large quantization errors per step. Without full-precision accumulators, these errors compound.
- **Stochastic rounding** provides unbiasedness (errors cancel in expectation) but variance grows over time.
- **Error feedback** (as in ECO) provides a principled way to track and compensate for accumulated quantization errors within a fixed memory budget.
- **The critical question:** Can error accumulation be bounded over the entire training trajectory (millions of steps) with only ~4 bytes of state per parameter?

---

## 6. Memory-Efficient Training of 1-bit / Ternary LLMs (2024-2026)

### 6a. BitNet b1.58 (2024)
- **Authors:** Shuming Ma, Hongyu Wang et al. (Microsoft Research)
- **Year:** 2024 (February)
- **ArXiv:** 2402.17764
- **Key Contribution:** Ternary {-1, 0, 1} weights matching full-precision LLaMA at 3B+ parameters.
- **Training:** Standard STE + Adam with BF16 shadow weights. **~16 bytes/param during training.**
- **Note:** Training is memory-expensive. Gradients and optimizer states kept in high precision. Shadow weights stored alongside ternary weights.
- **Source:** [arXiv](https://arxiv.org/abs/2402.17764)

### 6b. BitNet b1.58 2B4T (2025)
- **Authors:** Shuming Ma, Hongyu Wang, Shaohan Huang et al. (Microsoft Research)
- **Year:** 2025 (April)
- **ArXiv:** 2504.12285
- **Key Contribution:** Open-weights 2B parameter model trained on 4T tokens. Competitive with FP16 models.
- **Training:** BF16 master weights + STE. Still ~16 bytes/param. No training memory innovation.
- **Key observation:** Distribution of normalized latent weights shows "quantization-valley" structure; ~42.3% of weights are zero. BitNet naturally converges to sparse ternary representation.
- **Source:** [arXiv](https://arxiv.org/abs/2504.12285), [HuggingFace](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

### 6c. TernaryLM (2026)
- **Authors:** (arxiv 2602.07374)
- **Year:** 2026 (February)
- **Key Contribution:** 132M-parameter transformer with native 1-bit ternary quantization during training. Uses STE and **adaptive per-layer scaling factors**.
- **Results:**
  - Validation perplexity 58.42 on TinyStories.
  - 82.47% F1 on MRPC paraphrase detection.
  - **2.4x memory reduction** (498MB vs 1197MB) with comparable inference latency.
  - Stable training dynamics across diverse corpora.
- **Key Insight:** Middle transformer layers exhibit highest compatibility with extreme quantization, informing non-uniform precision strategies.
- **Limitation:** Only 132M parameters; unclear if it scales to billions.
- **Source:** [arXiv](https://arxiv.org/abs/2602.07374)

### 6d. Direct Quantized Training (DQT) with Stochastic Rounding (2024)
- **See Section 4a above.**
- Directly trains ternary weights without latent weights. Most directly relevant to the memory-efficient training goal.
- **Source:** [arXiv](https://arxiv.org/abs/2412.04787)

### 6e. ECO: Quantized Training without Full-Precision Master Weights (2025)
- **See Section 4c above.**
- Error-feedback mechanism eliminates master weights. Proven convergence guarantees.
- **Source:** [arXiv](https://arxiv.org/abs/2601.22101)

### 6f. Continual Quantization-Aware Pre-Training (2025)
- **ArXiv:** 2502.11895
- **Key Contribution:** Investigates when to transition from 16-bit to 1.58-bit pre-training for BitNet language models.
- **Source:** [arXiv](https://arxiv.org/abs/2502.11895)

### 6g. "Every Bit Counts: A Theoretical Study of Precision-Expressivity Tradeoffs in Quantized Transformers"
- **Year:** 2026 (February)
- **ArXiv:** 2602.02707
- **Key Contribution:** Theoretical analysis of how precision affects expressivity in quantized transformers.
- **Source:** [arXiv](https://arxiv.org/abs/2602.02707)

### 6h. Sparse-BitNet (2026)
- **ArXiv:** 2603.05168
- **Key Contribution:** Shows that 1.58-bit LLMs are naturally friendly to semi-structured sparsity, enabling further compression.
- **Source:** [arXiv](https://arxiv.org/abs/2603.05168)

---

## 7. GXNOR-Net and Direct Ternary Weight Updates Without Full-Precision Shadows

### 7a. GXNOR-Net
- **Authors:** Lei Deng, Peng Jiao, Jing Pei, Zhenzhi Wu, Guoqi Li
- **Year:** 2018 (Neural Networks journal; arXiv 2017)
- **ArXiv:** 1705.09283
- **Key Contribution:** Unified discretization framework for both weights AND activations. Uses a **Discrete State Transition (DST)** methodology with a **probabilistic projection operator** to constrain weights in discrete space without storing full-precision hidden weights.
- **Mechanism:**
  - Multi-step neuronal activation discretization with derivative approximation for backpropagation on discrete DNNs.
  - Probabilistic projection operator directly realizes DST: weights transition between discrete states {-1, 0, +1} based on gradient-derived transition probabilities.
  - No full-precision hidden weights stored during the entire training phase.
  - When both weights and activations are ternary, DNNs reduce to sparse binary networks (GXNOR-Nets): only non-zero weight AND non-zero activation enables XNOR logic.
- **Mathematical Results:**
  - The paper provides the probabilistic framework but **no formal convergence proof** for the DST methodology.
  - The approach is empirically validated on MNIST, CIFAR-10, and SVHN.
  - Flexible: can modify state number of weights/activations for various hardware platforms (not limited to binary/ternary).
- **Key Limitation:** Validated only on small vision benchmarks. No LLM-scale experiments. No theoretical convergence guarantees.
- **Source:** [arXiv](https://arxiv.org/abs/1705.09283), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608018300108), [GitHub](https://github.com/AcrossV/Gated-XNOR)

### 7b. Other Papers with Direct Discrete Weight Updates

**SGDAT: SGD with Adaptive Threshold (2023)**
- Suppresses weight flip frequency via adaptive thresholds per parameter.
- Significantly improves SGD performance in BNNs to be comparable to Adam.
- **Source:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005544)

**Fast and Slow Gradient Approximation for BNNs (FSG, 2024)**
- **ArXiv:** 2412.11777
- Proposes gradient approximation with superior convergence: faster descent rates, lower loss values.
- **Source:** [arXiv](https://arxiv.org/abs/2412.11777)

**Annealing-Inspired Training of Optical Neural Networks with Ternary Weights (2025)**
- Uses annealing-inspired approach for ternary weight training in optical hardware.
- Published in Communications Physics (Nature).
- **Source:** [Nature](https://www.nature.com/articles/s42005-025-01972-y)

---

## Summary of Key Mathematical Convergence Results

| Paper | Setting | Convergence Result | Limitations |
|---|---|---|---|
| Yin et al. 2019 | 2-layer BNN, STE | Converges to critical point of population loss if STE is properly chosen | 2-layer only |
| Beyond Discreteness 2025 | 2-layer BNN, STE | O(n^2) samples for ergodic convergence (tight); O(n^4) for last-iterate | 2-layer only |
| High-Dim Dynamics 2025 | Linear model, STE | ODE convergence; plateau then sharp drop; stability regimes identified | Linear/shallow |
| Bridging Discrete 2023 | General discrete | STE = first-order approx; ReinMax = second-order | No convergence rate |
| Gupta et al. 2015 | SGD + SR | Unbiased; converges at same rate as full precision in expectation | Convex/simple |
| Ozkara et al. 2025 | Adam + SR | Lower quantization error than NR in convergence bound | LLM empirical |
| ECO 2025 | Error-feedback | Converges to constant-radius neighborhood with decaying LR; naive removal fails | FP8/INT4, not ternary |
| Bop 2019 | Binary weights | No formal proof; empirical convergence | Binary only, small-scale |
| GXNOR-Net 2018 | Ternary weights+acts | No formal proof; probabilistic DST framework | Small-scale vision |

---

## Key Research Gaps Identified

1. **No convergence theory for ternary-specific optimizers.** All theoretical results are for binary or continuous quantization. The ternary case {-1, 0, +1} with its unique three-state transition structure has no dedicated convergence analysis.

2. **No Bop extension to ternary.** Despite Bop's success for binary networks and its memory efficiency, no published work extends the EMA+threshold framework to ternary weights.

3. **No LLM-scale validation of latent-free discrete training.** GXNOR-Net's DST and Bop are only validated on small vision models. DQT (2412.04787) is the closest but still uses Adam optimizer states.

4. **ECO's error-feedback mechanism has not been applied to ternary training.** ECO works with FP8/INT4 quantization but has not been tested with extreme ternary quantization.

5. **The interaction between stochastic rounding, momentum, and discrete weight spaces is poorly understood.** Ozkara et al. (2025) analyze SR+Adam for continuous weights; the discrete case remains open.

6. **No paper combines Bop's threshold mechanism with stochastic rounding for ternary LLMs.** This hybrid approach is the most promising unexplored direction.

---

## Sources

### Topic 1: Convergence Theory for Quantized Training
- [Understanding STE - Yin et al. 2019](https://openreview.net/forum?id=Skh4jRcKQ)
- [Beyond Discreteness - Finite Sample Analysis 2025](https://arxiv.org/abs/2505.18113)
- [High-Dimensional Learning Dynamics 2025](https://arxiv.org/abs/2510.10693)
- [Bridging Discrete and Backpropagation - NeurIPS 2023](https://arxiv.org/abs/2304.08612)
- [Overcoming Oscillations in QAT - ICML 2022](https://arxiv.org/abs/2203.11086)
- [Coarse Gradient for Nonlinear Classification](https://link.springer.com/article/10.1007/s40687-021-00281-4)
- [Relaxed Variable Splitting Convergence](https://www.frontiersin.org/articles/10.3389/fams.2020.00013/full)

### Topic 2: Stochastic Rounding
- [Deep Learning with Limited Numerical Precision - Gupta 2015](https://arxiv.org/abs/1502.02551)
- [Stochastic Rounding for LLM Training - Ozkara 2025](https://arxiv.org/abs/2502.20566)
- [Training with Fewer Bits - Edge LLMs 2025](https://arxiv.org/abs/2511.00874)
- [Low-Bit Optimizer EMA Dynamics 2025](https://arxiv.org/html/2505.00347)
- [Efficient Stochastic Rounding Method](https://arxiv.org/pdf/2103.13445)

### Topic 3: Bop and Extensions
- [Latent Weights Do Not Exist - Helwegen 2019](https://arxiv.org/abs/1906.02107)
- [Bop2ndOrder - CVPR 2021](https://arxiv.org/abs/2104.05124)
- [Comprehensive Study on Binary Optimizer](https://openreview.net/pdf?id=y53aaSM5o)
- [Probabilistic Optimizer for BNNs 2025](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009695)

### Topic 4: Ternary/Discrete Training Without Latent Weights
- [Direct Quantized Training with SR - 2024](https://arxiv.org/abs/2412.04787)
- [Training BNNs in Binary Weight Space - 2024](https://openreview.net/forum?id=Dm4qrBuFKH)
- [ECO: Quantized Training without Master Weights - 2025](https://arxiv.org/abs/2601.22101)
- [Ternary Weight Networks - Li 2016](https://arxiv.org/abs/1605.04711)
- [Sparsity-Control Ternary Weight Networks](https://arxiv.org/abs/2011.00580)

### Topic 5: Quantization Error Accumulation
- [White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
- [Quantization Error Propagation (QEP) 2025](https://arxiv.org/abs/2504.09629)
- [Error Propagation in Quantized Diffusion Models](https://arxiv.org/abs/2508.12094)
- [Survey of Quantization Methods](https://arxiv.org/pdf/2103.13630)

### Topic 6: Memory-Efficient 1-bit/Ternary LLM Training
- [BitNet b1.58 - 2024](https://arxiv.org/abs/2402.17764)
- [BitNet b1.58 2B4T - 2025](https://arxiv.org/abs/2504.12285)
- [TernaryLM - 2026](https://arxiv.org/abs/2602.07374)
- [Continual QAT Pre-Training](https://arxiv.org/abs/2502.11895)
- [Every Bit Counts - Precision-Expressivity 2026](https://arxiv.org/abs/2602.02707)
- [Sparse-BitNet 2026](https://arxiv.org/abs/2603.05168)

### Topic 7: GXNOR-Net and Direct Ternary Updates
- [GXNOR-Net - Deng et al. 2018](https://arxiv.org/abs/1705.09283)
- [SGDAT - 2023](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005544)
- [Fast and Slow Gradient for BNNs - 2024](https://arxiv.org/abs/2412.11777)
- [Annealing-Inspired Ternary Training - 2025](https://www.nature.com/articles/s42005-025-01972-y)
