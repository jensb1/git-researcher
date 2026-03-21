# Research Problem

## Title
Memory-Efficient Training Methods for BitNet 1.58-bit LLMs Without STE Latent Weight Overhead

## Problem Statement
BitNet b1.58 achieves remarkable inference efficiency with ternary {-1, 0, 1} weights (~0.2 bytes/param), but training requires ~16 bytes per parameter due to the Straight-Through Estimator (STE) approach. STE necessitates maintaining full-precision (FP32/BF16) "latent weights" as gradient accumulators, plus Adam optimizer states (first and second moments in FP32). This creates an 80x gap between training and inference memory per parameter.

**The core question:** Can we develop novel training methods for BitNet-style ternary networks that eliminate or drastically reduce the STE latent weight overhead, constraining extra memory to **at most 4 bytes per parameter** beyond the ternary weights themselves?

The 4-byte constraint means: the ternary weight (~0.2 bytes) plus at most 4 bytes of training state = ~4.2 bytes total per parameter. Compare this to the current 16 bytes/param with standard STE + Adam.

## Context
**BitNet Architecture:**
- Replaces nn.Linear with BitLinear layers
- Weights quantized to {-1, 0, 1} via absmean quantization: `w_q = RoundClip(w / (|w|_mean + ε), -1, 1)`
- Activations quantized to 8-bit using absmax per-token quantization
- Matrix multiply becomes pure addition/subtraction (no multiplications)
- Matches FP16 LLaMA performance at 3B+ parameters (BitNet b1.58)

**Current STE Training:**
1. Maintain FP32/BF16 latent weights (the "real" weights)
2. Forward: quantize latent weights to ternary on-the-fly
3. Backward: pass gradients through quantization as if identity (STE)
4. Update: apply gradients to latent weights via Adam
5. Repeat — latent weights accumulate small gradient steps that ternary weights cannot represent

**Memory breakdown (mixed-precision BF16 + Adam):**
| Component | Bytes/Param |
|---|---|
| BF16 latent weights | 2 |
| BF16 gradients | 2 |
| FP32 optimizer master copy | 4 |
| FP32 first moment (m) | 4 |
| FP32 second moment (v) | 4 |
| **Total** | **16** |

**Key papers:**
- BitNet (2310.11453) — original 1-bit Transformer
- BitNet b1.58 (2402.17764) — ternary weights, matches full-precision at scale
- BitNet b1.58-2B-4T (2504.12285) — open-source 2B model trained on 4T tokens
- "Latent Weights Do Not Exist" / Bop (1906.02107) — binary optimizer without latent weights
- Direct Quantized Training with Stochastic Rounding (2412.04787) — eliminates latent weights via stochastic rounding
- QuEST (2502.05003) — trust gradient estimator replacing STE
- FP4 training (2505.19115) — fully quantized training of LLMs

## Known Information
- **STE works but is memory-expensive.** It is the standard approach for all quantization-aware training of 1-bit and ternary networks.
- **Bop (Binary Optimizer)** demonstrated that latent weights are not strictly necessary for binary networks. It uses an exponential moving average of gradients (~4 bytes/param for momentum) to decide when to flip binary weights. However, it was only validated on small vision models (CIFAR-10/ImageNet with ResNets), not LLM-scale ternary training.
- **Stochastic rounding** (2412.04787) can eliminate full-precision latent weights by directly updating quantized weights, using randomness to preserve gradient information in expectation. Early results show ternary-only training is feasible.
- **QuEST** replaces STE with a trust gradient estimator that weights gradient components inversely by their quantization error. Achieves stable W1A1 training.
- **8-bit optimizers** (bitsandbytes) reduce Adam states from 8 to ~2 bytes/param but still require latent weights.
- **FP4 training** shows weights, activations, gradients, and optimizer states can all be 4-bit.
- The ternary case {-1, 0, 1} has a unique structure: weight transitions are limited (e.g., -1→0, 0→1, 1→0, etc.), which could be exploited by a specialized optimizer.
- Gradient information is fundamentally continuous — the challenge is encoding enough of it in ≤4 bytes to make good discrete update decisions.

## Success Criteria
A successful outcome would be one or more concrete training method proposals that:
1. **Memory constraint:** ≤ 4 bytes extra per parameter during training (beyond the ternary weights)
2. **Convergence:** Theoretically motivated and plausibly converges for LLM-scale training
3. **Quality:** Expected to approach STE+Adam training quality (within 5-10% on standard benchmarks)
4. **Practicality:** Implementable with existing hardware (GPUs/TPUs), no exotic requirements
5. **Novelty:** Goes beyond simply combining existing techniques (e.g., "just use 8-bit Adam" is not novel enough)

## Scope

### In Scope
- Novel optimizer designs for ternary weight networks
- Gradient compression/accumulation strategies that fit in ≤4 bytes/param
- Stochastic methods (stochastic rounding, probabilistic weight updates)
- Approaches that exploit the discrete structure of {-1, 0, 1}
- Hybrid approaches (e.g., different strategies for different training phases)
- Methods that reuse or repurpose existing per-param memory differently
- Theoretical analysis of why these methods should work

### Out of Scope
- Post-training quantization (PTQ) — we want training from scratch
- Inference optimization — already well-solved by bitnet.cpp
- Activation memory reduction (focus is on weight/optimizer memory)
- Changing the BitNet architecture itself (keep ternary {-1, 0, 1} weights)
- Distributed training memory optimizations (focus on per-device per-param cost)

## Supporting Files
- BitNet repo: https://github.com/microsoft/BitNet
- BitNet b1.58 paper: https://arxiv.org/abs/2402.17764
- Bop paper: https://arxiv.org/abs/1906.02107
- Stochastic rounding training: https://arxiv.org/abs/2412.04787
- QuEST paper: https://arxiv.org/abs/2502.05003
