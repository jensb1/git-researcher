# Empirical Results: EC-DQT-T (Idea 002)

## Experiment Setup
- **Model:** Tiny GPT (4 layers, 4 heads, 128 embd) with BitLinear layers
- **Dataset:** TinyShakespeare (1M chars, char-level)
- **Config:** ctx=64, batch=12, 2000 iters, dropout=0.0
- **Device:** Apple MPS (FP32, no BF16)

## Results

### STE Baseline
| Step | Train Loss | Val Loss |
|------|-----------|---------|
| 0 | 4.3069 | 4.3047 |
| 200 | 3.2930 | 3.3589 |
| 400 | 3.0276 | 3.0341 |
| 1000 | 2.5302 | 2.5426 |
| 1999 | 2.3750 | 2.3815 |

### EC-DQT-T
| Step | Train Loss | Val Loss | Weight Dist (-1/0/+1) |
|------|-----------|---------|----------------------|
| 0 | 4.1804 | 4.1872 | 20% / 61% / 20% |
| 200 | 3.3061 | 3.3483 | 30% / 40% / 30% |
| 1000 | 3.3254 | 3.3526 | 41% / 19% / 41% |
| 1999 | 3.3247 | 3.3423 | 42% / 16% / 42% |

**Final gap: +40.4%** (val loss 3.34 vs 2.38)

## Failure Analysis

### Root Cause: Model did not converge
EC-DQT-T loss plateaued at ~3.3 from step 200 onward. Weights flipped actively early on (36K flips at step 200) but settled into a non-learning state.

### Issues Identified

1. **Accumulator dynamics mismatch with ternary grid spacing.**
   - The `(1-beta)` dampening in the original formula `a = beta*a - (1-beta)*lr*g` produces steady-state accumulator values ~0.0005 with standard LR — far too small relative to the ternary grid spacing of 1.0.
   - Removing dampening and using LR=0.1 produced accumulator values ~0.5, which created flips but didn't lead to convergence — weights flipped but the model didn't learn meaningful patterns.

2. **Adafactor preconditioner instability at small scale.**
   - The factored second moment (row/col outer product) is designed for large matrices (4096x4096+). At 128x384 (this model), the approximation is poor and the preconditioner distorted gradient signals.
   - Clipping at 10x helped prevent explosion but couldn't fix the fundamentally noisy scaling.

3. **Stochastic rounding noise dominates at small scale.**
   - With 786K ternary params and batch size 12, the gradient signal-to-noise ratio is very low. SR adds additional variance (up to 0.25/param) on top of already noisy gradients.
   - The research correctly predicted that EC-DQT-T would work better at scale (larger models, larger batches), but at this toy scale the noise overwhelms the signal.

4. **No STE in backward pass.**
   - The original method computes gradients w.r.t. ternary weights directly. But without the STE providing a continuous gradient landscape, the gradients through ternary weights are extremely noisy and carry limited directional information.

### Key Lessons
- The theoretical analysis holds (accumulator mechanics, error compensation, convergence frameworks) but the **hyperparameter regime** for ternary is fundamentally different from continuous training.
- The method may require much larger scale (3B+ params, batch 2048+) to show its strengths, as the DQT paper showed the gap narrows from 130M to 1B.
- The `(1-beta)` dampening term in the momentum formula is a critical implementation detail not highlighted in the theoretical pseudocode — it reduces effective accumulation by 10x.
- **Stochastic rounding-based methods may not be suitable for very small models** where gradient noise already dominates.
