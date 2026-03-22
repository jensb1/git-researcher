# TBop: Ternary Binary Optimizer for Memory-Efficient LLM Training

## Abstract

We present TBop (Ternary Binary Optimizer), a latent-weight-free training method for ternary {-1, 0, +1} neural networks that reduces per-parameter training memory from 16 bytes (STE + Adam) to 2.25 bytes — a 7.1x reduction. TBop extends the Binary Optimizer (Bop) framework from binary to ternary weights by modeling each parameter as a 3-state finite state machine with hysteresis: weights transition only through the zero state (no direct -1 ↔ +1 jumps), with asymmetric thresholds that make activation easier than deactivation. An exponential moving average of gradients accumulates directional evidence, and weight flips occur only when this evidence exceeds a per-layer auto-calibrated threshold. We identify three implementation details critical for convergence of any STE-free ternary training method: (1) a learnable per-layer scale parameter, (2) gradient compensation for scale-induced suppression, and (3) per-layer threshold normalization for scale invariance across model sizes. On character-level language modeling with GPT-style transformers, TBop matches STE+Adam quality within 4.6% at 25M parameters (val loss 1.60 vs 1.53) when given 4x more training iterations, and surpasses STE by 15.5% at 800K parameters. TBop trades compute for memory: it requires approximately 4x more iterations than STE but uses 7.1x less optimizer memory per parameter, enabling training of proportionally larger models on fixed hardware.

## Keywords

ternary neural networks, BitNet, 1-bit LLMs, binary optimizer, quantization-aware training, memory-efficient training, latent-weight-free optimization, straight-through estimator

## Search Queries

To verify novelty, search for:
- "ternary binary optimizer" OR "ternary Bop"
- "ternary weight training without latent weights"
- "three-state finite state machine neural network training"
- "threshold-based ternary weight optimization"
- "Bop ternary extension"
- "latent-weight-free ternary training"
- "discrete optimizer ternary neural network"
- "EMA threshold ternary weight flip"
- "binary optimizer extended to ternary"
- "FSM hysteresis neural network weight training"
