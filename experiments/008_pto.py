"""
PTO: Probabilistic Ternary Optimizer
=====================================
Each weight stores two INT8 logits for a categorical distribution
over {-1, 0, +1}. Gradients update logits via stochastic rounding.
Weights sampled via Gumbel-softmax early, argmax late.

Memory: 2.0 bytes/param (two INT8 logits) — cheapest method.

Runs tiny model (800K) then scaled model (25M).
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

# Will be set per run
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
BLOCK_SIZE = 64
BATCH_SIZE = 12
MAX_ITERS = 8000
LEARNING_RATE = 1e-3
LR_DECAY_ITERS = 8000
WARMUP_ITERS = 100
MIN_LR = 1e-4
EVAL_INTERVAL = 500
EVAL_ITERS = 50

# PTO specific
PTO_LOGIT_SCALE = 25.0     # Maps INT8 [-128,127] to real logits [-5.12, 5.08]
PTO_GRAD_SCALE = 500.0     # Amplifies gradient to produce meaningful INT8 steps
PTO_TAU_MAX = 2.0          # Temperature: high early (exploration)
PTO_TAU_MIN = 0.1          # Temperature: low late (exploitation)

torch.manual_seed(1337)


# ─── Data ──────────────────────────────────────────────────────────

def get_shakespeare_data():
    data_path = os.path.join(os.path.dirname(__file__) or ".", "data", "shakespeare.txt")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        print("Downloading Shakespeare dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, itos


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
    return x, y


# ─── PTO BitLinear Layer ──────────────────────────────────────────

class BitLinearPTO(nn.Module):
    """Probabilistic Ternary Optimizer layer.

    Stores two INT8 logits per weight: logit_neg (for -1) and logit_pos (for +1).
    logit_zero is implicitly 0 (reference class).

    Memory: 2 bytes/param (INT8 + INT8) + learnable scale (~0 B/p)
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # Initialize logits to 0 (uniform prior: equal probability for -1, 0, +1)
        self.register_buffer("logit_neg", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("logit_pos", torch.zeros(out_features, in_features, dtype=torch.int8))

        # Learnable per-layer scale
        init_scale = math.sqrt(2.0 / in_features) * 0.5  # rough Kaiming scale
        self.scale = nn.Parameter(torch.tensor(init_scale))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def _get_weights(self, tau=1.0, sample=True):
        """Derive ternary weights from logits."""
        S = PTO_LOGIT_SCALE
        # Three logits: (logit_neg/S, 0, logit_pos/S) divided by temperature
        ln = self.logit_neg.float() / (S * tau)
        lz = torch.zeros_like(ln)
        lp = self.logit_pos.float() / (S * tau)
        logits_3 = torch.stack([ln, lz, lp], dim=0)  # [3, out, in]

        if sample:
            # Gumbel-max trick for sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(logits_3).clamp(min=1e-10) + 1e-10))
            idx = (logits_3 + gumbel).argmax(dim=0)  # {0, 1, 2}
        else:
            idx = logits_3.argmax(dim=0)

        w = idx.float() - 1.0  # map {0,1,2} -> {-1,0,+1}
        return w

    def forward(self, x):
        # Get temperature from training progress (set externally)
        tau = getattr(self, '_tau', 1.0)
        sample = getattr(self, '_sample', True)
        w = self._get_weights(tau=tau, sample=sample)
        w_grad = w.requires_grad_(True)
        self._w_for_grad = w_grad
        return F.linear(x, w_grad * self.scale, self.bias)


# ─── Transformer (same as before) ─────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, linear_cls):
        super().__init__()
        self.c_attn = linear_cls(N_EMBD, 3 * N_EMBD)
        self.c_proj = linear_cls(N_EMBD, N_EMBD)
        self.n_head = N_HEAD

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, linear_cls):
        super().__init__()
        self.c_fc = linear_cls(N_EMBD, 4 * N_EMBD)
        self.c_proj = linear_cls(4 * N_EMBD, N_EMBD)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, linear_cls):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention(linear_cls)
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP(linear_cls)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, linear_cls):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block(linear_cls) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def get_temperature(it):
    """Exponential decay from tau_max to tau_min."""
    ratio = min(it / LR_DECAY_ITERS, 1.0)
    return PTO_TAU_MAX * (PTO_TAU_MIN / PTO_TAU_MAX) ** ratio


# ─── PTO Optimizer Step ───────────────────────────────────────────

@torch.no_grad()
def pto_step(model, lr, step):
    """Update INT8 logits based on gradients via stochastic rounding."""

    for module in model.modules():
        if not isinstance(module, BitLinearPTO):
            continue
        if not hasattr(module, '_w_for_grad') or module._w_for_grad.grad is None:
            continue

        grad = module._w_for_grad.grad.detach()

        # Gradient scale compensation (same fix as TBop)
        scale_val = module.scale.detach().abs().clamp(min=1e-8)
        grad = grad / scale_val

        # Compute continuous logit deltas
        delta = lr * grad * PTO_GRAD_SCALE  # amplify to INT8 scale

        # Stochastic round to integer
        delta_floor = delta.floor()
        frac = delta - delta_floor
        sr_noise = torch.rand_like(frac)
        delta_int = (delta_floor + (sr_noise < frac).float()).to(torch.int32)

        # Update logits:
        # If grad > 0 (loss wants w to decrease): increase logit_neg, decrease logit_pos
        # If grad < 0 (loss wants w to increase): decrease logit_neg, increase logit_pos
        new_pos = module.logit_pos.to(torch.int32) - delta_int
        new_neg = module.logit_neg.to(torch.int32) + delta_int

        # Clamp to INT8 range
        module.logit_pos.copy_(new_pos.clamp(-128, 127).to(torch.int8))
        module.logit_neg.copy_(new_neg.clamp(-128, 127).to(torch.int8))


# ─── Helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    # Use argmax (no sampling) for evaluation
    for m in model.modules():
        if isinstance(m, BitLinearPTO):
            m._sample = False
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    for m in model.modules():
        if isinstance(m, BitLinearPTO):
            m._sample = True
    model.train()
    return out


def get_weight_dist(model):
    counts = {"-1": 0, "0": 0, "+1": 0}
    total = 0
    for m in model.modules():
        if isinstance(m, BitLinearPTO):
            w = m._get_weights(tau=0.01, sample=False)  # near-deterministic
            counts["-1"] += (w == -1).sum().item()
            counts["0"] += (w == 0).sum().item()
            counts["+1"] += (w == 1).sum().item()
            total += w.numel()
    if total == 0:
        return {}
    return {k: f"{v / total * 100:.0f}%" for k, v in counts.items()}


def get_entropy(model):
    """Mean per-weight entropy of the categorical distribution."""
    S = PTO_LOGIT_SCALE
    entropies = []
    for m in model.modules():
        if isinstance(m, BitLinearPTO):
            ln = m.logit_neg.float() / S
            lz = torch.zeros_like(ln)
            lp = m.logit_pos.float() / S
            logits = torch.stack([ln, lz, lp], dim=0)
            probs = F.softmax(logits, dim=0)
            ent = -(probs * (probs + 1e-10).log()).sum(dim=0).mean()
            entropies.append(ent.item())
    return sum(entropies) / max(len(entropies), 1)


def decode(tokens, itos):
    return "".join([itos[t] for t in tokens])


# ─── Training ─────────────────────────────────────────────────────

def train_pto(train_data, val_data, vocab_size, itos):
    print(f"\n{'=' * 65}")
    print(f"  PTO: Probabilistic Ternary Optimizer (2.0 B/p)")
    print(f"{'=' * 65}")

    model = GPT(vocab_size, BitLinearPTO).to(DEVICE)
    ternary_params = sum(m.logit_pos.numel() for m in model.modules() if isinstance(m, BitLinearPTO))
    print(f"Ternary params: {ternary_params:,} | Memory: {ternary_params * 2 / 1e6:.2f} MB")

    non_ternary_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(non_ternary_params, lr=LEARNING_RATE, betas=(0.9, 0.95))

    losses_log = []
    t0 = time.time()

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        tau = get_temperature(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Set temperature on all PTO layers
        for m in model.modules():
            if isinstance(m, BitLinearPTO):
                m._tau = tau
                m._sample = True

        x, y = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(non_ternary_params, 1.0)
        optimizer.step()

        # PTO logit update
        pto_step(model, lr, it)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            metrics = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            dist = get_weight_dist(model)
            ent = get_entropy(model)
            print(f"  step {it:5d} | train {metrics['train']:.4f} | "
                  f"val {metrics['val']:.4f} | tau {tau:.3f} | "
                  f"H {ent:.3f} | {elapsed:.0f}s | {dist}")
            losses_log.append((it, metrics["train"], metrics["val"]))

    print("\n--- Sample ---")
    for m in model.modules():
        if isinstance(m, BitLinearPTO):
            m._sample = False
    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    gen = model.generate(ctx, 300)
    print(decode(gen[0].tolist(), itos))

    return losses_log


# ─── Main ─────────────────────────────────────────────────────────

def main():
    global N_LAYER, N_HEAD, N_EMBD, BLOCK_SIZE, BATCH_SIZE
    global MAX_ITERS, LEARNING_RATE, LR_DECAY_ITERS, WARMUP_ITERS, MIN_LR, EVAL_INTERVAL

    train_data, val_data, vocab_size, itos = get_shakespeare_data()
    print(f"Device: {DEVICE}")
    print(f"Dataset: {len(train_data):,} train, {len(val_data):,} val, {vocab_size} vocab")

    # ── Tiny model ──
    print(f"\n{'#' * 65}")
    print(f"  Tiny Model: 4L 4H 128D (800K params)")
    print(f"{'#' * 65}")
    N_LAYER, N_HEAD, N_EMBD = 4, 4, 128
    BLOCK_SIZE, BATCH_SIZE = 64, 12
    MAX_ITERS, LR_DECAY_ITERS = 8000, 8000
    LEARNING_RATE, MIN_LR = 1e-3, 1e-4
    WARMUP_ITERS = 100

    pto_tiny = train_pto(train_data, val_data, vocab_size, itos)
    print(f"\n  Tiny: PTO val={pto_tiny[-1][2]:.4f} | STE ~2.29 | TBop ~2.01")

    # ── Scaled model ──
    print(f"\n{'#' * 65}")
    print(f"  Scaled Model: 8L 8H 512D (25M params)")
    print(f"{'#' * 65}")
    N_LAYER, N_HEAD, N_EMBD = 8, 8, 512
    BLOCK_SIZE, BATCH_SIZE = 256, 64
    MAX_ITERS, LR_DECAY_ITERS = 8000, 8000
    LEARNING_RATE, MIN_LR = 3e-4, 3e-5
    WARMUP_ITERS = 200
    EVAL_INTERVAL = 250

    pto_scaled = train_pto(train_data, val_data, vocab_size, itos)
    print(f"\n  Scaled: PTO val={pto_scaled[-1][2]:.4f} | STE ~1.53 | TBop ~2.20")

    print(f"\n{'=' * 65}")
    print(f"  ALL METHODS COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Method':<15} {'Memory':>8} {'Tiny val':>10} {'Scaled val':>12}")
    print(f"  {'STE+Adam':<15} {'16 B/p':>8} {'~2.29':>10} {'~1.53':>12}")
    print(f"  {'TBop':<15} {'2.25 B/p':>8} {'~2.01':>10} {'~2.20':>12}")
    print(f"  {'EC-DQT-T':<15} {'2.25 B/p':>8} {'~2.51':>10} {'~2.51':>12}")
    print(f"  {'PTO':<15} {'2.0 B/p':>8} {pto_tiny[-1][2]:>10.4f} {pto_scaled[-1][2]:>12.4f}")


if __name__ == "__main__":
    main()
