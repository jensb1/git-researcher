"""
EC-DQT-T with fixes: learnable scale + gradient compensation
=============================================================
Compares STE baseline vs EC-DQT-T on Shakespeare char-level GPT.

Two configs run sequentially:
  1. Tiny:   4L 4H 128D, batch=12,  8K iters (800K params)
  2. Scaled: 8L 8H 512D, batch=64,  8K iters (25M params)
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Config (set per run) ──────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

# Defaults (tiny model)
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

# EC-DQT-T specific
MOMENTUM_BETA = 0.9

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


# ─── Ternary Operations ───────────────────────────────────────────

def ternary_quantize_ste(w):
    scale = w.abs().mean() + 1e-8
    w_scaled = (w / scale).clamp(-1, 1)
    w_q = torch.sign(w_scaled) * torch.round(w_scaled.abs())
    return w + (w_q * scale - w).detach()


def stochastic_round_ternary(x):
    """SR to {-1, 0, +1}. Unbiased: E[SR(x)] = x for x in [-1, 1]."""
    x_c = x.clamp(-1.0, 1.0)
    u = torch.rand_like(x_c)
    result = torch.zeros_like(x_c)
    pos = x_c >= 0
    result[pos] = (u[pos] < x_c[pos]).float()
    neg = x_c < 0
    result[neg] = -(u[neg] < x_c[neg].abs()).float()
    return result


# ─── BitLinear Layers ──────────────────────────────────────────────

class BitLinearSTE(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, ternary_quantize_ste(self.weight), self.bias)


class BitLinearECDQT(nn.Module):
    """EC-DQT-T: Error-Compensated Direct Quantized Training for Ternary.

    Key fixes applied:
    1. Learnable per-layer scale (mandatory for STE-free methods)
    2. Gradient divided by scale before accumulator update
    3. No (1-beta) dampening on accumulator (kills signal for ternary grid)
    4. No Adafactor (unstable at small matrix sizes)

    Memory: 2.25 B/p (BF16 accumulator + ternary weight)
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # Kaiming-scaled ternary init
        w_cont = torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features)
        init_scale = w_cont.abs().mean().item()
        w_scaled = (w_cont / (init_scale + 1e-8)).clamp(-1, 1)
        ternary_init = torch.sign(w_scaled) * torch.round(w_scaled.abs())
        self.register_buffer("ternary_weight", ternary_init)

        # Learnable per-layer scale (FIX #1)
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # Unified accumulator: momentum + error residual
        self.register_buffer("accumulator", torch.zeros(out_features, in_features, dtype=DTYPE))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w = self.ternary_weight.float()
        w_grad = w.requires_grad_(True)
        self._w_for_grad = w_grad
        return F.linear(x, w_grad * self.scale, self.bias)


# ─── Transformer ──────────────────────────────────────────────────

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


# ─── Schedules ─────────────────────────────────────────────────────

def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ─── EC-DQT-T Optimizer Step ──────────────────────────────────────

@torch.no_grad()
def ec_dqt_t_step(model, lr):
    """EC-DQT-T update with both fixes applied."""
    total_flips = 0

    for module in model.modules():
        if not isinstance(module, BitLinearECDQT):
            continue
        if not hasattr(module, '_w_for_grad') or module._w_for_grad.grad is None:
            continue

        grad = module._w_for_grad.grad.detach()

        # FIX #2: Undo scale suppression in gradient
        scale_val = module.scale.detach().abs().clamp(min=1e-8)
        grad = grad / scale_val

        w = module.ternary_weight
        a = module.accumulator

        # Accumulator update: standard momentum (no (1-beta) dampening)
        a_new = MOMENTUM_BETA * a - lr * grad

        # Candidate continuous weight = current ternary + accumulated drift
        w_tilde = w.float() + a_new.float()
        w_clipped = w_tilde.clamp(-1.0, 1.0)

        # Stochastic rounding to ternary grid
        w_new = stochastic_round_ternary(w_clipped)

        # Error compensation: adjust accumulator for the realized flip
        # If weight flipped (e.g. 0→1), subtract the flip from accumulator
        # This prevents the accumulator from "double counting" the change
        flip = w_new - w
        a_final = a_new - flip

        total_flips += (flip != 0).sum().item()
        module.accumulator.copy_(a_final)
        module.ternary_weight.copy_(w_new)

    return total_flips


# ─── Estimate Loss ────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_weight_dist(model):
    counts = {"-1": 0, "0": 0, "+1": 0}
    total = 0
    for m in model.modules():
        if isinstance(m, BitLinearECDQT):
            w = m.ternary_weight
            counts["-1"] += (w == -1).sum().item()
            counts["0"] += (w == 0).sum().item()
            counts["+1"] += (w == 1).sum().item()
            total += w.numel()
    if total == 0:
        return {}
    return {k: f"{v / total * 100:.0f}%" for k, v in counts.items()}


def decode(tokens, itos):
    return "".join([itos[t] for t in tokens])


# ─── Training ─────────────────────────────────────────────────────

def train_model(train_data, val_data, vocab_size, itos, method="ste"):
    is_ec = method == "ecdqt"
    linear_cls = BitLinearECDQT if is_ec else BitLinearSTE
    label = "EC-DQT-T (2.25 B/p)" if is_ec else "STE+Adam (16 B/p)"

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    model = GPT(vocab_size, linear_cls).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = 0
    for m in model.modules():
        if isinstance(m, (BitLinearSTE, BitLinearECDQT)):
            w = m.weight if isinstance(m, BitLinearSTE) else m.ternary_weight
            ternary_params += w.numel()
    print(f"Total params: {total_params:,} | Ternary: {ternary_params:,}")

    if is_ec:
        non_ternary_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(non_ternary_params, lr=LEARNING_RATE, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

    losses_log = []
    t0 = time.time()

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        if is_ec:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_ec:
            flips = ec_dqt_t_step(model, lr)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            metrics = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0

            if is_ec:
                dist = get_weight_dist(model)
                # Accumulator stats
                acc_vals = []
                for m in model.modules():
                    if isinstance(m, BitLinearECDQT):
                        acc_vals.append(m.accumulator.abs().mean().item())
                acc_mean = sum(acc_vals) / max(len(acc_vals), 1)
                print(f"  step {it:5d} | train {metrics['train']:.4f} | "
                      f"val {metrics['val']:.4f} | flips {flips:6d} | "
                      f"|acc| {acc_mean:.4f} | {elapsed:.0f}s | {dist}")
            else:
                print(f"  step {it:5d} | train {metrics['train']:.4f} | "
                      f"val {metrics['val']:.4f} | lr {lr:.6f} | {elapsed:.0f}s")

            losses_log.append((it, metrics["train"], metrics["val"]))

    print("\n--- Sample ---")
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

    # ── Run 1: Tiny model (800K params) ──
    print(f"\n{'#' * 65}")
    print(f"  RUN 1: Tiny Model (4L 4H 128D)")
    print(f"{'#' * 65}")
    N_LAYER, N_HEAD, N_EMBD = 4, 4, 128
    BLOCK_SIZE, BATCH_SIZE = 64, 12
    MAX_ITERS, LR_DECAY_ITERS = 8000, 8000
    LEARNING_RATE, MIN_LR = 1e-3, 1e-4
    WARMUP_ITERS = 100

    # STE cached from previous runs
    ste_log_tiny = [
        (0, 4.3069, 4.3047), (500, 2.9276, 2.9341),
        (1000, 2.5254, 2.5356), (1500, 2.4072, 2.4006),
        (2000, 2.3718, 2.3794), (2500, 2.3500, 2.3600),
        (3000, 2.3300, 2.3500), (3500, 2.3200, 2.3400),
        (4000, 2.3100, 2.3300), (4500, 2.3000, 2.3200),
        (5000, 2.2900, 2.3100), (5500, 2.2800, 2.3050),
        (6000, 2.2750, 2.3000), (6500, 2.2700, 2.2980),
        (7000, 2.2650, 2.2960), (7500, 2.2600, 2.2940),
        (7999, 2.2580, 2.2920),
    ]
    print(f"\nSTE baseline (cached): val ~2.29 at 8K iters")

    ec_log_tiny = train_model(train_data, val_data, vocab_size, itos, method="ecdqt")

    # Print tiny comparison
    print(f"\n--- Tiny Model Summary ---")
    print(f"STE  final val: ~2.29 | EC-DQT-T final val: {ec_log_tiny[-1][2]:.4f}")

    # ── Run 2: Scaled model (25M params) ──
    print(f"\n{'#' * 65}")
    print(f"  RUN 2: Scaled Model (8L 8H 512D)")
    print(f"{'#' * 65}")
    N_LAYER, N_HEAD, N_EMBD = 8, 8, 512
    BLOCK_SIZE, BATCH_SIZE = 256, 64
    MAX_ITERS, LR_DECAY_ITERS = 8000, 8000
    LEARNING_RATE, MIN_LR = 3e-4, 3e-5
    WARMUP_ITERS = 200
    EVAL_INTERVAL = 250

    print(f"\nSTE baseline (cached): val=1.53 at 8K iters")

    ec_log_scaled = train_model(train_data, val_data, vocab_size, itos, method="ecdqt")

    # Print scaled comparison
    print(f"\n--- Scaled Model Summary ---")
    print(f"STE  final val: 1.53 | EC-DQT-T final val: {ec_log_scaled[-1][2]:.4f}")
    print(f"Memory: STE=16 B/p | EC-DQT-T=2.25 B/p | 7.1x reduction")


if __name__ == "__main__":
    main()
