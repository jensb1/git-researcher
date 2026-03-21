"""
BitNet Training Comparison: STE vs TBop — Scaled Model
=======================================================
Character-level Shakespeare model with ternary {-1, 0, 1} weights.

Scaled config:
  - 8 layers, 8 heads, 512 embedding dim (~26M params)
  - context length 256, batch size 64
  - 8000 iterations, dropout 0.0
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Config ────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Model — scaled up
N_LAYER = 8
N_HEAD = 8
N_EMBD = 512
BLOCK_SIZE = 256  # context length
DROPOUT = 0.0

# Training
BATCH_SIZE = 64
MAX_ITERS = 8000
LEARNING_RATE = 3e-4
LR_DECAY_ITERS = 8000
WARMUP_ITERS = 200
MIN_LR = 3e-5
EVAL_INTERVAL = 250
EVAL_ITERS = 50

# TBop with auto-calibrated thresholds
TBOP_GAMMA = 0.1            # EMA adaptivity rate
# Thresholds expressed as multipliers of per-layer mean |EMA|
# tau = multiplier * layer_mean_abs_ema
TBOP_TAU_ACT_MULT = 1.0     # flip when EMA > 1x layer average (consistent signal)
TBOP_TAU_DEACT_MULT = 3.0   # deactivate when EMA > 3x average (very strong counter-signal)
TBOP_TAU_MULT_FINAL = 0.3   # final multiplier after cosine decay

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


# ─── BitLinear Layers ──────────────────────────────────────────────

class BitLinearSTE(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, ternary_quantize_ste(self.weight), self.bias)


class BitLinearTBop(nn.Module):
    """TBop with auto-calibrated thresholds.
    Memory: 2B (BF16 EMA) + 0.25B (ternary) = 2.25 B/p"""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        w_cont = torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features)
        init_scale = w_cont.abs().mean().item()
        w_scaled = (w_cont / (init_scale + 1e-8)).clamp(-1, 1)
        ternary_init = torch.sign(w_scaled) * torch.round(w_scaled.abs())
        self.register_buffer("ternary_weight", ternary_init)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.register_buffer("ema", torch.zeros(out_features, in_features, dtype=DTYPE))
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

def get_lr(it, max_lr=LEARNING_RATE, min_lr=MIN_LR):
    if it < WARMUP_ITERS:
        return max_lr * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return min_lr
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ─── TBop Optimizer Step ──────────────────────────────────────────

@torch.no_grad()
def tbop_step(model, step):
    """TBop with auto-calibrated per-layer thresholds.
    tau = multiplier * mean(|EMA|) per layer — scales automatically."""
    # Cosine decay on the multiplier
    ratio = min(step / LR_DECAY_ITERS, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    act_mult = TBOP_TAU_MULT_FINAL + coeff * (TBOP_TAU_ACT_MULT - TBOP_TAU_MULT_FINAL)
    deact_mult = TBOP_TAU_MULT_FINAL + coeff * (TBOP_TAU_DEACT_MULT - TBOP_TAU_MULT_FINAL)

    total_flips = 0

    for module in model.modules():
        if not isinstance(module, BitLinearTBop):
            continue
        if not hasattr(module, '_w_for_grad') or module._w_for_grad.grad is None:
            continue

        grad = module._w_for_grad.grad.detach()
        # Undo scale suppression: forward is w*scale, so grad_w = grad_out*scale
        # We want the raw directional signal, not the scale-suppressed one
        scale_val = module.scale.detach().abs().clamp(min=1e-8)
        grad = grad / scale_val

        w = module.ternary_weight
        m = module.ema

        m_new = (1.0 - TBOP_GAMMA) * m + TBOP_GAMMA * grad

        # Per-layer threshold: scaled by this layer's mean |EMA|
        layer_ema_scale = m_new.abs().mean() + 1e-10
        tau_act = act_mult * layer_ema_scale
        tau_deact = deact_mult * layer_ema_scale

        # FSM transitions
        to_pos = (w == 0) & (m_new < -tau_act)
        to_neg = (w == 0) & (m_new > tau_act)
        deact_pos = (w == 1) & (m_new > tau_deact)
        deact_neg = (w == -1) & (m_new < -tau_deact)

        w_new = w.clone()
        w_new[to_pos] = 1.0
        w_new[to_neg] = -1.0
        w_new[deact_pos] = 0.0
        w_new[deact_neg] = 0.0

        transitioned = to_pos | to_neg | deact_pos | deact_neg
        m_new[transitioned] = 0.0
        total_flips += transitioned.sum().item()

        module.ternary_weight.copy_(w_new)
        module.ema.copy_(m_new)

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
        if isinstance(m, BitLinearTBop):
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
    is_tbop = method == "tbop"
    linear_cls = BitLinearTBop if is_tbop else BitLinearSTE
    label = "TBop (2.25 B/p)" if is_tbop else "STE+Adam (16 B/p)"

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    model = GPT(vocab_size, linear_cls).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = 0
    for m in model.modules():
        if isinstance(m, (BitLinearSTE, BitLinearTBop)):
            w = m.weight if isinstance(m, BitLinearSTE) else m.ternary_weight
            ternary_params += w.numel()

    print(f"Total params: {total_params:,}")
    print(f"Ternary params: {ternary_params:,}")

    if is_tbop:
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

        if is_tbop:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_tbop:
            flips = tbop_step(model, it)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            metrics = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0

            if is_tbop:
                dist = get_weight_dist(model)
                # Compute current effective multiplier from cosine schedule
                ratio = min(it / LR_DECAY_ITERS, 1.0)
                c = 0.5 * (1.0 + math.cos(math.pi * ratio))
                act_m = TBOP_TAU_MULT_FINAL + c * (TBOP_TAU_ACT_MULT - TBOP_TAU_MULT_FINAL)
                ema_vals = []
                for m in model.modules():
                    if isinstance(m, BitLinearTBop):
                        ema_vals.append(m.ema.abs().mean().item())
                ema_mean = sum(ema_vals) / max(len(ema_vals), 1)
                print(f"  step {it:5d} | train {metrics['train']:.4f} | "
                      f"val {metrics['val']:.4f} | flips {flips:6d} | "
                      f"|ema| {ema_mean:.5f} | mult {act_m:.2f} | "
                      f"{elapsed:.0f}s | {dist}")
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
    print("BitNet Scaled Training Comparison: STE vs TBop")
    print(f"Device: {DEVICE}")
    print(f"Config: {N_LAYER}L {N_HEAD}H {N_EMBD}D | ctx={BLOCK_SIZE} batch={BATCH_SIZE} iters={MAX_ITERS}")

    train_data, val_data, vocab_size, itos = get_shakespeare_data()
    print(f"Dataset: {len(train_data):,} train, {len(val_data):,} val, {vocab_size} vocab")

    # STE baseline already measured at this scale (8L 8H 512D, 25M params, 8K iters)
    ste_log = [
        (0,    4.3133, 4.3139), (250,  2.6963, 2.7046), (500,  2.4277, 2.4310),
        (750,  2.2601, 2.2784), (1000, 2.1269, 2.1638), (1250, 2.0273, 2.0903),
        (1500, 1.9072, 2.0006), (1750, 1.8036, 1.9324), (2000, 1.7248, 1.8692),
        (2250, 1.6501, 1.7993), (2500, 1.5979, 1.7688), (2750, 1.5541, 1.7320),
        (3000, 1.5143, 1.7086), (3250, 1.4801, 1.6821), (3500, 1.4534, 1.6563),
        (3750, 1.4327, 1.6377), (4000, 1.3980, 1.6240), (4250, 1.3830, 1.6004),
        (4500, 1.3624, 1.5850), (4750, 1.3450, 1.5771), (5000, 1.3243, 1.5844),
        (5250, 1.3103, 1.5708), (5500, 1.2976, 1.5522), (5750, 1.2854, 1.5593),
        (6000, 1.2717, 1.5493), (6250, 1.2628, 1.5351), (6500, 1.2540, 1.5389),
        (6750, 1.2410, 1.5361), (7000, 1.2380, 1.5337), (7250, 1.2279, 1.5342),
        (7500, 1.2210, 1.5388), (7750, 1.2178, 1.5239), (7999, 1.2129, 1.5276),
    ]
    print(f"\nSTE baseline (cached): final val loss = 1.5276")
    tbop_log = train_model(train_data, val_data, vocab_size, itos, method="tbop")

    # Summary
    print(f"\n{'=' * 75}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Step':>6} | {'STE train':>10} {'STE val':>10} | {'TBop train':>10} {'TBop val':>10} | {'Gap':>7}")
    print("-" * 75)

    # Align logs by step
    ste_dict = {s: (t, v) for s, t, v in ste_log}
    tbop_dict = {s: (t, v) for s, t, v in tbop_log}
    for step in sorted(set(ste_dict.keys()) & set(tbop_dict.keys())):
        t1, v1 = ste_dict[step]
        t2, v2 = tbop_dict[step]
        gap = ((v2 - v1) / v1) * 100
        print(f"{step:6d} | {t1:10.4f} {v1:10.4f} | {t2:10.4f} {v2:10.4f} | {gap:+6.1f}%")

    ste_final = ste_log[-1][2]
    tbop_final = tbop_log[-1][2]
    gap_pct = ((tbop_final - ste_final) / ste_final) * 100
    print(f"\nFinal val loss: STE={ste_final:.4f} | TBop={tbop_final:.4f} | gap={gap_pct:+.1f}%")
    print(f"Memory: STE=16 B/p | TBop=2.25 B/p | 7.1x reduction")
    print(f"Model: {N_LAYER}L {N_HEAD}H {N_EMBD}D = ~{sum(p.numel() for p in GPT(vocab_size, BitLinearSTE).parameters()) / 1e6:.1f}M params")


if __name__ == "__main__":
    main()
