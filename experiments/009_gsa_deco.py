"""
GSA-AT & DECO-T: Two more STE-free methods
============================================
GSA-AT:  INT8 gradient sign counter + threshold (1.25 B/p)
DECO-T:  Damped error-compensated optimizer (2.25 B/p)

Both use learnable scale + gradient compensation fixes.
Runs tiny model (800K, 8K iters) for each.
"""

import math, os, time, torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

N_LAYER, N_HEAD, N_EMBD = 4, 4, 128
BLOCK_SIZE, BATCH_SIZE = 64, 12
MAX_ITERS, LR_DECAY_ITERS = 8000, 8000
LEARNING_RATE, MIN_LR = 1e-3, 1e-4
WARMUP_ITERS, EVAL_INTERVAL, EVAL_ITERS = 100, 500, 50

# GSA-AT
GSA_GAMMA = 0.97       # counter decay
GSA_TAU_ACT = 15       # INT8 counter threshold for activation
GSA_TAU_DEACT = 25     # higher threshold for deactivation

# DECO-T
DECO_BETA = 0.9        # momentum coefficient
DECO_GAMMA_MAX = 0.3   # max error injection fraction

torch.manual_seed(1337)


def get_shakespeare_data():
    data_path = os.path.join(os.path.dirname(__file__) or ".", "data", "shakespeare.txt")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path)
    with open(data_path) as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, itos


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    return (torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(DEVICE),
            torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(DEVICE))


def stochastic_round_ternary(x):
    x_c = x.clamp(-1, 1)
    u = torch.rand_like(x_c)
    r = torch.zeros_like(x_c)
    pos = x_c >= 0; r[pos] = (u[pos] < x_c[pos]).float()
    neg = x_c < 0; r[neg] = -(u[neg] < x_c[neg].abs()).float()
    return r


# ─── BitLinear variants ───────────────────────────────────────────

class BitLinearGSA(nn.Module):
    """GSA-AT: INT8 gradient sign counter. 1.25 B/p."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        w = torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)
        s = w.abs().mean().item()
        ws = (w / (s + 1e-8)).clamp(-1, 1)
        self.register_buffer("ternary_weight", torch.sign(ws) * torch.round(ws.abs()))
        self.scale = nn.Parameter(torch.tensor(s))
        self.register_buffer("counter", torch.zeros(out_f, in_f, dtype=torch.int8))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        w = self.ternary_weight.float().requires_grad_(True)
        self._w_for_grad = w
        return F.linear(x, w * self.scale, self.bias)


class BitLinearDECO(nn.Module):
    """DECO-T: Damped error-compensated optimizer. 2.25 B/p."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        w = torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)
        s = w.abs().mean().item()
        ws = (w / (s + 1e-8)).clamp(-1, 1)
        self.register_buffer("ternary_weight", torch.sign(ws) * torch.round(ws.abs()))
        self.scale = nn.Parameter(torch.tensor(s))
        self.register_buffer("momentum", torch.zeros(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        w = self.ternary_weight.float().requires_grad_(True)
        self._w_for_grad = w
        return F.linear(x, w * self.scale, self.bias)


# ─── Transformer ──────────────────────────────────────────────────

class Attn(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.c_attn, self.c_proj = cls(N_EMBD, 3*N_EMBD), cls(N_EMBD, N_EMBD)
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(N_EMBD, 2)
        nh = N_HEAD; hs = C//nh
        q,k,v = [t.view(B,T,nh,hs).transpose(1,2) for t in (q,k,v)]
        return self.c_proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C))

class MLP(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.fc, self.proj, self.act = cls(N_EMBD, 4*N_EMBD), cls(4*N_EMBD, N_EMBD), nn.GELU()
    def forward(self, x): return self.proj(self.act(self.fc(x)))

class Block(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.mlp = nn.LayerNorm(N_EMBD), Attn(cls), nn.LayerNorm(N_EMBD), MLP(cls)
    def forward(self, x): return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

class GPT(nn.Module):
    def __init__(self, vs, cls):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vs, N_EMBD), nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks, self.ln_f = nn.ModuleList([Block(cls) for _ in range(N_LAYER)]), nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vs, bias=False); self.wte.weight = self.lm_head.weight
    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.lm_head(self.ln_f(x))
        return logits, (F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1)) if targets is not None else None)
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits[:,-1,:], -1), 1)], 1)
        return idx


def get_lr(it):
    if it < WARMUP_ITERS: return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS: return MIN_LR
    c = 0.5 * (1 + math.cos(math.pi * (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)))
    return MIN_LR + c * (LEARNING_RATE - MIN_LR)


# ─── GSA-AT Step ──────────────────────────────────────────────────

@torch.no_grad()
def gsa_step(model, step):
    """GSA-AT: accumulate gradient signs in INT8 counter, flip on threshold."""
    # Cosine threshold decay
    ratio = min(step / LR_DECAY_ITERS, 1.0)
    c = 0.5 * (1 + math.cos(math.pi * ratio))
    tau_a = max(3, int(3 + c * (GSA_TAU_ACT - 3)))
    tau_d = max(3, int(3 + c * (GSA_TAU_DEACT - 3)))
    flips = 0

    for m in model.modules():
        if not isinstance(m, BitLinearGSA): continue
        if not hasattr(m, '_w_for_grad') or m._w_for_grad.grad is None: continue

        grad = m._w_for_grad.grad.detach()
        scale_val = m.scale.detach().abs().clamp(min=1e-8)
        grad = grad / scale_val  # gradient scale fix

        w = m.ternary_weight
        cnt = m.counter.to(torch.int32)

        # Counter update: decay + sign accumulation
        sign_g = grad.sign().to(torch.int32)
        cnt = (GSA_GAMMA * cnt.float()).round().to(torch.int32) + sign_g

        # FSM transitions (same convention as TBop)
        to_pos  = (w == 0)  & (cnt < -tau_a)
        to_neg  = (w == 0)  & (cnt > tau_a)
        deact_p = (w == 1)  & (cnt > tau_d)
        deact_n = (w == -1) & (cnt < -tau_d)

        w_new = w.clone()
        w_new[to_pos] = 1.0; w_new[to_neg] = -1.0
        w_new[deact_p] = 0.0; w_new[deact_n] = 0.0

        trans = to_pos | to_neg | deact_p | deact_n
        cnt[trans] = 0
        flips += trans.sum().item()

        m.ternary_weight.copy_(w_new)
        m.counter.copy_(cnt.clamp(-128, 127).to(torch.int8))

    return flips


# ─── DECO-T Step ──────────────────────────────────────────────────

@torch.no_grad()
def deco_step(model, lr):
    """DECO-T: momentum + stochastic rounding + adaptive error feedback."""
    flips = 0
    eps = 1e-8

    for m in model.modules():
        if not isinstance(m, BitLinearDECO): continue
        if not hasattr(m, '_w_for_grad') or m._w_for_grad.grad is None: continue

        grad = m._w_for_grad.grad.detach()
        scale_val = m.scale.detach().abs().clamp(min=1e-8)
        grad = grad / scale_val  # gradient scale fix

        w = m.ternary_weight
        mom = m.momentum

        # Step 2: Momentum update (error from previous step already in mom)
        mom_new = DECO_BETA * mom + (1 - DECO_BETA) * grad

        # Step 3: Continuous candidate
        w_tilde = w.float() - lr * mom_new

        # Step 4: Stochastic rounding to ternary
        w_new = stochastic_round_ternary(w_tilde)

        # Step 5: Quantization error
        error = w_tilde - w_new.float()

        # Step 6: Adaptive damped error injection
        g_mag = grad.abs().mean()
        e_mag = error.abs().mean() + eps
        gamma_t = min(DECO_GAMMA_MAX, (g_mag / e_mag).item())
        mom_new = mom_new + gamma_t * error

        flips += (w_new != w).sum().item()
        m.ternary_weight.copy_(w_new)
        m.momentum.copy_(mom_new)

    return flips


# ─── Eval & Train ─────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, tr, va):
    model.eval()
    out = {}
    for s, d in [("train", tr), ("val", va)]:
        out[s] = sum(model(*get_batch(d))[1].item() for _ in range(EVAL_ITERS)) / EVAL_ITERS
    model.train()
    return out


def get_dist(model, cls):
    c = {"-1": 0, "0": 0, "+1": 0}; t = 0
    for m in model.modules():
        if isinstance(m, cls):
            w = m.ternary_weight
            c["-1"] += (w==-1).sum().item(); c["0"] += (w==0).sum().item(); c["+1"] += (w==1).sum().item()
            t += w.numel()
    return {k: f"{v/t*100:.0f}%" for k,v in c.items()} if t else {}


def train(tr, va, vs, itos, method):
    cls = {"gsa": BitLinearGSA, "deco": BitLinearDECO}[method]
    step_fn = {"gsa": lambda m, lr, it: gsa_step(m, it), "deco": lambda m, lr, it: deco_step(m, lr)}[method]
    label = {"gsa": "GSA-AT (1.25 B/p)", "deco": "DECO-T (2.25 B/p)"}[method]

    print(f"\n{'='*65}\n  {label}\n{'='*65}")
    model = GPT(vs, cls).to(DEVICE)
    tp = sum(m.ternary_weight.numel() for m in model.modules() if isinstance(m, cls))
    print(f"Ternary params: {tp:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, betas=(0.9, 0.95))
    log = []; t0 = time.time()

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for pg in opt.param_groups: pg["lr"] = lr
        opt.zero_grad()
        _, loss = model(*get_batch(tr))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        flips = step_fn(model, lr, it)

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            met = estimate_loss(model, tr, va)
            dist = get_dist(model, cls)
            print(f"  step {it:5d} | train {met['train']:.4f} | val {met['val']:.4f} | "
                  f"flips {flips:6d} | {time.time()-t0:.0f}s | {dist}")
            log.append((it, met["train"], met["val"]))

    print("\n--- Sample ---")
    print("".join(itos[t] for t in model.generate(torch.zeros(1,1,dtype=torch.long,device=DEVICE), 200)[0].tolist()))
    return log


def main():
    tr, va, vs, itos = get_shakespeare_data()
    print(f"Device: {DEVICE} | {N_LAYER}L {N_HEAD}H {N_EMBD}D | 8K iters")

    gsa_log = train(tr, va, vs, itos, "gsa")
    deco_log = train(tr, va, vs, itos, "deco")

    print(f"\n{'='*65}\n  RESULTS (Tiny Model, 800K params, 8K iters)\n{'='*65}")
    print(f"  {'Method':<15} {'Memory':>8} {'Final val':>10}")
    print(f"  {'STE+Adam':<15} {'16 B/p':>8} {'~2.29':>10}")
    print(f"  {'TBop':<15} {'2.25 B/p':>8} {'~2.01':>10}")
    print(f"  {'EC-DQT-T':<15} {'2.25 B/p':>8} {'~2.51':>10}")
    print(f"  {'PTO':<15} {'2.0 B/p':>8} {'~3.05':>10}")
    print(f"  {'GSA-AT':<15} {'1.25 B/p':>8} {gsa_log[-1][2]:>10.4f}")
    print(f"  {'DECO-T':<15} {'2.25 B/p':>8} {deco_log[-1][2]:>10.4f}")


if __name__ == "__main__":
    main()
