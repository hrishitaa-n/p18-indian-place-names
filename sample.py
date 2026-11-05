
import torch
from model import Config, MiniGPT

def load_model(path="out/mini_gpt.pt", dev="cpu"):
    ck = torch.load(path, map_location=dev)
    m = MiniGPT(Config(**ck["cfg"])).to(dev); m.load_state_dict(ck["model"]); m.eval()
    stoi = ck["stoi"]; itos = {i: c for c, i in stoi.items()}
    return m, stoi, itos

@torch.no_grad()
def generate(model, stoi, itos, prefix="", n=10, temperature=0.8, seed=None, max_len=24):
    """Autoregressive sampling. temperature<1 = safer/repetitive, >1 = more random."""
    if seed is not None: torch.manual_seed(seed)
    prefix = prefix.lower().strip()
    bad = [c for c in prefix if c not in stoi]
    if bad: raise ValueError(f"characters not in vocabulary: {bad}")
    dev = next(model.parameters()).device
    start = [1] + [stoi[c] for c in prefix]
    idx = torch.tensor([start] * n, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(max_len - len(prefix)):
        logits, _ = model(idx[:, -model.c.block_size:])
        logits = logits[:, -1, :] / max(temperature, 1e-3)
        logits[:, 0] = float("-inf")                       # never emit PAD
        nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        nxt[done] = 1
        idx = torch.cat([idx, nxt[:, None]], 1)
        done |= nxt == 1
        if done.all(): break
    out = []
    for row in idx.tolist():
        s = ""
        for t in row[1:]:
            if t == 1: break
            s += itos[t]
        out.append(s.strip().title())
    return out

if __name__ == "__main__":
    m, stoi, itos = load_model()
    for t in (0.5, 0.8, 1.2):
        print(t, generate(m, stoi, itos, n=10, temperature=t, seed=42))
