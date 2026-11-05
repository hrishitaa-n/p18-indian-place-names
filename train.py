
import os, math, time, json, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Config, MiniGPT

PAD, EOS = 0, 1          # "." in the text view; id 1 is both start and end token
BLOCK = 32

def load(path):
    return open(path).read().split("\n")

def build_vocab(names):
    chars = sorted(set("".join(names)))
    return {c: i + 2 for i, c in enumerate(chars)}

def encode(names, stoi, block=BLOCK):
    out = torch.full((len(names), block + 1), PAD, dtype=torch.long)
    for i, n in enumerate(names):
        s = [EOS] + [stoi[c] for c in n] + [EOS]
        out[i, :len(s)] = torch.tensor(s)
    x, y = out[:, :-1].clone(), out[:, 1:].clone()
    y[y == PAD] = -100
    return x, y

@torch.no_grad()
def evaluate(model, x, y, dev, bs=1024):
    model.eval(); tot, cnt = 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i+bs].to(dev), y[i:i+bs].to(dev)
        logits, _ = model(xb)
        l = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                                              ignore_index=-100, reduction="sum")
        tot += l.item(); cnt += (yb != -100).sum().item()
    model.train()
    return tot / cnt          # nats per character

def main(epochs=6, bs=256, lr=2e-3, out="out"):
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tr, va, te = load("data/train.txt"), load("data/val.txt"), load("data/test.txt")
    stoi = build_vocab(tr + va + te)
    xt, yt = encode(tr, stoi); xv, yv = encode(va, stoi); xe, ye = encode(te, stoi)
    cfg = Config(vocab_size=len(stoi) + 2)
    model = MiniGPT(cfg).to(dev)
    print(f"device={dev} params={model.n_params():,} vocab={cfg.vocab_size}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = math.ceil(len(xt) / bs); total = epochs * steps_per_epoch; warm = 200
    sched = lambda s: lr * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    log_steps, tr_l, va_l = [], [], []
    best, step, t0, run = 1e9, 0, time.time(), 0.0
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            ix = perm[i:i+bs]
            xb, yb = xt[ix].to(dev), yt[ix].to(dev)
            for g in opt.param_groups: g["lr"] = sched(step)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            run = 0.9 * run + 0.1 * loss.item() if step else loss.item()
            step += 1
            if step % 200 == 0 or step == total:
                v = evaluate(model, xv, yv, dev)
                log_steps.append(step); tr_l.append(run); va_l.append(v)
                print(f"ep {ep+1} step {step}/{total} train {run:.3f} val {v:.3f} ({v/math.log(2):.3f} bpc)")
                if v < best:
                    best = v
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "stoi": stoi}, f"{out}/mini_gpt.pt")
    secs = time.time() - t0
    ck = torch.load(f"{out}/mini_gpt.pt", map_location=dev); model.load_state_dict(ck["model"])
    test = evaluate(model, xe, ye, dev)
    m = {"params": model.n_params(), "train_seconds": round(secs, 1), "val_bpc": best / math.log(2),
         "test_bpc": test / math.log(2), "epochs": epochs, "train_names": len(tr)}
    json.dump(m, open(f"{out}/metrics_transformer.json", "w"), indent=2); print(m)
    plt.figure(figsize=(6, 4))
    plt.plot(log_steps, tr_l, label="train (smoothed)"); plt.plot(log_steps, va_l, label="validation")
    plt.xlabel("step"); plt.ylabel("loss (nats/char)"); plt.title("Mini-Transformer training loss")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{out}/loss_curve.png", dpi=150)

if __name__ == "__main__":
    main()
