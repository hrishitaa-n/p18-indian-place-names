import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch, gradio as gr
from sample import load_model, generate
from safety import filter_names

torch.set_num_threads(2)

# ZeroGPU Spaces require at least one @spaces.GPU function at startup. This placeholder is never
# called (the model is tiny and runs on CPU); it only satisfies the check. Harmless elsewhere.
try:
    import spaces
    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:
    pass

def find(name):
    """Locate a file anywhere under the app folder (works even if folders were flattened on upload)."""
    for root, _, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        if name in files:
            return os.path.join(root, name)
    return None

MODEL_PATH = find("mini_gpt.pt")
if MODEL_PATH is None:
    raise FileNotFoundError("mini_gpt.pt is missing from the Space. Upload it via Files > Add file > Upload files.")
MODEL, STOI, ITOS = load_model(MODEL_PATH)
MP = find("metrics_transformer.json")
METRICS = json.load(open(MP)) if MP else {"params": 1785408, "train_names": 295055, "epochs": 6,
                                          "train_seconds": 337, "val_bpc": 2.492, "test_bpc": 2.492}
LOSS_IMG = find("loss_curve.png")
KNOWN = set()                                    # training names, for the novelty tag
TP = find("train.txt")
if TP:
    KNOWN = {l.strip() for l in open(TP) if l.strip()}

def run(prefix, temperature, count, seed):
    count = int(count)
    seed = random.randrange(10**6) if seed in (None, "") else int(seed)
    try:
        cands = generate(MODEL, STOI, ITOS, prefix=prefix or "", n=max(count * 4, 24),
                         temperature=float(temperature), seed=seed)
    except ValueError:
        return "Please use only letters a-z and spaces in the starting letters.", ""
    uniq = list(dict.fromkeys(c for c in cands if len(c) >= 3))
    safe, blocked = filter_names(uniq)
    shown = safe[:count]
    lines = []
    for n in shown:
        tag = "" if not KNOWN else ("  _(exists in dataset)_" if n.lower() in KNOWN else "  _(new)_")
        lines.append(f"{len(lines)+1}. **{n}**{tag}")
    stats = (f"seed {seed} · sampled {len(cands)} · duplicates removed {len(cands)-len(uniq)} · "
             f"blocked by safety filter {blocked}")
    return "\n".join(lines) or "No safe names generated, try a different prefix or temperature.", stats

with gr.Blocks(title="Indian Place Name Generator") as demo:
    gr.Markdown("# Indian Place Name Generator\nA 1.8M-parameter decoder-only transformer, written from scratch in PyTorch "
                "and trained on GeoNames India (CC BY 4.0). Names are invented; any resemblance to real places is coincidence.")
    with gr.Tab("Generate"):
        with gr.Row():
            prefix = gr.Textbox(label="Starting letters (optional)", placeholder="e.g. Ban")
            temp = gr.Slider(0.3, 1.5, value=0.8, step=0.05, label="Temperature")
        with gr.Row():
            count = gr.Slider(1, 10, value=10, step=1, label="How many names")
            seed = gr.Textbox(label="Seed (optional, for reproducible output)")
        btn = gr.Button("Generate", variant="primary")
        out = gr.Markdown(); info = gr.Markdown()
        btn.click(run, [prefix, temp, count, seed], [out, info])
    with gr.Tab("Training results"):
        if LOSS_IMG:
            gr.Image(LOSS_IMG, label="Training and validation loss")
        gr.Markdown(f"**Parameters:** {METRICS['params']:,} · **Train names:** {METRICS['train_names']:,} · "
                    f"**Epochs:** {METRICS['epochs']} · **Train time:** {METRICS['train_seconds']:.0f}s  \n"
                    f"**Validation:** {METRICS['val_bpc']:.3f} bits/char · **Test:** {METRICS['test_bpc']:.3f} bits/char")
    gr.Markdown("Data: GeoNames (geonames.org), CC BY 4.0. Safety filter blocks profanity and slurs; it can have false positives.")

if __name__ == "__main__":
    demo.launch()
