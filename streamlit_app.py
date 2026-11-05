import os, sys, json, random
import streamlit as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "src"))
from sample import load_model, generate
from safety import filter_names

st.set_page_config(page_title="Indian Place Name Generator", page_icon="🏘️")
torch.set_num_threads(2)

def find(name):
    """Locate a file anywhere in the repo (robust to folder layout)."""
    for root, _, files in os.walk(HERE):
        if name in files:
            return os.path.join(root, name)
    return None

@st.cache_resource
def get_model():
    return load_model(find("mini_gpt.pt"))

@st.cache_data
def get_known():
    p = find("train.txt")
    return {l.strip() for l in open(p) if l.strip()} if p else set()

@st.cache_data
def get_metrics():
    p = find("metrics_transformer.json")
    return json.load(open(p)) if p else None

model, stoi, itos = get_model()
KNOWN, METRICS = get_known(), get_metrics()

st.title("Indian Place Name Generator")
st.write("A 1.8M-parameter decoder-only transformer, written from scratch in PyTorch and trained on "
         "GeoNames India (CC BY 4.0). Names are invented; any resemblance to real places is coincidence.")

tab_gen, tab_res = st.tabs(["Generate", "Training results"])

with tab_gen:
    c1, c2 = st.columns(2)
    prefix = c1.text_input("Starting letters (optional)", placeholder="e.g. Ban")
    temperature = c2.slider("Temperature", 0.3, 1.5, 0.8, 0.05)
    c3, c4 = st.columns(2)
    count = c3.slider("How many names", 1, 10, 10)
    seed_txt = c4.text_input("Seed (optional, for reproducible output)")
    if st.button("Generate", type="primary"):
        try:
            seed = int(seed_txt) if seed_txt.strip() else random.randrange(10**6)
        except ValueError:
            st.error("Seed must be a whole number."); st.stop()
        try:
            cands = generate(model, stoi, itos, prefix=prefix, n=max(count * 4, 24),
                             temperature=temperature, seed=seed)
        except ValueError:
            st.error("Please use only letters a-z and spaces in the starting letters."); st.stop()
        uniq = list(dict.fromkeys(c for c in cands if len(c) >= 3))
        safe, blocked = filter_names(uniq)
        shown = safe[:count]
        if not shown:
            st.warning("No safe names generated, try a different prefix or temperature.")
        for i, n in enumerate(shown, 1):
            tag = "" if not KNOWN else (" _(exists in dataset)_" if n.lower() in KNOWN else " _(new)_")
            st.markdown(f"{i}. **{n}**{tag}")
        st.caption(f"seed {seed} · sampled {len(cands)} · duplicates removed {len(cands)-len(uniq)} · "
                   f"blocked by safety filter {blocked}")

with tab_res:
    img = find("loss_curve.png")
    if img:
        st.image(img, caption="Training and validation loss")
    if METRICS:
        a, b, c = st.columns(3)
        a.metric("Parameters", f"{METRICS['params']:,}")
        b.metric("Validation bits/char", f"{METRICS['val_bpc']:.3f}")
        c.metric("Test bits/char", f"{METRICS['test_bpc']:.3f}")
        st.write(f"Trained on {METRICS['train_names']:,} names for {METRICS['epochs']} epochs "
                 f"({METRICS['train_seconds']:.0f}s on a Colab T4).")

st.caption("Data: GeoNames (geonames.org), CC BY 4.0. The safety filter blocks profanity and slurs; it can have false positives.")
