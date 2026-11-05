import os, sys, json, random
import streamlit as st
import torch
import pandas as pd

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

_m = find("mascot.png")
_ca, _cb = st.columns([1, 5])
if _m: _ca.image(_m, width=100)
_cb.title("Indian Place Name Generator")
st.write("A 1.8M-parameter decoder-only transformer, written from scratch in PyTorch and trained on "
         "GeoNames India (CC BY 4.0). Names are invented; any resemblance to real places is coincidence.")

tab_gen, tab_res, tab_cmp, tab_rai = st.tabs(["Generate", "Training results", "Model comparison", "Responsible AI"])

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

with tab_cmp:
    st.subheader("Mini-Transformer vs Char-LSTM vs GPT-2 small")
    st.table(pd.DataFrame({
        "Mini-Transformer": ["1.79M", "2.492", "2.492", "351 s", "99.9%", "84.4%", "0.918", "49.7"],
        "Char-LSTM":        ["1.76M", "2.442", "2.443", "270 s", "100%", "80.6%", "not measured", "41.5"],
        "GPT-2 small (fine-tuned)": ["124.4M", "2.544", "2.546", "772 s", "99.9%", "79.6%", "0.907", "323.7"],
    }, index=["Parameters", "Val bits/char", "Test bits/char", "Train time", "Unique (of 1,000)",
              "Novel (of unique)", "BERT plausibility", "CPU ms per name"]))
    st.markdown("**Prompting vs fine-tuning (GPT-2)**")
    st.table(pd.DataFrame({"Valid names": ["124 / 200", "196 / 200", "1000 / 1000"],
                           "BERT plausibility": [0.752, 0.803, 0.907]},
                          index=["Zero-shot prompt", "Few-shot prompt (8 examples)", "Fine-tuned"]))
    st.write("The two small models trained from scratch beat the 70x larger GPT-2 on bits per character. Single run per model, "
             "so small gaps may be noise. Plausibility comes from a DistilBERT real-vs-corrupted classifier "
             "(91.9% held-out accuracy); it rewards typical names and cannot detect plausible fakes.")
    st.caption("Failure cases: gibberish at T=1.2 (e.g. 'Ugaikkakavundh'), repeated endings ('Fatehpurpur'), and zero-shot GPT-2 "
               "returning sentence fragments ('Cat Was The Name').")

with tab_rai:
    st.subheader("Bias: regional suffix families, corpus vs generated")
    bc = find("bias_chart.png")
    if bc: st.image(bc)
    bt = find("bias_suffix_shares.csv")
    if bt: st.dataframe(pd.read_csv(bt, index_col=0).round(2))
    st.write("At low temperature (T=0.5) the model over-produces easy-to-predict endings: Tamil 5.1% vs 2.0% in the corpus, "
             "Kerala 1.2% vs 0.2%. At T=0.8 the mix is close to the corpus. Suffix families only approximate regions.")
    st.subheader("Memorisation and safety filter")
    st.table(pd.DataFrame({"Copy of a real name": ["31.1%", "17.1%", "8.3%"], "Unique": ["90.1%", "99.5%", "99.9%"]},
                          index=["T=0.5", "T=0.8", "T=1.2"]))
    st.write("The safety filter wrongly blocks 114 of 295,055 real place names (0.04%) and blocks 2 of 5,000 generated names.")
    st.caption("Licences: data GeoNames CC BY 4.0; code MIT; mascot made with sd-turbo (check the model's licence terms).")

st.caption("Data: GeoNames (geonames.org), CC BY 4.0. The safety filter blocks profanity and slurs; it can have false positives.")
