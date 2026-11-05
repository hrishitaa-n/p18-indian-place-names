---
title: Indian Place Name Generator
emoji: 🏘️
colorFrom: orange
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---
# P18 · Mini Transformer from Scratch

Character-level decoder-only transformer (4 layers, 6 heads, d=192, 1.79M params) with sinusoidal positional
encoding and hand-written causal self-attention, trained on Indian place names. Generates new names with
temperature sampling, behind a profanity/slur filter.

- **Data:** GeoNames India dump, populated places, 295,055 train names. © GeoNames, licensed CC BY 4.0.
- **Result:** validation 2.492 bits/char, test 2.492 bits/char (6 epochs, 5.6 min on a Colab T4).
- **Run locally:** `pip install torch gradio && python app.py`
- **Retrain:** `python src/data.py && python src/train.py`

See `MODEL_CARD.md` for intended use, limits and bias notes. No API keys are used.
