# P18 · Mini Transformer from Scratch: Indian Place Name Generator

A tiny decoder-only transformer, written from scratch in PyTorch, that invents new Indian place names. It is compared with a character LSTM and a fine-tuned GPT-2 small, and deployed as a public Streamlit app with a safety filter.

**Live app:** [https://YOUR-APP-NAME.streamlit.app](https://your-app-name.streamlit.app/)  <!-- replace with your Streamlit URL -->

## What it does
Type optional starting letters, choose a temperature (0.3 to 1.5) and how many names (1 to 10), and get new place names. Add a seed to reproduce the same output. Generated names pass a blocklist filter first, and the app shows how many names it sampled, how many duplicates it removed and how many the filter blocked.

## Model
| Item | Value |
|---|---|
| Architecture | Decoder-only transformer, 4 layers, 6 heads, d=192 |
| Parameters | 1,785,408 |
| Details | Sinusoidal positional encoding, hand-written masked (causal) self-attention, tied embeddings, character vocabulary |
| Training | PyTorch, AdamW, cosine schedule, 6 epochs on a Colab T4 (about 6 min) |
| Data | GeoNames India, populated places, 295,055 training names |

## Results
Single run per model, same task, bits per character (lower is better).

| Metric | Mini-Transformer | Char-LSTM | GPT-2 small, fine-tuned |
|---|---|---|---|
| Parameters | 1.79M | 1.76M | 124.4M |
| Validation bits/char | 2.492 | 2.442 | 2.544 |
| Test bits/char | 2.492 | 2.443 | 2.546 |
| Train time | 351 s | 270 s | 772 s |
| Unique names (of 1,000) | 99.9% | 100% | 99.9% |
| Novel names (of unique) | 84.4% | 80.6% | 79.6% |
| DistilBERT plausibility | 0.918 | not measured | 0.907 |
| CPU ms per name | 49.7 | 41.5 | 323.7 |

Prompting vs fine-tuning (GPT-2 small): zero-shot 0.752, few-shot 0.803, fine-tuned 0.907 plausibility.
The LSTM scored best; the small models beat GPT-2 here. Differences are small and from one run each.

**10 generated names** (temperature 0.8, seed 42): Araikkal, Pacharupani, Mangalpadar, Chak Shivai, Kirare, Pura Harsan, Kusma, Fatehpur Gujar, Bhakhpura, Bhadkot.

Training loss curve: `assets/loss_curve.png`.

## Responsible AI tests
| Test | Result |
|---|---|
| Bias (regional suffix families, 5,000 names) | Close to the corpus at T=0.8 (distance 2.4 points); over-produces easy endings at T=0.5 (Tamil 5.1% vs 2.0%, Kerala 1.2% vs 0.2%) |
| Memorisation (exact copies of real names) | 31% at T=0.5, 17% at T=0.8, 8% at T=1.2 |
| Safety filter | Blocks 114 of 295,055 real names (0.04% false positives) and 2 of 5,000 generated names |

## Failure cases
- Gibberish at high temperature (T=1.2), for example "Ugaikkakavundh".
- Repeated endings, for example "Fatehpurpur Khatu".
- Copies real places at low temperature (31% at T=0.5).
- Zero-shot GPT-2 returns sentence fragments or state names.
- One diffusion mascot candidate contained garbled text.

## Project layers
| Layer | Where |
|---|---|
| I Foundations | Generative (transformer) vs discriminative (DistilBERT scorer); zero-shot vs few-shot vs fine-tuned; temperature and softmax; bias and copy-rate by temperature |
| II GANs and LLMs | Transformer core, masked attention, positional encoding, LSTM comparison, GPT-2 comparison |
| III GPT, BERT, multimodal | GPT-2 small, DistilBERT plausibility scorer, sd-turbo mascot |
| IV Tools and generation | PyTorch from scratch; diffusion mascot (`assets/mascot.png`) |
| V Responsible AI | Safety filter, bias and memorisation tests, licences, `MODEL_CARD.md` |

## Repository
```
streamlit_app.py     main app (Streamlit)
app.py               optional Gradio version (pip install gradio)
requirements.txt
MODEL_CARD.md
src/                 data.py  model.py  train.py  sample.py  safety.py
out/                 mini_gpt.pt  metrics_transformer.json  loss_curve.png
assets/              loss_curve.png  mascot.png  bias_chart.png  gpt2_loss_curve.png
results/             comparison, bias, memorisation and filter result files
notebooks/           Colab notebooks: core training, LSTM, GPT-2 + BERT, bias + mascot
```

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Retrain
Run the notebooks in `notebooks/` on a Colab T4 GPU, or locally: `python src/data.py` then `python src/train.py`.

## Secrets
No API keys are used. If you add any, store them in the hosting platform's secrets settings and never commit them.

## Licences and attribution
- **Data:** this work uses data from [GeoNames](https://www.geonames.org), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Code:** MIT (see `LICENSE`).
- **Models:** GPT-2 is MIT-licensed and DistilBERT is Apache 2.0. The mascot was made with sd-turbo; see its model card on Hugging Face for terms.

## Limitations
Names are invented and may coincide with real places or carry unintended meanings. The filter is a blocklist and is not complete. Results are single runs, so small gaps may be noise. Suffix families only approximate regions. See `MODEL_CARD.md` for more.
