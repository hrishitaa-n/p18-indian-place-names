# Model card: Mini-Transformer place-name generator

- **Model:** decoder-only transformer, 1,785,408 parameters, character vocabulary, trained from scratch (PyTorch).
- **Intended use:** coursework demo and creative name generation (fiction, games). Not for official naming or geographic facts.
- **Data:** GeoNames India populated places (feature class P), ASCII names, deduplicated, 90/5/5 split. CC BY 4.0, attribution to GeoNames.
- **Metrics:** val 2.492 bpc, test 2.492 bpc.
- **Safety:** blocklist filter (English + transliterated Hindi + slurs); whole-word and substring matching. Known limit: can block real, harmless names and miss novel offensive strings.
- **Bias / limits:** TODO fill in after eval (region-typical suffix share vs. corpus, novelty %). Names reflect GeoNames coverage, which is uneven across states; ASCII transliteration drops native scripts.
- **Licence:** code MIT; data CC BY 4.0 (GeoNames).
