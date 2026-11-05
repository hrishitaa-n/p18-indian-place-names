# Model card: Mini-Transformer place-name generator

- **Model:** decoder-only transformer, 1,785,408 parameters, character vocabulary, trained from scratch (PyTorch).
- **Intended use:** coursework demo and creative name generation (fiction, games). Not for official naming or geographic facts.
- **Data:** GeoNames India populated places (feature class P), ASCII names, deduplicated, 90/5/5 split. CC BY 4.0, attribution to GeoNames.
- **Metrics:** val 2.492 bpc, test 2.492 bpc.
- **Safety:** blocklist filter (English + transliterated Hindi + slurs); whole-word and substring matching. Known limit: can block real, harmless names and miss novel offensive strings.
- **Bias test (5,000 names per temperature):** at T=0.8 the regional suffix mix is close to the corpus (total variation distance 2.4 points); at T=0.5 it over-produces easy endings (Tamil 5.1% vs 2.0%, Kerala 1.2% vs 0.2%, distance 7.1). Small groups (Kerala, Bengal, Marathi) have few samples, single seed. Suffix families only approximate regions.
- **Memorisation:** 31% of names at T=0.5, 17% at T=0.8 and 8% at T=1.2 are exact copies of real training names.
- **Filter:** blocks 114 of 295,055 real names (0.04% false positives) and 2 of 5,000 generated names.
- **Comparison:** Char-LSTM (2.443 test bpc) beat the Mini-Transformer (2.492) and GPT-2 small fine-tuned (2.546). Single runs.
- **Limits:** names reflect GeoNames coverage, which is uneven across states; ASCII transliteration drops native scripts; the BERT plausibility scorer cannot detect plausible fakes.
- **Licence:** code MIT; data CC BY 4.0 (GeoNames).
