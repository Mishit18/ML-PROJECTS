# Resume Bullets

- Fine-tuned `Mistral-7B-v0.3` with QLoRA rank-8 on Alpaca-cleaned, training only `6.82M` parameters (`0.0939%` of 7.25B weights) while freezing `99.9061%` of the model.
- Implemented 4-bit NF4 QLoRA with double quantization and paged AdamW, fitting Mistral 7B fine-tuning on a single RTX 4060 Laptop GPU under about `7.5GB` observed VRAM.
- Reduced held-out validation perplexity from `4.97` to `2.62` across 1,000 validation examples, a `47.28%` perplexity reduction over the base model.
- Ablated LoRA rank and target modules on real 7B runs; rank-16 reached `1.0106` eval loss at 100 steps vs rank-8 `1.0143`, while q/v-only underperformed at `1.0353`, validating q/k/v/o rank-8 as the practical full-run choice.
