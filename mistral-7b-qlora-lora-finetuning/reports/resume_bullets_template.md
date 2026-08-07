# Resume Bullets

Use measured values only. The current run is still in progress, so final
perplexity, peak VRAM, and ablation metrics should be added after completion.

- Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA on Alpaca-cleaned,
  training 6.82M adapter parameters across `q_proj`, `k_proj`, `v_proj`, and
  `o_proj` while keeping the 7B-class base model frozen.
- Resumed the main single-GPU run from checkpoint 600 and progressed beyond
  checkpoint 900 / 1,200 target optimizer steps; latest observed validation
  loss improved from 1.0143 early in training to 0.9648 during the resumed run.
- Implemented manual LoRA from first principles in PyTorch and production QLoRA
  with PEFT, bitsandbytes NF4, double quantization, gradient checkpointing, and
  paged AdamW.
- Built evaluation hooks for held-out perplexity, qualitative prompt regression,
  and future rank/target-module ablations.
