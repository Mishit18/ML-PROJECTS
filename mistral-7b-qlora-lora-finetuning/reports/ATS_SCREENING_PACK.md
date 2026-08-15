# ATS Screening Pack

## Best-Fit Roles

- AI/ML Engineer Intern
- Generative AI Intern
- LLM Fine-Tuning Intern
- Applied Scientist Intern
- Machine Learning Research Intern

## Strong Resume Bullets

- Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA on Alpaca-cleaned,
  training 6.82M adapter parameters across attention projection modules while
  keeping the 7B-class base model frozen.
- Implemented LoRA from first principles in PyTorch and production QLoRA using
  PEFT, bitsandbytes NF4, double quantization, gradient checkpointing, and
  paged AdamW for single-GPU fine-tuning.
- Completed the 1,200-step run and same-split 1,000-sample evaluation; reduced
  held-out perplexity from 4.97 for the base model to 2.62 for the adapter.
- Built evaluation support for held-out perplexity, adapter inference, prompt
  regression, and future ablations over rank, alpha, target modules, and
  learning rate.
- Saved 5-prompt deterministic base-vs-adapter qualitative regression and
  documented remaining factuality and overclaiming limitations.
- Completed a small rank-4 ablation: 3.41M trainable parameters, 100 steps, and
  eval loss 1.0312.

## ATS Keywords

Mistral-7B, LLM Fine-Tuning, LoRA, QLoRA, PEFT, bitsandbytes, NF4, Double
Quantization, Paged AdamW, Gradient Checkpointing, Instruction Tuning, Alpaca,
Parameter-Efficient Fine-Tuning, Hugging Face Transformers, PyTorch,
Perplexity, Adapter Tuning, Generative AI.

## Claims To Avoid

- Do not claim MMLU, safety, or hallucination improvement until those specific
  benchmarks are measured.
- Do not claim MMLU improvement from Alpaca instruction tuning without measuring
  it.
- Do not commit adapter weights or checkpoints to git.

## Upgrade Path To 100/100

- Expand qualitative regression from 5 to 10-20 prompts with pass/fail rubrics.
- Run rank ablations for 4, 8, 16, and 32.
- Compare attention-only targets against all-linear targets.
- Record peak VRAM, training time, and tokens/sec.
