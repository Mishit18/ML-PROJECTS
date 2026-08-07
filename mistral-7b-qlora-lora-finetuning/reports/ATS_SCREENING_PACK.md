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
- Resumed the main run from checkpoint 600 and progressed beyond checkpoint 900
  / 1,200 target steps; validation loss improved from 1.0143 early in training
  to 0.9648 in the resumed run.
- Built evaluation support for held-out perplexity, adapter inference, prompt
  regression, and future ablations over rank, alpha, target modules, and
  learning rate.

## ATS Keywords

Mistral-7B, LLM Fine-Tuning, LoRA, QLoRA, PEFT, bitsandbytes, NF4, Double
Quantization, Paged AdamW, Gradient Checkpointing, Instruction Tuning, Alpaca,
Parameter-Efficient Fine-Tuning, Hugging Face Transformers, PyTorch,
Perplexity, Adapter Tuning, Generative AI.

## Claims To Avoid

- Do not claim the run is complete until it reaches 1,200 steps.
- Do not claim final perplexity or benchmark improvement until final evaluation
  is saved.
- Do not claim MMLU improvement from Alpaca instruction tuning without measuring
  it.
- Do not commit adapter weights or checkpoints to git.

## Upgrade Path To 100/100

- Finish the 1,200-step run and save final eval metrics.
- Add base model vs adapter perplexity on the same validation split.
- Add 10 fixed qualitative prompts with base/adapted outputs.
- Run rank ablations for 4, 8, 16, and 32.
- Compare attention-only targets against all-linear targets.
- Record peak VRAM, training time, and tokens/sec.
