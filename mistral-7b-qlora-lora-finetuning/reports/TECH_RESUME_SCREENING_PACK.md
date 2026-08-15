# Tech Resume Screening Pack

## Resume Positioning

- Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA on Alpaca-cleaned, training 6.82M adapter parameters while freezing the base model.
- Implemented manual LoRA from first principles and production QLoRA with PEFT, bitsandbytes, double quantization, gradient checkpointing, and paged AdamW.
- Completed a 1,200-step single-GPU run on RTX 4060 Laptop GPU with about 7.5GB observed VRAM.
- Reduced held-out validation perplexity from 4.9675 to 2.6190 on 1,000 samples, a 47.28% reduction.
- Ran real 7B ablations for rank and target modules; rank-16 reached 1.0106 eval loss at 100 steps, rank-8 reached 1.0143, and q/v-only underperformed at 1.0353.

## Verified Evidence

- Test suite: 2/2 passed in 14.80s.
- Trainable parameters: 6,815,744.
- Trainable fraction: 0.0939%.
- Base eval loss/perplexity: 1.6029 / 4.9675.
- Adapter eval loss/perplexity: 0.9628 / 2.6190.
- Perplexity reduction: 47.28%.
- Final adapter: `outputs/mistral7b-alpaca-cleaned-qlora-r8`.

## Interview Defense

This project is strong because it proves memory-efficient fine-tuning of a real 7B-class model under consumer-GPU constraints. The defensible story is PEFT engineering: LoRA math, QLoRA memory savings, NF4 quantization, target-module selection, evaluation, and ablation discipline.

## Honest Scope

Do not claim MMLU improvement, hallucination reduction, production deployment, or full fine-tuning equivalence. Alpaca-cleaned improves instruction-following style and validation likelihood; it does not make the model a domain expert.

## Resume-Safe Bullets

- Fine-tuned Mistral-7B-v0.3 with rank-8 NF4 QLoRA, training 6.82M adapter parameters while freezing 99.9061% of the 7B-class base model.
- Implemented PEFT/bitsandbytes QLoRA with double quantization, gradient checkpointing, and paged AdamW; fit 1,200-step training under about 7.5GB VRAM.
- Reduced held-out validation perplexity from 4.97 to 2.62 on 1,000 samples and ablated LoRA rank/target modules across real 7B runs.
