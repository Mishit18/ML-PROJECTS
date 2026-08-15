# Mini-GPT Model Card

## Model

Mini-GPT is a small decoder-only causal language model implemented in PyTorch. The main modern configuration uses RoPE, RMSNorm, SwiGLU, Grouped Query Attention, PyTorch scaled dot-product attention, tied token/LM-head weights, and KV-cache decoding.

## Intended Use

This model is a learning, portfolio, and systems-benchmark artifact. It is intended to demonstrate transformer implementation, language-model evaluation, LoRA adapters, and inference benchmarking. It is not intended for production text generation or safety-critical use.

## Training Data

- TinyStories: final quality run and qualitative samples.
- WikiText-2: harder benchmark run for honest comparison.
- Synthetic toy instruction data: LoRA adapter smoke test only, explicitly labeled synthetic.

## Hardware

Measured runs were executed on an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch CUDA.

## Current Metrics

See `reports/PROJECT_EVIDENCE.md` for the generated evidence table. Current best TinyStories checkpoint:

- Pre-continuation best checkpoint tokens: 15,277,755
- Low-LR continuation tokens: 1,018,517
- Total staged training tokens to best checkpoint: approximately 16.3M
- Best validation loss: 2.2848
- Best validation perplexity: 9.82
- Best measured batched inference throughput: 1,014.4 tokens/sec

## Known Limitations

- The model is small and trained for a limited number of tokens compared with production LMs.
- Generated stories can be locally incoherent, repeat entities, or end abruptly.
- Automated qualitative rubric scores are proxy metrics, not human preference labels.
- KV-cache benefits are workload-dependent; single-sequence generation is often break-even at this model size.
- Checkpoints are ignored by Git to keep the repository lightweight.

## Safe Use

Do not use this model for medical, legal, financial, or safety-critical advice. Outputs should be treated as untrusted generated text.

## Failure Modes

- Contradictory character names or pronouns.
- Repetition in longer generations.
- Incomplete endings when the generation budget cuts off.
- Common-sense errors due to small scale and limited training.
