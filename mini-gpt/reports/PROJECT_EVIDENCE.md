# Mini-GPT Evidence Report

This report is generated from local JSON/CSV artifacts. Final rows use trained-checkpoint artifacts when present; smoke rows are labeled separately.

## Training And Evaluation

| Run | Dataset | Synthetic | Train loss | Val loss | Perplexity | Eval tok/s | Tokens trained |
|---|---|---:|---:|---:|---:|---:|---:|
| small_tinystories_continue_low_lr | tinystories | False | 1.6100 | 2.2848 | 9.82 | 47823.9 | 1018517 |

The best checkpoint was reevaluated on 99,625 validation tokens with 2,000 deterministic batch-bootstrap resamples; the 95% perplexity interval was 9.20-10.48.
| small_wikitext2 | wikitext-2 | False | 7.0267 | 7.2372 | 1390.22 | 16197.5 | 447594 |
| small_tinystories | tinystories | False | 2.7068 | 2.6565 | 14.25 | 47119.0 | 5092585 |
| small_tinystories_20m | tinystories | False | 1.9133 | 2.3053 | 10.03 | 46689.6 | 15277755 |

## KV Cache Benchmark

| Generated tokens | No-cache tok/s | KV-cache tok/s | Speedup |
|---:|---:|---:|---:|
| 32 | 138.2 | 122.9 | 0.89x |
| 64 | 127.0 | 133.0 | 1.05x |
| 128 | 134.1 | 125.9 | 0.94x |
| 256 | 127.9 | 124.9 | 0.98x |

## Inference Optimization

| Batch | Cache | Tok/s p50 | Latency p50 (s) | Latency p95 (s) | Peak GPU MB | KV cache MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | False | 132.1 | 0.9687 | 1.2531 | 184.1 | 0.7734 |
| 1 | True | 122.9 | 1.0418 | 1.1318 | 136.7 | 0.7734 |
| 2 | False | 262.3 | 0.9759 | 1.0548 | 234.5 | 1.5469 |
| 2 | True | 273.9 | 0.9346 | 1.0064 | 139.2 | 1.5469 |
| 4 | False | 458.8 | 1.1160 | 1.1806 | 336.0 | 3.0938 |
| 4 | True | 520.1 | 0.9844 | 1.0456 | 144.2 | 3.0938 |
| 8 | False | 446.5 | 2.2936 | 6.0782 | 539.7 | 6.1875 |
| 8 | True | 1014.4 | 1.0094 | 1.0277 | 155.2 | 6.1875 |

## Long-Prompt KV Cache Matrix

| Prompt tokens | Batch | Generated tokens | No-cache p95 (s) | KV-cache p95 (s) | Speedup | Memory saved MB |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 128 | 25.739 | 2.386 | 10.79x | 70.5 |
| 128 | 8 | 128 | 75.903 | 1.886 | 40.25x | 572.1 |
| 512 | 1 | 128 | 2.877 | 1.120 | 2.57x | 139.1 |
| 512 | 8 | 128 | 47.229 | 2.664 | 17.73x | 1120.1 |

## LoRA Smoke Fine-Tune

| Dataset | Synthetic | Trainable params | Total params | Trainable % | Final train loss |
|---|---:|---:|---:|---:|---:|
| synthetic toy instruction dataset | True | 3,584 | 3,295,104 | 0.109% | 10.6865 |

## Qualitative Samples

- Prompt `Once upon a time` -> Once upon a time, there was a little boy named Timmy. Timmy liked to play with his toy cars and trucks. One day, Timmy's mom told him they were going to a hospital. Timmy didn't want to take a bath because he was scared he could leave his r
- Prompt `Lily found a small` -> Lily found a small box in her room. It was shiny and had a shiny bow. She liked to put it on her shirt and pretend to be a real person. She loved her dress and wanted to use it to take it.  One day, Lily went to the kitchen. She saw her fri
- Prompt `Tom wanted to help` -> Tom wanted to help his mom in the kitchen. He ran to the kitchen and put some delicious vegetables in the oven. He put some oven in a oven and put some cheese in it. He put some clothes in the oven and put it on the oven.  He started to dra

## Qualitative Rubric

Automated 50-prompt proxy rubric average: 3.84 / 5 (coherence 5, repetition 4.16, entity consistency 2.94, ending completeness 3.32, grammar 3.78).

## Honesty Notes

- Tiny smoke metrics are intentionally labeled as smoke-test results.
- Final resume claims should use generated JSON artifacts and should distinguish best-checkpoint metrics from final-epoch metrics.
- Synthetic data is only used in the LoRA toy adapter smoke run unless explicitly enabled with `--allow-synthetic-fallback`.
- The qualitative rubric is an automated proxy, not a human preference evaluation.
