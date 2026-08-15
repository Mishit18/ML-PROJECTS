# Qualitative Generation Rubric

The project includes `qualitative_eval.py` to generate completions and score them into CSV/JSON artifacts.

Scores are automated proxy scores from 1-5:

| Dimension | What It Checks |
|---|---|
| Coherence | Sufficient length, no obvious broken tokens, basic story continuity proxy |
| Repetition | Repeated trigram ratio |
| Entity consistency | Number and stability of capitalized character names |
| Ending completeness | Whether the sample ends with sentence-like punctuation |
| Grammar | Basic sentence-length and quote-balance checks |

Target command:

```bash
python qualitative_eval.py \
  --checkpoint checkpoints/small_tinystories_20m/checkpoint_epoch_15.pt \
  --num-prompts 50 \
  --max-new-tokens 48 \
  --batch-size 10 \
  --csv-output reports/qualitative_scores_tinystories_20m.csv \
  --json-output reports/qualitative_scores_tinystories_20m.json
```

Status: the evaluator script is implemented, but the full 50-prompt generated CSV did not complete during the interactive audit pass. Do not claim qualitative rubric scores until the CSV/JSON files exist.
