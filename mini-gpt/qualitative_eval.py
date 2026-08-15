"""Generate and score qualitative samples with a simple transparent rubric.

The rubric is an automated proxy, not a substitute for human annotation. It is
useful for consistent before/after comparisons and interview discussion.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import load_sample_data
from inference.text_generation import load_model
from model.utils import get_device, set_seed
from tokenizer.tokenizer import create_tokenizer


NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
END_RE = re.compile(r"[.!?\"]\s*$")


def prompt_from_text(text, max_words=12):
    words = text.strip().split()
    return " ".join(words[:max_words])


def encode_prompt_batch(tokenizer, prompts, device):
    encoded = [torch.tensor(tokenizer.encode(prompt), dtype=torch.long) for prompt in prompts]
    max_len = max(len(ids) for ids in encoded)
    padded = [F.pad(ids, (0, max_len - len(ids)), value=tokenizer.pad_token_id) for ids in encoded]
    return torch.stack(padded).to(device)


def ngram_repetition_score(words, n=3):
    if len(words) < n * 2:
        return 5
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    ratio = repeated / max(1, len(grams))
    if ratio < 0.02:
        return 5
    if ratio < 0.05:
        return 4
    if ratio < 0.10:
        return 3
    if ratio < 0.20:
        return 2
    return 1


def score_completion(prompt, completion):
    text = completion.strip()
    words = re.findall(r"\b\w+\b", text)
    lower = text.lower()
    unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
    repetition = ngram_repetition_score([w.lower() for w in words])

    coherence = 5 if len(words) >= 50 and unique_ratio > 0.35 else 4 if len(words) >= 35 else 3 if len(words) >= 20 else 2
    if any(bad in lower for bad in ["<|endoftext|>", "�"]):
        coherence = max(1, coherence - 2)

    names = NAME_RE.findall(text)
    name_counts = Counter(names)
    entity_consistency = 5 if not names else 4
    if len(name_counts) > 4:
        entity_consistency = 3
    if len(name_counts) > 6:
        entity_consistency = 2

    ending_completeness = 5 if END_RE.search(text) else 3
    if text.endswith((",", "and", "the", "to", "a")):
        ending_completeness = 2

    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    grammar = 4 if sentences and all(len(s.split()) >= 3 for s in sentences[:5]) else 3
    if text.count('"') % 2 == 1:
        grammar = max(2, grammar - 1)

    return {
        "coherence": coherence,
        "repetition": repetition,
        "entity_consistency": entity_consistency,
        "ending_completeness": ending_completeness,
        "grammar": grammar,
        "average": mean([coherence, repetition, entity_consistency, ending_completeness, grammar]),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Qualitative generation rubric")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="small_tinystories")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--csv-output", default="reports/qualitative_scores_tinystories.csv")
    parser.add_argument("--json-output", default="reports/qualitative_scores_tinystories.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = create_tokenizer()
    model = load_model(args.checkpoint, device)
    _, val_texts, metadata = load_sample_data(
        tokenizer,
        num_train=0,
        num_val=max(args.num_prompts, 50),
        dataset_name="tinystories",
        return_metadata=True,
    )
    prompts = [prompt_from_text(text) for text in val_texts[: args.num_prompts]]

    rows = []
    for start in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[start : start + args.batch_size]
        input_ids = encode_prompt_batch(tokenizer, prompt_batch, device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True,
        )
        completions = tokenizer.batch_decode(output_ids.tolist())
        for offset, (prompt, completion) in enumerate(zip(prompt_batch, completions)):
            scores = score_completion(prompt, completion)
            rows.append({
                "id": start + offset,
                "prompt": prompt,
                "completion": completion,
                **scores,
            })

    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    score_keys = ["coherence", "repetition", "entity_consistency", "ending_completeness", "grammar", "average"]
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": "tinystories validation prompts",
        "synthetic": metadata["synthetic"],
        "num_prompts": len(rows),
        "rubric": "automated proxy scores from 1-5; higher is better",
        "mean_scores": {key: mean(row[key] for row in rows) for key in score_keys},
    }
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
