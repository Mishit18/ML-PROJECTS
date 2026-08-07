from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import ExperimentConfig


def format_alpaca(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    if input_text:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt + output


def format_chatml(example: dict) -> str:
    conversations = example.get("conversations") or example.get("messages")
    if not conversations:
        raise ValueError("Expected OpenHermes-style 'conversations' or 'messages' field.")

    rendered: list[str] = []
    for message in conversations:
        role = message.get("from") or message.get("role")
        content = (message.get("value") or message.get("content") or "").strip()
        if role in {"human", "user"}:
            rendered.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role in {"gpt", "assistant"}:
            rendered.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "system":
            rendered.append(f"<|im_start|>system\n{content}<|im_end|>")
    return "\n".join(rendered)


def get_formatter(dataset_format: str) -> Callable[[dict], str]:
    if dataset_format == "alpaca":
        return format_alpaca
    if dataset_format in {"chatml", "openhermes"}:
        return format_chatml
    raise ValueError(f"Unsupported dataset_format={dataset_format!r}")


def load_and_format_dataset(config: ExperimentConfig) -> DatasetDict:
    dataset = load_dataset(config.dataset_name)
    split_name = "train" if "train" in dataset else next(iter(dataset))
    train_dataset = dataset[split_name]

    if config.max_train_samples:
        train_dataset = train_dataset.shuffle(seed=config.seed).select(range(config.max_train_samples))

    formatter = get_formatter(config.dataset_format)

    def render(example: dict) -> dict[str, str]:
        return {"text": formatter(example)}

    keep_columns = train_dataset.column_names
    formatted = train_dataset.map(render, remove_columns=keep_columns)
    split = formatted.train_test_split(test_size=config.validation_size, seed=config.seed, shuffle=True)
    return DatasetDict(train=split["train"], validation=split["test"])


def tokenize_for_lm(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> DatasetDict:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokens = tokenizer(
            batch["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
