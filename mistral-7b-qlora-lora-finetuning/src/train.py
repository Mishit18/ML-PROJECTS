from __future__ import annotations

import argparse
from pathlib import Path

from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from src.config import load_config, save_config
from src.data import load_and_format_dataset
from src.model import count_parameters, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory to resume optimizer, scheduler, and trainer state from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_and_format_dataset(config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        bf16=config.bnb_4bit_compute_dtype == "bfloat16",
        fp16=config.bnb_4bit_compute_dtype == "float16",
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=[] if config.report_to == "none" else [config.report_to],
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=config.packing,
        args=training_args,
    )

    stats = count_parameters(model)
    print(f"Parameter counts: {stats}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    save_config(config, Path(config.output_dir) / "resolved_config.json")


if __name__ == "__main__":
    main()
