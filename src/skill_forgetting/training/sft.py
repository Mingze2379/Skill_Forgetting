"""Sequential skill-augmented math SFT through an external LlamaFactory install."""
from __future__ import annotations

import argparse
from pathlib import Path
from skill_forgetting.training.single_sft import SFTSpec, train_once

from skill_forgetting.math.constants import TRAIN_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_root", type=Path, default=Path("runs/math"))
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--finetuning_type", choices=["full", "lora"], default="full")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_subdir", default="per_type_data_skill_augmented")
    parser.add_argument("--types", default=",".join(TRAIN_TYPES))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    model = args.model
    for index, task in enumerate(args.types.split(","), 1):
        data = args.run_root.resolve() / args.data_subdir / task
        destination = args.run_root.resolve() / "checkpoints" / f"step_{index:02d}_{task}"
        train_dir = destination.with_name(destination.name + "_adapter") if args.finetuning_type == "lora" else destination
        model = str(train_once(SFTSpec(
            model=model, data_dir=data, output_dir=train_dir,
            max_steps=args.max_steps, seed=args.seed,
            template=args.template, finetuning_type=args.finetuning_type,
            learning_rate=args.learning_rate, epochs=args.epochs,
            batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
            cutoff_len=args.cutoff_len,
            merge_dir=destination if args.finetuning_type == "lora" else None,
        ), dry_run=args.dry_run))


if __name__ == "__main__":
    main()
