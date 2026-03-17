#!/usr/bin/env python3
"""Prepare stage1 train/eval splits from alpaca-gpt4 style data."""

import argparse
import json
import random
from pathlib import Path


def build_query(sample):
    instruction = (sample.get("instruction") or "").strip()
    input_text = (sample.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to alpaca-style json file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save split files.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--eval-size", type=int, default=100, help="Optional hard cap for eval set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON list.")

    cleaned = []
    for idx, sample in enumerate(data):
        instruction = (sample.get("instruction") or "").strip()
        output = (sample.get("output") or "").strip()
        if not instruction or not output:
            continue
        row = {
            "id": sample.get("id", idx),
            "instruction": instruction,
            "input": (sample.get("input") or "").strip(),
            "output": output,
        }
        cleaned.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    split_idx = int(len(cleaned) * args.train_ratio)
    train_data = cleaned[:split_idx]
    eval_data = cleaned[split_idx:]
    if args.eval_size > 0:
        eval_data = eval_data[:args.eval_size]

    eval_records = []
    for sample in eval_data:
        eval_records.append(
            {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "input": sample["input"],
                "reference": sample["output"],
                "query": build_query(sample),
            }
        )

    stats = {
        "input_path": str(input_path),
        "total_samples": len(cleaned),
        "train_samples": len(train_data),
        "eval_samples": len(eval_records),
        "seed": args.seed,
    }

    with (output_dir / "stage1_train.json").open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with (output_dir / "stage1_eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_records, f, ensure_ascii=False, indent=2)
    with (output_dir / "stage1_meta.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
